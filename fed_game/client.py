"""Client-side local training logic for FL algorithms (FedAvg / FedProx / MOON).

Highlights
---------
* Streams weighted aggregation: builds a single in-place weighted sum dict
    instead of retaining every local model (memory friendly for large m).
* Supports heterogeneous per-client epochs (cfg.NE list) derived from
    game-theoretic NE (loaded externally) or uniform integer.
* Optional MOON contrastive regularization with a small buffer of previous
    local models per client.
"""

import copy
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fed_game.dataset import DatasetSplit
from fed_game.federate import cal_accu
from utils.utils import *


class Clients(object):
    """Encapsulates all clients' local update workflow.

    Parameters
    ----------
    cfg : EasyDict
        Global experiment configuration.
    server_model : torch.nn.Module
        Current global model broadcast by the server.
    dataset : dict
        Dataset dict with keys: 'train', 'val', 'test', 'split_dict'.
    """

    def __init__(self, cfg, server_model, dataset):
        self.cfg = cfg  # scenario config
        self.device = cfg.device  # gpu or cpu
        self.datasets = dataset  # clients' datasets
        self._init_params(cfg, server_model)  # initialize parameters

    def local_update(self, model_server, selected_clients, t):
        """Execute local training for sampled clients and build weighted sum state_dict.

        Returns
        -------
        dict
            Aggregated (already weighted) parameter tensors ready to load
            directly into the server model.
        """
        self.model_server = model_server
        self.t = t
        rho = active_rho(self.cfg.rho, selected_clients)
        # Initialize new_model for aggregation
        global_model_inplace = {}
        for key, value in model_server.state_dict().items():
            global_model_inplace[key] = torch.zeros_like(value)

        for idx, self.i in enumerate(selected_clients):
            # Load client i's local settings
            self.model_i = copy.deepcopy(self.model_server).train()
            self.dataset_i = DatasetSplit(self.datasets, self.i)
            data_loader = init_data_loader(self.cfg, self.dataset_i)
            optimizer = init_optimizer(self.cfg, self.model_i, self.t)
            count_E = 0  # completed epochs
            if self.cfg.alg == "moon":
                self.prev_models_active = self._prepare_prev_models(self.i)
            else:
                self.prev_models_active = []
            # Start local training
            if isinstance(self.cfg.NE, int):  # expand scalar NE only once per round
                self.cfg.NE = [self.cfg.NE for _ in range(self.cfg.m)]
            while count_E < self.cfg.NE[self.i]:  # for each epoch
                for batch in data_loader:  # iterate over batches
                    if isinstance(batch[0], list):  # two items for NLP dataset
                        self.data = [item.to(self.device) for item in batch[0]]
                    else:
                        self.data = batch[0].to(self.device)
                    self.labels = batch[1].to(self.device)
                    self._local_step(self.cfg)  # perform single FGD/SGD step
                    optimizer.step()  # update client's model parameters u
                count_E += 1
            if self.cfg.alg == "moon":
                self._release_prev_models(self.prev_models_active)
                self._store_prev_model_state(self.i, self.model_i)
                self.prev_models_active = []
            # In-place streaming aggregation (FedAvg weighted sum)
            for key in global_model_inplace:
                if "num_batches_tracked" in key:  # for BN batches count
                    continue
                elif "running" in key:  # for BN running mean/var
                    global_model_inplace[key] += rho[idx] * self.model_i.state_dict()[key]
                else:  # for other layers
                    global_model_inplace[key] += rho[idx] * self.model_i.state_dict()[key]

            self.steps[self.i][self.t] = count_E
            self.res_dict["steps"] = self.steps  # record local FGD/SGD steps

        return global_model_inplace

    def _local_step(self, cfg):
        """One backward pass per mini-batch under selected FL algorithm."""
        if cfg.alg == "fedavg":
            self._grad(self.model_i, self.data, self.labels)
        elif cfg.alg in ["fedprox"]:
            self._fedprox_u(self.model_i)
        elif cfg.alg in ["moon"]:
            self._moon_step(self.model_i)
        else:
            raise ValueError(f"Invalid algorithm.")

    def _grad(self, model, data, labels):
        """Plain supervised gradient calculation (no optimizer step)."""
        model.train()
        model.zero_grad()
        pred = model(data)
        loss = model.loss(pred, labels)  # default reduction == "mean"
        loss.backward()  # compute the gradients of f_i(u_i)

    def _fedprox_u(self, model):
        """FedProx gradient: vanilla grad + proximal term mu(u_i - z)."""
        self._grad(model, self.data, self.labels)
        u_state = model.state_dict()  # u_i
        z_state = self.model_server.state_dict()  # z
        for name, param in model.named_parameters():
            param.grad += self.cfg.mu * (u_state[name] - z_state[name])

    def _moon_step(self, model):
        """MOON step: classification loss + contrastive loss vs global & cached past models."""
        model.train()
        model.zero_grad()
        logits_local, feat_local = self._logits_and_features(model, self.data)
        loss_cls = model.loss(logits_local, self.labels)

        with torch.no_grad():
            was_training = self.model_server.training
            self.model_server.eval()
            _, feat_global = self._logits_and_features(self.model_server, self.data)
            if was_training:
                self.model_server.train()

        feat_local = feat_local.reshape(feat_local.size(0), -1)
        feat_global = feat_global.reshape(feat_global.size(0), -1)

        logits = F.cosine_similarity(feat_local, feat_global, dim=-1).unsqueeze(1)

        for prev_model in self.prev_models_active:
            with torch.no_grad():
                _, feat_prev = self._logits_and_features(prev_model, self.data)
            feat_prev = feat_prev.reshape(feat_prev.size(0), -1)
            neg = F.cosine_similarity(feat_local, feat_prev, dim=-1).unsqueeze(1)
            logits = torch.cat((logits, neg), dim=1)

        logits /= self.cfg.moon_temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        loss_contrast = self.cfg.moon_mu * self.contrastive_loss(logits, labels)
        (loss_cls + loss_contrast).backward()

    def _logits_and_features(self, model, data):
        """Return (logits, penultimate_features). If no Linear layer, features==logits."""
        last_linear = self._get_last_linear(model)
        if last_linear is None:
            logits = model(data)
            return logits, logits

        features = {}

        def _pre_hook(_, inputs):
            feat = inputs[0] if isinstance(inputs, tuple) else inputs
            features["value"] = feat

        handle = last_linear.register_forward_pre_hook(_pre_hook)
        logits = model(data)
        handle.remove()
        feat = features.get("value")
        if isinstance(feat, tuple):
            feat = feat[0]
        return logits, feat

    def _get_last_linear(self, model):
        """Scan modules to find last nn.Linear (None if absent)."""
        last_linear = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        return last_linear

    def _prepare_prev_models(self, client_id):
        """Instantiate previous local model snapshots on active device for contrastive use."""
        if not self.prev_model_states:
            return []
        models = []
        for state in self.prev_model_states[client_id]:
            model = copy.deepcopy(self.model_server)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            models.append(model)
        return models

    def _release_prev_models(self, models):
        """Return previous snapshot models to CPU (memory hygiene)."""
        for model in models:
            model.to("cpu")

    def _store_prev_model_state(self, client_id, model):
        """Save detached CPU copy of current local model (MOON buffer)."""
        if self.prev_model_states is None:
            return
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.prev_model_states[client_id].append(state)

    def _init_params(self, cfg, model_server):
        """Allocate tracking arrays & initialize (optional) MOON buffers."""
        self.steps = np.full((cfg.m, cfg.T + 1), 0)  # to save local FGD/SGD steps
        self.accus = np.full((cfg.m, cfg.T + 1), 0, dtype=float)  # local model accuracy
        self.accus_glob = np.full((cfg.m, cfg.T + 1), 0, dtype=float)  # local model accuracy
        self.contrastive_loss = None
        self.prev_models_active = []
        if cfg.alg == "moon":
            if not hasattr(cfg, "moon_mu"):
                cfg.moon_mu = 1.0
            if not hasattr(cfg, "moon_temperature"):
                cfg.moon_temperature = 0.5
            if not hasattr(cfg, "moon_buffer_size"):
                cfg.moon_buffer_size = 1
            buffer_size = max(1, int(cfg.moon_buffer_size))
            self.prev_model_states = [deque(maxlen=buffer_size) for _ in range(cfg.m)]
            self.contrastive_loss = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            self.prev_model_states = None
        if cfg.T0 > 0:  # load previous results
            res_path = os.path.join(cfg.dir_res, f"0_results.npy")
            res_dict = np.load(res_path, allow_pickle=True).tolist()
            try:
                self.steps[:, : cfg.T0] = res_dict["steps"][:, : cfg.T0]
                self.accus[:, : cfg.T0] = res_dict["accu_clients"][:, : cfg.T0]
                self.accus_glob[:, : cfg.T0] = res_dict["accu_clients_glob"][:, : cfg.T0]
            except:
                pass


###############################################################################
# Auxiliary helpers
###############################################################################
def init_optimizer(cfg, model, t):
    """Build optimizer with round-wise power LR decay (cfg.lr_decay) if provided."""

    lr = cfg.lr * np.power(cfg.get("lr_decay", 1), t - 1)
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=cfg.w_decay)
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def init_data_loader(cfg, dataset_i, shuffle=True):
    """Create DataLoader; if cfg.bs falsy -> Full Gradient (single batch)."""
    if cfg.bs:  # SGD
        data_loader = DataLoader(dataset_i, batch_size=cfg.bs, shuffle=shuffle)
    else:  # FGD
        data_loader = DataLoader(dataset_i, batch_size=len(dataset_i), shuffle=shuffle)
    return data_loader
