# Clients' local training processes.

import copy, torch
import numpy as np
from torch.utils.data import DataLoader
from fed_game.dataset import DatasetSplit


class Clients(object):
    """Clients' local update processes."""

    def __init__(self, cfg, server_model, dataset):
        self.cfg = cfg  # scenario config
        self.device = cfg.device  # gpu or cpu
        self.datasets = dataset  # clients' datasets
        self._init_params(cfg, server_model)  # initialize parameters

    def local_update(self, model_server, selected_clients, t):
        """Clients' local update processes."""
        self.model_server = model_server
        self.t = t
        for self.i in range(self.cfg.m):
            if self.i in selected_clients:
                # Load client i's local settings
                self.model_i = copy.deepcopy(self.model_server).train()
                self.dataset_i = DatasetSplit(self.datasets, self.i)
                data_loader = init_data_loader(self.cfg, self.dataset_i)
                optimizer = init_optimizer(self.cfg, self.model_i)
                count_s, count_e = 0, 0  # finished SGD steps or epochs
                # Start local training
                train_flag = True  # flag for keep local training
                while train_flag:  # update client i's model parameters u_i
                    # check the criterion after each epoch
                    if self._stop_check(count_e):
                        break
                    for data, labels in data_loader:  # iterate over batches
                        self.data, self.labels = data.to(self.device), labels.to(self.device)
                        self._local_step(self.cfg)  # perform single FGD/SGD step
                        optimizer.step()  # update client's model parameters u
                        count_s += 1
                    count_e += 1
                # Personalized actions of different algorithms.
                self.models[self.i] = self.model_i.state_dict()
                self.steps[self.i][self.t] = count_e
                self.res_dict["steps"] = self.steps  # record local FGD/SGD steps
        return self.send_back

    def _stop_check(self, count):
        """Determine whether to stop client's local training."""
        stop_training = False
        # Check the maximum number.
        if self.cfg.stg == "epoch":
            # In FL-Game, each client's rational effort based on the NE.
            if count >= self.cfg.NE[self.i]:
                stop_training = True
        return stop_training

    def _local_step(self, cfg):
        """Client i performs single FGD/SGD step."""
        if cfg.alg in ["fedavg"]:
            self._grad(self.model_i, self.data, self.labels)

    def _grad(self, model, data, labels):
        """Calculate the gradients of the local update."""
        model.train()
        model.zero_grad()
        pred = model(data)
        loss = model.loss(pred, labels)  # default reduction == "mean"
        loss.backward()  # compute the gradients of f_i(u_i)

    def _init_params(self, cfg, model_server):
        """Initialize parameters and hyperparameters."""
        model_state = model_server.state_dict()
        self.models = [copy.deepcopy(model_state) for _ in range(cfg.m)]
        self.send_back = {}  # Results to send back to the server
        self.send_back["models"] = self.models  # local model parameters
        self.steps = np.full((cfg.m, cfg.T + 1), 0)  # to save local FGD/SGD steps
        self.accus = np.full((cfg.m, cfg.T + 1), 0, dtype=float)  # local model accuracy


# auxiliary functions
def init_optimizer(cfg, model):
    """Initialize the optimizer."""
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    return optimizer


def init_data_loader(cfg, dataset_i, shuffle=True):
    """Initialize the dataloader."""
    if cfg.bs:  # SGD
        data_loader = DataLoader(dataset_i, batch_size=cfg.bs, shuffle=shuffle)
    else:  # FGD
        data_loader = DataLoader(dataset_i, batch_size=len(dataset_i), shuffle=shuffle)
    return data_loader
