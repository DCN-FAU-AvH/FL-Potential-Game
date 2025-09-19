"""Server-side orchestration: sampling clients and aggregating model updates.

Supports FedAvg-style weighted aggregation. The Clients.local_update method can
return either:
    * a pre-weighted in-place summed state_dict (fast path), or
    * a list of raw client state_dicts (legacy path) for explicit aggregation here.
"""

import copy
import numpy as np
from fed_game.net import init_model
from utils.utils import *


class Server(object):
    """Holds global model and performs client sampling + weighted aggregation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = init_model(cfg)  # model z_k
        self.state = self.model.state_dict()  # state z_k

    def select_clients(self, frac):
        """Uniformly sample max(1, ceil(frac * m)) distinct client IDs."""
        num = max(np.ceil(frac * self.cfg.m).astype(int), 1)  # number of clients to sample
        self.active_clients = np.random.choice(range(self.cfg.m), num, replace=False)
        return self.active_clients

    def aggregate(self, model_u: dict):
        """Incorporate client updates into global model state.

        Parameters
        ----------
        model_u : dict | list[dict]
            - dict: already an aggregated weighted sum produced client-side.
            - list: individual client state_dicts to be combined here.
        """
        if not isinstance(model_u, list):  # updated in the client.py
            self.model.load_state_dict(model_u)  # update server's model
        else:
            rho = active_rho(self.cfg.rho, self.active_clients)
            model_z = copy.deepcopy(model_u[0])  # init model_server_new
            m = len(model_u)  # number of clients to aggregate
            # Iterate over model layers for aggregation.
            for key in model_z.keys():
                model_z[key].zero_()  # reset model parameters
                if "num_batches_tracked" in key:  # for BN batches count
                    continue
                elif "running" in key:  # for BN running mean/var
                    for i in range(m):
                        model_z[key] += rho[i] * model_u[i][key]
                else:  # for other layers
                    # FedAvg server aggregation.
                    if self.cfg.alg in ["fedavg", "fedprox", "moon"]:
                        for i in range(m):  # iterate over clients
                            model_z[key] += rho[i] * model_u[i][key]
                    else:
                        raise ValueError(f"Invalid algorithm.")
            self.model.load_state_dict(model_z)  # update server's model
