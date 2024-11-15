# Server aggregation process.

import copy
import numpy as np
from fed_game.net import init_model


class Server(object):
    """Server side operations."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = init_model(cfg)  # model z_k
        self.state = self.model.state_dict()  # state z_k

    def select_clients(self, frac):
        """Randomly select a subset of clients."""
        num = max(np.ceil(frac * self.cfg.m).astype(int), 1)  # number of clients to sample
        self.active_clients = np.random.choice(range(self.cfg.m), num, replace=False)
        return self.active_clients

    def aggregate(self, res_clients: dict):
        """Server aggregation process."""
        rho = self.cfg.rho
        model_u = res_clients["models"]  # list of clients' models
        model_z = copy.deepcopy(model_u[0])  # init model_server_new
        m = len(model_u)  # number of clients to aggregate
        # Iterate over model layers for aggregation.
        for key in model_z.keys():
            model_z[key].zero_()  # reset model parameters
            # FedAvg server aggregation.
            if self.cfg.alg in ["fedavg"]:
                for i in range(m):  # iterate over clients
                    model_z[key] += rho[i] * model_u[i][key]
            else:
                raise ValueError(f"Invalid algorithm.")
        self.model.load_state_dict(model_z)  # update server's model
