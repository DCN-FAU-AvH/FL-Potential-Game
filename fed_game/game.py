# FL-Game and best-response algorithm.

import copy, os
import numpy as np
from utils.utils import larger, save_config


class PotentialGame(object):
    """Initialize the FL potential game."""

    def __init__(self, cfg, res_dict, log):
        self.cfg = cfg
        self.res_dict = res_dict
        self.log = log
        self.D = self._init_dataset()  # dataset size for clients
        self.q = self._init_reso_q(cfg.reso_distri)  # clients' minimal efforts q
        self.Q = self._init_reso_Q(self.q, cfg.reso_distri)  # clients' max efforts Q
        self.alpha = self._init_alpha()  # runn cost coefficient for clients
        self.res_dict["cfg"] = self.cfg
        save_config(cfg, cfg.dir_res)

    def _init_dataset(self):
        """Initialize each client's dataset size."""
        lower = self.cfg.data_vol[0]
        upper = self.cfg.data_vol[1]
        D = np.random.randint(lower, upper + 1, self.cfg.m)
        self.res_dict["data_vol"] = D
        return D

    def _init_alpha(self):
        """Initialize each client's local cost coefficient alpha_i."""
        lower = self.cfg.alpha[0]
        upper = self.cfg.alpha[1]
        alpha = np.random.uniform(lower, larger(upper), self.cfg.m).tolist()
        self.res_dict["alpha"] = alpha
        return alpha

    def _init_reso_q(self, case="uniform"):
        """Initialize each client's minimal effort q_i."""
        q_min = self.cfg.reso_q[0]
        q_max = self.cfg.reso_q[1]
        if case == "uniform":
            reso_q = np.random.uniform(q_min, larger(q_max), self.cfg.m)
        self.res_dict["reso_q"] = reso_q
        return reso_q

    def _init_reso_Q(self, reso_q, case="uniform"):
        """Initialize each client's maximal resource Q_i."""
        m = self.cfg.m
        Q_min = self.cfg.reso_Q[0]
        Q_max = self.cfg.reso_Q[1]
        reso_Q = np.zeros(m)  # maximal efforts for clients
        if case == "uniform":
            for i in range(m):
                Q_i_min = max(reso_q[i], Q_min)  # to ensure Q_i >= q_i
                reso_Q[i] = np.random.uniform(Q_i_min, larger(Q_max))
        self.res_dict["reso_Q"] = reso_Q
        return reso_Q

    def _init_s(self):
        """Initialize s0 for the first BR iteration."""
        s = np.zeros((self.cfg.m, self.cfg.T))
        for i in range(self.cfg.m):
            s[i] = np.random.uniform(self.q[i], self.Q[i])
        # save results
        P_FL = self._potential(s)
        epsi_NE = self._epsilon_NE(s)
        self.res_dict["P_FL"].append(P_FL)  # potential
        self.res_dict["epsi_NE"].append(epsi_NE)  # epsilon NE
        self.res_dict["s"].append(copy.deepcopy(s))  # strategy
        for i in range(self.cfg.m):
            P_i_max = self._payoff(s, i)
            self.res_dict[f"p_{i}"].append(P_i_max)  # save payoff
        return s

    def _best_response(self, s, i):
        """
        Return the best strategy of the i-th client in the continuous homo game.
        """
        lamda = self.cfg.lamda
        rho = self.D / sum(self.D)
        s_avg = np.sum(s * rho[:, np.newaxis], axis=0)[0]
        s_avg_exclude_i = s_avg - rho[i] * s[i][0]
        s_tmp = (lamda * s_avg_exclude_i) / (2 * (self.alpha[i] - lamda * rho[i]))  # maxmizer
        s[i] = min(max(self.q[i], s_tmp), self.Q[i])  # consider constrains
        return s

    def best_response(self):
        """Best-response algorithm for obtaining the NE."""
        s = self._init_s()
        iter_BR = 1
        while True:
            for i in range(self.cfg.m):
                s = self._best_response(s, i)
                P_i_max = self._payoff(s, i)
                self.res_dict[f"p_{i}"].append(P_i_max)  # payoff
            P_FL = self._potential(s)
            epsi_NE = self._epsilon_NE(s)
            self.res_dict["P_FL"].append(P_FL)  # potential
            self.res_dict["epsi_NE"].append(epsi_NE)  # epsilon NE
            self.res_dict["s"].append(copy.deepcopy(s))  # strategy
            if epsi_NE < self.cfg.stop_e or iter_BR >= self.cfg.stop_iter:
                break
            iter_BR += 1
        # save results
        dir_res_dict = os.path.join(self.cfg.dir_res, "0_res_dict.npy")
        np.save(dir_res_dict, np.asarray(self.res_dict, dtype=object))
        return s

    def _unit_price(self, s, t):
        """Calculate the unit price."""
        p = self.cfg.lamda * np.average(s[:, t])
        return p

    def _cmp_cost(self, s_i_t):
        """Calculate the local computational cost."""
        return s_i_t**2

    def _payoff(self, s, i):
        """Calculate the payoff function."""
        P_i = 0
        for t in range(self.cfg.T):
            s_i_t = s[i][t]
            P_i += self._unit_price(s, t) * s_i_t  # running reward
            P_i -= self.alpha[i] * self._cmp_cost(s_i_t)  # running cost
        return P_i

    def _potential(self, s):
        """Calculate the potential function."""
        P = 0
        for i in range(self.cfg.m):
            for t in range(self.cfg.T):
                s_i_t = s[i][t]
                P += (self.cfg.lamda / 2 / self.cfg.m - self.alpha[i]) * (s_i_t**2)
        for t in range(self.cfg.T):
            P += self.cfg.lamda / 2 / self.cfg.m * np.square(np.sum(s[:, t]))
        return P

    def _epsilon_NE(self, s_ori):
        """Calculate the epsilon NE."""
        epsilon_max = 0
        for i in range(self.cfg.m):
            s = copy.deepcopy(s_ori)
            payoff_old = self._payoff(s, i)
            s = self._best_response(s, i)
            payoff_best = self._payoff(s, i)
            epsilon = payoff_best - payoff_old
            if epsilon > epsilon_max:
                epsilon_max = epsilon
        return epsilon_max
