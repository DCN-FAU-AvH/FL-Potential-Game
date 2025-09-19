"""FL-Game potential formulation and best-response solver.

This module defines the potential game for FL and provides an implementation
of the best-response dynamics to compute (approximate) Nash equilibria.
"""

import copy, os
import yaml
import numpy as np
from utils.utils import larger, save_config


class PotentialGame(object):
    """FL potential game and best-response routine."""

    def __init__(self, cfg, res_dict):
        self.cfg = cfg
        self.res_dict = res_dict
        self.D = self._init_dataset()  # dataset sizes per client
        self.q = self._init_reso_q()  # per-client minimal efforts q
        self.Q = self._init_reso_Q(self.q)  # per-client maximal efforts Q
        self.alpha = self._init_alpha()  # per-client running cost coefficient

    def _init_dataset(self):
        """Initialize each client's dataset size and compute rho.

        Priority order:
          1. If cfg.femnist_D exists and the file is present, load its contents (key=client_id, value=sample count).
          2. Otherwise, sample uniformly within self.cfg.data_vol.
        """
        if self.cfg.femnist_D:  # external yaml for FEMNIST provided
            with open(self.cfg.femnist_D, "r") as f:
                D_dict = yaml.safe_load(f) or {}
            D_list = []
            for idx in range(self.cfg.m):
                D_list.append(int(D_dict[idx]))
            D = np.array(D_list[: self.cfg.m])
        else:  # Random generation within specified range
            lower, upper = self.cfg.data_vol
            D = np.random.randint(lower, upper + 1, self.cfg.m)
        self.res_dict["data_vol"] = D
        self.rho = D / np.sum(D)  # normalized weights by data size
        self.res_dict["rho"] = self.rho.tolist()
        return D

    def _init_alpha(self):
        """Sample each client's local cost coefficient alpha_i."""
        lower = self.cfg.alpha[0]
        upper = self.cfg.alpha[1]
        alpha = np.random.uniform(lower, larger(upper), self.cfg.m).tolist()
        self.res_dict["alpha"] = alpha
        return alpha

    def _init_reso_q(self):
        """Sample each client's minimal effort q_i."""
        q_min = self.cfg.reso_q[0]
        q_max = self.cfg.reso_q[1]
        reso_q = np.random.uniform(q_min, larger(q_max), self.cfg.m)
        self.res_dict["reso_q"] = reso_q
        return reso_q

    def _init_reso_Q(self, reso_q):
        """Sample each client's maximal allowed effort Q_i (Q_i >= q_i)."""
        m = self.cfg.m
        Q_min = self.cfg.reso_Q[0]
        Q_max = self.cfg.reso_Q[1]
        reso_Q = np.zeros(m)
        for i in range(m):
            Q_i_min = max(reso_q[i], Q_min)  # ensure Q_i >= q_i
            reso_Q[i] = np.random.uniform(Q_i_min, larger(Q_max))
        self.res_dict["reso_Q"] = reso_Q
        return reso_Q

    def _init_s(self):
        """Initialize s0 for the first best-response iteration."""
        s = np.zeros((self.cfg.m, self.cfg.T))  # shape (m, T)
        for i in range(self.cfg.m):
            s[i] = np.random.uniform(self.q[i], self.Q[i])
        # Seed traces with the initial profile
        P_FL = self._potential(s)
        epsi_NE = self._epsilon_NE(s)
        self.res_dict["P_FL"].append(P_FL)
        self.res_dict["epsi_NE"].append(epsi_NE)
        self.res_dict["s"].append(copy.deepcopy(s))
        for i in range(self.cfg.m):
            P_i_max = self._payoff(s, i)
            self.res_dict[f"p_{i}"].append(P_i_max)
        return s

    def _best_response(self, s, i):
        """
        Return the best response of client i for a fixed s_{-i}.
        """
        lamda = self.cfg.lamda
        rho = self.rho
        s_avg = np.sum(s * rho[:, np.newaxis], axis=0)[0]
        s_avg_exclude_i = s_avg - rho[i] * s[i][0]
        s_tmp = (lamda * s_avg_exclude_i) / (2 * (self.alpha[i] - lamda * rho[i]))  # maximizer
        if self.alpha[i] - lamda * rho[i] <= 0:
            # P_i becomes convex; the quadratic opens upward and the argmax hits the boundary.
            s[i] = self.Q[i]
        else:
            s[i] = min(max(self.q[i], s_tmp), self.Q[i])  # project to [q_i, Q_i]
        return s

    def best_response(self):
        """Run best-response dynamics until epsilon-NE or max-iterations."""
        save_config(self.cfg, self.cfg.dir_res)  # save cfg for this run
        s = self._init_s()
        self.log.info(f"Strategy s0: {np.round(s[:,0],1)}\n")
        iter_BR = 1
        while True:
            self.log.info(f"\nBRA iter: {iter_BR}")
            for i in range(self.cfg.m):
                s = self._best_response(s, i)
                P_i_max = self._payoff(s, i)
                self.res_dict[f"p_{i}"].append(P_i_max)
            P_FL = self._potential(s)
            epsi_NE = self._epsilon_NE(s)
            self.res_dict["P_FL"].append(P_FL)
            self.res_dict["epsi_NE"].append(epsi_NE)
            self.res_dict["s"].append(copy.deepcopy(s))
            self.log.info(f"s_avg: {np.mean(s):.5}")
            self.log.info(f"Epsilon_NE: {epsi_NE}")
            # self.log.info(f"Potential: {P_FL:.3}")
            self.log.info(f"Strategy: {np.round(s[:5,0],2)}")
            if epsi_NE < self.cfg.stop_e or iter_BR >= self.cfg.stop_iter:
                self.log.info(f"End BRA with {iter_BR} iterations.\n")
                break
            iter_BR += 1
        # Save final profile; round to avoid precision issues in serialization
        self.res_dict["s"][-1] = np.round(self.res_dict["s"][-1], decimals=5)
        dir_res_dict = os.path.join(self.cfg.dir_res, "0_res_dict.npy")
        np.save(dir_res_dict, np.asarray(self.res_dict, dtype=object))
        return s

    def _unit_price(self, s, t):
        """Unit price p_t as a function of average effort at round t."""
        p = self.cfg.lamda * np.average(s[:, t])
        return p

    def _cmp_cost(self, s_i_t):
        """Local computational cost term for a single effort value."""
        return s_i_t**2

    def _payoff(self, s, i):
        """Client i's payoff given joint effort profile s."""
        P_i = 0
        for t in range(self.cfg.T):
            s_i_t = s[i][t]
            P_i += self._unit_price(s, t) * s_i_t  # revenue
            P_i -= self.alpha[i] * self._cmp_cost(s_i_t)  # cost
        return P_i

    def _potential(self, s):
        """Potential function value for joint effort profile s."""
        P = 0
        for i in range(self.cfg.m):
            for t in range(self.cfg.T):
                s_i_t = s[i][t]
                P += (self.cfg.lamda / 2 / self.cfg.m - self.alpha[i]) * (s_i_t**2)
        for t in range(self.cfg.T):
            P += self.cfg.lamda / 2 / self.cfg.m * np.square(np.sum(s[:, t]))
        return P

    def _epsilon_NE(self, s_ori):
        """Maximum unilateral improvement over all players (epsilon-NE gap)."""
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
