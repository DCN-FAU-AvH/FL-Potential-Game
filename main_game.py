# Calculate the Nash equilibrium of the FL-Game using the best-response algorithm.

import os
import numpy as np
from collections import defaultdict
from fed_game.game import PotentialGame
from utils.utils import *
from utils.args import args_parser_NE

# load args and config
args = args_parser_NE()
cfg = load_config(args)

# create a folder to save the results
cfg.tag = get_label(cfg)
dir_res = create_folder(cfg)  # dir for results

# range of the reward factor lambda to test
start, stop, interval = cfg.lamda[0], larger(cfg.lamda[1]), cfg.lamda[2]
cfg.lamda_list = np.arange(start, stop, interval).tolist()
s_avg = np.zeros(len(cfg.lamda_list))  # array to save results
save_config(cfg, dir_res)

# main
dir_res_BR = os.path.join(dir_res, "raw_data")  # dir for BR resutls
log_trial = get_logger(dir_res)
for idx_lamda, cfg.lamda in enumerate(cfg.lamda_list):
    # setup
    set_seed(cfg.seed)  # set seed
    log_trial.info(f"Reward factor : {cfg.lamda:.3}")
    res_dict = defaultdict(list)  # dict to save results
    cfg.dir_res = os.path.join(dir_res_BR, f"lambda-{cfg.lamda}")
    os.makedirs(cfg.dir_res)
    log_test = get_logger(cfg.dir_res)  # get logger

    # game scenario
    FL_GAME = PotentialGame(cfg, res_dict, log_test)
    s_star = FL_GAME.best_response()  # get the NE using the best-response algorithm

    # save results
    s_avg[idx_lamda] = np.average(s_star)
    log_trial.info(f"Average effort: {np.average(s_star):.4}\n")

# plot resutls
plot_NE(cfg, dir_res, res_dict, s_avg)
np.save(dir_res + f"/0_res_avg_s.npy", s_avg)
