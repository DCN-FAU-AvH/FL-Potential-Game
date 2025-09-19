"""Compute NE strategies via Best-Response for selected reward factors (lambda).

Outputs:
    * raw_data/lambda-*/0_res_dict.npy per lambda (trajectory + final s)
    * 0_res_avg_s.npy average effort curve
    * Case YAMLs for downstream training (main_train.py)
"""

import os
import numpy as np
from collections import defaultdict
from fed_game.game import PotentialGame
from utils.utils import *
from utils.args import args_parser_NE

# Load CLI arguments and config
args = args_parser_NE()
cfg = load_config(args)

# Setup experiment directories, logging, and random seed
cfg.tag = get_label(cfg)
dir_res = create_folder(cfg)  # dir for results
dir_res_BR = os.path.join(dir_res, "raw_data")  # dir for BR results
log_trial = get_logger(dir_res)
set_seed(cfg.seed)
res_dict = defaultdict(list)

# Construct and solve the potential game
FL_GAME = PotentialGame(cfg, res_dict)
four_cases_lambda = cal_lambdas(dir_res, res_dict, plot=True)

get_s_at_four_cases = 1  # only compute NE at the four cases
# Build the test list for the reward factor (lambda)
if get_s_at_four_cases:  # only compute NE at the four cases
    cfg.lamda_list = four_cases_lambda
else:  # full sweep
    start, stop, interval = cfg.lamda[0], larger(cfg.lamda[1]), cfg.lamda[2]
    cfg.lamda_list = np.round(np.arange(start, stop, interval), 5).tolist()
s_avg = np.zeros(len(cfg.lamda_list))  # container for average efforts per lambda
save_config(cfg, dir_res)

# Main loop over lambda values
for idx_lamda, cfg.lamda in enumerate(cfg.lamda_list):
    # Setup: fix randomness and prepare logging
    set_seed(cfg.seed)
    cfg.dir_res = os.path.join(dir_res_BR, f"lambda-{cfg.lamda}")
    os.makedirs(cfg.dir_res)
    log_trial.info(f"Reward factor : {cfg.lamda:.3}")
    FL_GAME.log = get_logger(cfg.dir_res, log_to_console=False)
    FL_GAME.res_dict = defaultdict(list)
    FL_GAME.cfg = cfg

    # NE via best-response algorithm
    s_star = FL_GAME.best_response()

    # Aggregate and report
    s_avg[idx_lamda] = np.average(s_star)
    log_trial.info(f"Average effort: {np.average(s_star):.4}\n")

# Persist artifacts
np.save(os.path.join(dir_res, "0_res_avg_s.npy"), s_avg)
plot_NE(cfg, dir_res, s_avg, save_cases=1)
