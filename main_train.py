"""Main entry for federated training using pre-computed rational local efforts (NE).

Workflow (when executed directly):
 1. Parse CLI args / load YAML (may include a chosen game Case id).
 2. Load the NE case file (generated previously by main_game.py) which sets cfg.NE.
 3. Build dataset / server / clients objects.
 4. Run FL rounds (delegated to fed_game.federate.FL_train).

Note: If the NE case yaml is missing, an explicit error from utils.utils.load_NE_case
    instructs the user to run main_game.py first.
"""

import os

import yaml
from fed_game.federate import FL_train  # main FL training loop
from fed_game.server import Server
from fed_game.client import Clients
from rich.progress import Progress
from fed_game.dataset import init_dataset
from utils.args import args_parser_Train
from utils.utils import load_config, set_seed, load_NE_case


def main(cfg):
    """Execute the full FL training pipeline.

    Parameters
    ----------
    cfg : EasyDict
        Configuration (merged YAML + CLI) including: dataset/model/optimizer,
        federated parameters (m, T, frac, NE list), and runtime options.
    """
    # 1) Deterministic behavior for reproducibility
    set_seed(cfg.seed)

    # 2) Build core FL components
    dataset = init_dataset(cfg)  # dict with train/val/test + split_dict
    server = Server(cfg)  # holds global model & sampling logic
    clients = Clients(cfg, server.model, dataset)  # local update handler

    # 3) Launch federated optimization rounds
    FL_train(cfg, server, clients, dataset)


if __name__ == "__main__":
    os.system("clear")  # Clear terminal for a clean run log (macOS/Linux).
    # Load the configuration.
    arg = args_parser_Train()  # Parse federated training arguments.
    cfg = load_config(arg)  # Merge YAML + CLI into EasyDict.
    load_NE_case(cfg)  # Inject cfg.NE list (throws if not generated yet).
    with Progress() as progress:  # progress bar
        task = progress.add_task("[green]Main loop:", total=1)  # main loop bar
        cfg.progress = progress  # make available to inner routines (optional use)
        main(cfg)  # run training
        progress.update(task, advance=1)  # finalize outer task
