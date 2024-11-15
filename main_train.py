# Conduct FL training with clients' rational efforts

from fed_game.federate import FL_train
from fed_game.server import Server
from fed_game.client import Clients
from rich.progress import Progress
from fed_game.dataset import init_dataset
from utils.args import args_parser_Train
from utils.utils import load_config, set_seed


def main(cfg):
    # Setup the random seed.
    set_seed(cfg.seed)

    # Initialize dataset, server and clients.
    dataset = init_dataset(cfg)  # Initialize datasets.
    server = Server(cfg)  # Initialize the server.
    clients = Clients(cfg, server.model, dataset)  # Initialize clients.

    # Start FL training.
    FL_train(cfg, server, clients, dataset)


if __name__ == "__main__":
    # Load the configuration.
    arg = args_parser_Train()  # Load arguments (--case X) from the command line.
    cfg = load_config(arg, save_cfg=True)

    with Progress() as progress:  # progress bar
        task = progress.add_task("[green]Main loop:", total=1)  # main loop bar
        cfg.progress = progress  # for the sub inner loop
        main(cfg)
        progress.update(task, advance=1)
