import argparse, os


def args_parser_NE():
    """
    Load parameters from the command line
    In general, no default values to avoid overwriting, see func load_config
    """
    args = argparse.ArgumentParser()
    path_cfg = os.path.join(os.getcwd(), "utils", "config_game.yaml")
    args.add_argument("--cfg", type=str, default=path_cfg, help="path to the config")
    args.add_argument("--seed", type=int, help="random seed")
    args.add_argument("--m", type=int, help="number of clients")
    args.add_argument("--T", type=int, help="number of rounds")
    args.add_argument("--lamda", type=float, help="reward factor")
    args.add_argument("--alpha", nargs="+", type=float, help="running cost factor")
    args.add_argument("--tag", type=str, help="tag for saving")
    args.add_argument("--stop_iter", type=int, help="number of BR iters")
    args = args.parse_args()
    return args


def args_parser_Train():
    """
    Load parameters from the command line
    In general, no default values to avoid overwriting, see func load_config
    """
    args = argparse.ArgumentParser()
    args.add_argument("--alg", type=str, help="FL algorithm")
    args.add_argument("--bs", type=int, help="size of local mini-batches")
    args.add_argument("--case", type=int, help="case 1-4 to test")
    args.add_argument("--dataset", type=str, help="name of the dataset")
    args.add_argument("--E", type=int, help="number of epochs")
    args.add_argument("--frac", type=float, help="fraction of active clients")
    args.add_argument("--iid", type=str2bool, help="whether iid or not")
    args.add_argument("--T", type=int, help="number of training rounds")
    args.add_argument("--lr", type=float, help="learning rate")
    args.add_argument("--m", type=int, help="number of users")
    args.add_argument("--model", type=str, help="name of the model")
    args.add_argument("--seed", type=int, help="seed")
    args.add_argument("--subset", type=str2bool, help="use a subset of the whole dataset")
    args.add_argument("--tag", type=str, help="tag for saving")
    args = args.parse_args()
    return args


def str2bool(str):
    """Parse true/false from the command line."""
    if isinstance(str, bool):
        return str
    if str.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif str.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
