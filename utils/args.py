import argparse, os
import sys


def args_parser_NE():
    """Parse CLI arguments for computing the Nash Equilibrium (used by main_game.py).

    Design principles:
    1. Do NOT set defaults for keys already defined in the YAML (except --cfg) to avoid accidental overrides.
    2. Only overwrite a YAML field if the user explicitly provides a CLI argument.
    3. Enforce fixed lengths for list/range style arguments via nargs to prevent index errors later.
    """

    parser = argparse.ArgumentParser()

    # Robust default config path (relative to this file rather than current working directory)
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_game.yaml")
    parser.add_argument("--cfg", type=str, default=default_cfg, help="Path to config file")

    # Basic experiment meta / randomness
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--tag", type=str, help="Experiment tag (used for result folder naming)")
    parser.add_argument("--tag0", type=str, help="Base directory tag for raw BR results")

    # Core game parameters (matching config_game.yaml)
    parser.add_argument("--m", type=int, help="Number of clients (m)")
    parser.add_argument("--T", type=int, help="Number of rounds (T)")

    # Reward factor sweep range [start, end, step] (exactly 3 floats)
    parser.add_argument(
        "--lamda",
        nargs=3,
        type=float,
        metavar=("START", "END", "STEP"),
        help="Reward factor sweep: start end step",
    )

    # Participant parameter ranges (closed intervals [low, high])
    parser.add_argument(
        "--alpha",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Running cost coefficient alpha range",
    )
    parser.add_argument(
        "--data_vol",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="Client data volume range",
    )
    parser.add_argument(
        "--reso_q",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="Lower bound range for minimal effort q",
    )
    parser.add_argument(
        "--reso_Q",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="Upper bound range for resource Q",
    )

    # Optional custom FEMNIST imbalanced data volume specification (empty string -> fallback to data_vol range)
    parser.add_argument(
        "--femnist_D",
        type=str,
        help="Custom FEMNIST imbalanced data volume file (leave empty to use data_vol range)",
    )
    parser.add_argument("--stop_e", type=float, help="Convergence tolerance for BR")
    parser.add_argument("--stop_iter", type=int, help="Maximum BR iterations")

    args = parser.parse_args()

    # Basic validation: only check length if user supplied --lamda
    if hasattr(args, "lamda") and args.lamda is not None and len(args.lamda) != 3:
        raise ValueError("--lamda expects exactly 3 values: start end step")

    return args


def args_parser_Train(cfg_file="config_train.yaml"):
    """
    Parse CLI arguments for FL training (used by main_train.py) based on utils/config_train.yaml.

    Principles:
    - Only set defaults for --cfg and auxiliary non-YAML helpers to avoid overriding YAML values.
    - Names and types mirror keys in utils/config_train.yaml.
    - Any CLI value provided explicitly overrides the YAML via utils.utils.load_config.
    """
    parser = argparse.ArgumentParser()

    # Default config path relative to this file
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_file)
    parser.add_argument("--cfg", type=str, default=default_cfg, help="Path to the config YAML")

    # FL scenario
    parser.add_argument("--m", type=int, help="Number of clients (m)")
    parser.add_argument("--T", type=int, help="Number of rounds (T)")
    parser.add_argument("--alg", type=str, help="Algorithm: fedavg | fedprox | moon")
    parser.add_argument("--case", type=int, help="Case id for pre-generated NE yaml (1-4)")

    # Dataset
    parser.add_argument(
        "--dataset", type=str, help="Dataset: mnist | cifar10 | cifar100 | imdb | femnist"
    )
    parser.add_argument("--subset", type=str2bool, help="Use subset of dataset (bool)")
    parser.add_argument("--data_train", type=int, help="Training subset size")
    parser.add_argument("--data_val", type=int, help="Validation subset size (from training set)")
    parser.add_argument("--data_test", type=int, help="Test subset size")
    parser.add_argument(
        "--iid",
        type=float,
        help="Data split: 1=iid; >1=labels per client; (0,1)=Dirichlet alpha",
    )

    # Server side
    parser.add_argument("--frac", type=float, help="Fraction of active clients per round")

    # Client side
    parser.add_argument("--optimizer", type=str, help="Optimizer: sgd | adam")
    parser.add_argument("--bs", type=int, help="Batch size (0/None for FGD)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--w_decay", type=float, help="Weight decay")
    parser.add_argument("--mu", type=float, help="FedProx proximal term weight (mu)")
    parser.add_argument("--moon_mu", type=float, help="MOON contrastive loss weight")
    parser.add_argument("--moon_temperature", type=float, help="MOON temperature")
    parser.add_argument("--moon_buffer_size", type=int, help="MOON buffer size (previous models)")

    # Model
    parser.add_argument(
        "--model", type=str, help="Model name (cnn/cnn2/cnn_xs|s|m|l|xl|resnetXX|tf)"
    )
    parser.add_argument("--activation", type=str, help="Activation: relu | sigmoid")
    parser.add_argument("--loss", type=str, help="Loss: mse | cn | bce")

    # Transformer-specific
    parser.add_argument("--max_length", type=int, help="Max sequence length (NLP)")
    parser.add_argument("--emb_dim", type=int, help="Embedding dim (NLP)")
    parser.add_argument("--nhead", type=int, help="Transformer heads (NLP)")
    parser.add_argument("--nhid", type=int, help="Transformer hidden dim (NLP)")
    parser.add_argument("--nlayers", type=int, help="Transformer layers (NLP)")

    # Other
    parser.add_argument("--save_freq", type=int, help="Evaluation/save frequency (rounds)")
    parser.add_argument("--device", type=str, help="Device: cuda[:id] | mps | cpu")
    parser.add_argument("--plot", type=str2bool, help="Plot curves during training")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--tag", type=str, help="Name tag for logging/results")
    parser.add_argument("--T0", type=int, help="Resume start round (T0)")
    parser.add_argument("--bs_idx", type=int, help="Centralized training batch index (-1 disable)")
    parser.add_argument("--EaSt", type=int, help="Early stop flag (0/1)")
    parser.add_argument("--NE", type=int, help="Local epochs per client (SGD steps)")

    # Aux (not defined in YAML; safe to keep defaults here)
    parser.add_argument("--debug", type=str2bool, help="Debug mode")
    parser.add_argument("--dir", type=str, default="cfgs", help="Directory for repeat cfg files")
    parser.add_argument("--jobid", type=int, default=0, help="Cluster job id")

    return parser.parse_args()


def str2bool(str):
    """Robust boolean parser accepting common textual truthy / falsy variants."""
    if isinstance(str, bool):
        return str
    if str.lower() in ["yes", "true", "t", "y", "1"]:
        return True
    elif str.lower() in ["no", "false", "f", "n", "0"]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
