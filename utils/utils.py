"""Utility toolkit: configuration merge, logging, plotting, FL/game helpers.

Functions here are mostly stateless; those that persist modify disk artifacts
 (save_config / update_cfg_arg) or mutate cfg in place (load_NE_case)."""

import re
import os, sys, time, random, logging, pytz, torch, yaml
import numpy as np
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import fsolve
from utils.args import args_parser_Train


def set_seed(seed):
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA) enabling deterministic runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(arg, save_cfg=False):
    """Merge base YAML (arg.cfg) with explicit CLI overrides (non-None values)."""
    arg_dict = vars(arg)  # parameters from the command line
    with open(arg.cfg, "r") as file:
        cfg = yaml.safe_load(file)  # parameters from the yaml file
    for key, value in arg_dict.items():  # do not set default values in args.py
        if value is not None:
            cfg[key] = value  # override yaml with args
    cfg = edict(cfg)  # allow to access dict values as attributes
    if save_cfg:
        cfg.dir_res = create_folder(cfg)  # folder to save the results
        print(f"Working in {cfg.dir_res}.\n")
        save_config(cfg, cfg.dir_res)  # save the test configuration
    return cfg


def load_NE_case(cfg, root_dir=None):
    """Load per-case NE (list of local epochs) into cfg.NE; raise informative error if absent."""
    try:
        if root_dir is None:
            # utils/ is under the project root; this file itself is inside utils/
            root_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
        dir_cases = os.path.join(root_dir, "utils", f"cases_m{cfg.m}")
        case_path = os.path.join(dir_cases, f"config_case_{cfg.case}.yaml")
        if not os.path.exists(case_path):
            raise FileNotFoundError(case_path)
        with open(case_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if "NE" not in data:
            raise KeyError("NE")
        cfg.NE = data["NE"]
        return cfg
    except Exception as e:
        raise ValueError(
            f"Failed to load NE case file (m={getattr(cfg,'m', '?')}, case={getattr(cfg,'case','?')}). "
            f"Please run main_game.py first to generate the case YAML. Original error: {e}"
        ) from e


def save_config(cfg, dir_res):
    """Serialize cfg to YAML, temporarily stripping non-serializable fields (e.g. progress)."""
    if hasattr(cfg, "progress"):
        temp = cfg.progress
        cfg.pop("progress", "")  # cannot be saved in yaml
        with open(os.path.join(dir_res, "config.yaml"), "w") as file:
            yaml.dump(dict(cfg), file, sort_keys=False)
        cfg.progress = temp  # restore the progress bar
    else:
        with open(os.path.join(dir_res, "config.yaml"), "w") as file:
            yaml.dump(dict(cfg), file, sort_keys=False)


def update_cfg_arg(dir_res, arg="T0", value=1):
    """Update a single field in the saved YAML (e.g., resume start iter).

    This does not mutate any in-memory cfg object; it edits the file only.
    """
    path = os.path.join(dir_res, "config.yaml")
    with open(path, "r") as file:
        cfg = yaml.safe_load(file)  # parameters from the yaml file
    cfg[arg] = value
    with open(path, "w") as file:
        yaml.dump(dict(cfg), file, sort_keys=False)


def get_logger(dir_res, log_to_console=True):
    """Create a simple logger writing to `0_log.txt` (and optionally console).

    Args:
        dir_res: Directory to store the log file.
        log_to_console: Also stream logs to stdout when True.
    """
    dir_log = os.path.join(dir_res, "0_log.txt")

    # Create logger instance
    log = logging.getLogger(dir_log)
    log.setLevel(logging.INFO)

    # Clear previous handlers to avoid duplication
    if log.handlers:
        log.handlers.clear()

    # Add file handler
    file_handler = logging.FileHandler(dir_log)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(file_handler)

    # Optionally add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(console_handler)

    # Disable propagation to root to prevent duplicated logs
    log.propagate = False

    return log


def get_time_stamp(time_zone="Europe/Berlin"):
    """Return a timestamp string for the specified time zone."""
    time_stamp = int(time.time())
    time_zone = pytz.timezone(time_zone)
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    return test_time


def create_folder(cfg, dir_parent="results", prefix="test"):
    """Create a result folder with a timestamp (Berlin time)."""
    if hasattr(cfg, "tag0"):
        dir_parent = os.path.join(dir_parent, cfg.tag0, "source")

    if hasattr(cfg, "dir_res"):
        # for continue training, use the same foldername
        path_folder = os.path.basename(cfg.dir_res)
    else:  # for new runs, create a new folder
        cfg.tag = get_label(cfg)
        test_time = get_time_stamp(time_zone="Europe/Berlin")
        path_folder = f"{prefix}_{cfg.tag}_{test_time}"
    dir_results = os.path.join(sys.path[0], dir_parent, path_folder)
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    return dir_results


def cal_E(cfg):
    """Return average effort (epochs) across all clients."""
    if isinstance(cfg.NE, list):
        s_avg = np.round(np.mean(np.array(cfg.NE), axis=0), 1)
    else:
        s_avg = cfg.NE
    return s_avg


def active_rho(rho, active_clients):
    """Compute normalized rho for active clients."""
    active_rhos = [rho[client_id] for client_id in active_clients]
    total_rho = sum(active_rhos)
    return [r / total_rho for r in active_rhos]


def get_label(cfg):
    """Build a descriptive tag for result directories (game/train)."""
    try:  # FL-Game BR
        label = "FL-Game_BRA_data-"
        label += "femnist_" if cfg.femnist_D else "balanced_"
        label += (
            f"m{cfg.m}"
            + f"_alpha{cfg.alpha[0]}-{cfg.alpha[1]}"
            + f"_q{cfg.reso_q[0]}-{cfg.reso_q[1]}"
            + f"_Q{cfg.reso_Q[0]}-{cfg.reso_Q[1]}"
            + f"_lambda{cfg.lamda[0]}-{cfg.lamda[1]}"
        )
    except:  # FL-Game Train
        lr = "{:.0e}".format(cfg["lr"]).replace("e-0", "e-")
        if cfg["w_decay"] == 0:
            w_decay = cfg["w_decay"]
        else:
            w_decay = "{:.0e}".format(cfg["w_decay"]).replace("e-0", "e-")

        if cfg.alg in ["fedprox"]:
            mu = cfg.get("mu", 0)
        elif cfg.alg in ["moon"]:
            mu = cfg.get("moon_mu", 0)
        else:
            mu = 0

        label = (
            f"{cfg['dataset']}_{cfg['model']}_{cfg['alg']}_mu-{mu}_m-{cfg['m']}_"
            f"D-{cfg['data_train']//1000}k_"
            f"frac-{cfg['frac']}_E-{cal_E(cfg)}_bs-{cfg['bs']}_lr-{lr}_Case-{cfg['case']}"
        )
    return label


def get_all_cases(dir_res, cfg, lm_1, lm_2):
    """Collect NEs for all lambdas between lm_1 and lm_2 and save YAML cases."""
    lm_a = get_adjacent_lm(cfg.lamda_list, lm_1, direc="small")  # Case 1
    lm_d = get_adjacent_lm(cfg.lamda_list, lm_2, direc="big")  # Case 4
    lm_list = []
    s_list = []
    for lamda in cfg.lamda_list:
        if lm_a <= lamda <= lm_d:
            s_list.append(get_s_at_lm(dir_res, lamda))
            lm_list.append(lamda)
    save_game_cases(lm_list, s_list)  # save yamls for FL training with main_Train.py


def get_four_cases(dir_res, cfg, lm_1, lm_x, lm_2):
    """Collect NEs at critical lambdas (lm_1, lm_x-, lm_x+, lm_2) and save cases."""
    lm_a = get_adjacent_lm(cfg.lamda_list, lm_1, direc="small")  # Case 1
    lm_b = get_adjacent_lm(cfg.lamda_list, lm_x, direc="small")  # Case 2
    lm_c = get_adjacent_lm(cfg.lamda_list, lm_x, direc="big")  # Case 3
    lm_d = get_adjacent_lm(cfg.lamda_list, lm_2, direc="big")  # Case 4
    lamda_list = [lm_a, lm_b, lm_c, lm_d]
    s_list = []  # corresponding NE at each Case
    for lamda in lamda_list:
        s_list.append(get_s_at_lm(dir_res, lamda))
    save_game_cases(lamda_list, s_list)  # save yamls for FL training with main_Train.py


def save_game_cases(lamda_list, s_list):
    """Save case YAMLs for FL training, embedding clients' rational efforts.

    YAML files are stored under `utils/cfg_cases` based on the base config.
    """
    dir_cfgs = os.path.join(os.getcwd(), "utils", f"cases_m{len(s_list[0])}")  # dir to save yamls
    os.makedirs(dir_cfgs, exist_ok=True)
    arg = args_parser_Train()
    cfg = load_config(arg)
    case = 1
    for cfg.lamda, cfg.NE in zip(lamda_list, s_list):
        if not isinstance(cfg.NE, list):
            cfg.NE = cfg.NE.tolist()
        cfg.tag = get_label(cfg)
        if len(lamda_list) == 4:  # four critical cases
            name = f"config_case_{case}"
        else:  # all cases between lm_1 and lm_2
            name = f"config_lm-{cfg.lamda}_"
        yml_tag = f"{name}.yaml"
        with open(os.path.join(dir_cfgs, yml_tag), "w") as file:
            yaml.dump(dict(cfg), file, sort_keys=False)
        case += 1
    print(f"Yamls for FL training are saved in {dir_cfgs}.")


def plot_acc_loss(cfg, data_path):
    """Plot FL training loss and accuracy curves from saved numpy data."""
    # figure formats
    label_font = FontProperties(family="sans-serif", weight="normal", size=12)
    data_dir = os.path.split(data_path)[0]
    data = np.load(data_path, allow_pickle=True).tolist()
    # remove nan and inf values
    data["loss"] = [1e2 if x == float("inf") else x for x in data["loss"]]
    data["loss"] = [1e2 if isinstance(x, float) and (x != x) else x for x in data["loss"]]
    # set plot range
    start = 1
    stop = len(data["loss"])
    xaxis = np.arange(stop)
    # plot training loss
    fig = plt.figure(dpi=100)
    plt.plot(xaxis[start:stop], data["loss"][start:stop])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Communication rounds", fontproperties=label_font)
    plt.ylabel("Loss", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("Training loss")
    fig.set_facecolor("white")
    plt.savefig(f"{data_dir}/res_loss.jpg", bbox_inches="tight")
    plt.close()

    # plot test accuracy
    fig = plt.figure(dpi=100)
    plt.plot(xaxis[start:stop], data["accu_test"][start:stop])
    plt.xlabel(f"Communication rounds", fontproperties=label_font)
    plt.ylabel("Test accuracy", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("Test accuracy.")
    plt.yticks(np.arange(0, 1.01, step=0.1), fontproperties=label_font)
    plt.grid(linestyle="--", linewidth=0.5)
    fig.set_facecolor("white")
    plt.savefig(f"{data_dir}/res_accu.jpg", bbox_inches="tight")
    plt.close()


def plot_NE(cfg, dir_res, s_avg, save_cases=1):
    """Plot NE statistics versus reward factor and mark critical points."""
    with open(os.path.join(dir_res, "0_lms.yaml")) as file:
        lms = yaml.safe_load(file)
    lm_1 = lms["lm_1"]
    lm_2 = lms["lm_2"]
    lm_x = lms["lm_x"]
    c_1 = lms["c_1"]
    c_2 = lms["c_2"]
    q_avg = lms["q_avg"]
    Q_avg = lms["Q_avg"]
    cases = (lms["case1"], lms["case2"], lms["case3"], lms["case4"])
    if save_cases:
        get_four_cases(dir_res, cfg, lm_1, lm_x, lm_2)  # save case yamls
        # get_all_cases(dir_res, cfg, lm_1, lm_2)

    # figure formats
    legend_font = FontProperties(family="sans-serif", style="normal", size=11)
    label_font = FontProperties(family="sans-serif", weight="normal", size=13)
    marker_size = 30
    file_name = "fig"
    # plot full scope
    start, stop, interval = 0, s_avg.shape[-1], 2  # plot every interval point
    fig = plt.figure(dpi=100)
    plt.scatter(
        cfg.lamda_list[start:stop][::interval],
        s_avg[start:stop][::interval],
        marker="o",
        s=marker_size,
        c="tab:purple",
        alpha=1,
    )
    plt.scatter(
        lm_1,
        q_avg,
        marker="s",
        s=marker_size,
        label=f"($\lambda_1, \\ \\ \\bar q$)=({lm_1:.3f}, {q_avg:.2f})",
    )
    plt.scatter(
        lm_x,
        c_1,
        marker="s",
        s=marker_size,
        label=f"($\lambda^*$, $ c_1$)=({lm_x:.3f}, {c_1:.2f})",
    )
    plt.scatter(
        lm_x,
        c_2,
        marker="s",
        s=marker_size,
        label=f"($\lambda^*$, $ c_2$)=({lm_x:.3f}, {c_2:.2f})",
    )
    plt.scatter(
        lm_2,
        Q_avg,
        marker="s",
        s=marker_size,
        alpha=1,
        label=f"($\lambda_2, \\ \\ \\bar Q$)=({lm_2:.3f}, {Q_avg:.2f})",
    )
    plt.legend(prop=legend_font, loc="best")
    plt.xlabel("Reward factor $\lambda$", fontproperties=label_font)
    plt.ylabel(r"Average effort $\bar s$", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("")
    fig.set_facecolor("white")
    plt.savefig(f"{dir_res}/{file_name}_NE.png", bbox_inches="tight")

    # plot zoom in at lm1
    start = get_lambda_index(cfg.lamda_list, lm_1) - 5
    stop = get_lambda_index(cfg.lamda_list, lm_1) + 6
    interval = 1
    fig = plt.figure(dpi=100)
    plt.scatter(
        cfg.lamda_list[start:stop][::interval],
        s_avg[start:stop][::interval],
        marker="o",
        s=marker_size,
        c="tab:purple",
        alpha=1,
    )
    plt.scatter(
        lm_1,
        q_avg,
        marker="s",
        s=marker_size,
        alpha=1,
        label=f"($\lambda_1, \\ \\ \\bar q$)=({lm_1:.3f}, {q_avg:.2f})",
    )
    plt.legend(prop=legend_font, loc="best")
    plt.xlabel("Reward factor $\lambda$", fontproperties=label_font)
    plt.ylabel(r"Average effort $\bar s$", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    plt.title("")
    fig.set_facecolor("white")
    plt.savefig(f"{dir_res}/{file_name}_NE_lm_1.png", bbox_inches="tight")

    # plot zoom in at lm_x
    start = get_lambda_index(cfg.lamda_list, lm_x) - 5
    stop = get_lambda_index(cfg.lamda_list, lm_x) + 25
    interval = 1
    fig = plt.figure(dpi=100)
    plt.scatter(
        cfg.lamda_list[start:stop][::interval],
        s_avg[start:stop][::interval],
        marker="o",
        s=marker_size,
        c="tab:purple",
        alpha=1,
    )
    plt.scatter(
        lm_x,
        c_1,
        marker="s",
        s=marker_size,
        c="tab:orange",
        alpha=1,
        label=f"($\lambda^*$, $ c_1$)=({lm_x:.3f}, {c_1:.2f})",
    )
    plt.scatter(
        lm_x,
        c_2,
        marker="s",
        s=marker_size,
        c="tab:green",
        alpha=1,
        label=f"($\lambda^*$, $ c_2$)=({lm_x:.3f}, {c_2:.2f})",
    )

    plt.legend(prop=legend_font, loc="lower right")
    plt.xlabel("Reward factor $\lambda$", fontproperties=label_font)
    plt.ylabel(r"Average effort $\bar s$", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("")
    fig.set_facecolor("white")
    plt.savefig(f"{dir_res}/{file_name}_NE_lm_x.png", bbox_inches="tight")

    # plot zoom in at lm_2
    start = get_lambda_index(cfg.lamda_list, lm_2) - 5
    stop = min(4.81, cfg.lamda_list[-2])  # max lambda to plot
    stop = min(s_avg.shape[-1], get_lambda_index(cfg.lamda_list, stop) + 1)
    interval = 1
    fig = plt.figure(dpi=100)
    plt.scatter(
        cfg.lamda_list[start:stop][::interval],
        s_avg[start:stop][::interval],
        marker="o",
        s=marker_size,
        c="tab:purple",
        alpha=1,
    )
    plt.scatter(
        lm_2,
        Q_avg,
        marker="s",
        s=marker_size,
        c="tab:red",
        alpha=1,
        label=f"($\lambda_2, \\ \\ \\bar Q$)=({lm_2:.3f}, {Q_avg:.2f})",
    )
    plt.legend(prop=legend_font, loc="best")
    plt.xlabel("Reward factor $\lambda$", fontproperties=label_font)
    plt.ylabel(r"Average effort $\bar s$", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    plt.title("")
    fig.set_facecolor("white")
    plt.savefig(f"{dir_res}/{file_name}_NE_lm_2.png", bbox_inches="tight")


def larger(x):
    """Return the next float after x to handle half-open sampling intervals."""
    x = np.nextafter(x, x + 1)
    return x


def get_adjacent_lm(lm_list, target, direc="small"):
    """Return the adjacent lambda to target from below (small) or above (big)."""
    for i, lamda in enumerate(lm_list):
        if lamda > target:
            if direc == "small":
                res = lm_list[i - 1]
            elif direc == "big":
                res = lm_list[i]
            break
    return round(res, 3)


def get_path_lm(dir_res, lamda=2):
    """Return the path to the saved result dict for a specific lambda."""
    for root, dirs, _ in os.walk(dir_res):
        for dir_name in dirs:
            if dir_name == f"lambda-{lamda}":
                target_folder = os.path.join(root, dir_name)
                target_file = os.path.join(target_folder, "0_res_dict.npy")
                if os.path.exists(target_file):
                    return target_file
    raise ValueError(f"Invalid lambda.")


def get_s_at_lm(dir_res, lamda):
    """Load the NE (s*) at the given reward factor from disk."""
    data = get_path_lm(dir_res, lamda)
    res_dict = np.load(data, allow_pickle=True).tolist()
    s_star = np.array(res_dict["s"])[-1][:, 0]  # get NE or strategy
    return s_star


def get_lm_star(alpha, rho):
    """Return lambda_star solving sum(lambda/(2a-lambda)) = 1 (scaled by rho)."""

    def eq2solve(lambda_val):
        return 1 - sum(lambda_val / (2 * a - lambda_val) for a in alpha / rho)

    lm_star = fsolve(eq2solve, x0=1)  # get lambda_star
    return lm_star[0]


def get_lambda_index(lm_list, target):
    """Return index of the largest lambda smaller than the target."""
    for i, lamda in enumerate(lm_list):
        if lamda > target:
            return i - 1


def cal_lambdas(dir_res, res_dict, plot=True):
    """Plot critical points for lambda without s_avg data."""
    # load game parameters
    alpha = np.array(res_dict["alpha"])
    reso_Q = np.array(res_dict["reso_Q"])
    reso_q = np.array(res_dict["reso_q"])
    D = np.array(res_dict["data_vol"])
    rho = D / np.sum(D)
    q_avg = np.average(reso_q)
    Q_avg = np.average(reso_Q)

    # calculate critical reward factors
    lm_1 = 2 * min(alpha * reso_q / (q_avg + reso_q * rho))
    lm_2 = 2 * max(alpha * reso_Q / (Q_avg + reso_Q * rho))
    lm_x = get_lm_star(alpha, rho)  # get lambda_star
    c_1 = max(2 * alpha * reso_q / lm_x - reso_q * rho)
    c_2 = min(2 * alpha * reso_Q / lm_x - reso_Q * rho)

    # save parameters to 0_lms.yaml
    lms_dict = {
        "lm_1": float(lm_1),
        "lm_2": float(lm_2),
        "lm_x": float(lm_x),
        "c_1": float(c_1),
        "c_2": float(c_2),
        "q_avg": float(q_avg),
        "Q_avg": float(Q_avg),
    }
    # compute four case lambdas directly from critical points (rounded two decimals)
    lm, case1, case2, case4 = 0, 0, 0, 0
    while True:
        lm += 0.01
        if lm > lm_1 and case1 == 0:
            case1 = round(lm - 0.01, 2)
            continue
        if lm > lm_x and case2 == 0:
            case2 = round(lm - 0.01, 2)
            case3 = round(lm, 2)
            continue
        if lm > lm_2 and case4 == 0:
            case4 = round(lm, 2)
            break
    lms_dict.update({"case1": case1, "case2": case2, "case3": case3, "case4": case4})
    with open(os.path.join(dir_res, "0_lms.yaml"), "w") as file:
        yaml.dump(lms_dict, file, sort_keys=False)

    if plot:  # plot critical points
        plot_lambdas(dir_res)

    return case1, case2, case3, case4


def plot_lambdas(dir_res):
    """Plot critical points for lambda without s_avg data."""
    # load precomputed lambda parameters from YAML
    with open(os.path.join(dir_res, "0_lms.yaml")) as file:
        lms = yaml.safe_load(file)
    lm_1 = lms["lm_1"]
    lm_2 = lms["lm_2"]
    lm_x = lms["lm_x"]
    c_1 = lms["c_1"]
    c_2 = lms["c_2"]
    q_avg = lms["q_avg"]
    Q_avg = lms["Q_avg"]

    # figure formats
    legend_font = FontProperties(family="sans-serif", style="normal", size=11)
    label_font = FontProperties(family="sans-serif", weight="normal", size=13)
    marker_size = 30
    file_name = "fig"

    # plot the four critical points
    fig = plt.figure(dpi=100)
    plt.scatter(
        lm_1,
        q_avg,
        marker="s",
        s=marker_size,
        label=f"($\lambda_1, \\ \\ \\bar q$)=({lm_1:.3f}, {q_avg:.2f})",
    )
    plt.scatter(
        lm_x,
        c_1,
        marker="s",
        s=marker_size,
        label=f"($\lambda^*$, $ c_1$)=({lm_x:.3f}, {c_1:.2f})",
    )
    plt.scatter(
        lm_x,
        c_2,
        marker="s",
        s=marker_size,
        label=f"($\lambda^*$, $ c_2$)=({lm_x:.3f}, {c_2:.2f})",
    )
    plt.scatter(
        lm_2,
        Q_avg,
        marker="s",
        s=marker_size,
        alpha=1,
        label=f"($\lambda_2, \\ \\ \\bar Q$)=({lm_2:.3f}, {Q_avg:.2f})",
    )
    plt.legend(prop=legend_font, loc="best")
    plt.xlabel("Reward factor $\lambda$", fontproperties=label_font)
    plt.ylabel(r"Average effort $\bar s$", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("")
    fig.set_facecolor("white")
    plt.savefig(f"{dir_res}/{file_name}_NE.png", bbox_inches="tight")
    plt.close()
