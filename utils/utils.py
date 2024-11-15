# Auxiliary functions
import os, sys, time, random, logging, pytz, torch, yaml
import numpy as np
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.optimize import fsolve
from utils.args import args_parser_Train


def set_seed(seed):
    """Setup the random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(arg, save_cfg=False):
    """Load params from yaml and overwrite them with command line arguments."""
    arg_dict = vars(arg)  # parameters from the command line
    try:  # for main_Train.py, choose from four cases
        if arg.case:
            arg.cfg = os.path.join(
                os.getcwd(), "utils", "cfg_cases", f"config_case_{arg.case}.yaml"
            )
    except:
        pass
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


def save_config(cfg, dir_res):
    """Save config to a yaml file, cfg here is an EasyDict."""
    cfg.pop("progress", "")  # cannot be saved in yaml
    with open(os.path.join(dir_res, "config.yaml"), "w") as file:
        yaml.dump(dict(cfg), file, default_flow_style=False)


def get_logger(dir_res):
    """Return the logger."""
    dir_log = os.path.join(dir_res, "0_log.txt")
    logging.basicConfig(
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
        format="%(message)s",
    )
    log = logging.getLogger(dir_log)
    log.addHandler(logging.FileHandler(dir_log))
    return log


def get_time_stamp(time_zone="Europe/Berlin"):
    """Return the time stamp of a specifid time zone."""
    time_stamp = int(time.time())
    time_zone = pytz.timezone(time_zone)
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    return test_time


def create_folder(cfg, dir_parent="results", prefix="test_", postfix=""):
    """Create a test folder with a timestamp in Berlin time zone."""
    prefix = f"test_{cfg.tag}_"
    test_time = get_time_stamp(time_zone="Europe/Berlin")
    dir_results = os.path.join(sys.path[0], dir_parent, f"{prefix}{test_time}{postfix}")
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    return dir_results


def get_label(cfg):
    """Setup the dir name for an FL-Game test."""
    try:  # FL-Game BR
        label = (
            f"FL-Game_BR_m{cfg.m}"
            + f"_alpha{cfg.alpha[0]}-{cfg.alpha[1]}"
            + f"_lambda{cfg.lamda[0]}-{cfg.lamda[1]}"
            + f"_q{cfg.reso_q[0]}-{cfg.reso_q[1]}"
            + f"_Q{cfg.reso_Q[0]}-{cfg.reso_Q[1]}"
        )
    except:  # FL-Game Train
        label = f"FL-Game_Train_Case-{cfg.case}_lambda-{cfg.lamda}"
    return label


def get_four_cases(dir_res, cfg, lm_1, lm_x, lm_2):
    """Return the NE at critical reward factors."""
    lm_a = get_adjacent_lm(cfg.lamda_list, lm_1, direc="small")  # Case 1
    lm_b = get_adjacent_lm(cfg.lamda_list, lm_x, direc="small")  # Case 4
    lm_c = get_adjacent_lm(cfg.lamda_list, lm_x, direc="big")  # Case 3
    lm_d = get_adjacent_lm(cfg.lamda_list, lm_2, direc="big")  # Case 4
    lamda_list = [lm_a, lm_b, lm_c, lm_d]
    s_list = []  # corresponding NE at each Case
    for lamda in lamda_list:
        s_list.append(get_s_at_lm(dir_res, lamda))
    save_four_cases(lamda_list, s_list)  # save yamls for FL training with main_Train.py


def plot_acc_loss(cfg, data_path):
    """Plot FL training results."""
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
    plt.ylabel("Accuracy", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("Test accuracy.")
    plt.yticks(np.arange(0, 1.01, step=0.1), fontproperties=label_font)
    plt.grid(linestyle="--", linewidth=0.5)
    fig.set_facecolor("white")
    plt.savefig(f"{data_dir}/res_accu.jpg", bbox_inches="tight")
    plt.close()


def plot_NE(cfg, dir_res, res_dict, s_avg):
    """Plot FL training results."""
    # figure formats
    legend_font = FontProperties(family="sans-serif", style="normal", size=11)
    label_font = FontProperties(family="sans-serif", weight="normal", size=13)
    marker_size = 30
    file_name = "fig"
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
    get_four_cases(dir_res, cfg, lm_1, lm_x, lm_2)  # Return NEs at critical reward factors.

    # plot full scope
    start, stop, interval = 0, s_avg.shape[-1], 2
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
        label=f"($\lambda_1, \\ \\ \\bar q$)=({lm_1:.2f}, {q_avg:.2f})",
    )
    plt.scatter(
        lm_x,
        c_1,
        marker="s",
        s=marker_size,
        label=f"($\lambda^*$, $ c_1$)=({lm_x:.2f}, {c_1:.2f})",
    )
    plt.scatter(
        lm_x,
        c_2,
        marker="s",
        s=marker_size,
        label=f"($\lambda^*$, $ c_2$)=({lm_x:.2f}, {c_2:.2f})",
    )
    plt.scatter(
        lm_2,
        Q_avg,
        marker="s",
        s=marker_size,
        alpha=1,
        label=f"($\lambda_2, \\ \\ \\bar Q$)=({lm_2:.2f}, {Q_avg:.2f})",
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
        label=f"($\lambda_1, \\ \\ \\bar q$)=({lm_1:.2f}, {q_avg:.2f})",
    )
    plt.legend(prop=legend_font, loc="best")
    plt.xlabel("Reward factor $\lambda$", fontproperties=label_font)
    plt.ylabel(r"Average effort $\bar s$", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
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
        label=f"($\lambda^*$, $ c_1$)=({lm_x:.2f}, {c_1:.2f})",
    )
    plt.scatter(
        lm_x,
        c_2,
        marker="s",
        s=marker_size,
        c="tab:green",
        alpha=1,
        label=f"($\lambda^*$, $ c_2$)=({lm_x:.2f}, {c_2:.2f})",
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
        label=f"($\lambda_2, \\ \\ \\bar Q$)=({lm_2:.2f}, {Q_avg:.2f})",
    )
    plt.legend(prop=legend_font, loc="best")
    plt.xlabel("Reward factor $\lambda$", fontproperties=label_font)
    plt.ylabel(r"Average effort $\bar s$", fontproperties=label_font)
    plt.xticks(fontproperties=label_font)
    plt.yticks(fontproperties=label_font)
    plt.title("")
    fig.set_facecolor("white")
    plt.savefig(f"{dir_res}/{file_name}_NE_lm_2.png", bbox_inches="tight")


def larger(x):
    """
    Return the next floating-point value after x.
    For solving the upper exclusive issue in np.random methods.
    """
    x = np.nextafter(x, x + 1)
    return x


def get_adjacent_lm(lm_list, target, direc="small"):
    """Return the largest lamda samller than the target or other way round."""
    for i, lamda in enumerate(lm_list):
        if lamda > target:
            if direc == "small":
                res = lm_list[i - 1]
            elif direc == "big":
                res = lm_list[i]
            break
    return round(res, 3)


def get_path_lm(dir_res, lamda=2):
    """Return the path of res under a specific lambda."""
    for root, dirs, _ in os.walk(dir_res):
        for dir_name in dirs:
            if dir_name.startswith(f"lambda-{lamda}"):
                target_folder = os.path.join(root, dir_name)
                target_file = os.path.join(target_folder, "0_res_dict.npy")
                if os.path.exists(target_file):
                    return target_file
    raise ValueError(f"Invalid lambda.")


def get_s_at_lm(dir_res, lamda):
    """Return the NE at reward factor lambda."""
    data = get_path_lm(dir_res, lamda)
    res_dict = np.load(data, allow_pickle=True).tolist()
    s_star = np.array(res_dict["s"])[-1][:, 0]  # get NE or strategy
    return s_star


def save_four_cases(lamda_list, s_list):
    """
    Save the yamls for FL training with clients' rational efforts.
    The yamls are storaged at "FL_Potential_Game_DCN\\utils\\cfgs"
    """
    base_cfg = os.path.join(os.getcwd(), "utils", "config_train.yaml")
    dir_cfgs = os.path.join(os.getcwd(), "utils", "cfg_cases")  # dir to save yamls
    if not os.path.exists(dir_cfgs):
        os.makedirs(dir_cfgs)
    arg = args_parser_Train()
    arg.cfg = base_cfg
    cfg = load_config(arg, save_cfg=False)
    cfg.m = len(s_list[0])  # number of clients
    cfg.data_train = 500 * cfg.m
    cfg.case = 1
    for cfg.lamda, cfg.NE in zip(lamda_list, s_list):
        cfg.NE = cfg.NE.tolist()
        cfg.tag = get_label(cfg)
        yml_tag = f"config_case_{cfg.case}.yaml"
        with open(os.path.join(dir_cfgs, yml_tag), "w") as file:
            yaml.dump(dict(cfg), file, default_flow_style=False)
        cfg.case += 1
    print(f"Yamls for FL training are saved in {dir_cfgs}.")


def get_lm_star(alpha, rho):
    """Return the value of lambda_star."""

    def eq2solve(lambda_val):
        return 1 - sum(lambda_val / (2 * a - lambda_val) for a in alpha / rho)

    lm_star = fsolve(eq2solve, x0=1)  # get lambda_star
    return lm_star[0]


def get_lambda_index(lm_list, target):
    """Return the index of the largest lamda samller than the target."""
    for i, lamda in enumerate(lm_list):
        if lamda > target:
            return i - 1
