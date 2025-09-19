"""Federated training loop + evaluation utilities.

FL_train orchestrates communication rounds:
    (1) sample clients, (2) local updates, (3) aggregate, (4) evaluate & persist.

Resume logic: cfg.T0 indicates next round to execute; artifacts (results/model)
are appended and lr decay stays consistent by incrementing cfg.T0 on each save.
"""

import os
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from utils.utils import *


def FL_train(cfg, server, clients, dataset):
    """Run federated optimization from cfg.T0 through cfg.T inclusive."""
    # Initialization.
    res_dict, res_path, log = prepare(cfg, clients, server, dataset)  # logging system
    # task2 = cfg.progress.add_task("[green] Sub loop:", total=cfg.T)  # progress bar
    # Start training.
    train_start_time = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(train_start_time))
    log.info(f"\n-----Start training ({cfg.zdevice})-----\n{local_time}\n")
    for t in range(cfg.T0, cfg.T + 1):  # Iterate over FL training rounds.
        round_start_time = time.time()
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log.info(f"\nRound: {t} ---{local_time}")
        # server selects clients
        selected_clients = server.select_clients(frac=cfg.frac)
        # clients' local updates
        res_clients = clients.local_update(server.model, selected_clients, t)
        # server aggregation
        server.aggregate(res_clients)
        # evaluation and save results
        log.info(f" Round time: {(time.time() - round_start_time)/60:.2f} min")
        if t % cfg.save_freq == 0 or t in [1, cfg.T]:
            loss = evaluate(cfg, server.model, dataset, res_dict, log)
            if cfg.plot:
                plot_acc_loss(cfg, res_path)
            if loss > 200 or np.isnan(loss):  # early stop
                log.info(f"\n# Early stop dut to the loss explosion #\n")
                break
        update_cfg_arg(cfg.dir_res, arg="T0", value=cfg.T0 + 1)
        cfg.T0 += 1
        # cfg.progress.update(task2, advance=1)  # complete one round
        log.info(f" Total time: {(time.time() - train_start_time)/60:.1f} min\n")
    # cfg.progress.remove_task(task2)  # complete all rounds
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    log.info(f"-----End training-----\n{local_time}\n")


def prepare(cfg, clients, server, dataset):
    """Initialize / load result dict and perform a baseline evaluation if fresh run."""
    res_path = os.path.join(cfg.dir_res, f"0_results.npy")
    log = get_logger(cfg.dir_res)  # logging system
    if cfg.T0 > 0:  # load previous results
        res_dict = np.load(res_path, allow_pickle=True).tolist()
    else:  # start from scratch
        res_dict = defaultdict(list)  # dict to save results
        evaluate(cfg, server.model, dataset, res_dict, log)
        update_cfg_arg(cfg.dir_res, arg="T0", value=cfg.T0 + 1)
        cfg.T0 += 1
    clients.res_dict = res_dict  # to save results during clients' local updates
    clients.log = log  # to save log
    # save_results(cfg, res_dict, server.model)
    return res_dict, res_path, log


def evaluate(cfg, model, dataset, res_dict, log):
    """Evaluate global model (train loss + val/test accuracy) and checkpoint state."""
    res_dict["round"].append(cfg.T0)
    loss = cal_loss(cfg, model, dataset)
    res_dict["loss"].append(loss)
    acc_test = cal_accu(cfg, model, dataset["test"])
    res_dict["accu_test"].append(acc_test)
    acc_val = cal_accu(cfg, model, dataset["val"])
    res_dict["accu_val"].append(acc_val)
    if cfg.T0 != 0:
        log.info(f" Loss: {loss:.3f}")
        log.info(f" Accu_val: {acc_val:.3f}")
        log.info(f" Accu_test: {acc_test:.3f}")
    save_results(cfg, res_dict, model)
    return loss


def save_results(cfg, res_dict, model):
    """Persist numpy results dict and model weights into cfg.dir_res."""
    res_path = os.path.join(cfg.dir_res, f"0_results.npy")
    np.save(res_path, np.asarray(res_dict, dtype=object))
    model_path = os.path.join(cfg.dir_res, f"0_model.pth")
    torch.save(model.state_dict(), model_path)


def cal_loss(cfg, model, dataset, bs=1000):
    """Compute mean loss over entire training split (batched)."""
    total_loss = 0.0
    total_samples = 0
    model.eval()
    data_loader = DataLoader(dataset["train"], batch_size=bs)
    with torch.no_grad():
        for batch in data_loader:
            features, labels = batch
            if isinstance(features, (list, tuple)):  # datasets that return multiple tensors
                data = [item.to(cfg.device) for item in features]
            else:
                data = features.to(cfg.device)
            labels = labels.to(cfg.device)
            preds = model(data)
            batch_loss = model.loss(preds, labels)  # reduction="mean"
            batch_size = labels.shape[0] if hasattr(labels, "shape") else len(labels)
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size
    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def cal_accu(cfg, model, dataset, bs=1000):
    """Compute accuracy over provided dataset (multi-class or binary)."""
    correct = 0
    model.eval()
    data_loader = DataLoader(dataset, batch_size=bs)
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch[0], list):  # two items for NLP dataset
                data = [item.to(cfg.device) for item in batch[0]]
            else:  # one item for vision dataset
                data = batch[0].to(cfg.device)
            label = batch[1].to(cfg.device)
            pred = model(data)
            if len(pred.shape) == 2:  # vision dataset
                _, pred = torch.max(pred, 1)  # predicted labels
                correct += (pred.view_as(label) == label).sum()  # correct predictions
            else:  # nlp dataset
                correct += ((pred > 0.5).int() == label).sum()
        accuracy = correct / len(dataset)
    return accuracy.item()
