"""Dataset initialization and federated partitioning utilities.

Supports:
    * MNIST / CIFAR10 / CIFAR100 (torchvision) with optional balanced subsetting + val split.
    * IMDB (HuggingFace datasets + Transformer tokenization) for binary text classification.
    * FEMNIST (HuggingFace flwrlabs/femnist) with per-writer client mapping, caching & YAML export
        of per-client data volumes for reproducibility and game-theoretic use.

Key design points:
    - Validation samples are removed from the training pool before optional train/test subsetting.
    - IID split uniformly partitions indices; non-IID split uses label shards per client.
    - When cfg.subset is True, balanced subsets are drawn to maintain class uniformity.
"""

import os
import torch
import numpy as np
import json
import yaml
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from PIL import Image
from typing import List, Dict
from collections import defaultdict


def init_dataset(cfg):
    """Load base dataset, carve out validation, optionally subset, then federate.

    Returns a dict with keys: train / test / val / split_dict.
    Also updates cfg.{num_classes, data_train, data_test, data_val, rho}.
    """
    # path to download the dataset
    path = os.path.join(os.path.dirname(os.getcwd()), "dataset")
    if cfg.dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans)
        dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans)
        cfg.num_classes = 10
    elif cfg.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        trans_tr = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        trans_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset_train = datasets.CIFAR10(root=path, train=True, download=True, transform=trans_tr)
        dataset_test = datasets.CIFAR10(root=path, train=False, download=True, transform=trans_te)
        cfg.num_classes = 10
    elif cfg.dataset == "cifar100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        dataset_train = datasets.CIFAR100(root=path, train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR100(root=path, train=False, download=True, transform=trans)
        cfg.num_classes = 100
    elif cfg.dataset == "imdb":
        dataset_train, dataset_test, cfg.vocab_size = load_IMDB_data(local_path=path)
        cfg.num_classes = 2
        cfg.model = "tf"  # manually set the model to tf
    elif cfg.dataset == "femnist":
        # Support both local LEAF JSON (local) and HuggingFace (hf) sources
        dataset_train, dataset_test, dataset_val, split_dict = load_FEMNIST_data_hf(cfg)
        cfg.num_classes = 62  # 62 classes
        dataset = {
            "train": dataset_train,
            "test": dataset_test,
            "val": dataset_val,
            "split_dict": split_dict,
        }
        cfg.rho = cal_rho(cfg, dataset)
        return dataset
    else:
        raise ValueError(f"Invalid dataset.")

    # First, create validation set from original training set
    if hasattr(cfg, "data_val") and cfg.data_val > 0:
        dataset_val = get_subset(dataset_train, cfg.data_val, cfg.num_classes)

        # Remove validation samples from training set
        val_indices = set(dataset_val.indices)
        remaining_train_indices = [i for i in range(len(dataset_train)) if i not in val_indices]
        dataset_train = Subset(dataset_train, remaining_train_indices)
    else:
        # If no validation set is configured, create an empty validation set
        dataset_val = Subset(dataset_train, [])
        cfg.data_val = 0

    # Then, if subset is needed, take subset from remaining training data
    if cfg.subset:
        dataset_train = get_subset(dataset_train, cfg.data_train, cfg.num_classes)
        dataset_test = get_subset(dataset_test, cfg.data_test, cfg.num_classes)

    cfg.data_train = len(dataset_train)
    cfg.data_test = len(dataset_test)
    cfg.data_val = len(dataset_val)

    # data separation
    if cfg.noniid == 0:
        split_dict = split_iid(dataset_train, cfg.m)
    else:  # each client has cfg.noniid categories of data
        split_dict = split_noniid(dataset_train, cfg)
    dataset = {
        "train": dataset_train,
        "test": dataset_test,
        "val": dataset_val,
        "split_dict": split_dict,
    }
    cfg.rho = cal_rho(cfg, dataset)  # percentage of clients' datasets
    return dataset


class IMDBDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=512):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input = encoding["input_ids"].flatten()
        mask = ~encoding["attention_mask"].flatten().bool()
        return (input, mask), torch.tensor(label, dtype=torch.float32)


def load_IMDB_data(max_length=256, local_path=""):
    if local_path:
        print("Loading IMDB dataset from local dir...\n")
        dataset = load_from_disk(os.path.join(local_path, "local_imdb_dataset"))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_path, "local_tokenizer"))
    else:
        print("Downloading the IMDB dataset from the internet...")
        dataset = load_dataset("imdb")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # Extract train and test splits
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Create dataset wrappers
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, max_length)

    return train_dataset, test_dataset, tokenizer.vocab_size


def split_iid(dataset, num_clients):
    """Split the dataset in an iid way."""
    data_per_client = len(dataset) // num_clients  # The same amount of data per customer.
    data_indices = range(len(dataset))
    # Create a split dict, where key: client idx, value: client's data indices.
    split_dict = {i: np.array([], dtype="int32") for i in range(num_clients)}
    for i in range(num_clients):
        # allocate data
        split_dict[i] = np.random.choice(data_indices, data_per_client, replace=False)
        data_indices = list(set(data_indices) - set(split_dict[i]))  # remaining data
    return split_dict


def split_noniid(dataset, cfg):
    """
    Split the dataset in a non-iid way: each client only has a limited categoris of data.
    """

    shards_per_client = cfg.noniid  # number of categories per client
    num_clients = cfg.m
    data_per_client = len(dataset) // num_clients  # The same amount of data per customer.
    data_per_shard = data_per_client // shards_per_client
    num_shards = num_clients * shards_per_client
    shards = [i for i in range(num_shards)]
    # Create a split dict, where key: client idx, value: client's data indices.
    split_dict = {i: np.array([], dtype="int32") for i in range(num_clients)}
    # Load and sort labels.
    labels = np.asarray([data[1] for data in dataset]).argsort()
    # Load indices of data in the order of sorted labels.
    data_sorted = np.arange(len(dataset))[labels]
    # Split the dataset to clients.
    for i in range(num_clients):
        # Randomly select shards for client i.
        shards_i = set(np.random.choice(shards, shards_per_client, replace=False))
        # Allocate data in selected shards to client i.
        for shard_i in shards_i:
            start = shard_i * data_per_shard  # index of the first data in shard_i
            end = start + data_per_shard  # index of the last data in shard_i
            split_dict[i] = np.concatenate((split_dict[i], data_sorted[start:end]))
        shards = list(set(shards) - shards_i)  # Get indices of remaining shards.
    return split_dict


class DatasetSplit(Dataset):
    """Return client i's dataset."""

    def __init__(self, dataset, client_i):
        self.dataset = dataset["train"]  # whole training set
        self.indices_i = list(dataset["split_dict"][client_i])  # list of indices of client's data

    def __len__(self):
        return len(self.indices_i)

    def __getitem__(self, index):
        data_index = self.indices_i[index]
        data, label = self.dataset[data_index]
        return data, label


def get_subset(dataset, set_size, classes):
    """Get a subset with the same number of training data in each class."""
    num_per_class = set_size // classes
    class_indices = {i: [] for i in range(classes)}
    for idx, data in enumerate(dataset):
        label = int(data[1])
        if len(class_indices[label]) < num_per_class:
            class_indices[label].append(idx)
        if all(len(indices) == num_per_class for indices in class_indices.values()):
            break
    subset_indices = sum(class_indices.values(), [])
    return Subset(dataset, subset_indices)


def cal_rho(cfg, dataset):
    "Calculate the percentage of each client's dataset."
    rho = []
    for i in range(cfg.m):
        dataset_i = DatasetSplit(dataset, i)
        rho_i = len(dataset_i) / len(dataset["train"])  # percentage of the client's data
        rho.append(rho_i)
    return rho


class FEMNISTDataset(Dataset):
    """FEMNIST dataset class for handling individual client data."""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # FEMNIST images are 28x28 grayscale
            image = Image.fromarray(image.astype("uint8"), "L")

        if self.transform:
            image = self.transform(image)

        return image, label


def load_FEMNIST_data_hf(cfg):
    """Load FEMNIST directly from HuggingFace and split clients by writer_id.

    Strategy:
      1. Download flwrlabs/femnist (train split only).
      2. Group by writer_id so each writer is treated as an original user.
      3. Filter writers whose sample counts fall within (min_samples_per_user, max_samples_per_user).
      4. Randomly select cfg.m writers as clients (shrink cfg.m if not enough writers).
      5. Aggregate all samples, then build split_dict using the aggregated index order.
      6. Sample centralized validation/test sets based on cfg.data_val / cfg.data_test
         (prefer writers not assigned to clients; otherwise sample from the aggregated pool).

    Note: The HF version only provides a train split, so we apply a simple partition:
        - Aggregate the selected writers to create the training set.
        - Draw test/val data from remaining writers (preferred) or from the train pool.
    """
    # ========================= Preprocessing cache / reuse mechanism =========================
    # Users can set force_reload_femnist = True to force regeneration
    force_reload = getattr(cfg, "force_reload_femnist", False)
    cache_dir = os.path.join(os.path.dirname(os.getcwd()), "dataset", "femnist")
    processed_root = os.path.join(cache_dir, "processed")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(processed_root, exist_ok=True)

    # Build a signature from key parameters that affect the split; extend if finer granularity is needed
    sig_items = {
        "m": getattr(cfg, "m", None),
        "seed": getattr(cfg, "seed", None),
        "min": getattr(cfg, "min_samples_per_user", getattr(cfg, "min_samples", 10)),
        "max": getattr(cfg, "max_samples_per_user", getattr(cfg, "max_samples", 1000)),
        "test": getattr(cfg, "data_test", 0) or 0,
        "val": getattr(cfg, "data_val", 0) or 0,
    }
    signature = "_".join([f"{k}{v}" for k, v in sig_items.items()])
    processed_dir = os.path.join(processed_root, signature)

    split_path = os.path.join(processed_dir, "split_dict.json")
    meta_path = os.path.join(processed_dir, "meta.yaml")
    np_files = {
        "train_x": os.path.join(processed_dir, "train_x.npy"),
        "train_y": os.path.join(processed_dir, "train_y.npy"),
        "test_x": os.path.join(processed_dir, "test_x.npy"),
        "test_y": os.path.join(processed_dir, "test_y.npy"),
        "val_x": os.path.join(processed_dir, "val_x.npy"),
        "val_y": os.path.join(processed_dir, "val_y.npy"),
    }

    def _all_exist():
        return all(os.path.isfile(p) for p in np_files.values()) and os.path.isfile(split_path)

    if (not force_reload) and _all_exist():
        try:
            print(f"[FEMNIST][Cache] Trying to load processed data from cache: {processed_dir}")
            # Read split_dict
            with open(split_path, "r") as fjs:
                split_dict_raw = json.load(fjs)
            # Keys may be strings; convert them to int
            split_dict = {int(k): np.array(v, dtype="int32") for k, v in split_dict_raw.items()}

            # Load arrays
            train_array = np.load(np_files["train_x"])  # (N,28,28)
            train_labels_array = np.load(np_files["train_y"]).astype(np.int64)
            test_array = (
                np.load(np_files["test_x"])
                if os.path.isfile(np_files["test_x"])
                else np.empty((0, 28, 28), dtype=np.uint8)
            )
            test_labels_array = (
                np.load(np_files["test_y"]).astype(np.int64)
                if os.path.isfile(np_files["test_y"])
                else np.empty((0,), dtype=np.int64)
            )
            val_array = (
                np.load(np_files["val_x"])
                if os.path.isfile(np_files["val_x"])
                else np.empty((0, 28, 28), dtype=np.uint8)
            )
            val_labels_array = (
                np.load(np_files["val_y"]).astype(np.int64)
                if os.path.isfile(np_files["val_y"])
                else np.empty((0,), dtype=np.int64)
            )

            # Use transforms consistent with the original logic
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            transform_eval = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

            dataset_train = FEMNISTDataset(train_array, train_labels_array, transform_train)
            dataset_test = FEMNISTDataset(test_array, test_labels_array, transform_eval)
            dataset_val = FEMNISTDataset(val_array, val_labels_array, transform_eval)

            cfg.data_train = len(dataset_train)
            cfg.data_test = len(dataset_test)
            cfg.data_val = len(dataset_val)
            cfg.num_classes = 62  # Fixed value

            print(
                f"[FEMNIST][Cache] Loaded successfully - Train:{cfg.data_train} Test:{cfg.data_test} Val:{cfg.data_val} (m={len(split_dict)})"
            )
            return dataset_train, dataset_test, dataset_val, split_dict
        except Exception as e:
            print(f"[FEMNIST][Cache][Warn] Failed to load cache; rebuilding from scratch: {e}")

    # ========================= If cache missing or forced refresh -> regular pipeline =========================

    # Cache directory: set femnist_cache_dir in the config; otherwise use the provided default path
    # (cache_dir is already created earlier)
    print(f"[HF] Loading flwrlabs/femnist via datasets.load_dataset ... cache_dir={cache_dir}")
    ds = load_dataset("flwrlabs/femnist", split="train", cache_dir=cache_dir)  # Dataset

    # Extract fields
    images = ds["image"]  # list of PIL Images
    labels = ds["character"]  # int labels 0..61
    writer_ids = ds["writer_id"]  # string id per writer

    # Group indices by writer
    writer_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, w in enumerate(writer_ids):
        writer_to_indices[w].append(idx)

    min_samples = getattr(cfg, "min_samples_per_user", 10)
    max_samples = getattr(cfg, "max_samples_per_user", 1000)

    # Filter writers based on sample counts
    filtered_writers = [
        w for w, inds in writer_to_indices.items() if min_samples <= len(inds) <= max_samples
    ]
    print(
        f"[HF] total writers={len(writer_to_indices)}, filtered={len(filtered_writers)} (range {min_samples}-{max_samples})"
    )

    if len(filtered_writers) == 0:
        raise RuntimeError("[HF] No writers satisfy sample count constraints.")

    if len(filtered_writers) < cfg.m:
        print(
            f"[HF] Warning: request m={cfg.m} > filtered writers {len(filtered_writers)}, shrink m."
        )
        cfg.m = len(filtered_writers)

    # Randomly choose writers for clients
    rng = np.random.default_rng(getattr(cfg, "seed", None))
    selected_writers = rng.choice(filtered_writers, cfg.m, replace=False)

    # Build federated training data
    all_train_imgs = []
    all_train_labels = []
    split_dict = {}
    cursor = 0
    selected_set = set(selected_writers)

    for cid, w in enumerate(selected_writers):
        inds = writer_to_indices[w]
        # Collect every sample written by the selected writer
        for i in inds:
            all_train_imgs.append(images[i])
            all_train_labels.append(labels[i])
        new_inds = list(range(cursor, cursor + len(inds)))
        split_dict[cid] = np.array(new_inds, dtype="int32")
        cursor += len(inds)
        print(f"[HF] Client {cid} writer={w} samples={len(inds)}")

    # ========== Export per-client sample counts to YAML ==========
    try:
        client_data_vols = {int(cid): int(len(idxs)) for cid, idxs in split_dict.items()}
        # Default export path: cache_dir with a name derived from m and seed; override via cfg.femnist_D
        if not hasattr(cfg, "femnist_D"):
            cfg.femnist_D = os.path.join(
                os.getcwd(), "utils", f"femnist_D_m{cfg.m}_seed{cfg.seed}.yaml"
            )
        export_path = cfg.femnist_D
        with open(export_path, "w") as f_yaml:
            yaml.dump(client_data_vols, f_yaml, sort_keys=True, allow_unicode=True)
        print(f"[HF] Saved client data volumes to {export_path}")
    except Exception as e:
        print(f"[HF][Warn] Failed to save client data volumes yaml: {e}")

    # Use remaining writers for centralized test/val splits (fallback to train pool if necessary)
    remaining_writers = [w for w in writer_to_indices.keys() if w not in selected_set]
    rem_indices = []
    for w in remaining_writers:
        rem_indices.extend(writer_to_indices[w])

    # Shuffle the remaining indices
    rng.shuffle(rem_indices)

    def sample_from_pool(pool: List[int], need: int) -> List[int]:
        if need <= 0:
            return []
        if len(pool) >= need:
            return pool[:need]
        # If insufficient, sample additional indices from the training pool (without strict uniqueness)
        extra = rng.choice(cursor, need - len(pool), replace=False).tolist()
        return pool + extra

    data_test_cfg = getattr(cfg, "data_test", 0) or 0
    data_val_cfg = getattr(cfg, "data_val", 0) or 0

    test_sel = sample_from_pool(rem_indices, data_test_cfg) if data_test_cfg else []
    # Remove indices chosen for the test set
    rem_after_test = [x for x in rem_indices if x not in set(test_sel)]
    val_sel = sample_from_pool(rem_after_test, data_val_cfg) if data_val_cfg else []

    # Build NumPy-formatted training data
    # Convert PIL images to 28x28 uint8 arrays
    def pil_to_np(img):
        if isinstance(img, Image.Image):
            return np.array(img.convert("L"), dtype="uint8")
        return np.array(img, dtype="uint8")

    train_array = np.stack([pil_to_np(im) for im in all_train_imgs])  # (N,28,28)
    train_labels_array = np.array(all_train_labels, dtype=np.int64)

    # Build test / val datasets
    test_array = (
        np.stack([pil_to_np(images[i]) for i in test_sel])
        if test_sel
        else np.empty((0, 28, 28), dtype=np.uint8)
    )
    test_labels_array = (
        np.array([labels[i] for i in test_sel], dtype=np.int64)
        if test_sel
        else np.empty((0,), dtype=np.int64)
    )
    val_array = (
        np.stack([pil_to_np(images[i]) for i in val_sel])
        if val_sel
        else np.empty((0, 28, 28), dtype=np.uint8)
    )
    val_labels_array = (
        np.array([labels[i] for i in val_sel], dtype=np.int64)
        if val_sel
        else np.empty((0,), dtype=np.int64)
    )

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset_train = FEMNISTDataset(train_array, train_labels_array, transform_train)
    dataset_test = FEMNISTDataset(test_array, test_labels_array, transform_eval)
    dataset_val = FEMNISTDataset(val_array, val_labels_array, transform_eval)

    # Update cfg with actual dataset sizes
    cfg.data_train = len(dataset_train)
    cfg.data_test = len(dataset_test)
    cfg.data_val = len(dataset_val)

    print(
        f"[HF] Final sizes - Train:{cfg.data_train} Test:{cfg.data_test} Val:{cfg.data_val} (writers {cfg.m})"
    )

    # ========================= Save processed data to cache =========================
    try:
        os.makedirs(processed_dir, exist_ok=True)
        # Save arrays
        np.save(np_files["train_x"], train_array)
        np.save(np_files["train_y"], train_labels_array)
        np.save(np_files["test_x"], test_array)
        np.save(np_files["test_y"], test_labels_array)
        np.save(np_files["val_x"], val_array)
        np.save(np_files["val_y"], val_labels_array)
        # Save the split_dict
        with open(split_path, "w") as fjs:
            json.dump({int(k): v.tolist() for k, v in split_dict.items()}, fjs)
        # Save metadata
        meta = {
            "signature": signature,
            "params": sig_items,
            "actual_m": cfg.m,
            "data_train": cfg.data_train,
            "data_test": cfg.data_test,
            "data_val": cfg.data_val,
            "num_classes": 62,
            "force_reload_used": force_reload,
        }
        with open(meta_path, "w") as fm:
            yaml.dump(meta, fm, allow_unicode=True)
        print(f"[FEMNIST][Cache] Stored processed data at {processed_dir}")
    except Exception as e:
        print(f"[FEMNIST][Cache][Warn] Failed to save cache: {e}")

    return dataset_train, dataset_test, dataset_val, split_dict
