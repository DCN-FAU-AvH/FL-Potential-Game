"""Model architectures and initialization helpers for FL experiments.

Includes:
    * CNN baseline (FedAvg style) for image datasets.
    * Lightweight Transformer classifier for IMDB (binary sentiment).
Per-run folder creation + resume logic handled inside init_model.
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils.utils import *


def init_model(cfg, requires_grad=True):
    """Instantiate model, configure loss, allocate result folder, select device & optionally resume."""
    if "tf" in cfg.model:
        model = TransformerModel(
            vocab_size=cfg.vocab_size,
            emb_dim=cfg.emb_dim,
            nhead=cfg.nhead,
            nhid=cfg.nhid,
            nlayers=cfg.nlayers,
        )
    elif "cnn" in cfg.model:
        if cfg.model == "cnn":
            model = CNN(cfg)
    else:
        raise ValueError(f"Invalid model.")

    model.loss = get_loss_func(cfg)

    # creat a folder and save the results
    cfg.dir_res = create_folder(cfg)  # folder to save the results
    print(f"Working in {cfg.dir_res}.\n")

    select_device(cfg)  # select the appropriate device (priority: CUDA > MPS > CPU)
    # If continuing work, load the model from the latest round
    if cfg.T0 > 0 :
        print(f"\nContinuing previous work from round {cfg.T0}.\n")
        path_model = os.path.join(cfg.dir_res, "0_model.pth")
        model.load_state_dict(torch.load(path_model, map_location=torch.device(cfg.device)))

    # save the test configuration before cfg.device is overwritten
    save_config(cfg, cfg.dir_res)
    cfg.device = torch.device(cfg.device)
    model.requires_grad_(requires_grad)  # requires gradients or not
    return model.to(cfg.device)


class CNN(nn.Module):
    """Two-convolution + two-linear CNN similar to FedAvg reference implementation."""

    def __init__(self, cfg):
        super().__init__()
        dim_in_channels = {"mnist": 1, "cifar10": 3, "cifar100": 3, "femnist": 1}
        dim_in_fc = {"mnist": 1024, "cifar10": 1600, "cifar100": 1600, "femnist": 1024}
        self.loss = get_loss_func(cfg)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in_channels[cfg.dataset], 32, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_in_fc[cfg.dataset], 512),
            nn.ReLU(True),
            nn.Linear(512, cfg.num_classes),  # output layer
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten the data from dim=1
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)

        encoder_layers = TransformerEncoderLayer(emb_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.emb_dim = emb_dim
        self.decoder = nn.Linear(emb_dim, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, data):
        # src shape: [batch_size, seq_len]
        # Need to transpose to [seq_len, batch_size] for the Transformer
        src = data[0]
        src_mask = data[1]
        src = src.transpose(0, 1)

        src = self.embedding(src) * np.sqrt(self.emb_dim)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        output = output[0, :, :]

        output = self.decoder(output)
        output = self.sigmoid(output)

        return output.squeeze()


def get_activation(cfg):
    """Return activation layer (ReLU inplace or Sigmoid) from cfg.activation."""
    if cfg.activation == "relu":
        return nn.ReLU(inplace=True)  # to save memory
    elif cfg.activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Invalid activation function.")


def get_loss_func(cfg):
    """Map dataset / cfg.loss to torch loss module (CrossEntropy | MSE | BCE)."""
    if cfg.dataset in ["imdb"]:
        cfg.loss = "bce"
        return nn.BCELoss()
    elif cfg.loss == "mse":
        return nn.MSELoss()
    elif cfg.loss == "cn":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function.")


def select_device(cfg):
    """
    Selects the appropriate device (CUDA, MPS, or CPU)
    based on availability and updates the configuration object.
    """
    if torch.cuda.is_available():
        if "cuda" not in cfg.device:
            cfg.device = "cuda"
    elif torch.backends.mps.is_available():
        cfg.device = "mps"
    else:
        cfg.device = "cpu"
    cfg.zdevice = cfg.device  # save the original device name
