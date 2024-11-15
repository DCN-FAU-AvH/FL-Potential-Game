import torch
from torch import nn


def init_model(cfg, requires_grad=True):
    """Initialize a model."""
    model = CNN(cfg)
    cfg.device = torch.device(cfg.device)
    model.requires_grad_(requires_grad)  # requires gradients or not
    return model.to(cfg.device)


class CNN(nn.Module):
    """CNN used in the FedAvg paper."""

    def __init__(self, cfg):
        super().__init__()
        dim_in_channels = {"mnist": 1, "cifar10": 3, "cifar100": 3}
        dim_in_fc = {"mnist": 1024, "cifar10": 1600, "cifar100": 1600}
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


def get_activation(cfg):
    """Selects the activation function based on the cfg."""
    if cfg.activation == "relu":
        return nn.ReLU(inplace=True)  # to save memory
    elif cfg.activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Invalid activation function.")


def get_loss_func(cfg):
    """Selects the loss function based on the cfg."""
    if cfg.loss == "mse":
        return nn.MSELoss()
    elif cfg.loss == "cn":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function.")
