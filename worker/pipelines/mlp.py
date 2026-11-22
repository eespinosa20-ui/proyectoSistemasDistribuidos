import torch
from torch import nn

from .common import objects, regression, seed_all, train


def run(config, data_dir, checkpoint_dir, progress_callback):
    seed_all(config["seed"])
    data = {key: (torch.from_numpy(x), torch.from_numpy(y)) for key, (x, y) in objects(data_dir).items()}
    sizes = [data["train"][0].shape[1], *config["hyperparameters"].get("hidden_layers", [32, 16])]
    layers = []
    for left, right in zip(sizes, sizes[1:]):
        layers.extend((nn.Linear(left, right), nn.ReLU()))
    model = nn.Sequential(*layers, nn.Linear(sizes[-1], 1), nn.Flatten(0))
    return train(config, checkpoint_dir, model, data, nn.MSELoss(), regression, progress_callback)
