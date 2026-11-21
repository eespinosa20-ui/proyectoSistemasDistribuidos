from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.nn.functional import interpolate

from .common import seed_all, train


def score(y, logits):
    predictions = logits.argmax(dim=1)
    return {"accuracy": float(accuracy_score(y, predictions)),
            "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1, 2]).tolist()}


def run(config, data_dir, checkpoint_dir, progress_callback):
    seed_all(config["seed"])
    hp = config["hyperparameters"]
    size, filters, count = (int(hp.get(key, default)) for key, default in
                            (("image_size", 32), ("num_filters", 8), ("num_layers", 2)))
    if min(size, filters, count) < 1:
        raise ValueError("CNN dimensions must be positive")
    root = Path(data_dir)
    images = np.load(root / "recortes.npy", mmap_mode="r")
    labels = np.load(root / "recortes_labels.npy")
    data = {}
    for split in ("train", "val", "test"):
        mask = labels["split"] == split
        x = torch.from_numpy(images[mask]).permute(0, 3, 1, 2).float() / 255
        data[split] = (interpolate(x, size=(size, size), mode="bilinear", align_corners=False),
                       torch.from_numpy(labels["label"][mask]))
    layers, channels = [], 3
    for _ in range(count):
        layers.extend((nn.Conv2d(channels, filters, 3, padding=1), nn.ReLU(),
                       nn.MaxPool2d(2, ceil_mode=True)))
        channels = filters
    model = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 3))
    return train(config, checkpoint_dir, model, data, nn.CrossEntropyLoss(), score, progress_callback)
