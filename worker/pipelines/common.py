import csv
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = ("width", "length", "height", "distance", "visibility", "num_radar_pts", "velocity")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as output:
            write(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def publish(config, root, suffix, write, metrics):
    path = Path(root) / "results" / config["job_id"] / str(config.get("attempt", 0)) / f"model.{suffix}"
    atomic(path, write)
    atomic(path.with_name("metrics.json"), lambda file: file.write(json.dumps(metrics, allow_nan=False).encode()))
    return {"model_path": str(path), "metrics": metrics}


def objects(data_dir):
    with (Path(data_dir) / "objetos.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    output = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        x = np.array([[float(row[key]) for key in FEATURES] for row in selected], dtype=np.float32)
        y = np.array([float(row["num_lidar_pts"]) for row in selected], dtype=np.float32)
        output[split] = (x, y)
    return output


def regression(y, predicted):
    return {"rmse": float(mean_squared_error(y, predicted) ** 0.5), "r2": float(r2_score(y, predicted))}


def train(config, root, model, data, loss, score, progress):
    hp = config["hyperparameters"]
    epochs = int(hp.get("epochs", 5))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hp.get("learning_rate", 0.001)))
    path = Path(root) / f'{config["job_id"]}.pt'
    attempt = config.get("attempt", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock", timeout=5)
    start = 0
    with lock:
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start = checkpoint["epoch"]
    x, y = data["train"]
    for epoch in range(start + 1, epochs + 1):
        progress(epoch - 1, (epoch - 1) / epochs)
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        for batch in torch.randperm(len(x), generator=generator).split(16):
            optimizer.zero_grad()
            loss(model(x[batch]), y[batch]).backward()
            optimizer.step()
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "attempt": attempt}
        with lock:
            progress(epoch, epoch / epochs)
            atomic(path, lambda file: torch.save(state, file))
    if start >= epochs:
        progress(start, 1.0)
    model.eval()
    with torch.no_grad():
        metrics = {split: score(data[split][1], model(data[split][0])) for split in ("val", "test")}
    return publish(config, root, "pt", lambda file: torch.save(
        {"model": model.state_dict(), "hyperparameters": hp}, file), metrics)
