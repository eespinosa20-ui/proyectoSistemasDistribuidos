import joblib
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import objects, publish, regression, seed_all


def run(config, data_dir, checkpoint_dir, progress_callback):
    seed_all(config["seed"])
    hp = config["hyperparameters"]
    variant = hp.get("variant", "ridge")
    estimator = {"ridge": Ridge, "lasso": Lasso}[variant](alpha=float(hp.get("alpha", 1)))
    model = make_pipeline(StandardScaler(), estimator) if hp.get("normalize", False) else estimator
    data = objects(data_dir)
    model.fit(*data["train"])
    metrics = {split: regression(y, model.predict(x)) for split, (x, y) in data.items() if split != "train"}
    progress_callback(1, 1.0)
    return publish(config, checkpoint_dir, "joblib", lambda file: joblib.dump(model, file), metrics)
