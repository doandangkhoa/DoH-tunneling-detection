import argparse
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.isolation_forest.preprocess import load_split, validate_training_data


def compute_anomaly_score(model: IsolationForest, X: np.ndarray, method: str) -> np.ndarray:
    if method == "negative_score_samples":
        return -model.score_samples(X)
    if method == "negative_decision_function":
        return -model.decision_function(X)
    raise ValueError(f"Không hỗ trợ score method: {method}")


def find_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    p = precisions[:-1]
    r = recalls[:-1]
    f1s = np.divide(2 * p * r, p + r, out=np.zeros_like(thresholds, dtype=float), where=(p + r) > 0)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest baseline")
    parser.add_argument("--config", type=str, default="configs/isolation_forest.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_data = load_split(data_cfg["directory"], "train", data_cfg)
    val_data = load_split(data_cfg["directory"], cfg["selection"]["split"], data_cfg)

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    validate_training_data(y_train, cfg["training"])

    model = IsolationForest(**cfg["model"])
    t0 = time.perf_counter()
    model.fit(X_train)
    train_seconds = time.perf_counter() - t0

    score_method = cfg["score"]["method"]
    val_scores = compute_anomaly_score(model, X_val, score_method)
    val_ap = float(average_precision_score(y_val, val_scores))
    best_threshold, best_val_f1 = find_best_f1_threshold(y_val, val_scores)

    val_pred = (val_scores >= best_threshold).astype(int)
    val_precision = float(precision_score(y_val, val_pred, zero_division=0))
    val_recall = float(recall_score(y_val, val_pred, zero_division=0))

    model_dir = Path(cfg["output"]["model_directory"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "isolation_forest.joblib"
    joblib.dump(model, model_path)

    exp_dir = Path(cfg["output"]["experiment_directory"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Copy preprocessing.json vào experiment_directory
    prep_src = Path(data_cfg["preprocessing_file"])
    if prep_src.exists():
        shutil.copy2(prep_src, exp_dir / "preprocessing.json")

    feature_names = train_data.get("feature_names")
    feature_order = feature_names.tolist() if feature_names is not None else []

    summary = {
        "model": "isolation_forest",
        "seed": cfg["model"].get("random_state"),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "num_features": int(X_train.shape[1]),
        "feature_order": feature_order,
        "train_seconds": round(train_seconds, 4),
        "model_size_bytes": model_path.stat().st_size,
        "val_average_precision": val_ap,
        "val_f1": best_val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "threshold": best_threshold,
        "score_method": score_method,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (exp_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Khóa threshold vào locked_config.yaml để dùng cho predict
    cfg["threshold"]["value"] = best_threshold
    with (exp_dir / "locked_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[DONE] Train xong trong {train_seconds:.3f}s | Model: {model_path}")
    print(f"[VAL]  AP={val_ap:.4f} | Best F1={best_val_f1:.4f} | Threshold={best_threshold:.6f}")


if __name__ == "__main__":
    main()