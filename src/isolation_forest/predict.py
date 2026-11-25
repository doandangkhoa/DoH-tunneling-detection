import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.isolation_forest.preprocess import load_split
from src.isolation_forest.train import compute_anomaly_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Isolation Forest inference on test split")
    parser.add_argument("--config", type=str, default="experiments/isolation_forest/locked_config.yaml")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    threshold = cfg["threshold"]["value"]
    if threshold is None:
        raise ValueError("threshold.value đang là null. Hãy chạy train.py trước để khóa ngưỡng!")
    threshold = float(threshold)

    model_path = Path(cfg["output"]["model_directory"]) / "isolation_forest.joblib"
    model = joblib.load(model_path)

    data_cfg = cfg["data"]
    test_data = load_split(data_cfg["directory"], args.split, data_cfg)
    X_test = test_data["X"]
    y_test = test_data["y"]
    sample_ids = test_data["sample_id"]

    score_method = cfg["score"]["method"]

    # Warm-up trước khi đo thời gian inference (theo Bước 5 README)
    _ = compute_anomaly_score(model, X_test[: min(256, len(X_test))], score_method)

    t0 = time.perf_counter()
    y_score = compute_anomaly_score(model, X_test, score_method)
    inference_seconds = time.perf_counter() - t0

    y_pred = (y_score >= threshold).astype(int)

    df_out = pd.DataFrame({
        "sample_id": sample_ids,
        "model": "isolation_forest",
        "y_true": y_test.astype(int),
        "y_score": y_score.astype(float),
        "y_pred": y_pred,
        "threshold": threshold,
        "split": args.split,
    })

    out_path = Path(cfg["output"]["predictions_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    exp_dir = Path(cfg["output"]["experiment_directory"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    infer_stats = {
        "split": args.split,
        "num_samples": int(len(X_test)),
        "batch_size": int(len(X_test)),
        "warmup_samples": int(min(256, len(X_test))),
        "inference_seconds": round(inference_seconds, 6),
        "ms_per_sample": round((inference_seconds / len(X_test)) * 1000, 6),
        "threshold_used": threshold,
        "predicted_positive_rate": round(float(y_pred.mean()), 6),
    }
    with (exp_dir / "inference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(infer_stats, f, indent=2)

    print(f"[DONE] Đã xuất {len(df_out)} dòng ra {out_path}")
    print(f"[TIME] Inference: {inference_seconds:.4f}s ({infer_stats['ms_per_sample']:.4f} ms/flow)")


if __name__ == "__main__":
    main()