"""D — Nạp checkpoint + threshold đã khóa, xuất prediction test và đo inference.

Chạy sau train (dùng config thực tế đã ghi threshold):
    python src/cnn1d/predict.py --config experiments/cnn1d/config.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cnn1d.build_sequences import ROOT, load_split, log_sequences, normalized_inputs, resolve, to_tensors  # noqa: E402
from src.cnn1d.model import build_model  # noqa: E402

MODEL_NAME = "cnn1d"
WARMUP_RUNS, TIMED_RUNS = 1, 3


def utf8_console():
    # Console Windows (cp1258) không in được tiếng Việt có dấu khi output bị pipe/tee
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def sync(device):
    # GPU chạy bất đồng bộ: phải đồng bộ thì mới đo đúng thời gian
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict_logits(model, X, mask, batch_size, device) -> torch.Tensor:
    model.eval()
    out = [model(X[i:i + batch_size].to(device), mask[i:i + batch_size].to(device)).float().cpu()
           for i in range(0, len(X), batch_size)]
    return torch.cat(out)


def prediction_frame(d: dict, scores: np.ndarray, threshold: float, split: str) -> pd.DataFrame:
    """Đúng schema chung ở TRAINING_GUIDE mục 4."""
    scores = scores.astype(float)
    assert np.isfinite(scores).all(), "score có NaN/inf"
    return pd.DataFrame({
        "sample_id": d["sample_id"], "model": MODEL_NAME, "y_true": d["y"].astype(int),
        "y_score": scores, "y_pred": (scores >= threshold).astype(int),
        "threshold": float(threshold), "split": split,
    })


def environment(device: torch.device) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    import sklearn
    return {
        "git_commit": commit,
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda"
        else (platform.processor() or platform.machine()),
        "device": str(device),
        "python_version": platform.python_version(),
        "library_versions": {"torch": torch.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/cnn1d/config.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))

    threshold = cfg["threshold"]["value"]
    if threshold is None:  # chưa khóa ngưỡng bằng validation thì không được chạy test
        sys.exit("threshold.value đang null: chạy train.py trước và dùng experiments/cnn1d/config.yaml")

    device = pick_device(cfg["training"]["device"])
    model = build_model(cfg["model"]).to(device)
    checkpoint = resolve(cfg["output"]["checkpoint_file"])
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    if cfg["threshold"]["method"] == "minimize_worst_browser_fpr":
        test = log_sequences(cfg["data"], "test")
        preprocessing = json.loads((checkpoint.parent / "preprocessing.json").read_text(encoding="utf-8"))
        X, mask = normalized_inputs(test["X"], test["mask"], preprocessing["mean"], preprocessing["std"])
    else:
        test = load_split(cfg["data"], "test")
        X, mask, _ = to_tensors(test)
    batch_size = cfg["training"]["batch_size"]

    # Warm-up rồi đo nhiều lượt predict toàn bộ test
    for _ in range(WARMUP_RUNS):
        predict_logits(model, X, mask, batch_size, device)
    times = []
    for _ in range(TIMED_RUNS):
        sync(device)
        start = time.perf_counter()
        logits = predict_logits(model, X, mask, batch_size, device)
        sync(device)
        times.append(time.perf_counter() - start)

    scores = torch.sigmoid(logits).numpy()
    frame = prediction_frame(test, scores, threshold, "test")
    out_file = resolve(cfg["output"]["predictions_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_file, index=False)

    # Bổ sung phần test/inference vào run_summary do train.py tạo
    summary_file = resolve(cfg["output"]["experiment_directory"]) / "run_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    inference = float(np.mean(times))
    summary.update({
        "n_test": int(len(frame)),
        "inference_time_seconds": inference,
        "inference_ms_per_flow": inference * 1000 / len(frame),
        "inference_runs_seconds": times,
        "batch_size": batch_size,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "includes_preprocessing": False,  # X đã nạp sẵn vào tensor, chỉ đo forward + chuyển device
        **environment(device),
    })
    write_json(summary_file, summary)
    print(f"Đã ghi {out_file} ({len(frame)} flow, threshold={threshold:.6f}); "
          f"inference {inference:.3f}s = {summary['inference_ms_per_flow']:.4f} ms/flow")

    y_true, y_pred = frame["y_true"], frame["y_pred"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = f"{fp / (fp + tn):.6f}" if fp + tn else "N/A"
    print("\nCNN1D - TEST EVALUATION")
    print("Confusion matrix (rows: actual, columns: predicted)")
    print("                 Benign  Tunnel")
    print(f"Actual benign    {tn:6d}  {fp:6d}")
    print(f"Actual tunnel    {fn:6d}  {tp:6d}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.6f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.6f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.6f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.6f}")
    print(f"FPR:       {fpr}")
    if y_true.nunique() == 2:
        print(f"ROC-AUC:   {roc_auc_score(y_true, scores):.6f}")
        print(f"AP:        {average_precision_score(y_true, scores):.6f}")
    else:
        print("ROC-AUC and AP: N/A (requires both classes)")
    print()
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Benign", "Tunnel"],
        digits=6, zero_division=0,
    ))


if __name__ == "__main__":
    main()
