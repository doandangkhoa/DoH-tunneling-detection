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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cnn1d.build_sequences import ROOT, load_split, resolve, to_tensors  # noqa: E402
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


if __name__ == "__main__":
    main()
