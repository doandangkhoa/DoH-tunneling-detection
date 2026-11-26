"""D — Train 1D-CNN; chọn checkpoint theo validation AP, threshold tối đa F1 trên val.

Chạy tại root repository:
    python src/cnn1d/train.py --config configs/cnn1d.yaml
Không đụng tới test. Sau đó chạy predict.py để xuất prediction test.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.cnn1d.build_sequences import load_split, resolve, to_tensors  # noqa: E402
from src.cnn1d.model import build_model, check_mask_invariance  # noqa: E402
from src.cnn1d.predict import (MODEL_NAME, environment, pick_device,  # noqa: E402
                               predict_logits, prediction_frame, utf8_console, write_json)

MASK_CHECK_SAMPLES = 1000  # số mẫu mỗi split (train, val) dùng kiểm tra padding


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def best_f1_threshold(y: np.ndarray, scores: np.ndarray):
    """Threshold làm F1 lớn nhất; quy ước dự đoán 1 khi score >= threshold."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def main(argv=None):
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cnn1d.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    assert not cfg["data"]["refit_preprocessing"], "Không fit lại preprocessing"

    seed = cfg["random_seed"]
    set_seed(seed)
    device = pick_device(cfg["training"]["device"])
    tcfg = cfg["training"]
    assert tcfg["optimizer"] == "adam" and tcfg["loss"] == "binary_cross_entropy_with_logits"

    # --- Dữ liệu: chỉ train + val ---
    train, val = load_split(cfg["data"], "train"), load_split(cfg["data"], "val")
    X_tr, M_tr, y_tr = to_tensors(train)
    X_va, M_va, y_va = to_tensors(val)
    print(f"train {tuple(X_tr.shape)} | val {tuple(X_va.shape)} | device {device}")

    loader = DataLoader(TensorDataset(X_tr, M_tr, y_tr), batch_size=tcfg["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])

    # --- Train loop: mỗi epoch đánh giá val, giữ checkpoint có AP cao nhất ---
    history, best = [], {"ap": -1.0, "epoch": None, "state": None}
    start = time.perf_counter()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb, mb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)

        logits = predict_logits(model, X_va, M_va, tcfg["batch_size"], device)
        val_loss = F.binary_cross_entropy_with_logits(logits, y_va).item()
        val_ap = float(average_precision_score(val["y"], torch.sigmoid(logits).numpy()))
        history.append({"epoch": epoch, "train_loss": total / len(y_tr),
                        "val_loss": val_loss, "val_average_precision": val_ap})
        if val_ap > best["ap"]:
            best = {"ap": val_ap, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        print(f"epoch {epoch:3d} | train_loss {total / len(y_tr):.4f} | "
              f"val_loss {val_loss:.4f} | val_AP {val_ap:.4f}" + (" *" if best["epoch"] == epoch else ""))
    train_time = time.perf_counter() - start

    # --- Lưu checkpoint được chọn ---
    out = cfg["output"]
    model_dir, exp_dir = resolve(out["model_directory"]), resolve(out["experiment_directory"])
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "best.pt"
    torch.save(best["state"], checkpoint)
    model.load_state_dict(best["state"])

    # --- Khóa threshold trên validation bằng checkpoint đã chọn ---
    val_scores = torch.sigmoid(predict_logits(model, X_va, M_va, tcfg["batch_size"], device)).numpy()
    threshold, val_f1 = best_f1_threshold(val["y"], val_scores)
    val_pred_file = resolve(out["predictions_file"]).with_name("validation_predictions.parquet")
    val_pred_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(val, val_scores, threshold, "val").to_parquet(val_pred_file, index=False)

    # --- Kiểm tra mask/padding ở eval trên mẫu train + val ---
    rng = np.random.default_rng(seed)
    pick_tr = rng.choice(len(y_tr), min(MASK_CHECK_SAMPLES, len(y_tr)), replace=False)
    pick_va = rng.choice(len(y_va), min(MASK_CHECK_SAMPLES, len(y_va)), replace=False)
    mask_check = check_mask_invariance(
        model, torch.cat([X_tr[pick_tr], X_va[pick_va]]).to(device),
        torch.cat([M_tr[pick_tr], M_va[pick_va]]).to(device), seed=seed)
    mask_check["splits"] = ["train", "val"]
    write_json(exp_dir / "mask_check.json", mask_check)

    # --- Các file bàn giao (TRAINING_GUIDE mục 4.1–4.2) ---
    with open(exp_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    shutil.copy2(resolve(cfg["data"]["preprocessing_file"]), model_dir / "preprocessing.json")
    write_json(model_dir / "feature_names.json",
               {"channels": train["channels"], "stored_layout": cfg["data"]["stored_layout"],
                "model_layout": cfg["data"]["model_layout"]})

    used = copy.deepcopy(cfg)
    used["threshold"]["value"] = threshold
    used["output"]["checkpoint_file"] = str(Path(out["model_directory"]) / "best.pt").replace("\\", "/")
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(used, sort_keys=False, allow_unicode=True), encoding="utf-8")

    write_json(exp_dir / "run_summary.json", {
        "model": MODEL_NAME, "data_profile": "common", "data_directory": cfg["data"]["directory"],
        "seed": seed,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": None,
        "selection_metric": "average_precision", "validation_average_precision": best["ap"],
        "threshold_method": "maximize_f1", "threshold": threshold, "validation_f1": val_f1,
        "selected_epoch": best["epoch"], "epochs_trained": tcfg["epochs"],
        "score_method": "sigmoid_logits",
        "architecture": {**cfg["model"], "input_layout": "N,C,L",
                         "mask_handling": "input*mask before each conv, zero padding after "
                                          "each conv+ReLU, masked mean pooling"},
        "train_time_seconds": train_time,
        "inference_time_seconds": None, "inference_ms_per_flow": None,  # predict.py điền
        "model_size_bytes": checkpoint.stat().st_size,
        "mask_check_passed": mask_check["passed"],
        **environment(device),
    })

    print(f"Chọn epoch {best['epoch']} (val AP {best['ap']:.4f}); threshold {threshold:.6f} "
          f"(val F1 {val_f1:.4f}); mask check: {'PASS' if mask_check['passed'] else 'FAIL'}")
    print(f"Tiếp theo: python src/cnn1d/predict.py --config {(exp_dir / 'config.yaml').as_posix()}")


if __name__ == "__main__":
    main()
