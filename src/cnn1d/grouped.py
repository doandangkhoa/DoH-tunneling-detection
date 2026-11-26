"""Train CNN1D with fixed epochs and select a threshold from grouped predictions."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.cnn1d.build_sequences import log_sequences, normalized_inputs, resolve
from src.cnn1d.model import build_model, check_mask_invariance
from src.cnn1d.predict import MODEL_NAME, environment, pick_device, predict_logits, write_json


def train_grouped(cfg, set_seed):
    data_cfg = cfg["data"]
    development = log_sequences(data_cfg, "development")
    values, mask, labels, ids = (
        development[key] for key in ("X", "mask", "y", "sample_id")
    )
    manifest = pd.read_parquet(resolve(data_cfg["fold_manifest"])).set_index("sample_id")
    if not manifest.index.is_unique or len(np.unique(ids)) != len(ids) or set(ids) != set(manifest.index):
        raise ValueError("Fold manifest must cover each development sample exactly once")
    folds = manifest.loc[ids].copy()
    if not np.array_equal(folds.label, labels) or folds[["split_group", "source", "fold"]].isna().any().any():
        raise ValueError("Invalid labels or metadata in fold manifest")
    if set(folds.fold) != {0, 1, 2} or folds.groupby("split_group").fold.nunique().ne(1).any():
        raise ValueError("Each collection group must belong to one of the three folds")

    training = cfg["training"]
    seed = cfg["random_seed"]
    device = pick_device(training["device"])
    history = []

    def fit(take, stage):
        set_seed(seed)
        packets = values[take][mask[take]]
        center, scale = packets.mean(axis=0), packets.std(axis=0)
        center[1], scale[1] = 0, 1
        scale[scale == 0] = 1
        x, m = normalized_inputs(values[take], mask[take], center, scale)
        y = torch.from_numpy(labels[take]).float()
        loader = DataLoader(
            TensorDataset(x, m, y), batch_size=training["batch_size"], shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        model = build_model(cfg["model"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
        for epoch in range(1, training["epochs"] + 1):
            model.train()
            total = 0.0
            for xb, mb, yb in loader:
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(xb.to(device), mb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()
                total += loss.item() * len(yb)
            history.append({"stage": stage, "epoch": epoch, "train_loss": total / len(y)})
            print(f"{stage} | epoch {epoch} | train_loss {total / len(y):.6f}", flush=True)
        return model, center, scale

    started = time.perf_counter()
    scores = np.empty(len(labels), dtype=np.float32)
    for fold in range(3):
        held = folds.fold.eq(fold).to_numpy()
        if set(labels[held]) != {0, 1}:
            raise ValueError(f"Fold {fold} must contain both labels")
        model, center, scale = fit(~held, f"fold_{fold}")
        x, m = normalized_inputs(values[held], mask[held], center, scale)
        scores[held] = torch.sigmoid(predict_logits(model, x, m, training["batch_size"], device)).numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite out-of-fold scores")

    thresholds = np.unique(scores)

    def positive_rate(take):
        ordered = np.sort(scores[take])
        if not ordered.size:
            raise ValueError("Threshold selection requires both browsers and tunnel samples")
        return 1 - np.searchsorted(ordered, thresholds) / len(ordered)

    recall = positive_rate(labels == 1)
    chrome = positive_rate((labels == 0) & folds.source.eq("Chrome").to_numpy())
    firefox = positive_rate((labels == 0) & folds.source.eq("Firefox").to_numpy())
    target = float(cfg["threshold"]["minimum_recall"])
    eligible = np.flatnonzero(recall >= target)
    if not 0 < target <= 1 or not eligible.size:
        raise ValueError("Minimum recall must be in (0, 1]")
    i = min(eligible, key=lambda j: (max(chrome[j], firefox[j]), (chrome[j] + firefox[j]) / 2,
                                     -recall[j], -thresholds[j]))
    threshold = float(thresholds[i])
    print(f"Locked threshold {threshold:.9f} | OOF recall {recall[i]:.6f}", flush=True)
    model, center, scale = fit(np.ones(len(labels), dtype=bool), "final")
    train_time = time.perf_counter() - started

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(labels), min(2000, len(labels)), replace=False)
    x, m = normalized_inputs(values[pick], mask[pick], center, scale)
    mask_check = check_mask_invariance(model, x.to(device), m.to(device), seed=seed)
    mask_check["splits"] = ["development"]

    output = cfg["output"]
    model_dir = resolve(output["model_directory"])
    exp_dir = resolve(output["experiment_directory"])
    model_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "best.pt"
    torch.save(model.state_dict(), checkpoint)
    write_json(model_dir / "preprocessing.json", {"mean": center.tolist(), "std": scale.tolist()})
    write_json(model_dir / "feature_names.json", {
        "channels": data_cfg["channels"], "stored_layout": data_cfg["stored_layout"],
        "model_layout": data_cfg["model_layout"],
    })
    write_json(exp_dir / "mask_check.json", mask_check)
    folds.assign(score=scores).to_parquet(exp_dir / "out_of_fold.parquet")
    pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)
    cfg["threshold"]["value"] = threshold
    output["checkpoint_file"] = (Path(output["model_directory"]) / "best.pt").as_posix()
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(exp_dir / "run_summary.json", {
        "model": MODEL_NAME, "data_profile": "common_grouped", "seed": seed,
        "n_train": len(labels), "epochs_trained_per_fit": training["epochs"],
        "threshold_method": cfg["threshold"]["method"], "threshold": threshold,
        "recall_target": target, "oof_recall": float(recall[i]),
        "oof_chrome_fpr": float(chrome[i]), "oof_firefox_fpr": float(firefox[i]),
        "oof_average_precision": float(average_precision_score(labels, scores)),
        "train_time_seconds": train_time, "model_size_bytes": checkpoint.stat().st_size,
        "mask_check_passed": mask_check["passed"], **environment(device),
    })
