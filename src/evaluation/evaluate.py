"""Regenerate tables and figures from the three saved test prediction files."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.evaluation.metrics import binary_metrics, ratio
from src.evaluation.plots import NAMES, render_figures

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "data/processed/cnn1d/test_metadata.parquet"


def read_predictions(path, model, reference):
    data = pd.read_parquet(path)
    required = {"sample_id", "model", "y_true", "y_pred", "y_score", "threshold", "split"}
    if not required <= set(data) or data.empty:
        raise ValueError(f"{path}: missing prediction columns or rows")
    if not data.model.eq(model).all() or not data.split.eq("test").all():
        raise ValueError(f"{path}: unexpected model or split")
    if not data.sample_id.is_unique or set(data.sample_id) != set(reference.index):
        raise ValueError(f"{path}: test IDs differ from reference")
    data = data.set_index("sample_id").loc[reference.index]
    if not np.array_equal(data.y_true, reference.label):
        raise ValueError(f"{path}: test labels differ from reference")
    if not np.isfinite(data[["y_score", "threshold"]]).all().all() or data.threshold.nunique() != 1:
        raise ValueError(f"{path}: invalid scores or threshold")
    if not data.y_pred.eq(data.y_score.ge(data.threshold)).all():
        raise ValueError(f"{path}: invalid predicted labels")
    return data.join(reference.drop(columns="label"))


def runtime_rows(metrics):
    for model, result in metrics.iterrows():
        directory = ROOT / "experiments" / model
        inference = {}
        if model == "isolation_forest":
            summary = json.loads((directory / "train_summary.json").read_text())
            inference = json.loads((directory / "inference_summary.json").read_text())
        else:
            summary = json.loads((directory / "run_summary.json").read_text())
        if not np.isclose(summary["threshold"], result.threshold, rtol=0, atol=1e-12):
            raise ValueError(f"{model}: runtime summary threshold differs from predictions")
        if inference and (inference["num_samples"] != result.n or not np.isclose(
            inference["threshold_used"], result.threshold, rtol=0, atol=1e-12
        )):
            raise ValueError(f"{model}: inference summary differs from predictions")
        checkpoint = ROOT / "models" / model / {
            "xgboost": "model.json", "isolation_forest": "isolation_forest.joblib", "cnn1d": "best.pt",
        }[model]
        yield {
            "model": model,
            "fit_samples_final": summary.get("n_train", summary.get("train_samples")),
            "train_seconds": summary.get("train_time_seconds", summary.get("train_seconds")),
            "model_size_bytes": summary.get("model_size_bytes", checkpoint.stat().st_size if checkpoint.exists() else None),
            "inference_seconds": summary.get("inference_time_seconds", inference.get("inference_seconds")),
            "inference_ms_per_flow": summary.get("inference_ms_per_flow", inference.get("ms_per_sample")),
            "batch_size": summary.get("batch_size", inference.get("batch_size")),
            "timed_runs": summary.get("timed_runs", 1 if inference else None),
            "warmup_runs": summary.get("warmup_runs", 1 if inference else None),
            "warmup_flows_per_run": inference.get("warmup_samples", result.n),
        }


def main():
    reference = pd.read_parquet(METADATA, columns=[
        "sample_id", "label", "source", "capture_id", "split_group",
    ]).set_index("sample_id")
    if not reference.index.is_unique or set(reference.label) != {0, 1}:
        raise ValueError("Test metadata must have unique IDs and both binary classes")
    rows, per_source, curves, errors, provenance = [], [], [], [], {}
    for model in NAMES:
        path = ROOT / "outputs" / model / "predictions.parquet"
        data = read_predictions(path, model, reference)
        overall = binary_metrics(data.y_true, data.y_pred, data.y_score)
        model_sources = []
        for (label, source), part in data.groupby(["y_true", "source"]):
            positive = int(part.y_pred.sum())
            model_sources.append({
                "model": model, "label": label, "source": source, "support": len(part),
                "captures": part.capture_id.nunique(), "split_groups": part.split_group.nunique(),
                "metric": "recall" if label else "fpr",
                "positive_predictions": positive, "rate": ratio(positive, len(part)),
            })
        per_source.extend(model_sources)
        rows.append({
            "model": model, "threshold": data.threshold.iloc[0], **overall,
            "macro_tunnel_recall": np.mean([row["rate"] for row in model_sources if row["label"] == 1]),
        })
        precision, recall, thresholds = precision_recall_curve(data.y_true, data.y_score)
        curves.append(pd.DataFrame(dict(
            model=model, precision=precision, recall=recall, threshold=np.r_[thresholds, np.nan],
        )))
        wrong = data[data.y_pred.ne(data.y_true)].copy()
        wrong["error"] = np.where(wrong.y_true.eq(0), "FP", "FN")
        errors.append(wrong.reset_index())
        provenance[model] = {"path": str(path.relative_to(ROOT)),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metrics = pd.DataFrame(rows).set_index("model")
    sources = pd.DataFrame(per_source)
    curve_points = pd.concat(curves, ignore_index=True)
    runtime = pd.DataFrame(runtime_rows(metrics))
    tables, figures = (ROOT / "reports" / name for name in ("tables", "figures"))
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "metrics.csv", na_rep="N/A")
    sources.to_csv(tables / "source_metrics.csv", index=False, na_rep="N/A")
    runtime.to_csv(tables / "runtime.csv", index=False, na_rep="N/A")
    curve_points.to_csv(tables / "pr_curve.csv", index=False, na_rep="N/A")
    pd.concat(errors, ignore_index=True).to_parquet(tables / "errors.parquet", index=False)
    (tables / "provenance.json").write_text(json.dumps({
        "dataset": "CIRA-CIC-DoHBrw-2020", "profile": "common", "split": "test",
        "test_flows": len(reference), "metadata": str(METADATA.relative_to(ROOT)),
        "metadata_sha256": hashlib.sha256(METADATA.read_bytes()).hexdigest(),
        "predictions": provenance,
    }, indent=2) + "\n")
    render_figures(metrics, sources, curve_points, figures)
    print(metrics[["f1", "recall", "fpr", "ap", "fp", "fn"]].to_string())


if __name__ == "__main__":
    main()
