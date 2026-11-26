"""D — Kiểm thử 1D-CNN trên dữ liệu giả cùng định dạng NPZ (không cần common.tar.gz)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cnn1d import predict, train  # noqa: E402
from src.cnn1d.model import MaskedConv1D, check_mask_invariance  # noqa: E402

CHANNELS = ["packet_length", "direction", "interarrival_time"]


def fake_split(n, rng, prefix):
    """Flow 4–32 packet thật ở đầu chuỗi, padding = 0 sau chuẩn hóa như pipeline của A."""
    y = rng.integers(0, 2, n)
    lengths = rng.integers(4, 33, n)
    mask = np.arange(32)[None, :] < lengths[:, None]
    X = rng.normal(size=(n, 32, 3)).astype(np.float32)
    X[:, :, 0] += y[:, None] * 1.5  # tunnel có packet dài hơn để mô hình học được
    X[:, :, 1] = np.sign(X[:, :, 1])
    X[~mask] = 0
    ids = np.array([f"{prefix}-{i}" for i in range(n)])
    raw = X.copy()
    raw[:, :, 0] = np.exp(X[:, :, 0])
    raw[:, :, 2] = np.exp(X[:, :, 2])
    raw[~mask] = 0
    return dict(X=X, X_raw=raw, mask=mask, y=y, sample_id=ids, feature_names=np.array(CHANNELS))


def test_padding_does_not_change_scores():
    torch.manual_seed(0)
    model = MaskedConv1D()
    rng = np.random.default_rng(0)
    d = fake_split(64, rng, "x")
    X = torch.from_numpy(d["X"].transpose(0, 2, 1).copy())
    result = check_mask_invariance(model, X, torch.from_numpy(d["mask"]))
    assert result["passed"] and result["max_abs_score_difference"] == 0.0
    assert result["n_samples_with_padding"] > 0


def test_mask_check_detects_unmasked_model():
    """Đảm bảo phép kiểm tra có tác dụng: model bỏ qua mask phải bị FAIL."""
    class Unmasked(MaskedConv1D):
        def forward(self, x, mask):
            return super().forward(x, torch.ones_like(mask))

    torch.manual_seed(0)
    d = fake_split(64, np.random.default_rng(0), "x")
    X = torch.from_numpy(d["X"].transpose(0, 2, 1).copy())
    assert not check_mask_invariance(Unmasked(), X, torch.from_numpy(d["mask"]))["passed"]


@pytest.mark.parametrize("grouped", [False, True])
def test_train_then_predict_end_to_end(tmp_path, grouped):
    rng = np.random.default_rng(1)
    data = tmp_path / "data"
    data.mkdir()
    manifest = []
    development = []
    for split, n in [("train", 300), ("val", 120), ("test", 150)]:
        arrays = fake_split(n, rng, split)
        np.savez_compressed(data / f"{split}.npz", **arrays)
        if split != "test":
            development.append(arrays)
            fold = np.arange(n) % 3
            source = np.where(arrays["y"] == 1, "iodine", np.where(fold == 0, "Chrome", "Firefox"))
            manifest.append(pd.DataFrame({
                "sample_id": arrays["sample_id"], "label": arrays["y"], "source": source,
                "fold": fold, "split_group": [f"group-{i}" for i in fold],
            }))
    (data / "preprocessing.json").write_text("{}")

    cfg = yaml.safe_load((ROOT / "configs/cnn1d.yaml").read_text(encoding="utf-8"))
    cfg["data"]["directory"] = str(data)
    if grouped:
        merged = {
            key: np.concatenate([split[key] for split in development])
            for key in ("X", "X_raw", "mask", "y", "sample_id")
        }
        np.savez_compressed(data / "development.npz", **merged, feature_names=CHANNELS)
        (data / "train.npz").unlink()
        (data / "val.npz").unlink()
        path = data / "folds.parquet"
        pd.concat(manifest).to_parquet(path, index=False)
        cfg["data"]["fold_manifest"] = str(path)
    else:
        cfg["data"].update(preprocessing_file=str(data / "preprocessing.json"), refit_preprocessing=False)
        cfg["threshold"].update(method="maximize_f1", split="val")
    cfg["training"].update(epochs=3, device="cpu")
    cfg["output"] = {"model_directory": str(tmp_path / "models"),
                     "predictions_file": str(tmp_path / "outputs/predictions.parquet"),
                     "experiment_directory": str(tmp_path / "exp")}
    config = tmp_path / "cnn1d.yaml"
    config.write_text(yaml.safe_dump(cfg))

    train.main(["--config", str(config)])
    used = tmp_path / "exp/config.yaml"
    assert yaml.safe_load(used.read_text(encoding="utf-8"))["threshold"]["value"] is not None
    predict.main(["--config", str(used)])

    validation_output = "exp/out_of_fold.parquet" if grouped else "outputs/validation_predictions.parquet"
    for name in ["models/best.pt", "models/preprocessing.json", "models/feature_names.json",
                 "exp/history.csv", "exp/mask_check.json", validation_output]:
        assert (tmp_path / name).is_file(), name
    assert len(pd.read_csv(tmp_path / "exp/history.csv")) == (12 if grouped else 3)
    assert json.loads((tmp_path / "exp/mask_check.json").read_text())["passed"]

    pred = pd.read_parquet(tmp_path / "outputs/predictions.parquet")
    assert list(pred.columns) == ["sample_id", "model", "y_true", "y_score", "y_pred", "threshold", "split"]
    with np.load(data / "test.npz") as z:
        assert np.array_equal(pred.sample_id, z["sample_id"]) and np.array_equal(pred.y_true, z["y"])
    assert (pred.y_pred == (pred.y_score >= pred.threshold)).all() and (pred.split == "test").all()
    assert pred.threshold.nunique() == 1 and pred.model.eq("cnn1d").all()

    summary = json.loads((tmp_path / "exp/run_summary.json").read_text(encoding="utf-8"))
    for key in ["threshold", "train_time_seconds",
                "inference_time_seconds", "inference_ms_per_flow", "model_size_bytes"]:
        assert summary[key] is not None, key
    assert summary["n_test"] == 150
    if grouped:
        assert summary["n_train"] == 420
        assert summary["oof_recall"] >= cfg["threshold"]["minimum_recall"]
    else:
        assert (summary["n_train"], summary["n_val"]) == (300, 120)
        assert summary["validation_average_precision"] is not None
        assert summary["selected_epoch"] is not None
