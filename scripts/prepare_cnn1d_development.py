"""Combine the original CNN1D train and validation archives in their existing order."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime-deps"))

import numpy as np
import pandas as pd


def main():
    folder = ROOT / "data/processed/cnn1d"
    with np.load(folder / "train.npz", allow_pickle=False) as train, np.load(
        folder / "val.npz", allow_pickle=False
    ) as val:
        columns = train["feature_names"]
        if not np.array_equal(columns, val["feature_names"]):
            raise ValueError("Train and validation channel order differs")
        arrays = {
            key: np.concatenate([train[key], val[key]])
            for key in ("X", "X_raw", "mask", "y", "sample_id")
        }
    ids = pd.Index(arrays["sample_id"], name="sample_id")
    manifest = pd.read_parquet(ROOT / "splits/common_folds.parquet").set_index("sample_id")
    if not ids.is_unique or not manifest.index.is_unique or set(ids) != set(manifest.index):
        raise ValueError("Fold manifest must cover each development sample exactly once")
    if not np.array_equal(manifest.loc[ids, "label"], arrays["y"]):
        raise ValueError("Fold manifest labels differ from the sequence archives")
    temporary = folder / "development.tmp.npz"
    np.savez_compressed(temporary, **arrays, feature_names=columns)
    temporary.replace(folder / "development.npz")
    print(f"Saved {len(ids):,} flows to {folder / 'development.npz'}")


if __name__ == "__main__":
    main()
