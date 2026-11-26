"""D — Nạp chuỗi packet đã xử lý sẵn cho 1D-CNN.

NPZ lưu X theo layout (N, L, C); Conv1d cần (N, C, L). Không chuẩn hóa lại X:
tham số chuẩn hóa đã fit từ train nằm trong preprocessing.json cạnh dữ liệu.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def resolve(path) -> Path:
    """Đường dẫn tương đối tính từ root repository."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_split(data_cfg: dict, split: str) -> dict:
    """Đọc một split, kiểm tra shape/nhãn/mask, trả về X ở layout (N, C, L)."""
    with np.load(resolve(data_cfg["directory"]) / f"{split}.npz", allow_pickle=False) as z:
        X = z[data_cfg["input_key"]].astype(np.float32)
        mask = z[data_cfg["mask_key"]].astype(bool)
        y = z[data_cfg["label_key"]].astype(np.int64)
        ids = z[data_cfg["id_key"]].astype(str)
        channels = [str(c) for c in z["feature_names"]]

    # Kiểm tra sớm để không train trên dữ liệu lỗi
    L, C = data_cfg["sequence_length"], len(data_cfg["channels"])
    assert X.shape == (len(y), L, C) and mask.shape == (len(y), L), (split, X.shape, mask.shape)
    assert channels == list(data_cfg["channels"]), (channels, data_cfg["channels"])
    assert np.isfinite(X).all(), f"{split}: X có NaN/inf"
    assert np.isin(y, [0, 1]).all(), f"{split}: nhãn ngoài 0/1"
    assert mask.any(axis=1).all(), f"{split}: có flow không có packet thật"
    assert len(np.unique(ids)) == len(ids), f"{split}: sample_id bị trùng"

    return {
        "X": np.ascontiguousarray(X.transpose(0, 2, 1)),  # (N, L, C) -> (N, C, L)
        "mask": mask,
        "y": y,
        "sample_id": ids,
        "channels": channels,
    }


def to_tensors(d: dict):
    """numpy -> tensor: X float32 (N,C,L), mask bool (N,L), y float (cho BCEWithLogits)."""
    return torch.from_numpy(d["X"]), torch.from_numpy(d["mask"]), torch.from_numpy(d["y"]).float()
