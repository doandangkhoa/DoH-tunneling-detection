from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def load_split(data_dir: str | Path, split: str, data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Nạp dữ liệu split (.npz) theo cấu hình trong isolation_forest.yaml."""
    path = Path(data_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    with np.load(path, allow_pickle=False) as npz:
        result: dict[str, Any] = {
            "X": npz[data_cfg["input_key"]],
            "y": npz[data_cfg["label_key"]],
        }
        id_key = data_cfg.get("id_key", "sample_id")
        if id_key in npz:
            result["sample_id"] = npz[id_key]
        feat_key = data_cfg.get("feature_names_key", "feature_names")
        if feat_key in npz:
            result["feature_names"] = npz[feat_key]

    return result


def validate_training_data(y_train: np.ndarray, train_cfg: dict[str, Any]) -> None:
    """Đảm bảo tập train chỉ chứa nhãn benign (0) theo đúng thiết kế IF."""
    if train_cfg.get("benign_only", True):
        benign_label = train_cfg.get("benign_label", 0)
        if not np.all(y_train == benign_label):
            raise ValueError(
                f"Isolation Forest yêu cầu tập train chỉ chứa nhãn {benign_label}, "
                f"nhưng phát hiện nhãn khác: {np.unique(y_train)}"
            )

def fit_preprocessing(benign: pd.DataFrame) -> dict:
    median = benign.median().fillna(0)
    filled = benign.fillna(median)
    scale = (filled.quantile(0.75) - filled.quantile(0.25)).replace(0, 1)
    return {
        "features": benign.columns.tolist(),
        "median": median.to_dict(),
        "center": filled.median().to_dict(),
        "scale": scale.to_dict(),
    }


def transform(features: pd.DataFrame, preprocessing: dict) -> np.ndarray:
    columns = preprocessing["features"]
    filled = features.loc[:, columns].fillna(preprocessing["median"])
    center = pd.Series(preprocessing["center"]).reindex(columns)
    scale = pd.Series(preprocessing["scale"]).reindex(columns)
    return ((filled - center) / scale).to_numpy(dtype=np.float32)
