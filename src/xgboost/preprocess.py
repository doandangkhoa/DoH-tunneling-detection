"""Chưa triển khai: XGBoost preprocessing.

Khung file chưa triển khai. Xem TODO.md và docs/TRAINING_GUIDE.md.
"""


from pathlib import Path
import numpy as np


def load_npz(file_path, input_key="X", label_key="y",
              id_key="sample_id", feature_names_key="feature_names"):

    with np.load(file_path, allow_pickle=False) as data:

        X = data[input_key]
        y = data[label_key]
        sample_ids = data[id_key]

        feature_names = None

        if feature_names_key in data.files:
            feature_names = data[feature_names_key]

    return X, y, sample_ids, feature_names


def validate_data(X, y, sample_ids, feature_names=None):

    assert X.ndim == 2, "X phải là ma trận 2 chiều"

    assert len(X) == len(y) == len(sample_ids), \
        "Số lượng X, y và ID không khớp"

    assert len(np.unique(sample_ids)) == len(sample_ids), \
        "sample_id bị trùng"

    assert np.isfinite(X).all(), \
        "X chứa NaN hoặc Inf"

    assert np.isin(y, [0, 1]).all(), \
        "Nhãn không hợp lệ"

    if feature_names is not None:
        assert len(feature_names) == X.shape[1], \
            "Số feature không khớp số cột X"


def load_dataset(config):

    folder = Path(config["directory"])

    input_key = config["input_key"]
    label_key = config["label_key"]
    id_key = config["id_key"]
    feature_names_key = config["feature_names_key"]

    dataset = {}

    for split in ["train", "val", "test"]:

        file_path = folder / f"{split}.npz"

        X, y, ids, feature_names = load_npz(
            file_path,
            input_key,
            label_key,
            id_key,
            feature_names_key
        )

        validate_data(X, y, ids, feature_names)

        dataset[split] = {
            "X": X,
            "y": y,
            "ids": ids,
            "feature_names": feature_names
        }

        print(f"\n[{split.upper()}]")
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("Benign:", np.sum(y == 0))
        print("Malicious:", np.sum(y == 1))
        print("Unique IDs:", len(np.unique(ids)))

    # Kiểm tra feature order giữa các tập nếu có lưu feature_names
    train_features = dataset["train"]["feature_names"]

    if train_features is not None:
        for split in ["val", "test"]:
            current = dataset[split]["feature_names"]

            if current is not None:
                assert np.array_equal(train_features, current), \
                    f"Feature order không khớp tại {split}"

    return dataset
