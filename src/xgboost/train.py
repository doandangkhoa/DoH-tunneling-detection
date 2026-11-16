"""Chưa triển khai: Train and tune XGBoost.

Khung file chưa triển khai. Xem TODO.md và docs/TRAINING_GUIDE.md.
"""

"""
Train XGBoost baseline cho bài toán DoH Tunneling Detection.

Chạy từ thư mục root:
    python -m src.xgboost.train

Quy trình:
    1. Đọc cấu hình YAML.
    2. Load train/validation/test NPZ.
    3. Train XGBoost bằng train set.
    4. Tính Average Precision trên validation.
    5. Chọn threshold tối đa F1 trên validation.
    6. Lưu model và artifacts.

Không sử dụng test để lựa chọn model hoặc threshold.
"""

import json
import time
import shutil
import platform
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    log_loss
)

from src.xgboost.preprocess import load_dataset


# ============================================================
# 1. ĐỌC CONFIG
# ============================================================

CONFIG_PATH = Path("configs/xgboost.yaml")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy config: {CONFIG_PATH}"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

data_config = config["data"]
model_config = config["model"]
selection_config = config["selection"]
threshold_config = config["threshold"]
output_config = config["output"]

print("=" * 60)
print("XGBOOST BASELINE TRAINING")
print("=" * 60)


# ============================================================
# 2. KIỂM TRA CẤU HÌNH
# ============================================================

if selection_config["split"] != "val":
    raise ValueError("Baseline yêu cầu selection.split = val")

if threshold_config["split"] != "val":
    raise ValueError("Baseline yêu cầu threshold.split = val")

if selection_config["metric"] != "average_precision":
    raise ValueError("Metric lựa chọn phải là average_precision")

if threshold_config["method"] != "maximize_f1":
    raise ValueError("Threshold method phải là maximize_f1")

if data_config.get("refit_preprocessing", False):
    raise ValueError(
        "Không được refit preprocessing trên dữ liệu common."
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n[1/6] Loading dataset...")

dataset = load_dataset(data_config)

X_train = dataset["train"]["X"]
y_train = dataset["train"]["y"]
train_ids = dataset["train"]["ids"]

X_val = dataset["val"]["X"]
y_val = dataset["val"]["y"]
val_ids = dataset["val"]["ids"]

X_test = dataset["test"]["X"]
y_test = dataset["test"]["y"]
test_ids = dataset["test"]["ids"]

feature_names = dataset["train"]["feature_names"]

# Kiểm tra dữ liệu đầu vào
if X_train.shape[1] != 29:
    raise ValueError(
        f"XGBoost yêu cầu 29 features, nhận được {X_train.shape[1]}"
    )

if feature_names is None:
    raise ValueError("Không tìm thấy feature_names trong train.npz")

if len(feature_names) != X_train.shape[1]:
    raise ValueError("Feature names không khớp số cột X")

# Train phải có cả hai lớp
if len(np.unique(y_train)) != 2:
    raise ValueError("Train phải có cả benign và malicious")

# Validation phải có cả hai lớp
if len(np.unique(y_val)) != 2:
    raise ValueError("Validation phải có cả benign và malicious")

print("\nDataset profile:")
print(f"Train:      {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")

print("\nTrain labels:")
print(f"Benign:     {np.sum(y_train == 0):,}")
print(f"Malicious:  {np.sum(y_train == 1):,}")


# ============================================================
# 4. KHỞI TẠO VÀ TRAIN MODEL
# ============================================================

print("\n[2/6] Training XGBoost...")

# Chỉ truyền phần model trong YAML vào estimator.
model = xgb.XGBClassifier(**model_config)

start_train = time.perf_counter()

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

train_time_seconds = time.perf_counter() - start_train

print("Training completed.")
print(f"Training time: {train_time_seconds:.2f} seconds")


# ============================================================
# 5. ĐÁNH GIÁ VALIDATION VÀ CHỌN THRESHOLD
# ============================================================

print("\n[3/6] Evaluating validation...")

# Xác suất thuộc lớp 1 = malicious/tunnel DoH
val_scores = model.predict_proba(X_val)[:, 1]

if not np.isfinite(val_scores).all():
    raise ValueError("Validation scores chứa NaN hoặc Inf")

# Metric lựa chọn model
val_ap = average_precision_score(y_val, val_scores)

val_logloss = log_loss(
    y_val,
    val_scores,
    labels=[0, 1]
)

# Tìm threshold tối đa F1
precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_scores
)

# precision và recall dài hơn thresholds một phần tử.
# Phần tử cuối không có threshold tương ứng nên bỏ qua.
f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

if len(thresholds) == 0:
    raise ValueError("Không tìm được threshold từ validation")

best_index = int(np.argmax(f1_values))

best_threshold = float(thresholds[best_index])
best_f1 = float(f1_values[best_index])

val_predictions = (val_scores >= best_threshold).astype(np.int32)

# Kiểm tra F1 sau khi áp dụng threshold
check_f1 = f1_score(y_val, val_predictions)

if not np.isclose(best_f1, check_f1):
    raise RuntimeError("F1 tại threshold được chọn không khớp")

print("\nValidation results:")
print(f"Average Precision: {val_ap:.6f}")
print(f"Log Loss:          {val_logloss:.6f}")
print(f"Best threshold:    {best_threshold:.8f}")
print(f"Best F1:           {best_f1:.6f}")


# ============================================================
# 6. TẠO THƯ MỤC OUTPUT
# ============================================================

print("\n[4/6] Saving model and preprocessing...")

model_dir = Path(output_config["model_directory"])
output_dir = Path("outputs/xgboost")
experiment_dir = Path(output_config["experiment_directory"])

model_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
experiment_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 7. LƯU MODEL JSON
# ============================================================

model_path = model_dir / "model.json"

model.save_model(str(model_path))

model_size_bytes = model_path.stat().st_size

print(f"Model saved: {model_path}")


# ============================================================
# 8. LƯU PREPROCESSING.JSON
# ============================================================

preprocessing_source = Path(data_config["preprocessing_file"])

if not preprocessing_source.exists():
    raise FileNotFoundError(
        f"Không tìm thấy preprocessing: {preprocessing_source}"
    )

shutil.copy2(
    preprocessing_source,
    model_dir / "preprocessing.json"
)

print("Preprocessing copied.")


# ============================================================
# 9. LƯU FEATURE NAMES
# ============================================================

feature_names_list = [
    str(name) for name in feature_names.tolist()
]

with open(
    model_dir / "feature_names.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        feature_names_list,
        file,
        ensure_ascii=False,
        indent=4
    )

print("Feature names saved.")


# ============================================================
# 10. XUẤT VALIDATION PREDICTIONS
# ============================================================

print("\n[5/6] Exporting validation predictions...")

val_result = pd.DataFrame({
    "sample_id": val_ids,
    "model": "xgboost",
    "y_true": y_val.astype(np.int32),
    "y_score": val_scores.astype(np.float64),
    "y_pred": val_predictions,
    "threshold": best_threshold,
    "split": "val"
})

if val_result["sample_id"].duplicated().any():
    raise ValueError("Validation prediction có sample_id trùng")

if not np.isfinite(val_result["y_score"]).all():
    raise ValueError("Validation prediction có score không hữu hạn")

if not np.array_equal(
    val_result["y_pred"].to_numpy(),
    (val_result["y_score"].to_numpy() >= best_threshold).astype(np.int32)
):
    raise RuntimeError("y_pred không khớp threshold")

val_prediction_path = output_dir / "validation_predictions.parquet"

val_result.to_parquet(
    val_prediction_path,
    index=False
)

print(f"Validation predictions saved: {val_prediction_path}")


# ============================================================
# 11. LƯU CONFIG THỰC NGHIỆM
# ============================================================

print("\n[6/6] Saving experiment configuration...")

experiment_config = dict(config)

# Không sửa file baseline gốc trong configs/
experiment_config["threshold"] = dict(threshold_config)
experiment_config["threshold"]["value"] = best_threshold

experiment_config["data"] = dict(data_config)
experiment_config["data"]["refit_preprocessing"] = False

experiment_config["experiment"] = {
    "model": "xgboost",
    "data_profile": "common",
    "seed": model_config["random_state"],
    "selected_epoch": None
}

with open(
    experiment_dir / "config.yaml",
    "w",
    encoding="utf-8"
) as file:
    yaml.safe_dump(
        experiment_config,
        file,
        allow_unicode=True,
        sort_keys=False
    )


# ============================================================
# 12. LƯU RUN SUMMARY
# ============================================================

def get_library_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


run_summary = {
    "model": "xgboost",
    "data_profile": "common",
    "data_directory": str(data_config["directory"]),
    "seed": model_config["random_state"],
    "git_commit": None,

    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),

    "selection_metric": "average_precision",
    "validation_average_precision": float(val_ap),

    "threshold_method": "maximize_f1",
    "threshold": best_threshold,
    "validation_f1_at_threshold": best_f1,

    "selected_epoch": None,
    "score_method": "predict_proba_class_1",

    "train_time_seconds": float(train_time_seconds),

    # Sẽ được cập nhật sau khi đo inference thực tế.
    "inference_time_seconds": None,
    "inference_ms_per_flow": None,

    "hardware": platform.platform(),
    "device": "CPU",
    "batch_size": int(len(X_test)),
    "warmup_runs": None,
    "timed_runs": None,
    "includes_preprocessing": False,

    "model_size_bytes": int(model_size_bytes),

    "python_version": platform.python_version(),

    "library_versions": {
        "numpy": get_library_version("numpy"),
        "pandas": get_library_version("pandas"),
        "scikit-learn": get_library_version("scikit-learn"),
        "xgboost": get_library_version("xgboost"),
        "pyyaml": get_library_version("PyYAML"),
        "pyarrow": get_library_version("pyarrow")
    }
}

with open(
    experiment_dir / "run_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        run_summary,
        file,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 13. HOÀN THÀNH
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST BASELINE COMPLETED")
print("=" * 60)

print(f"Validation AP:      {val_ap:.6f}")
print(f"Validation F1:      {best_f1:.6f}")
print(f"Locked threshold:   {best_threshold:.8f}")
print(f"Training time:      {train_time_seconds:.2f}s")
print(f"Model size:         {model_size_bytes:,} bytes")

print("\nArtifacts:")
print(f"- {model_dir / 'model.json'}")
print(f"- {model_dir / 'preprocessing.json'}")
print(f"- {model_dir / 'feature_names.json'}")
print(f"- {val_prediction_path}")
print(f"- {experiment_dir / 'config.yaml'}")
print(f"- {experiment_dir / 'run_summary.json'}")

print("\nTest set has NOT been predicted.")
print("Model and threshold are ready to be locked.")
