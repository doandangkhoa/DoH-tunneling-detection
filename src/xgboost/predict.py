"""Chưa triển khai: Export predictions contract.

Khung file chưa triển khai. Xem TODO.md và docs/TRAINING_GUIDE.md.
"""

"""
Predict XGBoost trên Test set.

Chạy từ thư mục root:
    python -m src.xgboost.predict

Quy trình:
    1. Đọc experiment config đã khóa threshold.
    2. Load model đã train.
    3. Load Test NPZ.
    4. Dự đoán xác suất malicious.
    5. Áp dụng threshold đã chọn từ Validation.
    6. Xuất predictions.parquet.
    7. Đo inference time và cập nhật run_summary.json.

Không train lại model.
Không lựa chọn lại threshold trên Test.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import classification_report

from src.evaluation.metrics import binary_metrics
from src.xgboost.preprocess import load_npz, validate_data


# ============================================================
# 1. ĐỌC EXPERIMENT CONFIG
# ============================================================

CONFIG_PATH = Path("experiments/xgboost/config.yaml")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy experiment config: {CONFIG_PATH}"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

data_config = config["data"]
output_config = config["output"]
threshold_config = config["threshold"]

threshold = threshold_config["value"]

if threshold is None:
    raise ValueError(
        "Threshold chưa được khóa. Hãy chạy train.py trước."
    )

threshold = float(threshold)

print("=" * 60)
print("XGBOOST TEST PREDICTION")
print("=" * 60)

print(f"Locked threshold: {threshold:.8f}")


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[1/5] Loading test dataset...")

test_path = Path(data_config["directory"]) / "test.npz"

X_test, y_test, test_ids, test_features = load_npz(
    test_path,
    input_key=data_config["input_key"],
    label_key=data_config["label_key"],
    id_key=data_config["id_key"],
    feature_names_key=data_config["feature_names_key"]
)

validate_data(
    X_test,
    y_test,
    test_ids,
    test_features
)

print("Test shape:", X_test.shape)
print("Number of samples:", len(test_ids))


# ============================================================
# 3. LOAD FEATURE NAMES VÀ MODEL
# ============================================================

print("\n[2/5] Loading trained model...")

model_dir = Path(output_config["model_directory"])

model_path = model_dir / "model.json"
feature_names_path = model_dir / "feature_names.json"

if not model_path.exists():
    raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

if not feature_names_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy feature names: {feature_names_path}"
    )

with open(feature_names_path, "r", encoding="utf-8") as file:
    feature_names = json.load(file)

if X_test.shape[1] != len(feature_names):
    raise ValueError(
        "Số feature của Test không khớp feature_names.json"
    )

if test_features is not None:
    test_feature_names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in test_features.tolist()
    ]

    if test_feature_names != feature_names:
        raise ValueError(
            "Thứ tự feature trong Test không khớp model."
        )

if threshold_config["method"] == "minimize_worst_browser_fpr":
    preprocessing = json.loads((model_dir / "preprocessing.json").read_text())
    raw = pd.read_parquet("data/interim/flow_features.parquet").set_index("sample_id")
    X_test = raw.loc[test_ids, preprocessing["features"]].fillna(
        preprocessing["median"]
    ).to_numpy()
    validate_data(X_test, y_test, test_ids, test_features)

model = xgb.XGBClassifier()
model.load_model(str(model_path))

if model.n_features_in_ != X_test.shape[1]:
    raise ValueError(
        "Số feature của model không khớp dữ liệu Test."
    )

print("Model loaded:", model_path)
print("Number of features:", model.n_features_in_)


# ============================================================
# 4. ĐO INFERENCE TIME VÀ DỰ ĐOÁN
# ============================================================

print("\n[3/5] Running inference...")

# Số lần chạy khởi động, không tính vào thời gian đo.
WARMUP_RUNS = 1

# Số lần đo inference trên toàn bộ Test.
TIMED_RUNS = 5

# Warm-up
for _ in range(WARMUP_RUNS):
    model.predict_proba(X_test)

# Đo thời gian inference
inference_times = []
y_scores = None

for run in range(TIMED_RUNS):
    start_time = time.perf_counter()

    probabilities = model.predict_proba(X_test)

    elapsed = time.perf_counter() - start_time

    inference_times.append(elapsed)

    # Lưu kết quả của lần chạy cuối.
    y_scores = probabilities[:, 1]

    print(
        f"Run {run + 1}/{TIMED_RUNS}: "
        f"{elapsed:.6f} seconds"
    )

if y_scores is None:
    raise RuntimeError("Không nhận được kết quả prediction.")

if not np.isfinite(y_scores).all():
    raise ValueError("Prediction scores chứa NaN hoặc Inf")

# Trung bình thời gian của các lần đo
inference_time_seconds = float(np.mean(inference_times))

n_test = len(X_test)

inference_ms_per_flow = (
    inference_time_seconds * 1000 / n_test
)

# Áp dụng threshold đã khóa từ Validation.
y_pred = (y_scores >= threshold).astype(np.int32)

print("\nInference results:")
print(f"Average inference time: {inference_time_seconds:.6f}s")
print(f"Inference per flow:     {inference_ms_per_flow:.8f}ms")
print(f"Warm-up runs:           {WARMUP_RUNS}")
print(f"Timed runs:             {TIMED_RUNS}")


# ============================================================
# 5. XUẤT TEST PREDICTIONS
# ============================================================

print("\n[4/5] Exporting predictions...")

prediction_path = Path(output_config["predictions_file"])

prediction_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

predictions = pd.DataFrame({
    "sample_id": test_ids,
    "model": "xgboost",
    "y_true": y_test.astype(np.int32),
    "y_score": y_scores.astype(np.float64),
    "y_pred": y_pred,
    "threshold": threshold,
    "split": "test"
})

predictions.to_parquet(
    prediction_path,
    index=False
)

print("Predictions saved:", prediction_path)
print("Prediction rows:", len(predictions))


# ============================================================
# 6. CẬP NHẬT RUN SUMMARY
# ============================================================

print("\n[5/5] Updating run_summary.json...")

experiment_dir = Path(output_config["experiment_directory"])

summary_path = experiment_dir / "run_summary.json"

if not summary_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy run summary: {summary_path}"
    )

with open(summary_path, "r", encoding="utf-8") as file:
    run_summary = json.load(file)

run_summary.update({
    "inference_time_seconds": inference_time_seconds,
    "inference_ms_per_flow": inference_ms_per_flow,
    "batch_size": int(n_test),
    "warmup_runs": WARMUP_RUNS,
    "timed_runs": TIMED_RUNS,
    "includes_preprocessing": False
})

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(
        run_summary,
        file,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 7. HOÀN THÀNH
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST TEST PREDICTION COMPLETED")
print("=" * 60)

print(f"Test samples:          {n_test:,}")
print(f"Threshold:             {threshold:.8f}")
print(f"Benign predictions:    {np.sum(y_pred == 0):,}")
print(f"Malicious predictions: {np.sum(y_pred == 1):,}")

print(f"\nInference time:        {inference_time_seconds:.6f}s")
print(f"Inference per flow:    {inference_ms_per_flow:.8f}ms")

print("\nArtifacts:")
print(f"- {prediction_path}")
print(f"- {summary_path}")

metrics = binary_metrics(y_test, y_pred, y_scores)
print("\nXGBOOST - TEST EVALUATION")
print("Confusion matrix (rows: actual, columns: predicted)")
print("                 Benign  Tunnel")
print(f"Actual benign    {metrics['tn']:6d}  {metrics['fp']:6d}")
print(f"Actual tunnel    {metrics['fn']:6d}  {metrics['tp']:6d}")
for label, key in (
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1"),
    ("FPR", "fpr"),
    ("ROC-AUC", "roc_auc"),
    ("AP", "ap"),
):
    value = metrics[key]
    formatted = f"{value:.6f}" if np.isfinite(value) else "N/A"
    print(f"{label + ':':<11}{formatted}")
print()
print(classification_report(
    y_test, y_pred, labels=[0, 1], target_names=["Benign", "Tunnel"],
    digits=6, zero_division=0,
))
