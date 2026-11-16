
import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Đọc kết quả dự đoán đã xuất từ predict.py
df = pd.read_parquet(
    "outputs/xgboost/predictions.parquet"
)

y_true = df["y_true"].to_numpy()
y_pred = df["y_pred"].to_numpy()
y_score = df["y_score"].to_numpy()

# Kiểm tra dữ liệu
if not np.isfinite(y_score).all():
    raise ValueError("y_score chứa NaN hoặc Inf")

# Confusion Matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("=" * 55)
print("XGBOOST - TEST EVALUATION")
print("=" * 55)

print("\nCONFUSION MATRIX")
print("                  Predicted")
print("                  Benign    Malicious")
print(f"Actual Benign     {tn:7d}    {fp:7d}")
print(f"Actual Malicious  {fn:7d}    {tp:7d}")

print("\nCLASSIFICATION METRICS")

print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.6f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.6f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.6f}")
print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.6f}")

print(f"ROC-AUC:   {roc_auc_score(y_true, y_score):.6f}")
print(f"PR-AUC:    {average_precision_score(y_true, y_score):.6f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malicious"],
        digits=6,
        zero_division=0
    )
)
