# Data & Output Contracts (bản khởi tạo)

> Cả nhóm phải review và khóa phiên bản trước khi xử lý dataset thật. Thông tin dưới đây là **đặc tả mong muốn**, chưa có file dữ liệu thật.

## Dataset

**`data/interim/master_dataset.parquet`**: `sample_id` (unique string), `capture_id`, `flow_id`, `start_time`, `label` (0=benign, 1=malicious), `split_group`.

**`data/interim/flow_features.parquet`**: `sample_id`, các thống kê flow numeric (length, duration, direction, IAT, rates...). Metadata chứa nhãn/IP/nguồn thu thập chỉ giữ cho QA, không đưa trực tiếp vào feature matrix mặc định.

**`data/interim/packet_metadata.parquet`**: `sample_id`, `packet_index` (0-based), `timestamp`, `packet_length`, `direction` (+1/-1 theo initiator), `interarrival_time` (giây).

**`splits/train_ids.csv`, `val_ids.csv`, `test_ids.csv`**: `sample_id`. Split không overlap, cân nhắc chia theo capture hoặc group để tránh leakage. `split_manifest.json` ghi rule, seed và phân phối nhãn.

## Chuẩn prediction (mọi mô hình)

`outputs/<model>/predictions.parquet` yêu cầu mỗi test flow một dòng:

| Field | Type | Ghi chú |
|---|---|---|
| `sample_id` | str | Thuộc tập test chuẩn |
| `model` | str | `xgboost`, `isolation_forest` hoặc `cnn1d` |
| `y_true` | int | 0/1, khớp master dataset |
| `y_score` | float | Điểm càng cao càng nghiêng malicious; IF có thể cần đảo chiều |
| `y_pred` | int | 0/1, chọn threshold từ validation |
| `threshold` | float | Được cố định *trước* test |
| `split` | str | `test` cho báo cáo cuối |

**Lưu ý:** IF `y_score` không nhất thiết là xác suất; evaluator chỉ coi đó là điểm xếp hạng.

## Chốt nghiệm thu

- 3 model outputs phải có đúng cùng bộ `sample_id` và nhãn `y_true`.
- Trong mỗi file prediction, `sample_id` không trùng và không thiếu.
- Fit preprocessing theo train; validation chọn threshold/hyperparameters, không đọc test để tuning.
- Thống nhất tên chỉ số `average_precision` (AP) nếu sử dụng `sklearn.metrics.average_precision_score`.
