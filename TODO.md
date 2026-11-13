# Công việc triển khai

Điểm bắt đầu: [README](README.md) → [Hướng dẫn dữ liệu](docs/TRAINING_GUIDE.md). B/C/D phụ trách huấn luyện và xuất kết quả mô hình; A phụ trách dữ liệu, đánh giá, so sánh và demo. Tiến độ trao đổi trong buổi họp nhóm.

## Phạm vi vòng triển khai đầu

Cả nhóm dùng common 190.100 flow. Mỗi mô hình một baseline, seed train 42; CNN dùng 32 packet. Validation chọn checkpoint/cấu hình và threshold; chỉ đánh giá test sau khi khóa mô hình/ngưỡng. Chưa cần train bộ full, nhiều seed, grid search hoặc sequence length khác.

## A — Dữ liệu, đánh giá và tổng hợp

Code: `src/data/`, `src/evaluation/`, `scripts/`, `dashboard/app.py`.

- Bàn giao `common.tar.gz` và hỗ trợ B/C/D nạp đúng dữ liệu. Pipeline hiện tại gồm `prepare_dataset.py`, `prepare_packets.py`, `finalize_training.py`; các script audit/validate/đóng gói đã có.
- Nhận model, prediction và thông tin thí nghiệm theo [quy ước đầu ra](docs/TRAINING_GUIDE.md#41-mỗi-người-cần-lưu-gì-và-để-ở-đâu).
- Kiểm tra ba prediction có đủ cùng 13.411 test IDs, nhãn đúng, không trùng/thiếu và score/threshold đúng quy ước; thử nạp lại model.
- Triển khai evaluator chung và kiểm thử metric/schema: Precision, Recall, F1, AP, FPR, TP/FN/FP/TN; Recall/support theo công cụ và FPR theo browser.
- Tổng hợp bảng vào `reports/tables/`, biểu đồ vào `reports/figures/` theo [mục 4.4](docs/TRAINING_GUIDE.md#44-a-nhận-kết-quả-và-tổng-hợp); phân tích FP/FN từ prediction và metadata.
- Đo lại inference của các model đã lưu trên cùng máy để so sánh chi phí; tổng hợp thời gian train và kích thước model từ run summary.
- Làm dashboard một trang đọc kết quả có sẵn: metric, PR curve, confusion matrix, chi phí và ví dụ dự đoán sai.
- Chạy tích hợp từ dữ liệu → model đã lưu → prediction → evaluator → dashboard; cập nhật lệnh chạy thực tế trong README và chuẩn bị phần so sánh/giới hạn từ kết quả thật.

## B — XGBoost

Code: `src/xgboost/`. Cấu hình: [xgboost.yaml](configs/xgboost.yaml). Đầu vào: `data/processed/xgboost/common/`.

- Triển khai loader/train/predict từ NPZ, giữ feature order và ID; dùng X đã xử lý sẵn.
- Train baseline, chọn cấu hình/ngưỡng bằng validation, khóa trước test.
- Xuất model, preprocessing, feature names, config, run summary, validation predictions và test predictions theo [mục 4.1–4.2](docs/TRAINING_GUIDE.md#41-mỗi-người-cần-lưu-gì-và-để-ở-đâu).
- Gửi A ba thư mục `models/xgboost/`, `outputs/xgboost/`, `experiments/xgboost/` và lệnh nạp model/predict.

## C — Isolation Forest

Code: `src/isolation_forest/`. Cấu hình: [isolation_forest.yaml](configs/isolation_forest.yaml). Đầu vào: `data/processed/isolation_forest/common/`.

- Triển khai train/predict; chỉ fit benign common train, dùng X đã scale sẵn.
- Chọn cấu hình/ngưỡng bằng validation; dùng `y_score = -score_samples(X)` để điểm cao nghĩa là nghi malicious hơn.
- Xuất model và thông tin theo cùng [mục 4.1–4.2](docs/TRAINING_GUIDE.md#41-mỗi-người-cần-lưu-gì-và-để-ở-đâu); predict đủ hai lớp trên val/test.
- Gửi A ba thư mục `models/isolation_forest/`, `outputs/isolation_forest/`, `experiments/isolation_forest/` và lệnh nạp model/predict.

## D — 1D-CNN

Code: `src/cnn1d/`. Cấu hình: [cnn1d.yaml](configs/cnn1d.yaml). Đầu vào: `data/processed/cnn1d/`.

- Nạp NPZ, chuyển `(N,32,3)` sang `(N,3,32)`, giữ mask/ID/nhãn; không chuẩn hóa X lần nữa.
- Xây Conv1d và train/validation loop; chọn checkpoint/ngưỡng bằng validation, giữ prefix 32 packet.
- Dùng mask trong forward/pooling; kiểm thử prediction ở eval không đổi khi thay padding nhưng giữ packet thật và mask.
- Xuất checkpoint và thông tin theo [mục 4.1–4.2](docs/TRAINING_GUIDE.md#41-mỗi-người-cần-lưu-gì-và-để-ở-đâu), thêm `history.csv` và `mask_check.json` để A phân tích/vẽ biểu đồ.
- Gửi A ba thư mục `models/cnn1d/`, `outputs/cnn1d/`, `experiments/cnn1d/` và lệnh nạp model/predict.

## Quy tắc chung

- B/C/D train song song và bàn giao đúng tên file/thư mục. A tổng hợp sau khi nhận đủ kết quả.
- Giữ nguyên split và preprocessing; không cân bằng val/test, không tuning theo test. Metadata chỉ để phân tích, không làm feature.
- Lưu cấu hình, log và artifact đủ để chạy lại mô hình đã chọn; không cần bảng tracking riêng.
