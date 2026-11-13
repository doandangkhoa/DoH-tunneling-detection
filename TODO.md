# Công việc triển khai

Điểm bắt đầu: [README](README.md) → [Hướng dẫn dữ liệu](docs/TRAINING_GUIDE.md). Danh sách dưới đây chỉ mô tả phần cần triển khai và đầu ra; trao đổi phân công chi tiết và tiến độ trong buổi họp nhóm.

## Phạm vi vòng triển khai đầu

Mỗi mô hình dùng một cấu hình baseline và seed 42; CNN dùng 32 packet. Cả ba dùng common 190.100 flow; chưa cần train bộ full. Xem [quy trình chung](docs/TRAINING_GUIDE.md#phạm-vi-huấn-luyện-đã-thống-nhất). Validation chọn threshold theo cấu hình, chỉ thử thêm ít tham số nếu baseline có vấn đề rõ ràng. Không cần grid search, nhiều seed, SHAP, sequence 64 packet hoặc phát hiện thời gian thực trong vòng này. Luồng và sản phẩm cuối xem [README](README.md#các-bước-tiếp-theo-và-sản-phẩm-cuối).

## A — Dữ liệu và tích hợp

Pipeline tạo master, flow features, sequence, split và common đã có. Pipeline chính nằm trong `src/data/prepare_dataset.py`, `src/data/prepare_packets.py` và `src/data/finalize_training.py`. Dùng `scripts/audit_training_data.py`, `scripts/audit_packet_mapping.py`, `scripts/validate_prepared_data.py` và `scripts/package_training_data.py` để kiểm tra và đóng gói khi cần sửa dữ liệu.

- Bàn giao `common.tar.gz` cho B/C/D; hỗ trợ xử lý lỗi đọc dữ liệu và giữ cùng bộ common trong một lần thí nghiệm.
- Bổ sung kiểm tra độc lập các tham số imputer/scaler từ đúng train của mỗi profile; IF chỉ fit benign train.
- Phối hợp B triển khai kiểm tra prediction: ID duy nhất, đủ 13.411 test flow, nhãn khớp dữ liệu và ba mô hình khớp nhau.
- Chạy tích hợp trên môi trường sạch: giải nén dữ liệu → nạp model → prediction → evaluator → dashboard. Cập nhật lệnh chạy thực tế vào tài liệu khi đã hoạt động.

## B — XGBoost và đánh giá chung

Code: `src/xgboost/`, `src/evaluation/`. Cấu hình: [xgboost.yaml](configs/xgboost.yaml). Đầu vào: `data/processed/xgboost/common/`.

- Triển khai loader/preprocess đọc NPZ sẵn có, giữ feature order và ID; viết train/predict có thể chạy từ CLI.
- Huấn luyện baseline; chọn cấu hình và threshold bằng validation. Lưu model, config, seed, feature names, validation scores và thời gian train/inference.
- Xuất `outputs/xgboost/predictions.parquet` theo [chuẩn chung](docs/TRAINING_GUIDE.md#4-xuất-kết-quả-để-so-sánh-chung).
- Viết một evaluator dùng cho cả ba mô hình: Precision, Recall, F1, AP, FPR, TP/FN/FP/TN; Recall/support theo công cụ và FPR theo browser. Kiểm thử các trường hợp tính metric và dữ liệu đầu vào lỗi.
- Tạo bảng metric, confusion matrix và PR curve từ prediction; thống nhất file kết quả với C để dashboard đọc trực tiếp. Feature importance có thể bổ sung sau baseline.

## C — Isolation Forest và dashboard

Code: `src/isolation_forest/`, `dashboard/app.py`. Cấu hình: [isolation_forest.yaml](configs/isolation_forest.yaml). Đầu vào: `data/processed/isolation_forest/common/`.

- Triển khai train/predict CLI; chỉ fit benign common train, dùng `X` đã scale sẵn.
- Chọn cấu hình và threshold bằng validation; đổi chiều anomaly score để điểm cao nghĩa là nghi malicious hơn. Lưu model, config, seed, validation scores và thời gian train/inference.
- Xuất `outputs/isolation_forest/predictions.parquet` theo [chuẩn chung](docs/TRAINING_GUIDE.md#4-xuất-kết-quả-để-so-sánh-chung); phân tích false positives bằng metadata.
- Làm dashboard đọc prediction và kết quả evaluator của B: bảng metric, PR curve, confusion matrix, thời gian inference và ví dụ prediction. Hiển thị thông báo dễ hiểu khi thiếu file; phân biệt anomaly score và xác suất.
- Bổ sung lệnh chạy dashboard vào README khi giao diện hoạt động với output thực.

## D — Sequence loader và 1D-CNN

Code: `src/cnn1d/`. Cấu hình: [cnn1d.yaml](configs/cnn1d.yaml). Đầu vào: `data/processed/cnn1d/`.

- Triển khai loader từ NPZ có sẵn: chuyển `(N,32,3)` sang `(N,3,32)` cho Conv1d, giữ mask/ID/nhãn; không trích lại PCAP hoặc chuẩn hóa `X` lần nữa.
- Xây Conv1d baseline và train/validation loop; lưu checkpoint, config, seed, loss curves và thời gian train/inference.
- Dùng mask trong forward/pooling; kiểm thử prediction ở eval không đổi khi thay giá trị padding nhưng giữ mask và packet thật.
- Giữ prefix 32 packet cho vòng baseline; chọn checkpoint và threshold bằng validation. Các thử nghiệm sequence length khác để sau khi luồng chính hoạt động.
- Viết predict CLI và xuất `outputs/cnn1d/predictions.parquet` theo [chuẩn chung](docs/TRAINING_GUIDE.md#4-xuất-kết-quả-để-so-sánh-chung).

## Sau khi train — hoàn thiện kết quả và demo

- **B/C/D:** khóa checkpoint/cấu hình và threshold từ validation; xuất prediction test theo đúng thứ tự ID. Giữ cả score để vẽ PR curve.
- **A + B:** kiểm tra ba bộ test IDs và nhãn khớp nhau trước khi tính metric.
- **B:** xuất bảng metric chung vào `reports/tables/`, confusion matrix từng model và PR curve so sánh vào `reports/figures/`.
- **B/C/D:** xem một số FP/FN thực tế; báo cáo Recall kèm support theo công cụ, FPR kèm FP/TN theo browser. Nếu không có lỗi thì ghi đúng kết quả, không tạo ví dụ giả.
- **B/C/D:** đo thời gian inference trên cùng máy, ghi batch size, warm-up, số mẫu và có tính preprocessing hay không; lưu thêm thời gian train và kích thước model.
- **C:** làm dashboard một trang từ kết quả có sẵn, gồm bảng so sánh, PR curve, confusion matrix và sample predictions. Không cần train trực tiếp hoặc nhận PCAP mới khi demo.
- **A:** chạy lại từ dữ liệu → model đã lưu → prediction → evaluator → dashboard; bổ sung lệnh thực tế trong README.
- **Cả nhóm:** chuẩn bị phần trình bày ngắn từ kết quả thật: bài toán, ba phương pháp, bảng/biểu đồ, đánh đổi Recall–FPR–chi phí và giới hạn dữ liệu. Không cần thêm bảng tracking hay báo cáo tiến độ.

## Quy tắc triển khai chung

- B/C/D có thể train song song. B và C thống nhất định dạng kết quả evaluator trước khi nối dashboard; A tích hợp sau khi có đủ output.
- Giữ nguyên split; chỉ dùng validation cho tuning, early stopping và threshold. Không cân bằng lại val/test.
- Metadata chỉ để truy vết/phân tích, không làm feature. Cả ba mô hình phải dùng common cho phép so sánh chính.
- Lưu artifact và cấu hình đủ để chạy lại mô hình đã chọn. Các log thực nghiệm phục vụ tái tạo, không cần bảng báo cáo tiến độ riêng.
- Cách đọc dữ liệu, giới hạn dataset và schema prediction dùng một nguồn tại [training guide](docs/TRAINING_GUIDE.md).
