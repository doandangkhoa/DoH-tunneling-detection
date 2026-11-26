# Kết quả thực nghiệm DoH Tunneling Detection

Báo cáo dùng 13.411 flow thuộc tập test chung của CIRA-CIC-DoHBrw-2020, gồm 6.792 benign và 6.619 tunnel. Trình đánh giá kiểm tra ID và nhãn của từng mô hình với cùng một metadata tham chiếu. Nhãn dương là tunnel.

| Mô hình | F1 | Precision | Recall | FPR | AP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 97,30% | 98,18% | 96,43% | 1,74% | 99,47% | 118 | 236 |
| Isolation Forest | 76,63% | 62,32% | 99,49% | 58,63% | 69,03% | 3.982 | 34 |
| 1D-CNN | 94,78% | 92,74% | 96,90% | 7,39% | 89,85% | 502 | 205 |

Trên tập test này, XGBoost có F1, AP và tỷ lệ báo nhầm tốt nhất. Isolation Forest phát hiện thêm 202 flow tunnel so với XGBoost, đồng thời tạo thêm 3.864 cảnh báo sai. 1D-CNN nằm giữa hai mô hình về Recall và FPR. Các kết quả phản ánh cả cách huấn luyện và biểu diễn đầu vào. 1D-CNN dùng tối đa 32 packet đầu, còn hai mô hình cây dùng thống kê toàn flow.

## Dùng số liệu trong bài

- [metrics.csv](tables/metrics.csv) có số đếm và chỉ số tổng thể. Các tỷ lệ lưu ở thang 0–1.
- [source_metrics.csv](tables/source_metrics.csv) có FPR theo trình duyệt và Recall theo công cụ, kèm số flow, capture và nhóm thu thập. Mỗi dòng chỉ mô tả nguồn trong tập test.
- [pr_curve.csv](tables/pr_curve.csv) có các điểm Precision–Recall. AP được tính bằng `average_precision_score`. Điểm cuối đường cong không có ngưỡng tương ứng.
- [errors.parquet](tables/errors.parquet) có các mẫu FP và FN để truy vết theo ID, nguồn và nhóm. Dùng bảng tổng hợp và biểu đồ khi đưa minh chứng vào tiểu luận.
- [runtime.csv](tables/runtime.csv) ghi thời gian đã lưu của từng lần chạy. Số lượt đo, batch và cách warm-up khác nhau. Thời gian train XGBoost không có trong báo cáo đã lưu và được ghi `N/A`. Các số này chưa phải benchmark cùng điều kiện và không bao gồm trích xuất đặc trưng từ lưu lượng mạng.
- [provenance.json](tables/provenance.json) ghi đường dẫn và SHA-256 của các file đầu vào.

Thư mục [figures](figures) có biểu đồ Precision–Recall, biểu đồ theo nguồn và ma trận nhầm lẫn cho từng mô hình. Trong ma trận, hàng là nhãn thật và cột là nhãn dự đoán.

## Tạo lại

Từ thư mục gốc, sau khi đã có prediction và báo cáo lần chạy của cả ba mô hình:

```bash
PYTHONPATH=.runtime-deps python -m src.evaluation.evaluate
```

Lệnh đọc kết quả đã lưu và không train lại mô hình. Bảng tóm tắt trong README phản ánh lần xuất hiện tại.
