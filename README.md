# TLS Anomaly Detection — Team Repository

**Đề tài:** Phát hiện bất thường trong lưu lượng TLS mã hóa bằng XGBoost, Isolation Forest và 1D-CNN.  
**Nhóm:** 4 thành viên (A/B/C/D) · **Kế hoạch:** 12 ngày làm việc + 2–3 ngày dự phòng.

> **Trạng thái:** Đây là **khung thư mục để bắt đầu phân công**, **không phải dự án đã cài đặt hoặc có mô hình huấn luyện**. Mọi tác vụ kỹ thuật cần triển khai theo [TODO.md](TODO.md).

## Giao việc theo owner

| Owner | Module được giao | Công việc |
|---|---|---|
| A — Data Lead | `src/data/`, `data/`, `splits/`, `docs/contracts.md` | Dataset audit, flow/packet join, feature extraction, split, QA, integration |
| B — ML + Evaluation | `src/xgboost/`, `src/evaluation/` | XGBoost, metric chuẩn, chung test set, comparison |
| C — ML + Dashboard | `src/isolation_forest/`, `dashboard/` | Benign-only IF, threshold trên validation, giao diện Streamlit |
| D — Deep Learning | `src/cnn1d/` | Packet sequences và CNN baseline, checkpoint, prediction |

## Bắt đầu ngay

**Nếu bạn vừa giải nén bản GitHub-ready:** mở **[START_HERE.md](START_HERE.md)** trước; repo đã được tạo sẵn Git và bốn nhánh A/B/C/D.

1. Giải nén ZIP, vào thư mục gốc của project.
2. Mở **[TODO.md](TODO.md)**, điền tên người, ngày kickoff và đánh dấu những task được giao.
3. Đọc **[docs/contracts.md](docs/contracts.md)** để thống nhất `sample_id`, dữ liệu đầu vào và kết quả dự đoán.
4. Tạo repository GitHub và các branch theo **[docs/git_workflow.md](docs/git_workflow.md)**.
5. Mỗi người code trong module mình sở hữu; tạo pull request với mã Task ID.

```bash
cd tls-anomaly-detection
# Repo đã có `.git` và commit đầu tiên: KHÔNG cần chạy lại `git init`.
git status
git branch
# Làm theo START_HERE.md để push GitHub.
```

Các thư viện trong `requirements.txt` là gợi ý cho giai đoạn sau; chưa có pipeline để chạy. Cài bằng `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` khi bắt đầu phát triển.

## Các artefact sẽ được tạo khi thực hiện task

- `data/interim/master_dataset.parquet` — mỗi flow một `sample_id`, nhãn và metadata nguồn (A).
- `data/interim/flow_features.parquet` — đặc trưng flow dùng chung cho XGBoost/IF (A).
- `data/interim/packet_metadata.parquet` — packet theo `sample_id` và thứ tự (A; D review).
- `splits/{train,val,test}_ids.csv` — ID chung cho cả ba mô hình (A).
- `outputs/{xgboost,isolation_forest,cnn1d}/predictions.parquet` — chuẩn đầu ra theo hợp đồng (B/C/D).
- `reports/` — kết quả so sánh cuối và hình minh họa (B; A review).

## Kiểm soát khoa học

- Bắt buộc xác nhận PCAP ↔ flow ↔ label **trước** khi xây CNN.
- Split theo group/capture nếu dữ liệu cho phép; không chia train/test riêng tại module của mô hình.
- Không dùng nhãn test để chọn features/hyperparameters/threshold.
- Isolation Forest huấn luyện benign-only trong thí nghiệm chính; `y_score` IF không phải xác suất.
- Ba mô hình **phải có cùng `sample_id` trên test** mới so sánh trực tiếp được.
- Không commit PCAP thật, dữ liệu được kiểm soát hoặc trọng số mô hình lớn lên Git mặc định.
