
# TLS Anomaly Detection

**Network Intrusion Detection Using XGBoost, Isolation Forest, and 1D-CNN**

> **Đề tài:** Phát hiện bất thường trong lưu lượng mạng mã hóa TLS bằng các phương pháp học máy: XGBoost, Isolation Forest và 1D-CNN.
>
> **Nhóm thực hiện:** 4 thành viên (A/B/C/D).
>
> **Trạng thái dự án:** 🟡 Planning & Development

---

## 1. Giới thiệu dự án

Dự án tập trung nghiên cứu và xây dựng một hệ thống phát hiện hành vi bất thường trong lưu lượng mạng mã hóa TLS (*Transport Layer Security*) mà không cần giải mã nội dung truyền tải (*payload*).

Hệ thống sử dụng các đặc trưng thống kê của luồng mạng (*flow-level features*) và chuỗi đặc trưng gói tin (*packet-level sequences*) để huấn luyện, đánh giá và so sánh ba thuật toán:

- **XGBoost:** Phân loại lưu lượng mạng bằng phương pháp học có giám sát (*Supervised Learning*).
- **Isolation Forest:** Phát hiện hành vi bất thường dựa trên mô hình học từ lưu lượng bình thường (*Anomaly Detection*).
- **1D-CNN:** Sử dụng mạng nơ-ron tích chập một chiều để học các đặc trưng từ chuỗi gói tin (*Deep Learning*).

### Mục tiêu nghiên cứu

1. Xây dựng quy trình thu thập, kiểm tra và tiền xử lý dữ liệu TLS.
2. Trích xuất các đặc trưng ở mức flow và packet.
3. Huấn luyện và tối ưu ba mô hình học máy.
4. Đánh giá hiệu quả phát hiện dựa trên các chỉ số thống nhất.
5. So sánh khả năng phát hiện, tỷ lệ cảnh báo sai và chi phí tính toán.
6. Xây dựng dashboard trực quan hóa kết quả thực nghiệm.

### Phạm vi triển khai

Giai đoạn đầu tập trung vào **phân tích dữ liệu offline**, sử dụng dataset có sẵn.

Hệ thống chưa hướng đến triển khai Network IDS thời gian thực.

---

## 2. Trạng thái hiện tại

> [!IMPORTANT]
> Repository hiện tại là **project scaffold** — bộ khung phục vụ phát triển và phân công công việc.
>
> Các thư mục và file khởi tạo đã được chuẩn bị, nhưng pipeline xử lý dữ liệu và các mô hình học máy **chưa được triển khai hoàn chỉnh**.

| Thành phần | Trạng thái |
|---|---|
| Repository Structure | ✅ Initialized |
| Team Responsibilities | ✅ Defined |
| Git Workflow | ✅ Documented |
| Dataset Inspection | ⬜ Pending |
| Data Preprocessing | ⬜ Pending |
| XGBoost Model | ⬜ Pending |
| Isolation Forest Model | ⬜ Pending |
| 1D-CNN Model | ⬜ Pending |
| Model Evaluation | ⬜ Pending |
| Streamlit Dashboard | ⬜ Pending |
| Final Report | ⬜ Pending |

Danh sách nhiệm vụ chi tiết được quản lý tại [TODO.md](TODO.md).

---

## 3. Phân công thành viên

Dự án được chia thành bốn nhóm trách nhiệm nhằm hạn chế xung đột mã nguồn và hỗ trợ phát triển song song.

| Owner | Role | Modules | Responsibilities |
|---|---|---|---|
| **A** | Data Engineer / Team Lead | `src/data/`, `data/`, `splits/`, `docs/contracts.md` | Dataset audit, flow/packet mapping, feature extraction, data splitting, QA, integration |
| **B** | ML Engineer / Evaluation | `src/xgboost/`, `src/evaluation/` | XGBoost training, hyperparameter tuning, metrics, model comparison |
| **C** | ML Engineer / Dashboard | `src/isolation_forest/`, `dashboard/` | Isolation Forest, anomaly scoring, threshold selection, Streamlit dashboard |
| **D** | Deep Learning Engineer | `src/cnn1d/` | Packet sequence construction, CNN architecture, training, evaluation |

### Nguyên tắc phân công

- Mỗi module có một thành viên chịu trách nhiệm chính.
- Các thành viên không tự ý thay đổi module của người khác.
- Những thay đổi liên quan đến dữ liệu hoặc cấu trúc dùng chung phải được thống nhất trước.
- Mọi thay đổi đưa vào nhánh `main` phải thông qua Pull Request.
- Mỗi nhiệm vụ cần có Task ID, người phụ trách và điều kiện hoàn thành.

---

## 4. Cấu trúc thư mục dự án

```text
tls-anomaly-detection/
│
├── .github/                     # GitHub Actions, Issue/PR templates
│
├── configs/                     # Cấu hình dataset và các mô hình
│
├── data/                        # Dữ liệu đầu vào và dữ liệu xử lý
│   ├── raw/                     # Dataset gốc (không commit)
│   ├── interim/                 # Dữ liệu trung gian
│   └── processed/               # Dữ liệu đã tiền xử lý
│
├── splits/                      # Train/Validation/Test manifests
│
├── src/
│   ├── data/                    # [A] Data processing pipeline
│   ├── xgboost/                 # [B] XGBoost model
│   ├── isolation_forest/        # [C] Isolation Forest model
│   ├── cnn1d/                   # [D] 1D-CNN model
│   └── evaluation/              # [B] Model evaluation
│
├── dashboard/                   # [C] Streamlit application
│
├── models/                      # Trained models/checkpoints
│
├── outputs/                     # Predictions and experiment outputs
│   ├── xgboost/
│   ├── isolation_forest/
│   └── cnn1d/
│
├── experiments/                 # Experiment configurations and logs
│
├── notebooks/                   # Exploratory data analysis
│
├── reports/                     # Evaluation reports and visualizations
│
├── scripts/                     # Utility and setup scripts
│
├── tests/                       # Unit and integration tests
│
├── docs/
│   ├── contracts.md             # Data & Model Output Contracts
│   ├── branch_guide.md          # Branch usage guide
│   └── git_workflow.md          # Team Git workflow
│
├── .gitignore
├── README.md
├── TEAM.md
├── TODO.md
└── requirements.txt
```

### Quy định quản lý thư mục

**Thành viên A — Data Pipeline**

Chịu trách nhiệm dữ liệu gốc, xử lý PCAP, chuẩn hóa nhãn, trích xuất đặc trưng và tạo các tập dữ liệu dùng chung.

**Thành viên B — XGBoost & Evaluation**

Chịu trách nhiệm huấn luyện XGBoost và phát triển các hàm đánh giá chung cho ba mô hình.

**Thành viên C — Isolation Forest & Dashboard**

Chịu trách nhiệm triển khai Isolation Forest, lựa chọn ngưỡng cảnh báo và xây dựng giao diện demo.

**Thành viên D — 1D-CNN**

Chịu trách nhiệm xây dựng chuỗi packet-level và triển khai mô hình 1D-CNN.

---

## 5. Hướng dẫn bắt đầu

### 5.1. Clone repository

```bash
git clone https://github.com/doandangkhoa/tls-anomaly-detection.git

cd tls-anomaly-detection
```

Kiểm tra trạng thái Git:

```bash
git status

git branch -a
```

> [!NOTE]
> Repository đã được khởi tạo trên GitHub. Không cần chạy lại `git init` sau khi clone.

### 5.2. Thiết lập môi trường Python

Yêu cầu đề xuất:

- Python 3.10 trở lên,