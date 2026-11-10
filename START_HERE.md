# BẮT ĐẦU Ở ĐÂY — Nhóm nghiên cứu TLS (4 người)

> Repo này đã có sẵn file, thư mục, tài liệu, TODO và **Git repository với commit đầu tiên trên nhánh `main`**.
> Toàn bộ mã xử lý dữ liệu/huấn luyện còn ở dạng khung, KHÔNG phải mô hình đã hoàn thiện.

## Bước 1 — Điền người và nhận việc

1. Mở [`TEAM.md`](TEAM.md), điền tên/GitHub username A/B/C/D.
2. Mở [`TODO.md`](TODO.md), chọn task theo vai trò, ghi link Issue/PR khi hoàn thành.
3. Cả nhóm đọc [`docs/contracts.md`](docs/contracts.md) trước khi thay đổi schema dữ liệu.

## Bước 2 — Tạo GitHub repo và push

Trên GitHub: **New repository** → đặt tên `tls-anomaly-detection` → chọn **Private** (khuyến nghị) → **KHÔNG** tích chọn *Add a README*, `.gitignore` hoặc License → Create repository.

Mở Terminal trong thư mục `tls-anomaly-detection` đã giải nén, chạy:

```bash
# Kiểm tra Git đã có sẵn
 git status
 git branch -a

# Thay YOUR_USERNAME bằng tài khoản GitHub của bạn
 git remote add origin https://github.com/YOUR_USERNAME/tls-anomaly-detection.git
 git push -u origin main
 git push origin feature/data-pipeline feature/xgboost feature/isolation-forest feature/cnn1d
```

GitHub sẽ yêu cầu xác thực theo phương thức đã cấu hình trên máy (ví dụ Git Credential Manager, GitHub CLI hoặc token phù hợp); không nhập mật khẩu GitHub thông thường vào lệnh.

**Nếu GitHub báo `remote origin already exists`:** dùng `git remote set-url origin <địa-chỉ-repo>`.

**Nếu bạn chỉ muốn 1 lệnh:** sau khi tạo repo trống, chạy `bash scripts/push_to_github.sh https://github.com/YOUR_USERNAME/tls-anomaly-detection.git`.

> Nếu giải nén bằng ứng dụng ẩn file ẩn, thư mục `.git` vẫn phải tồn tại. Chạy `ls -la` để kiểm tra trên Linux. Nếu mất `.git`, xem lệnh dự phòng trong [`docs/git_workflow.md`](docs/git_workflow.md).

## Bước 3 — Mời thành viên vào GitHub

Repo → Settings → Collaborators (hoặc Collaborators and teams) → Add people → mời 3 thành viên còn lại.

Mỗi thành viên `git clone` repo từ GitHub, sau đó làm việc theo nhánh:

| Người | Nhánh | Thư mục chính |
|---|---|---|
| A | `feature/data-pipeline` | `src/data/`, `data/`, `splits/` |
| B | `feature/xgboost` | `src/xgboost/`, `src/evaluation/` |
| C | `feature/isolation-forest` | `src/isolation_forest/`, `dashboard/` |
| D | `feature/cnn1d` | `src/cnn1d/` |

Ví dụ đối với B:

```bash
 git clone https://github.com/YOUR_USERNAME/tls-anomaly-detection.git
 cd tls-anomaly-detection
 git fetch origin
 git switch --track origin/feature/xgboost
 # viết code -> git add -> git commit -> git push
```

## Bước 4 — Kiểm tra project

```bash
python3 scripts/verify_repository.py
python3 -m compileall -q src dashboard
```

Các kiểm tra này chỉ kiểm tra **bộ khung và cú pháp Python**, chưa chứng minh pipeline ML chạy được.

## Các mốc nhóm

- **D2:** PCAP, TLS, nhãn và khả năng ghép flow↔packet được chứng minh.
- **D4:** master dataset, flow features, packet metadata, tập split dùng chung.
- **D9:** ba mô hình xuất cùng định dạng `predictions.parquet` trên cùng test IDs.
- **D12:** evaluator chung, bảng so sánh, Streamlit demo và báo cáo.

