# TEAM — Phân công 4 thành viên

**Tên đề tài:** Phát hiện bất thường trong lưu lượng TLS mã hóa bằng XGBoost, Isolation Forest và 1D-CNN.

**Ngày kickoff:** ____________  
**Repository URL:** ______________________________  
**Thời hạn dự kiến:** 12 ngày làm việc + 2–3 ngày dự phòng.

| Ký hiệu | Họ và tên | GitHub username | Vai trò | Nhánh làm việc |
|---|---|---|---|---|
| A | __________ | __________ | Data Engineer / Team Lead | `feature/data-pipeline` |
| B | __________ | __________ | XGBoost + Evaluation | `feature/xgboost` |
| C | __________ | __________ | Isolation Forest + Dashboard | `feature/isolation-forest` |
| D | __________ | __________ | Packet sequences + 1D-CNN | `feature/cnn1d` |

## Quy ước

- **A** sở hữu dữ liệu chuẩn, split và tích hợp cuối; **D** review phần packet↔flow.
- **B** sở hữu chuẩn metrics/evaluator, tất cả mô hình xuất theo `docs/contracts.md`.
- **C** sở hữu module IF và dashboard chỉ đọc kết quả chuẩn hóa.
- **D** sở hữu sequence representation/CNN; cần kiểm tra sample alignment từ ngày đầu.

**Check-in ngắn mỗi ngày:** đã làm / sẽ làm / blocker / thay đổi contract.

**Các mốc đánh giá:** D2 (dataset), D4 (data), D9 (3 predictions), D12 (demo).
