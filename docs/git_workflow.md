# Git workflow — nhóm 4 người

| Nhánh | Owner | Khu vực code |
|---|---|---|
| `feature/data-pipeline` | A | `src/data/`, `data/`, `splits/` |
| `feature/xgboost` | B | `src/xgboost/`, `src/evaluation/` |
| `feature/isolation-forest` | C | `src/isolation_forest/`, `dashboard/` |
| `feature/cnn1d` | D | `src/cnn1d/` |

**Quy trình:** branch → commit → pull request → review → checks → merge `main`.

Ví dụ:

```bash
git switch -c feature/xgboost
# lập trình trên src/xgboost
git add src/xgboost
git commit -m "feat(xgboost): add baseline training"
git push -u origin feature/xgboost
```

- **Không commit** dữ liệu PCAP/thật, `.venv`, kết quả lớn hay model weights.
- Khi sửa schema dùng chung, trao đổi với A/B/C/D; người nhận dữ liệu phải review.
- Người review kiểm tra PR mô tả input/output, hướng dẫn kiểm thử và chuẩn đầu ra.
- Nếu dùng một branch dài hạn/người, vẫn nên tạo feature branch ngắn cho mỗi thay đổi khi đã quen thao tác Git.

Xem [phân công triển khai](../TODO.md), [training guide](TRAINING_GUIDE.md) và [thao tác branch](branch_guide.md).
