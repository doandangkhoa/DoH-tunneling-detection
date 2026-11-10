# Git workflow — nhóm 4 người

| Nhánh | Owner | Khu vực code |
|---|---|---|
| `feature/data-pipeline` | A | `src/data/`, `data/`, `splits/` |
| `feature/xgboost` | B | `src/xgboost/`, `src/evaluation/` |
| `feature/isolation-forest` | C | `src/isolation_forest/`, `dashboard/` |
| `feature/cnn1d` | D | `src/cnn1d/` |

**Quy trình:** issue → branch → commit nhỏ theo Task ID → pull request → review → checks → merge `main`.

Ví dụ:

```bash
git switch -c feature/xgboost
# lập trình trên src/xgboost
git add src/xgboost
git commit -m "feat(XGB-01): add baseline scaffold"
git push -u origin feature/xgboost
```

- **Không commit** dữ liệu PCAP/thật, `.venv`, kết quả lớn hay model weights.
- Khi sửa schema dùng chung, tạo issue thông báo A/B/C/D; người nhận dữ liệu phải review.
- Người review kiểm tra PR mô tả input/output, hướng dẫn kiểm thử, evidence và contract.
- Nếu dùng một branch dài hạn/người, vẫn nên tạo feature branch ngắn cho mỗi issue khi đã quen thao tác Git.
