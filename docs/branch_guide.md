# Hướng dẫn làm việc theo branch

## Công việc hằng ngày

```bash
git fetch origin
git switch feature/xgboost     # thay bằng nhánh được giao
git pull --ff-only origin feature/xgboost
# sửa code; kiểm tra thay đổi
git status
git add src/xgboost
git commit -m "feat(XGB-01): implement baseline preprocessing"
git push origin feature/xgboost
```

Trên GitHub, mở **Pull request** từ nhánh tính năng vào `main`. Chỉ merge khi reviewer đã duyệt và kiểm tra tự động chạy thành công.

## Đồng bộ với main

Sau khi PR của người khác đã được merge, có thể dùng:

```bash
git fetch origin
git switch feature/xgboost
git merge origin/main
# nếu có conflict thì giải quyết cẩn thận, test rồi push
```

Tránh `git push --force` lên nhánh chung. Nếu phải thay đổi các file chung như `docs/contracts.md`, `requirements.txt` hoặc tên cột, báo trước cho các thành viên liên quan.

## Quy tắc đóng task

Chỉ tick `[x]` trong `TODO.md` khi PR đã merge, output hợp lệ và có bằng chứng nghiệm thu. Link issue/PR ngay dưới Task ID.
