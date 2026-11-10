#!/usr/bin/env bash
# Đẩy repo đã git init lên một GitHub repository TRỐNG do bạn tạo trước.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/push_to_github.sh https://github.com/USER/tls-anomaly-detection.git"
  exit 2
fi
REPO_URL="$1"
if [[ "$REPO_URL" != https://github.com/* && "$REPO_URL" != git@github.com:* ]]; then
  echo "Repo URL must be a GitHub HTTPS or SSH URL."
  exit 2
fi
cd "$(dirname "$0")/.."
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "The .git directory is missing; unzip the complete project folder again."
  exit 1
fi
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi
printf 'Pushing local main and four team branches to %s\n' "$REPO_URL"
git push -u origin main
git push origin feature/data-pipeline feature/xgboost feature/isolation-forest feature/cnn1d
printf '\nDone. Invite collaborators in GitHub repository Settings.\n'
