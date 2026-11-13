"""Kiểm tra tài liệu và các module triển khai, chỉ dùng standard library.

Không kiểm tra tính đúng đắn hay hiệu năng mô hình học máy.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "TODO.md", "docs/TRAINING_GUIDE.md", ".gitignore",
    "docs/git_workflow.md", "docs/branch_guide.md",
    "src/data/prepare_dataset.py", "src/data/prepare_packets.py",
    "src/data/finalize_training.py", "scripts/package_training_data.py",
    "src/xgboost/train.py", "src/xgboost/predict.py",
    "src/isolation_forest/train.py", "src/isolation_forest/predict.py",
    "src/cnn1d/model.py", "src/cnn1d/build_sequences.py", "src/cnn1d/train.py",
    "src/evaluation/metrics.py", "src/evaluation/evaluate.py",
    "dashboard/app.py", "requirements.txt",
    ".github/workflows/scaffold-check.yml",
]
missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
if missing:
    print("Missing files:", *missing, sep="\n  - ")
    sys.exit(1)
print("Repository structure OK. Training guide and implementation modules are present.")
