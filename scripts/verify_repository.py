"""Kiểm tra độ đầy đủ của scaffold, chỉ dùng standard library.

Không kiểm tra tính đúng đắn hay hiệu năng mô hình học máy.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "START_HERE.md", "README.md", "TODO.md", "TEAM.md", ".gitignore",
    "docs/contracts.md", "docs/git_workflow.md", "docs/branch_guide.md",
    "src/data/inspect_dataset.py", "src/data/extract_flows.py",
    "src/data/extract_packets.py", "src/data/split_dataset.py",
    "src/xgboost/train.py", "src/xgboost/predict.py",
    "src/isolation_forest/train.py", "src/isolation_forest/predict.py",
    "src/cnn1d/model.py", "src/cnn1d/build_sequences.py", "src/cnn1d/train.py",
    "src/evaluation/metrics.py", "src/evaluation/evaluate.py",
    "dashboard/app.py", "requirements.txt", ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/scaffold-check.yml",
]
missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
if missing:
    print("Missing files:", *missing, sep="\n  - ")
    sys.exit(1)
content = (ROOT / "TODO.md").read_text(encoding="utf-8")
task_ids = re.findall(r"\*\*(?:DATA|XGB|IF|CNN|EVAL|UI|DOC|INT|QA)-\d+", content)
task_ids = [i.strip("*") for i in task_ids]
print(f"Found {len(task_ids)} task references ({len(set(task_ids))} unique IDs).")
if len(task_ids) != len(set(task_ids)) or len(task_ids) != 34:
    print("Expected exactly 34 unique Task IDs in TODO.md.")
    sys.exit(1)
print("Scaffold structure OK. Model functionality is not yet implemented.")
