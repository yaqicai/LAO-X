#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/check_empty_annotations.sh [DIR]
# Default DIR is current working directory.

TARGET_DIR="${1:-.}"

python - <<'PY' "${TARGET_DIR}"
import json
import os
import sys

root = sys.argv[1]
empty_files = []
invalid_files = []

for dirpath, _, filenames in os.walk(root):
    for name in filenames:
        if not name.endswith(".json"):
            continue
        path = os.path.join(dirpath, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            invalid_files.append(path)
            continue
        anns = data.get("annotations")
        if anns is None or len(anns) == 0:
            empty_files.append(path)

if invalid_files:
    print("[invalid json]")
    for p in invalid_files:
        print(p)

if empty_files:
    print("[empty annotations]")
    for p in empty_files:
        print(p)
        try:
            os.remove(p)
        except Exception as exc:
            print(f"[delete failed] {p}: {exc}")
else:
    print("No empty annotations found.")
PY
