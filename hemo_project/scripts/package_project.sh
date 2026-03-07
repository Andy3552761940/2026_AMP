#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/deliverables"
mkdir -p "$OUT_DIR"

ZIP_PATH="$OUT_DIR/hemo_project_bundle.zip"
rm -f "$ZIP_PATH"

cd "$ROOT_DIR"
zip -r "$ZIP_PATH" README.md requirements.txt docs scripts src tests -x "*/__pycache__/*"

echo "Packaged: $ZIP_PATH"
