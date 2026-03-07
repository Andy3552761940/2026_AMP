#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline for hemolysis project:
# 1) create venv
# 2) install deps
# 3) check dataset files
# 4) train
# 5) evaluate (if test label exists)
# 6) predict
# 7) package deliverable zip

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# defaults
PYTHON_BIN="python3"
VENV_DIR=".venv"
TRAIN_CSV="data/processed/train.csv"
TEST_CSV="data/processed/test.csv"
OUT_DIR="outputs/one_click"
SEQ_COL="sequence"
LABEL_COL="label"
FOLDS=5
SEED=42
DEVICE="cpu"
THR=0.5
SKIP_INSTALL=0
SKIP_EVAL=0

usage() {
  cat <<USAGE
Usage: bash scripts/one_click_run.sh [options]

Options:
  --python_bin PATH         Python executable to use (default: python3)
  --venv_dir DIR            Virtual env directory (default: .venv)
  --train_csv PATH          Train CSV path (default: data/processed/train.csv)
  --test_csv PATH           Test CSV path (default: data/processed/test.csv)
  --out_dir DIR             Output directory (default: outputs/one_click)
  --seq_col NAME            Sequence column name (default: sequence)
  --label_col NAME          Label column name (default: label)
  --folds N                 K-folds (default: 5)
  --seed N                  Random seed (default: 42)
  --device DEVICE           cpu|cuda (default: cpu)
  --thr FLOAT               Prediction threshold (default: 0.5)
  --skip_install 0|1        Skip pip install (default: 0)
  --skip_eval 0|1           Skip evaluate step (default: 0)
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --venv_dir) VENV_DIR="$2"; shift 2 ;;
    --train_csv) TRAIN_CSV="$2"; shift 2 ;;
    --test_csv) TEST_CSV="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --seq_col) SEQ_COL="$2"; shift 2 ;;
    --label_col) LABEL_COL="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --thr) THR="$2"; shift 2 ;;
    --skip_install) SKIP_INSTALL="$2"; shift 2 ;;
    --skip_eval) SKIP_EVAL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

echo "[1/7] Checking python: $PYTHON_BIN"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN"; exit 1; }

echo "[2/7] Creating/activating virtual environment: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[3/7] Installing dependencies"
if [[ "$SKIP_INSTALL" == "0" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  echo "Skipped dependency installation"
fi

echo "[4/7] Validating input files"
[[ -f "$TRAIN_CSV" ]] || { echo "Missing train csv: $TRAIN_CSV"; exit 1; }
[[ -f "$TEST_CSV" ]] || { echo "Missing test csv: $TEST_CSV"; exit 1; }
mkdir -p "$OUT_DIR"

echo "[5/7] Training model"
python scripts/train.py \
  --train_csv "$TRAIN_CSV" \
  --seq_col "$SEQ_COL" \
  --label_col "$LABEL_COL" \
  --out_dir "$OUT_DIR" \
  --folds "$FOLDS" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[6/7] Evaluating and predicting"
if [[ "$SKIP_EVAL" == "0" ]]; then
  python scripts/evaluate.py \
    --model_dir "$OUT_DIR" \
    --test_csv "$TEST_CSV" \
    --seq_col "$SEQ_COL" \
    --label_col "$LABEL_COL" \
    --thr "$THR" \
    --device "$DEVICE"
else
  echo "Skipped evaluation"
fi

python scripts/predict.py \
  --model_dir "$OUT_DIR" \
  --input_csv "$TEST_CSV" \
  --output_csv "$OUT_DIR/predictions.csv" \
  --seq_col "$SEQ_COL" \
  --thr "$THR" \
  --device "$DEVICE"

echo "[7/7] Packaging deliverable"
bash scripts/package_project.sh

echo "Done. Key outputs:"
echo "- models/metrics: $OUT_DIR"
echo "- predictions:   $OUT_DIR/predictions.csv"
echo "- bundle:        $ROOT_DIR/deliverables/hemo_project_bundle.zip"
