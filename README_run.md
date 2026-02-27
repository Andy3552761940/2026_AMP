# Light PLM Pipeline (Student Distill + Sampler)

This repo contains a lightweight student model training pipeline with:
- hard loss + distillation loss (split any/subheads)
- optional weighted sampling for rare subheads
- evaluation utilities

## 0) Environment
Tested on Linux with GPU (CUDA). Recommended:
- Python 3.10+
- PyTorch + CUDA
- pip install -r requirements.txt

Optional:
export PYTHONPATH=$(pwd)

## 1) Data
Expected CSV files in DATA_DIR:
- Fold1.csv
- Fold2.csv
- Test.csv

Required columns:
- Sequence
- Antimicrobial   (this is the "any" label)
- Antibacterial
- Antifungal
- Antiviral
- Antiparasitic

Label relation:
any == OR(all four subheads)

## 2) Paths (example)
export DATA_DIR=/root/autodl-tmp/Algorithm_Competition/dataverse_files
export MODEL_DIR=/root/autodl-tmp/models/esm2_t12_35M_UR50D
export TEACHER_RUN=runs/teacher_esm2_150M_lora_v2

## 3) Best Student Training (recommended)
python scripts/train_student_distill.py \
  --student_model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --train_csv Fold1.csv --val_csv Fold2.csv \
  --teacher_logits_train $TEACHER_RUN/logits_fold1.parquet \
  --teacher_logits_val $TEACHER_RUN/logits_fold2.parquet \
  --output_dir runs/student35m_distill_sampler_subpos_a2_ls03_pw20 \
  --lambda_soft 0.3 --epochs 8 --batch_size 16 --lr 3e-4 --fp16 1 \
  --alpha_sub 2.0 --pos_weight_mode auto --pos_weight_max 20 \
  --lambda_soft_any 0.1 --lambda_soft_sub 0.9 \
  --soft_subhead_weights 1,1,1,4 \
  --sampler subpos --sampler_max_mult 20 --sampler_power 0.5

Notes:
- --sampler options: none | antiparasitic | subpos
- --soft_subhead_weights order: Antibacterial, Antifungal, Antiviral, Antiparasitic
- pooler warning from ESM can be ignored (not used for this task)

## 4) Evaluation
Step A: find best threshold on val (Fold2)
python scripts/eval.py \
  --ckpt_dir runs/student35m_distill_sampler_subpos_a2_ls03_pw20/best \
  --base_model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --csv Fold2.csv \
  --out runs/student35m_distill_sampler_subpos_a2_ls03_pw20/val_metrics.json \
  --thr_from self

Step B: evaluate Test with fixed threshold from val (example 0.95)
python scripts/eval.py \
  --ckpt_dir runs/student35m_distill_sampler_subpos_a2_ls03_pw20/best \
  --base_model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --csv Test.csv \
  --out runs/student35m_distill_sampler_subpos_a2_ls03_pw20/test_metrics.json \
  --thr_from fixed --fixed_thr 0.95

## 5) Output Files
- runs/<exp>/best/                best checkpoint
- runs/<exp>/val_metrics.json     validation metrics
- runs/<exp>/test_metrics.json    test metrics

## 6) Reproducibility
Use --seed to fix randomness:
  --seed 42

## 7) Known Warnings
"You should probably TRAIN this model..." and pooler init warnings are expected for ESM.
They do not affect this fine-tuning pipeline.


## 高性能推理（推荐）

使用脚本：`scripts/predict_fast.py`，用于批量候选肽序列预测（输出 `Antimicrobial` + 4 子头概率）。

### 推荐参数（RTX 4090 实测）
- `--amp fp16`
- `--batch_size 128`
- `--sort_by_len 1`
- `--num_workers 4`
- `--chunk_size 50000`
- `--resume 1`

### 输入 CSV 要求
- 必需列：`Sequence`
- 可选列：`Hash`（推荐，若缺失则使用行号作为 `seq_id`）
- 若列名不同，可用：`--seq_col`、`--id_col`

### chunk / resume 用法
- `--chunk_size`：按块处理大 CSV，降低内存占用（如 `50000`）。
- `--resume 1`：支持断点续跑，会跳过已完成 chunk。
- 若修改 `amp` / `batch_size` / `sort_by_len` 等关键参数，请更换 `--pred_out` 文件名，避免误复用旧 chunk。

### 推荐命令（Test.csv 示例）
`python scripts/predict_fast.py --ckpt_dir runs/student35m_distill_sampler_subpos_a2_ls03_pw20_rerun/best --base_model_dir /root/work_runtime/models/esm2_t12_35M_UR50D --data_dir /root/work_runtime/data/dataverse_files --csv Test.csv --pred_out runs/predictions/test_fast_fp16_bs128_sort1.csv --batch_size 128 --num_workers 4 --amp fp16 --sort_by_len 1 --chunk_size 50000 --resume 1 --include_labels 1 --thr_any 0.95 --summary_out runs/predictions/test_fast_fp16_bs128_sort1_summary.json`

### 自定义候选文件（无标签）
将 `--in_csv /path/to/candidates.csv` 替换为真实路径；建议先按 `pred_any` 或 `p_Antimicrobial` 筛选，再按 `p_Antibacterial / p_Antifungal / p_Antiviral` 排序。


## 溶血性（Hemolysis）二分类：训练 + 评估 + 高性能预测

你可以直接复用本仓库的 ESM2 backbone + mean pooling 思路做溶血性预测。

### 1) 数据格式
准备 CSV（放在 `DATA_DIR` 下），至少包含：
- `Sequence`：肽序列
- `Hemolysis`：二分类标签（0/1）

如果列名不同，可在脚本里通过 `--seq_col` / `--label_col` 指定；如果你的 ID 列是 `id`（而不是本仓库 AMP 任务里的 `Hash`），建议额外加上 `--id_col id`，便于输出/回溯。

> 备注：有些数据文件虽然叫 `.csv` 但实际是 `\t` 分隔（TSV）。本仓库的溶血性脚本已做了基本的 CSV/TSV 自动兼容；如果你仍遇到列名识别问题，可将 TSV 显式转换为逗号分隔 CSV。

### 2) 训练（单标签）
```bash
export DATA_DIR=/path/to/hemo_data
export MODEL_DIR=/path/to/esm2_t12_35M_UR50D

python scripts/train_hemo.py \
  --model_dir $MODEL_DIR \
  --data_dir $DATA_DIR \
  --train_csv train.csv --val_csv val.csv \
  --label_col Hemolysis \
  --output_dir runs/hemo_esm2_35m \
  --epochs 5 --batch_size 16 --lr 3e-4 --fp16 1 \
  --pos_weight_mode auto --pos_weight_max 50
```

输出：
- `runs/hemo_esm2_35m/best/`：用于推理的 checkpoint（权重 + tokenizer）

### 3) 评估（可选）
```bash
python scripts/eval_hemo.py \
  --ckpt_dir runs/hemo_esm2_35m/best \
  --base_model_dir $MODEL_DIR \
  --data_dir $DATA_DIR --csv val.csv \
  --label_col Hemolysis \
  --out runs/hemo_esm2_35m/val_metrics.json \
  --thr_from self
```

### 4) 高性能预测（fp16 + length-sort + chunk/resume）
```bash
python scripts/predict_hemo_fast.py \
  --ckpt_dir runs/hemo_esm2_35m/best \
  --base_model_dir $MODEL_DIR \
  --in_csv /path/to/candidates.csv \
  --pred_out runs/hemo_pred_fast.csv \
  --amp fp16 --batch_size 128 --sort_by_len 1 --num_workers 4 \
  --chunk_size 50000 --resume 1 \
  --label_name Hemolysis --thr 0.5 \
  --summary_out runs/hemo_pred_fast_summary.json
```
