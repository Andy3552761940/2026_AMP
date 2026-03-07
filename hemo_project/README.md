# Hemolysis-Pred: 抗菌肽溶血性预测项目

这是一个从零构建的、可复现实验工程，目标是预测抗菌肽是否具有溶血性（binary classification）。

## 项目目标
- 在标准 benchmark 数据集上实现 **Accuracy >= 90%**（建议在 HemoPI-1 / HemoPI-2 子集 + 严格划分测试集评估）。
- 兼顾泛化性能：
  - 使用 **分层 K 折交叉验证**
  - 提供 **独立测试集评估**
  - 提供 **模型集成（stacking）**

## 方法概览
我们采用两路特征 + 一层集成：
1. **Handcrafted 特征分支**
   - AAC（二十种氨基酸组成）
   - 物化特征（长度、净电荷估计、疏水比例、芳香族比例、分子量估计）
   - 模型：LightGBM
2. **Protein LM 表征分支**
   - 预训练模型：`facebook/esm2_t12_35M_UR50D`（默认，可通过参数切换）
   - 序列 mean pooling 嵌入
   - 模型：Logistic Regression
3. **Stacking 融合**
   - 用两路分支输出概率作为二级模型输入
   - 元学习器：Logistic Regression

> 该方案在公开文献中常见范式基础上做了轻量改进（双分支 + stacking + class_weight + repeated CV），通常可稳定优于单一特征模型。

## 快速开始

### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 下载 benchmark 数据（示例）
```bash
python scripts/download_benchmarks.py --out_dir data/raw
```

> 部分来源站点可能提供网页下载入口（需手工点击）。脚本会尽可能自动下载可直链资源，并生成下载指引。

### 3) 准备训练数据
将 CSV 放到 `data/processed/train.csv`，至少包含：
- `sequence`: 肽序列（字符串）
- `label`: 0/1（1=溶血）


### 一键运行（推荐）
```bash
bash scripts/one_click_run.sh \
  --train_csv data/processed/train.csv \
  --test_csv data/processed/test.csv \
  --out_dir outputs/one_click \
  --device cpu
```

常用可选参数：
- `--skip_install 1`：已装好依赖时跳过安装
- `--skip_eval 1`：只有待预测数据、没有标签时跳过评估
- `--seq_col / --label_col`：当你的列名不是 `sequence/label` 时覆盖默认值

### 4) 训练 + 交叉验证
```bash
python scripts/train.py \
  --train_csv data/processed/train.csv \
  --out_dir outputs/exp1 \
  --folds 5 --seed 42 \
  --esm_model_name facebook/esm2_t12_35M_UR50D
```

### 5) 独立测试评估
```bash
python scripts/evaluate.py \
  --model_dir outputs/exp1 \
  --test_csv data/processed/test.csv
```

### 6) 单文件预测
```bash
python scripts/predict.py \
  --model_dir outputs/exp1 \
  --input_csv data/processed/test.csv \
  --output_csv outputs/exp1/predictions.csv
```

## 输出
- `outputs/<exp>/cv_metrics.json`：交叉验证指标（含阈值调优结果）
- `outputs/<exp>/stacking_model.joblib`：融合模型
- `outputs/<exp>/decision_threshold.json`：自动搜索的最佳分类阈值
- `outputs/<exp>/model_config.json`：训练时的 ESM 配置
- `outputs/<exp>/branch_*.joblib`：分支模型
- `outputs/<exp>/predictions.csv`：预测结果

## Benchmark 来源
见：`docs/benchmark_sources.md`

## 备注
- 若你希望冲击更高上限（>92%），建议把 ESM backbone 替换为 `esm2_t12_35M_UR50D` 或 `prot_t5_xl_uniref50` 并增加温度缩放校准。
