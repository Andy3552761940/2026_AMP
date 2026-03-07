# 溶血性预测 Benchmark 数据集来源与下载地址

> 以下列出社区常用数据源（研究/工业中最常引用），并给出来源说明与下载入口。

## 1) HemoPI-1 数据集
- **来源**：Raghava Lab（HemoPI server / 相关论文配套）
- **用途**：早期经典溶血性二分类 benchmark
- **下载入口**：
  - https://webs.iiitd.edu.in/raghava/hemopi/
  - https://webs.iiitd.edu.in/raghava/hemopi/dataset.php
- **说明**：部分数据通过网页按钮导出，建议下载后统一转成 UTF-8 CSV。

## 2) HemoPI-2 / HemoPI2.0 相关数据
- **来源**：Raghava Lab 后续版本（更新样本与模型）
- **用途**：更现代的数据分布，常用于验证泛化性能
- **下载入口**：
  - https://webs.iiitd.edu.in/raghava/hemopi2/
  - 若主站调整，可在论文 supplementary 或作者 GitHub 中检索“hemolytic peptides dataset”。

## 3) DBAASP（可筛选溶血实验）
- **来源**：DBAASP 数据库
- **用途**：构建外部验证集（跨来源泛化）
- **下载入口**：
  - https://dbaasp.org/
- **说明**：可按 hemolytic activity 字段筛选，需注意实验条件（浓度、细胞来源）的一致性。

## 4) SATPdb / APD3（可用于补充 AMP 序列）
- **来源**：公共肽数据库
- **用途**：负样本清洗、外部泛化检验
- **下载入口**：
  - SATPdb: http://crdd.osdd.net/raghava/satpdb/
  - APD3: https://aps.unmc.edu/

---

## 推荐 benchmark 评估协议
1. **主 benchmark**：HemoPI-1 官方划分（若提供）或按文献协议复现。
2. **泛化测试**：在 DBAASP 筛选出的外部集上做 zero-shot 测试。
3. **防信息泄露**：
   - 序列去重（完全重复 + 高相似冗余）
   - 训练/测试集按聚类划分（可选）
4. **指标**：Accuracy, AUROC, F1, MCC（优先报告 MCC 反映不平衡鲁棒性）。
