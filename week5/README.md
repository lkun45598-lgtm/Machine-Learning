# 实验五：基于决策树的鸢尾花分类

## 数据集

Iris Dataset（150 条，4 个特征 + 1 个标签）

- 特征：sepal length / sepal width / petal length / petal width
- 标签：setosa / versicolor / virginica（各 50 条）

## 实验流程

1. 加载 iris，查看缺失值，打印前 5 个样本
2. 训练 / 测试集划分 8:2，以 entropy 为分裂准则，对 `max_depth ∈ {2, 5, 10, 15, None}` × `min_samples_leaf ∈ {8, 6, 4, 2, 1}` 共 25 组参数遍历
3. 同 2，以 gini 为分裂准则
4. 在训练集每个特征上随机置 20 个 NaN，统计缺失值数量
5. 三种缺失值处理（丢弃 / 均值填充 / KNN 填充），重复步骤 3 的参数遍历
6. 缺失值标记法（用 -1 替代 NaN），重复步骤 3 的参数遍历
7. 可视化步骤 2、3 中性能最好的决策树各 1 张

> 数据预处理顺序：先 `train_test_split`，再仅用训练集统计量做填充 / 标准化，避免数据泄漏。
> 测试集不引入缺失值，保持原始数据。

## 主要结果

| 步骤 | 配置 | 最佳参数 | Train Acc | Test Acc |
|---|---|---|---|---|
| Step 2 | entropy | max_depth=5, min_samples_leaf=4 | 0.9833 | 0.9667 |
| Step 3 | gini | max_depth=5, min_samples_leaf=4 | 0.9833 | 0.9667 |
| Step 5a | gini + 丢弃 | max_depth=5, min_samples_leaf=2 | 0.9825 | 0.9667 |
| Step 5b | gini + 均值填充 | max_depth=5, min_samples_leaf=6 | 0.9500 | 0.9667 |
| Step 5c | gini + KNN 填充 | max_depth=2, min_samples_leaf=8 | 0.9500 | 0.9333 |
| Step 6 | gini + -1 标记 | max_depth=5, min_samples_leaf=8 | 0.9167 | 0.9667 |

- Step 4 引入 NaN 后，120 条训练样本中有 63 条至少含 1 个缺失值
- 丢弃法将训练集压缩到 57 条；其余三种方法保留 120 条
- 均值填充与 -1 标记在测试集上同样达到 96.67%，且训练集准确率不再 100%，泛化更稳健

## 输出文件

- `step2_entropy.csv` / `step3_gini.csv`：步骤 2/3 的 5×5 准确率表
- `step5_drop.csv` / `step5_mean.csv` / `step5_knn.csv`：步骤 5 三种缺失值处理的 5×5 准确率表
- `step6_mark.csv`：步骤 6 缺失值标记法的 5×5 准确率表
- `best_entropy_tree.png` / `best_gini_tree.png`：步骤 7 可视化最佳决策树
- `run_log.txt`：完整运行日志（含混淆矩阵）

## 运行方式

```bash
cd week5
conda run -n pytorch python experiment5.py
```
