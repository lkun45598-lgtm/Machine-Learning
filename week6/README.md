# 实验六：基于随机森林、XGBoost 的糖尿病预测

## 数据集

Pima Indians Diabetes Dataset（`diabetes.csv`，768 条样本 × 8 个特征 + 1 个标签）

- 特征：`Pregnancies` / `Glucose` / `BloodPressure` / `SkinThickness` / `Insulin` / `BMI` / `DiabetesPedigreeFunction` / `Age`
- 标签：`Outcome`（0=非糖尿病 500 条，1=糖尿病 268 条）
- 原始数据无 NaN，但 `Glucose` / `BloodPressure` / `SkinThickness` / `Insulin` / `BMI` 这 5 列中的 `0` 不合生理常识，视为缺失值（其余如 `Pregnancies=0` 是合法值，不动）

## 实验流程

1. 加载数据，查看形状 / 前 5 行 / 数据类型 / NaN 数 / 各列 `0` 的数量 / 类别分布
2. 把 5 列不合生理常识的 `0` 替换为 NaN，再 `train_test_split` 8:2 (stratify=y)，**仅用训练集的中位数**做 `SimpleImputer` 填充
3. 默认参数 `RandomForestClassifier` 训练 + 评估（模型参数、训练 / 测试集混淆矩阵、分类报告）
4. 网格遍历 `n_estimators ∈ {100, 200, 400}` × `max_depth ∈ {3, 5, 8, None}` × `min_samples_split ∈ {2, 5, 10}`（共 36 组），选 test_acc 最高的为最优模型
5. 原数据 `0` 设为 NaN（不做填充，XGBoost 内部用 `missing=nan` 处理），切分后用默认 `XGBClassifier` 训练 + 评估
6. 在步骤 5 基础上做 9 维参数网格搜索（`learning_rate`、`n_estimators`、`max_depth`、`min_child_weight`、`subsample`、`colsample_bytree`、`gamma`、`reg_alpha`、`reg_lambda`，每维 2 个取值共 512 组），3 折 `StratifiedKFold`、`accuracy` 评分，输出 CV 最优参数 + 测试集表现
7. 步骤 4 与步骤 6 的最优模型特征重要性对比，柱状图可视化

> 数据预处理顺序：先 `train_test_split`，再仅用训练集统计量做填充，避免数据泄漏。

## 主要结果

| 模型 | 关键参数 | Train Acc | Test Acc |
|---|---|---|---|
| RF default | n_estimators=100, max_depth=None, min_samples_split=2 | 1.0000 | 0.7662 |
| RF best    | n_estimators=100, max_depth=None, min_samples_split=2 | 1.0000 | 0.7662 |
| XGB default| 库默认 (objective=binary:logistic, missing=NaN) | 1.0000 | 0.7338 |
| XGB best   | lr=0.05, n_est=100, max_depth=3, min_child_weight=1, subsample=0.8, colsample_bytree=1.0, gamma=0, reg_alpha=0.1, reg_lambda=1.0 | 0.8550 | 0.7532 |

XGBoost 网格搜索 best CV acc = **0.7867**。

随机森林网格搜索最佳组合与默认参数完全一致（test_acc=0.7662），说明在该数据上 sklearn 的 RF 默认配置已较接近最优；但 `train_acc=1.0` 表明仍存在明显过拟合。

XGBoost 在网格搜索后，训练 / 测试准确率从 (1.0000 / 0.7338) 收敛到 (0.8550 / 0.7532)——**正则化（小学习率 + 浅树 + 子采样 + L1 / L2）显著降低了过拟合**，泛化能力变好。

特征重要性两个模型基本一致：**Glucose > BMI > Age / DiabetesPedigreeFunction > Insulin / BloodPressure > Pregnancies > SkinThickness**。其中 `DiabetesPedigreeFunction` 在 RF 中排第 3、XGB 中排第 6，是两个模型分歧最大的特征。

## 输出文件

- `step4_rf_grid.csv`：步骤 4 RF 网格 36 组的 train / test 准确率
- `step6_xgb_grid.csv`：步骤 6 XGB 512 组 GridSearchCV 的全部 cv 结果
- `step7_feature_importance.csv`：两个最优模型的特征重要性 + 排名
- `summary.csv`：四个模型 train / test acc 汇总
- `feature_importance.png`：特征重要性对比柱状图
- `run_log.txt`：完整运行日志（含所有混淆矩阵 / 分类报告）

## 运行方式

```bash
cd week6
conda run -n pytorch python experiment6.py
```

> 依赖：`pandas / numpy / matplotlib / scikit-learn / xgboost`
