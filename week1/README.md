# Machine Learning Experiments

华南农业大学 机器学习课程实验代码

---

## Week 1 — 线性回归实现波士顿房价预测

### 实验内容

使用波士顿房价数据集，完整走一遍机器学习基本流程：

- 数据加载与探索（`head` / `info` / `describe` / `corr`）
- 数据可视化（前8列分布直方图）
- 缺失值处理（先切分，再用训练集中位数填充，避免数据泄漏）
- 特征标准化（StandardScaler）
- 建模与对比：LinearRegression / Ridge / Lasso / ElasticNet
- 模型评估：MSE、MAE、R²
- 正则化参数调优（alpha = 1 / 10 / 20，l1_ratio = 0.2 / 0.8）

### 文件说明

| 文件 | 说明 |
|---|---|
| `experiment1.py` | 实验主代码 |
| `HousingData.csv` | 波士顿房价数据集（506条，14列，含缺失值） |
| `histograms.png` | 前8列特征分布直方图（运行后生成） |

### 数据集说明

| 列名 | 含义 |
|---|---|
| CRIM | 人均犯罪率 |
| ZN | 大面积住宅用地比例 |
| INDUS | 非零售商业用地比例 |
| CHAS | 是否毗邻查尔斯河（0/1） |
| NOX | 氮氧化物浓度 |
| RM | 平均房间数 |
| AGE | 1940年前建成的自住房比例 |
| DIS | 到就业中心的加权距离 |
| RAD | 公路可达性指数 |
| TAX | 房产税率 |
| PTRATIO | 师生比 |
| B | 黑人居民比例相关指数 |
| LSTAT | 低收入人口比例 |
| MEDV | 房价中位数（目标变量，单位：千美元） |

### 运行方式

```bash
conda activate pytorch
python experiment1.py
```

### 实验结果摘要

| 模型 | MSE | MAE | R² |
|---|---|---|---|
| LinearRegression | 24.9834 | 3.1476 | 0.6593 |
| Ridge(alpha=1) | 24.9870 | 3.1446 | 0.6593 |
| Ridge(alpha=10) | 25.0309 | 3.1293 | 0.6587 |
| Ridge(alpha=20) | 25.0821 | 3.1220 | 0.6580 |
| Lasso(alpha=1) | 27.7621 | 3.4452 | 0.6214 |
| Lasso(alpha=10) | 75.0454 | 6.2558 | -0.0233 |
| Lasso(alpha=20) | 75.0454 | 6.2558 | -0.0233 |
| ElasticNet(alpha=1, l1=0.2) | 28.1807 | 3.3613 | 0.6157 |
| ElasticNet(alpha=1, l1=0.8) | 27.8847 | 3.4087 | 0.6198 |
| ElasticNet(alpha=10, l1=0.2) | 58.1125 | 5.3625 | 0.2076 |
| ElasticNet(alpha=10, l1=0.8) | 75.0454 | 6.2558 | -0.0233 |
| ElasticNet(alpha=20, l1=0.2) | 71.1444 | 6.0627 | 0.0299 |
| ElasticNet(alpha=20, l1=0.8) | 75.0454 | 6.2558 | -0.0233 |

> Ridge 正则化最为稳健；Lasso/ElasticNet 在 alpha 过大时会严重欠拟合。

### 依赖环境

```
pandas
numpy
matplotlib
scikit-learn
```
