# 实验二：乳腺癌良恶性分类

## 数据集

Wisconsin Breast Cancer Dataset（699 条，11 列）

| 列名 | 说明 |
|------|------|
| Sample code number | 样本编号（不参与建模） |
| Clump Thickness | 肿块厚度 |
| Uniformity of Cell Size | 细胞大小均匀性 |
| Uniformity of Cell Shape | 细胞形状均匀性 |
| Marginal Adhesion | 边缘粘附 |
| Single Epithelial Cell Size | 单上皮细胞大小 |
| Bare Nuclei | 裸核（含 16 个缺失值） |
| Bland Chromatin | 平淡染色质 |
| Normal Nucleoli | 正常核仁 |
| Mitoses | 有丝分裂 |
| Class | 类别：2=良性，4=恶性 |

## 实验流程

1. 加载数据，查看前 5 行
2. `info()` 识别缺失值（Bare Nuclei 含 16 个 "?"）
3. `describe()` 查看统计量
4. 类别分布（良性 458 / 恶性 241）
5. 9 个特征分布直方图
6. 数据预处理：**先切分 → 用训练集中位数填充缺失值 → StandardScaler 标准化**
7. 多模型训练与评估
8. 最优模型混淆矩阵 & ROC 曲线可视化

## 模型评估结果

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|-----------|--------|----|-----|
| LogisticRegression(C=1)    | 0.9500 | 0.9362 | 0.9167 | 0.9263 | 0.9950 |
| LogisticRegression(C=0.1)  | 0.9571 | 0.9375 | 0.9375 | 0.9375 | 0.9946 |
| LogisticRegression(C=10)   | 0.9500 | 0.9362 | 0.9167 | 0.9263 | 0.9948 |
| KNN(k=3)                   | 0.9429 | 0.9167 | 0.9167 | 0.9167 | 0.9579 |
| KNN(k=5)                   | 0.9429 | 0.9167 | 0.9167 | 0.9167 | 0.9735 |
| KNN(k=11)                  | 0.9571 | 0.9375 | 0.9375 | 0.9375 | 0.9792 |
| SVM(C=1, rbf)              | 0.9571 | 0.9375 | 0.9375 | 0.9375 | 0.9896 |
| SVM(C=10, rbf)             | 0.9357 | 0.9333 | 0.8750 | 0.9032 | 0.9889 |
| SVM(C=1, linear)           | 0.9571 | 0.9375 | 0.9375 | 0.9375 | 0.9952 |
| DecisionTree(depth=3)      | 0.9357 | 0.8679 | 0.9583 | 0.9109 | 0.9413 |
| DecisionTree(depth=5)      | 0.9214 | 0.8936 | 0.8750 | 0.8842 | 0.9698 |
| DecisionTree(depth=None)   | 0.9286 | 0.9318 | 0.8542 | 0.8913 | 0.9108 |
| RandomForest(n=50)         | 0.9500 | 0.9362 | 0.9167 | 0.9263 | 0.9774 |
| RandomForest(n=100)        | 0.9500 | 0.9184 | 0.9375 | 0.9278 | 0.9882 |
| RandomForest(n=200)        | 0.9500 | 0.9184 | 0.9375 | 0.9278 | 0.9907 |
| **GaussianNB**             | **0.9571** | **0.9200** | **0.9583** | **0.9388** | **0.9906** |

最优模型（按 F1）：**GaussianNB**，F1=0.9388，AUC=0.9906

## 输出文件

- `histograms.png`：9 个特征分布直方图
- `confusion_roc.png`：最优模型混淆矩阵 & ROC 曲线

## 运行方式

```bash
cd week2
conda run -n pytorch python experiment2.py
```
