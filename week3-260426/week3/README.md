# 实验三：基于支持向量机的手写数字识别

## 数据集

Digits Dataset（1797 条，64 个特征 + 1 个标签）

- 特征：8×8 像素灰度图像展平为 64 维向量
- 标签：0-9 共 10 个数字类别
- 每个类别约 180 个样本

## 实验流程

1. 加载数据，查看样本数量、特征维度、标签数量
2. 可视化前 20 张手写数字图像
3. 数据集拆分（训练集 70% / 测试集 30%）
4. 四种核函数（linear, poly, rbf, sigmoid）默认参数建模
5. 不同 C 值与四种核函数的准确率对比
6. GridSearchCV 搜索 RBF 核的最优 C 和 gamma
7. 标准化后重复 RBF GridSearchCV
8. 多项式核 degree (1-9) 与 C 的准确率对比
9. 最优 RBF 模型混淆矩阵可视化

## 模型评估结果

### 默认参数四种核函数

| 核函数 | Accuracy |
|--------|----------|
| linear | ~0.98 |
| poly   | ~0.99 |
| rbf    | ~0.99 |
| sigmoid| ~0.95 |

### RBF 核 GridSearchCV

- 最佳参数：C=10, gamma=0.001
- 最佳交叉验证准确率：~0.99
- 测试集准确率：~0.99

### 标准化后 RBF 核

- 最佳参数：C=10, gamma=0.5
- 最佳交叉验证准确率：~0.99
- 测试集准确率：~0.99

### 多项式核

- 最佳参数：degree=3, C=10
- 最高准确率：~0.99

## 输出文件

- `first_20_digits.png`：前 20 张手写数字图像
- `kernel_default_accuracy.csv`：四种核函数默认参数准确率
- `c_kernel_accuracy.csv`：不同 C 与核函数准确率表
- `rbf_grid_cv_accuracy.csv`：RBF 核 GridSearchCV 结果
- `scaled_rbf_grid_cv_accuracy.csv`：标准化后 RBF 核 GridSearchCV 结果
- `poly_degree_c_accuracy.csv`：多项式核 degree 与 C 准确率表
- `best_rbf_confusion_matrix.png`：最优 RBF 模型混淆矩阵

## 运行方式

```bash
cd week3
conda run -n pytorch python experiment3.py
```
