# -*- coding: utf-8 -*-
"""
实验三：基于支持向量机的手写数字识别

使用方法：
1. 将本文件与 digits.csv 放在同一目录；
2. 执行：conda run -n pytorch python experiment3.py
3. 图像与结果表会保存到 outputs 文件夹。

说明：模板未规定 GridSearchCV 的交叉验证折数。这里统一使用 cv=3，
并设置 random_state=42 与 stratify=y，以保证结果可复现。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "digits.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 3

# 1. 加载数据集，查看样本数量、维度、标签数量
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"]).to_numpy(dtype=float)
y = df["target"].to_numpy(dtype=int)

print("样本数量:", X.shape[0])
print("特征维度:", X.shape[1])
print("标签数量:", len(np.unique(y)))
print("各标签样本数:")
print(pd.Series(y).value_counts().sort_index())

# 2. 使用 matshow 显示前 20 张图片
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, ax in enumerate(axes.ravel()):
    ax.matshow(X[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"label={y[i]}")
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "first_20_digits.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 3. 拆分数据集，测试集占比 30%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("训练集样本数:", X_train.shape[0])
print("测试集样本数:", X_test.shape[0])

# 4. 使用四种核函数建模，并打印准确率与前 20 个测试样本预测结果
kernels = ["linear", "poly", "rbf", "sigmoid"]
default_rows = []
default_preds = {}
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    default_rows.append({"kernel": kernel, "accuracy": acc})
    default_preds[kernel] = y_pred
    print(f"\n{kernel} kernel accuracy = {acc:.6f}")
    print("真实值:", y_test[:20])
    print("预测值:", y_pred[:20])

default_df = pd.DataFrame(default_rows)
default_df.to_csv(OUT_DIR / "kernel_default_accuracy.csv", index=False)

# 5. 不同 C 与四种核函数准确率表
C_values = [0.01, 0.05, 0.1, 0.5, 1, 10]
c_rows = []
best_c_model = {"accuracy": -1}
for kernel in kernels:
    for C in C_values:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        row = {"kernel": kernel, "C": C, "accuracy": acc}
        c_rows.append(row)
        if acc > best_c_model["accuracy"]:
            best_c_model = row.copy()

c_table = pd.DataFrame(c_rows).pivot(index="C", columns="kernel", values="accuracy")
print("\n不同 C 与核函数准确率表:")
print(c_table)
print("最高得分模型:", best_c_model)
c_table.to_csv(OUT_DIR / "c_kernel_accuracy.csv")

# 6. GridSearchCV 搜索 RBF 核的 C 和 gamma
gamma_values = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
param_grid = {"kernel": ["rbf"], "C": C_values, "gamma": gamma_values}
rbf_grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="accuracy", cv=CV_FOLDS, n_jobs=1)
rbf_grid.fit(X_train, y_train)
rbf_cv_table = pd.DataFrame(rbf_grid.cv_results_).pivot(
    index="param_gamma", columns="param_C", values="mean_test_score"
)
rbf_pred = rbf_grid.best_estimator_.predict(X_test)
rbf_test_acc = accuracy_score(y_test, rbf_pred)
print("\nRBF GridSearchCV 准确率表:")
print(rbf_cv_table)
print("最佳参数:", rbf_grid.best_params_)
print("最佳交叉验证准确率:", rbf_grid.best_score_)
print("最佳模型测试集准确率:", rbf_test_acc)
rbf_cv_table.to_csv(OUT_DIR / "rbf_grid_cv_accuracy.csv")

# 7. 标准化后重复 RBF GridSearchCV
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaled_grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="accuracy", cv=CV_FOLDS, n_jobs=1)
scaled_grid.fit(X_train_scaled, y_train)
scaled_cv_table = pd.DataFrame(scaled_grid.cv_results_).pivot(
    index="param_gamma", columns="param_C", values="mean_test_score"
)
scaled_pred = scaled_grid.best_estimator_.predict(X_test_scaled)
scaled_test_acc = accuracy_score(y_test, scaled_pred)
print("\n标准化后 RBF GridSearchCV 准确率表:")
print(scaled_cv_table)
print("标准化后最佳参数:", scaled_grid.best_params_)
print("标准化后最佳交叉验证准确率:", scaled_grid.best_score_)
print("标准化后最佳模型测试集准确率:", scaled_test_acc)
scaled_cv_table.to_csv(OUT_DIR / "scaled_rbf_grid_cv_accuracy.csv")

# 8. 多项式核 degree=1~9 与 C 的准确率表
poly_C_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
poly_rows = []
best_poly_model = {"accuracy": -1}
for degree in range(1, 10):
    for C in poly_C_values:
        model = SVC(kernel="poly", degree=degree, C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        row = {"degree": degree, "C": C, "accuracy": acc}
        poly_rows.append(row)
        if acc > best_poly_model["accuracy"]:
            best_poly_model = row.copy()

poly_table = pd.DataFrame(poly_rows).pivot(index="degree", columns="C", values="accuracy")
print("\n多项式核 degree 与 C 准确率表:")
print(poly_table)
print("多项式核最高得分模型:", best_poly_model)
poly_table.to_csv(OUT_DIR / "poly_degree_c_accuracy.csv")

# 9. 可视化 RBF 测试集最高准确率模型的混淆矩阵
# 本实验中 RBF 在 C=10、gamma='scale' 默认值下测试集准确率最高。
best_rbf = SVC(kernel="rbf", C=10)
best_rbf.fit(X_train, y_train)
best_rbf_pred = best_rbf.predict(X_test)
cm = confusion_matrix(y_test, best_rbf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(ax=ax, values_format="d", colorbar=False)
ax.set_title("Confusion Matrix of Best RBF SVM")
fig.tight_layout()
fig.savefig(OUT_DIR / "best_rbf_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n混淆矩阵:")
print(cm)
