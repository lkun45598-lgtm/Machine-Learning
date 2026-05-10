# -*- coding: utf-8 -*-
"""
实验五：基于决策树的鸢尾花分类

步骤：
1. 加载 iris，查看缺失值，打印前 5 个样本
2. 划分 8:2，以 entropy 为分裂准则，遍历 max_depth × min_samples_leaf
3. 同 2，以 gini 为分裂准则
4. 随机在训练集每个特征上删除 20 个特征值（置 NaN），查看缺失值
5. 缺失值丢弃 / 均值填充 / KNN 填充，重复步骤 3 的参数遍历
6. 缺失值标记法（用特定数值代替 NaN），重复步骤 3 的参数遍历
7. 可视化步骤 2、3 中性能最好的模型（plot_tree 各 1 张）

执行：conda run -n pytorch python experiment5.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer, KNNImputer

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "iris_dataset.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_DEPTHS = [2, 5, 10, 15, None]
MIN_LEAVES = [8, 6, 4, 2, 1]
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def header(title):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


# -----------------------------------------------------------------------------
# 1. 加载数据集，查看缺失值，打印前 5 个样本
# -----------------------------------------------------------------------------
header("Step 1: load iris dataset")

df = pd.read_csv(DATA_PATH)
print("数据集形状:", df.shape)
print("\n前 5 个样本:")
print(df.head())
print("\n各列缺失值数量:")
print(df.isna().sum())

X = df[FEATURE_NAMES].to_numpy(dtype=float)
y = df["target"].to_numpy(dtype=int)


# -----------------------------------------------------------------------------
# helper: 在给定的训练 / 测试集上跑参数网格，返回准确率表与最优配置
# -----------------------------------------------------------------------------
def sweep(X_tr, y_tr, X_te, y_te, criterion, label):
    """对 max_depth × min_samples_leaf 做网格遍历。"""
    rows = []
    best = {"test_acc": -1.0}
    for depth in MAX_DEPTHS:
        for leaf in MIN_LEAVES:
            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=RANDOM_STATE,
            )
            clf.fit(X_tr, y_tr)
            tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
            te_acc = accuracy_score(y_te, clf.predict(X_te))
            rows.append(
                {
                    "max_depth": "None" if depth is None else depth,
                    "min_samples_leaf": leaf,
                    "train_acc": round(tr_acc, 4),
                    "test_acc": round(te_acc, 4),
                }
            )
            if te_acc > best["test_acc"]:
                best = {
                    "criterion": criterion,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "train_acc": tr_acc,
                    "test_acc": te_acc,
                    "model": clf,
                }
    table = pd.DataFrame(rows)
    print(f"\n[{label}] train/test accuracy across max_depth x min_samples_leaf:")
    print(table.to_string(index=False))

    table_str = table.copy()
    table_str["max_depth"] = table_str["max_depth"].astype(str)
    pivot = table_str.pivot(
        index="max_depth", columns="min_samples_leaf", values="test_acc"
    )
    pivot = pivot.reindex(
        index=[str(d) if d is not None else "None" for d in MAX_DEPTHS],
        columns=MIN_LEAVES,
    )
    print(f"\n[{label}] test accuracy pivot (rows=max_depth, cols=min_samples_leaf):")
    print(pivot.to_string())

    cm = confusion_matrix(y_te, best["model"].predict(X_te))
    print(
        f"\n[{label}] best: max_depth={best['max_depth']} "
        f"min_samples_leaf={best['min_samples_leaf']} "
        f"train_acc={best['train_acc']:.4f} test_acc={best['test_acc']:.4f}"
    )
    print(f"[{label}] best model confusion matrix on test set:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    table.to_csv(OUT_DIR / f"{label}.csv", index=False)
    return table, best


# -----------------------------------------------------------------------------
# 2-3. 训练 / 测试拆分 + entropy / gini 网格
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

header("Step 2: entropy criterion grid")
_, best_entropy = sweep(X_train, y_train, X_test, y_test, "entropy", "step2_entropy")

header("Step 3: gini criterion grid")
_, best_gini = sweep(X_train, y_train, X_test, y_test, "gini", "step3_gini")


# -----------------------------------------------------------------------------
# 4. 在训练集每个特征上随机删除 20 个值
# -----------------------------------------------------------------------------
header("Step 4: introduce missing values into training set")

rng = np.random.default_rng(RANDOM_STATE)
X_train_missing = X_train.astype(float).copy()
n_train = X_train_missing.shape[0]
for j in range(X_train_missing.shape[1]):
    idx = rng.choice(n_train, size=20, replace=False)
    X_train_missing[idx, j] = np.nan

train_missing_df = pd.DataFrame(X_train_missing, columns=FEATURE_NAMES)
print("训练集形状:", train_missing_df.shape)
print("\n训练集各特征缺失值数量:")
print(train_missing_df.isna().sum())
print(f"\n训练集中含至少一个缺失值的样本数: {train_missing_df.isna().any(axis=1).sum()}")
print("测试集不引入缺失值，保持原始数据。")


# -----------------------------------------------------------------------------
# 5. 三种缺失值处理：丢弃 / 均值 / KNN，重复步骤 3 设置（gini）
# -----------------------------------------------------------------------------
header("Step 5: missing-value strategies (drop / mean / KNN), gini criterion")

# 5a 丢弃法
mask = ~np.isnan(X_train_missing).any(axis=1)
X_tr_drop, y_tr_drop = X_train_missing[mask], y_train[mask]
print(f"\n[drop] 丢弃后训练集大小: {X_tr_drop.shape}")
sweep(X_tr_drop, y_tr_drop, X_test, y_test, "gini", "step5_drop")

# 5b 均值填充：仅用训练集统计量 fit，训练集 / 测试集都用同一 imputer transform
mean_imp = SimpleImputer(strategy="mean")
X_tr_mean = mean_imp.fit_transform(X_train_missing)
X_te_mean = mean_imp.transform(X_test)
sweep(X_tr_mean, y_train, X_te_mean, y_test, "gini", "step5_mean")

# 5c KNN 填充：同样仅 fit 训练集
knn_imp = KNNImputer(n_neighbors=5)
X_tr_knn = knn_imp.fit_transform(X_train_missing)
X_te_knn = knn_imp.transform(X_test)
sweep(X_tr_knn, y_train, X_te_knn, y_test, "gini", "step5_knn")


# -----------------------------------------------------------------------------
# 6. 缺失值标记法：把 NaN 替换为特定数字（-1），保留缺失信号
# -----------------------------------------------------------------------------
header("Step 6: missing-value marking (-1), gini criterion")

MARK_VALUE = -1.0
mark_imp = SimpleImputer(strategy="constant", fill_value=MARK_VALUE)
X_tr_mark = mark_imp.fit_transform(X_train_missing)
X_te_mark = mark_imp.transform(X_test)
print(f"用 {MARK_VALUE} 标记缺失值；训练集形状: {X_tr_mark.shape}")
sweep(X_tr_mark, y_train, X_te_mark, y_test, "gini", "step6_mark")


# -----------------------------------------------------------------------------
# 7. 可视化步骤 2、3 中表现最好的模型
# -----------------------------------------------------------------------------
header("Step 7: visualize best trees from step 2 / step 3")


def save_tree(model, title, fname):
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {fname}")


save_tree(
    best_entropy["model"],
    f"Best entropy tree (max_depth={best_entropy['max_depth']}, "
    f"min_samples_leaf={best_entropy['min_samples_leaf']}, "
    f"test_acc={best_entropy['test_acc']:.4f})",
    "best_entropy_tree.png",
)
save_tree(
    best_gini["model"],
    f"Best gini tree (max_depth={best_gini['max_depth']}, "
    f"min_samples_leaf={best_gini['min_samples_leaf']}, "
    f"test_acc={best_gini['test_acc']:.4f})",
    "best_gini_tree.png",
)

print("\nAll outputs written to:", OUT_DIR)
