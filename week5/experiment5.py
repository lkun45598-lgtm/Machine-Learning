# -*- coding: utf-8 -*-
"""
实验五：基于决策树的鸢尾花分类（按 7 步流水线写法）

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


# =============================================================================
# Step 1: 加载数据集，查看缺失值，打印前 5 个样本
# =============================================================================
print("\n" + "=" * 70 + "\nStep 1: load iris dataset\n" + "=" * 70)

df = pd.read_csv(DATA_PATH)
print("数据集形状:", df.shape)
print("\n前 5 个样本:")
print(df.head())
print("\n各列缺失值数量:")
print(df.isna().sum())

X = df[FEATURE_NAMES].to_numpy(dtype=float)
y = df["target"].to_numpy(dtype=int)


# =============================================================================
# 训练 / 测试拆分（步骤 2、3 共用）
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")


# =============================================================================
# Step 2: 以 entropy 为分裂准则，遍历 max_depth × min_samples_leaf
# =============================================================================
print("\n" + "=" * 70 + "\nStep 2: entropy criterion grid\n" + "=" * 70)

rows_entropy = []
best_entropy_acc = -1.0
best_entropy_model = None
best_entropy_params = None

for depth in MAX_DEPTHS:
    for leaf in MIN_LEAVES:
        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, clf.predict(X_train))
        te_acc = accuracy_score(y_test, clf.predict(X_test))
        rows_entropy.append({
            "max_depth": "None" if depth is None else depth,
            "min_samples_leaf": leaf,
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
        })
        if te_acc > best_entropy_acc:
            best_entropy_acc = te_acc
            best_entropy_model = clf
            best_entropy_params = (depth, leaf)

table_entropy = pd.DataFrame(rows_entropy)
print("\n[entropy] train/test accuracy across max_depth x min_samples_leaf:")
print(table_entropy.to_string(index=False))

pivot_entropy = table_entropy.copy()
pivot_entropy["max_depth"] = pivot_entropy["max_depth"].astype(str)
pivot_entropy = pivot_entropy.pivot(
    index="max_depth", columns="min_samples_leaf", values="test_acc"
).reindex(
    index=[str(d) if d is not None else "None" for d in MAX_DEPTHS],
    columns=MIN_LEAVES,
)
print("\n[entropy] test accuracy pivot (rows=max_depth, cols=min_samples_leaf):")
print(pivot_entropy.to_string())

cm_entropy = confusion_matrix(y_test, best_entropy_model.predict(X_test))
print(
    f"\n[entropy] best: max_depth={best_entropy_params[0]} "
    f"min_samples_leaf={best_entropy_params[1]} test_acc={best_entropy_acc:.4f}"
)
print("[entropy] best model confusion matrix on test set:")
print(pd.DataFrame(cm_entropy, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

table_entropy.to_csv(OUT_DIR / "step2_entropy.csv", index=False)


# =============================================================================
# Step 3: 以 gini 为分裂准则，遍历 max_depth × min_samples_leaf
# =============================================================================
print("\n" + "=" * 70 + "\nStep 3: gini criterion grid\n" + "=" * 70)

rows_gini = []
best_gini_acc = -1.0
best_gini_model = None
best_gini_params = None

for depth in MAX_DEPTHS:
    for leaf in MIN_LEAVES:
        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, clf.predict(X_train))
        te_acc = accuracy_score(y_test, clf.predict(X_test))
        rows_gini.append({
            "max_depth": "None" if depth is None else depth,
            "min_samples_leaf": leaf,
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
        })
        if te_acc > best_gini_acc:
            best_gini_acc = te_acc
            best_gini_model = clf
            best_gini_params = (depth, leaf)

table_gini = pd.DataFrame(rows_gini)
print("\n[gini] train/test accuracy across max_depth x min_samples_leaf:")
print(table_gini.to_string(index=False))

pivot_gini = table_gini.copy()
pivot_gini["max_depth"] = pivot_gini["max_depth"].astype(str)
pivot_gini = pivot_gini.pivot(
    index="max_depth", columns="min_samples_leaf", values="test_acc"
).reindex(
    index=[str(d) if d is not None else "None" for d in MAX_DEPTHS],
    columns=MIN_LEAVES,
)
print("\n[gini] test accuracy pivot (rows=max_depth, cols=min_samples_leaf):")
print(pivot_gini.to_string())

cm_gini = confusion_matrix(y_test, best_gini_model.predict(X_test))
print(
    f"\n[gini] best: max_depth={best_gini_params[0]} "
    f"min_samples_leaf={best_gini_params[1]} test_acc={best_gini_acc:.4f}"
)
print("[gini] best model confusion matrix on test set:")
print(pd.DataFrame(cm_gini, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

table_gini.to_csv(OUT_DIR / "step3_gini.csv", index=False)


# =============================================================================
# Step 4: 在训练集每个特征上随机删除 20 个值（置 NaN），查看缺失值
# =============================================================================
print("\n" + "=" * 70 + "\nStep 4: introduce missing values into training set\n" + "=" * 70)

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


# =============================================================================
# Step 5a: 缺失值丢弃法 + gini 网格
# =============================================================================
print("\n" + "=" * 70 + "\nStep 5a: drop rows with NaN, gini criterion\n" + "=" * 70)

mask = ~np.isnan(X_train_missing).any(axis=1)
X_tr_drop = X_train_missing[mask]
y_tr_drop = y_train[mask]
print(f"丢弃后训练集大小: {X_tr_drop.shape}")

rows_drop = []
for depth in MAX_DEPTHS:
    for leaf in MIN_LEAVES:
        clf = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr_drop, y_tr_drop)
        tr_acc = accuracy_score(y_tr_drop, clf.predict(X_tr_drop))
        te_acc = accuracy_score(y_test, clf.predict(X_test))
        rows_drop.append({
            "max_depth": "None" if depth is None else depth,
            "min_samples_leaf": leaf,
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
        })

table_drop = pd.DataFrame(rows_drop)
print("\n[drop] train/test accuracy across max_depth x min_samples_leaf:")
print(table_drop.to_string(index=False))
table_drop.to_csv(OUT_DIR / "step5_drop.csv", index=False)


# =============================================================================
# Step 5b: 均值填充 + gini 网格（仅 fit 训练集，对训练/测试集都 transform）
# =============================================================================
print("\n" + "=" * 70 + "\nStep 5b: mean imputation, gini criterion\n" + "=" * 70)

mean_imp = SimpleImputer(strategy="mean")
X_tr_mean = mean_imp.fit_transform(X_train_missing)
X_te_mean = mean_imp.transform(X_test)

rows_mean = []
for depth in MAX_DEPTHS:
    for leaf in MIN_LEAVES:
        clf = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr_mean, y_train)
        tr_acc = accuracy_score(y_train, clf.predict(X_tr_mean))
        te_acc = accuracy_score(y_test, clf.predict(X_te_mean))
        rows_mean.append({
            "max_depth": "None" if depth is None else depth,
            "min_samples_leaf": leaf,
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
        })

table_mean = pd.DataFrame(rows_mean)
print("\n[mean] train/test accuracy across max_depth x min_samples_leaf:")
print(table_mean.to_string(index=False))
table_mean.to_csv(OUT_DIR / "step5_mean.csv", index=False)


# =============================================================================
# Step 5c: KNN 填充 + gini 网格
# =============================================================================
print("\n" + "=" * 70 + "\nStep 5c: KNN imputation, gini criterion\n" + "=" * 70)

knn_imp = KNNImputer(n_neighbors=5)
X_tr_knn = knn_imp.fit_transform(X_train_missing)
X_te_knn = knn_imp.transform(X_test)

rows_knn = []
for depth in MAX_DEPTHS:
    for leaf in MIN_LEAVES:
        clf = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr_knn, y_train)
        tr_acc = accuracy_score(y_train, clf.predict(X_tr_knn))
        te_acc = accuracy_score(y_test, clf.predict(X_te_knn))
        rows_knn.append({
            "max_depth": "None" if depth is None else depth,
            "min_samples_leaf": leaf,
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
        })

table_knn = pd.DataFrame(rows_knn)
print("\n[knn] train/test accuracy across max_depth x min_samples_leaf:")
print(table_knn.to_string(index=False))
table_knn.to_csv(OUT_DIR / "step5_knn.csv", index=False)


# =============================================================================
# Step 6: 缺失值标记法（用 -1 替代 NaN）+ gini 网格
# =============================================================================
print("\n" + "=" * 70 + "\nStep 6: missing-value marking (-1), gini criterion\n" + "=" * 70)

MARK_VALUE = -1.0
mark_imp = SimpleImputer(strategy="constant", fill_value=MARK_VALUE)
X_tr_mark = mark_imp.fit_transform(X_train_missing)
X_te_mark = mark_imp.transform(X_test)
print(f"用 {MARK_VALUE} 标记缺失值；训练集形状: {X_tr_mark.shape}")

rows_mark = []
for depth in MAX_DEPTHS:
    for leaf in MIN_LEAVES:
        clf = DecisionTreeClassifier(
            criterion="gini", max_depth=depth, min_samples_leaf=leaf,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr_mark, y_train)
        tr_acc = accuracy_score(y_train, clf.predict(X_tr_mark))
        te_acc = accuracy_score(y_test, clf.predict(X_te_mark))
        rows_mark.append({
            "max_depth": "None" if depth is None else depth,
            "min_samples_leaf": leaf,
            "train_acc": round(tr_acc, 4),
            "test_acc": round(te_acc, 4),
        })

table_mark = pd.DataFrame(rows_mark)
print("\n[mark] train/test accuracy across max_depth x min_samples_leaf:")
print(table_mark.to_string(index=False))
table_mark.to_csv(OUT_DIR / "step6_mark.csv", index=False)


# =============================================================================
# Step 7: 可视化步骤 2、3 中性能最好的模型（plot_tree 各 1 张）
# =============================================================================
print("\n" + "=" * 70 + "\nStep 7: visualize best trees from step 2 / step 3\n" + "=" * 70)

# 步骤 2 最佳：entropy
fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    best_entropy_model,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    ax=ax,
)
ax.set_title(
    f"Best entropy tree (max_depth={best_entropy_params[0]}, "
    f"min_samples_leaf={best_entropy_params[1]}, "
    f"test_acc={best_entropy_acc:.4f})"
)
fig.tight_layout()
fig.savefig(OUT_DIR / "best_entropy_tree.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved: best_entropy_tree.png")

# 步骤 3 最佳：gini
fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    best_gini_model,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    ax=ax,
)
ax.set_title(
    f"Best gini tree (max_depth={best_gini_params[0]}, "
    f"min_samples_leaf={best_gini_params[1]}, "
    f"test_acc={best_gini_acc:.4f})"
)
fig.tight_layout()
fig.savefig(OUT_DIR / "best_gini_tree.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved: best_gini_tree.png")

print("\nAll outputs written to:", OUT_DIR)
