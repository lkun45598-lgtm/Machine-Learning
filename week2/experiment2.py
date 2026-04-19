"""
实验二：威斯康星乳腺癌数据集分类实验

实验流程：
1. 加载数据，显示前5行
2. 清洗 '?' 异常值，查看缺失值情况
3. 切分训练集 / 测试集
4. 用训练集中位数填充缺失值（避免数据泄漏）
5. StandardScaler 标准化（只对训练集 fit）
6. 多模型训练与交叉验证评估
7. 最优模型在测试集上评估一次，绘制诊断图

运行方式：
    python experiment2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE = 42

# ============================================================
# 1. 加载数据，显示前5行
# ============================================================
print("=" * 70)
print("1. 加载数据，显示前5行")
print("=" * 70)

data_file = BASE_DIR / "（加表头）breast-cancer-wisconsin.csv"
df = pd.read_csv(data_file)
print(df.head())

# ============================================================
# 2. 清洗 '?' 异常值，查看缺失值
# ============================================================
print("\n" + "=" * 70)
print("2. 清洗 '?' 异常值，查看数据基本信息")
print("=" * 70)

# 原始文件中 Bare Nuclei 含 '?'，用 errors="coerce" 将其转为 NaN
feature_cols = [c for c in df.columns if c not in ["Sample code number", "Class"]]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.info()
print("\n各列缺失值数量：")
print(df.isnull().sum())
print("\n类别分布（2=良性，4=恶性）：")
print(df["Class"].value_counts().sort_index())

# ============================================================
# 3. 切分训练集 / 测试集
# ============================================================
print("\n" + "=" * 70)
print("3. 切分训练集 / 测试集（8:2，分层抽样）")
print("=" * 70)

X = df[feature_cols].copy()
# 将标签映射为 {0, 1}：2（良性）→ 0，4（恶性）→ 1
y = df["Class"].map({2: 0, 4: 1}).astype(int)

# stratify=y 保证训练集和测试集的良恶性比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"训练集：{len(X_train)} 条 | 测试集：{len(X_test)} 条")
print("训练集类别分布：")
print(y_train.value_counts().rename(index={0: "良性", 1: "恶性"}))

# ============================================================
# 4. 填充缺失值（只用训练集统计量，避免数据泄漏）
# ============================================================
print("\n" + "=" * 70)
print("4. 用训练集中位数填充缺失值")
print("=" * 70)

# fit 只看训练集，transform 同时处理训练集和测试集
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=feature_cols)

print(f"填充后训练集缺失值：{X_train.isnull().sum().sum()} 个")
print(f"填充后测试集缺失值：{X_test.isnull().sum().sum()} 个")

# ============================================================
# 5. 标准化（只对训练集 fit，测试集只 transform）
# ============================================================
print("\n" + "=" * 70)
print("5. StandardScaler 标准化")
print("=" * 70)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # 训练集：fit + transform
X_test_s  = scaler.transform(X_test)        # 测试集：只 transform

print("标准化后训练集前5行：")
print(pd.DataFrame(X_train_s[:5], columns=feature_cols).round(4).to_string())

# ============================================================
# 6. 多模型交叉验证（仅在训练集上进行）
# ============================================================
print("\n" + "=" * 70)
print("6. 多模型交叉验证评估（5折×10次，仅在训练集上）")
print("=" * 70)

# 所有候选模型
models = {
    "LogisticRegression(C=0.1)": LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE),
    "LogisticRegression(C=1)":   LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
    "LogisticRegression(C=10)":  LogisticRegression(C=10.0, max_iter=2000, random_state=RANDOM_STATE),
    "KNN(k=3)":                  KNeighborsClassifier(n_neighbors=3),
    "KNN(k=5)":                  KNeighborsClassifier(n_neighbors=5),
    "KNN(k=11)":                 KNeighborsClassifier(n_neighbors=11),
    "SVM(C=1, rbf)":             SVC(C=1.0,  kernel="rbf",    probability=True),
    "SVM(C=10, rbf)":            SVC(C=10.0, kernel="rbf",    probability=True),
    "SVM(C=1, linear)":          SVC(C=1.0,  kernel="linear", probability=True),
    "DecisionTree(depth=3)":     DecisionTreeClassifier(max_depth=3,    random_state=RANDOM_STATE),
    "DecisionTree(depth=5)":     DecisionTreeClassifier(max_depth=5,    random_state=RANDOM_STATE),
    "DecisionTree(depth=None)":  DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE),
    "RandomForest(n=50)":        RandomForestClassifier(n_estimators=50,  random_state=RANDOM_STATE),
    "RandomForest(n=100)":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "RandomForest(n=200)":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "GaussianNB":                GaussianNB(),
}

# 重复分层 K 折：5折×10次=50个评估结果，比单次更稳定
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)

scoring = {
    "accuracy":  "accuracy",
    "precision": "precision",
    "recall":    "recall",
    "f1":        "f1",
    "roc_auc":   "roc_auc",
}

rows = []
for name, model in models.items():
    scores = cross_validate(model, X_train_s, y_train, cv=cv, scoring=scoring)
    rows.append({
        "模型":      name,
        "Accuracy":  round(scores["test_accuracy"].mean(),  4),
        "Precision": round(scores["test_precision"].mean(), 4),
        "Recall":    round(scores["test_recall"].mean(),    4),
        "F1":        round(scores["test_f1"].mean(),        4),
        "F1_std":    round(scores["test_f1"].std(),         4),
        "ROC_AUC":   round(scores["test_roc_auc"].mean(),  4),
    })

cv_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
print(cv_df.to_string(index=False))

# ============================================================
# 7. 最优模型在测试集上评估一次
# ============================================================
print("\n" + "=" * 70)
print("7. 最优模型测试集评估（按 CV F1 选出，仅此一次）")
print("=" * 70)

best_name = cv_df.iloc[0]["模型"]
best_model = models[best_name]
best_model.fit(X_train_s, y_train)

y_pred  = best_model.predict(X_test_s)
# ROC/PR 曲线需要连续概率值
y_score = best_model.predict_proba(X_test_s)[:, 1] if hasattr(best_model, "predict_proba") \
          else best_model.decision_function(X_test_s)

print(f"最优模型：{best_name}")
print(f"测试集准确率：{accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_pred,
                             target_names=["Benign", "Malignant"],
                             digits=4))

# ============================================================
# 8. 绘制诊断图：混淆矩阵 + ROC 曲线 + PR 曲线
# ============================================================
print("\n" + "=" * 70)
print("8. 绘制诊断图")
print("=" * 70)

OUTPUT_DIR.mkdir(exist_ok=True)

# —— 特征分布图 ——
x_values = np.arange(1, 11)
bar_width = 0.38
fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
axes = axes.flatten()
class_map = {2: "Benign", 4: "Malignant"}

for ax, col in zip(axes, feature_cols):
    for class_value, offset in [(2, -bar_width / 2), (4, bar_width / 2)]:
        values = pd.to_numeric(df.loc[df["Class"] == class_value, col], errors="coerce").dropna()
        counts = values.astype(int).value_counts().reindex(x_values, fill_value=0)
        # 转为类内百分比，消除样本数量差异的干扰
        pct = counts / counts.sum() * 100 if counts.sum() > 0 else counts
        ax.bar(x_values + offset, pct, width=bar_width, label=class_map[class_value])
    ax.set_title(col)
    ax.set_xticks(x_values)
    ax.set_xlabel("Score")
    ax.set_ylabel("% within class")
    ax.grid(axis="y", alpha=0.25)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle("Feature Distributions by Class (Benign vs Malignant)", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
feat_path = OUTPUT_DIR / "feature_distributions_by_class.png"
fig.savefig(feat_path, dpi=180, bbox_inches="tight")
plt.close(fig)

# —— 混淆矩阵 + ROC + PR 三合一 ——
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
row_totals = cm.sum(axis=1, keepdims=True)
cm_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 混淆矩阵
im = axes[0].imshow(cm)
axes[0].set_title(f"Confusion Matrix\n{best_name}")
axes[0].set_xlabel("Predicted label")
axes[0].set_ylabel("True label")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Benign", "Malignant"])
axes[0].set_yticklabels(["Benign", "Malignant"])
threshold = cm.max() / 2 if cm.max() > 0 else 0
for i in range(2):
    for j in range(2):
        # viridis 低值深紫→白字，高值亮黄→黑字
        color = "black" if cm[i, j] > threshold else "white"
        axes[0].text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]*100:.1f}%",
                     ha="center", va="center", color=color)
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)
axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
axes[1].plot([0, 1], [0, 1], linestyle="--", label="Random")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate (FPR)")
axes[1].set_ylabel("True Positive Rate (TPR)")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.25)

# PR 曲线
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_score)
ap = average_precision_score(y_test, y_score)
axes[2].plot(recall_vals, precision_vals, label=f"AP = {ap:.4f}")
axes[2].set_title("Precision-Recall Curve")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].legend(loc="lower left")
axes[2].grid(alpha=0.25)

fig.tight_layout()
diag_path = OUTPUT_DIR / "confusion_roc_pr.png"
fig.savefig(diag_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"特征分布图已保存：{feat_path}")
print(f"诊断图已保存：    {diag_path}")
print("\n实验完成！")
