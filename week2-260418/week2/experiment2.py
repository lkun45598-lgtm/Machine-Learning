"""
实验二：威斯康星乳腺癌分类（逻辑回归）

实验步骤：
1. 读取数据集，显示前5行
2. 替换 ? 为 NaN，查看缺失值和类别分布
3. 丢弃含缺失值的行，查看数据基本信息
4. y 转换为 {0,1}，标准化特征，显示前5行
5. LogisticRegression 和 LogisticRegressionCV 默认参数建模
6. score() 准确率 + classification_report()
7. 不同 C 值（1, 0.1, 0.01, 0.001）使用 SAGA 求解器
8. 十折交叉验证
9. 绘制步骤7的4条 P-R 曲线和4条 ROC 曲线

运行方式：
    conda run -n pytorch python experiment2.py
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)
from pathlib import Path

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_FILE = BASE_DIR / "（加表头）breast-cancer-wisconsin.csv"
RANDOM_STATE = 42

# ============================================================
# 步骤1：读取数据集，显示前5行
# ============================================================
print("=" * 65)
print("步骤1. 读取数据集，显示前5行")
print("=" * 65)

df = pd.read_csv(DATA_FILE)
print(df.head().to_string())

# ============================================================
# 步骤2：替换 ? 为 NaN，查看缺失值和类别分布
# ============================================================
print("\n" + "=" * 65)
print("步骤2. 替换 ? 为 NaN，查看数据基本信息")
print("=" * 65)

# 将所有列中的 '?' 替换为 NaN
df.replace("?", np.nan, inplace=True)
# 确保所有列都是数值类型
df = df.apply(pd.to_numeric, errors="coerce")

df.info()
print("\n各列缺失值数量：")
print(df.isnull().sum())
print("\n类别分布（2=良性，4=恶性）：")
print(df["Class"].value_counts().sort_index())

# ============================================================
# 步骤3：丢弃含缺失值的行，查看数据基本信息
# ============================================================
print("\n" + "=" * 65)
print("步骤3. 丢弃含缺失值的行，查看数据基本信息")
print("=" * 65)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.info()
print(f"\n丢弃后样本数：{len(df)}（原699，丢弃16行含缺失值记录）")

# ============================================================
# 步骤4：y 转换为 {0,1}，标准化特征，显示前5行
# ============================================================
print("\n" + "=" * 65)
print("步骤4. y 转换为 {0,1}，标准化特征，显示前5行")
print("=" * 65)

feature_cols = [c for c in df.columns if c not in ["Sample code number", "Class"]]
X = df[feature_cols].values
# 2（良性）→ 0，4（恶性）→ 1
y = (df["Class"] == 4).astype(int).values

# 先切分，再标准化（只对训练集 fit，避免数据泄漏）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 训练集：fit + transform
X_test  = scaler.transform(X_test)       # 测试集：只 transform

print("标准化后训练集前5行：")
print(pd.DataFrame(X_train[:5], columns=feature_cols).round(4).to_string())

# ============================================================
# 步骤5：LogisticRegression 和 LogisticRegressionCV 默认参数
# ============================================================
print("\n" + "=" * 65)
print("步骤5. LogisticRegression 和 LogisticRegressionCV 默认参数")
print("=" * 65)

# --- LogisticRegression ---
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
correct_lr = (pred_lr == y_test).sum()

print("\n[LogisticRegression 默认参数]")
print(f"  测试样本数：{len(y_test)}，预测正确数：{correct_lr}")
print(f"  模型权重（9个）：{np.round(lr.coef_[0], 4)}")
print(f"  截距：{np.round(lr.intercept_, 4)}")

# --- LogisticRegressionCV（内置交叉验证自动选 C）---
lrcv = LogisticRegressionCV(cv=10, max_iter=1000, random_state=RANDOM_STATE)
lrcv.fit(X_train, y_train)
pred_lrcv = lrcv.predict(X_test)
correct_lrcv = (pred_lrcv == y_test).sum()

print("\n[LogisticRegressionCV 默认参数]")
print(f"  测试样本数：{len(y_test)}，预测正确数：{correct_lrcv}")
print(f"  自动选出的最优 C：{lrcv.C_[0]:.4f}")
print(f"  模型权重（9个）：{np.round(lrcv.coef_[0], 4)}")
print(f"  截距：{np.round(lrcv.intercept_, 4)}")

# ============================================================
# 步骤6：模型评估 —— score() + classification_report()
# ============================================================
print("\n" + "=" * 65)
print("步骤6. 模型评估")
print("=" * 65)

print(f"\n[LogisticRegression] 测试集准确率：{lr.score(X_test, y_test):.4f}")
print(classification_report(y_test, pred_lr,
                             target_names=["Benign(0)", "Malignant(1)"]))

print(f"\n[LogisticRegressionCV] 测试集准确率：{lrcv.score(X_test, y_test):.4f}")
print(classification_report(y_test, pred_lrcv,
                             target_names=["Benign(0)", "Malignant(1)"]))

# ============================================================
# 步骤7：不同 C 值 + SAGA 求解器
# ============================================================
print("\n" + "=" * 65)
print("步骤7. 不同 C 值的 LogisticRegression（SAGA 求解器）")
print("=" * 65)

C_values = [1, 0.1, 0.01, 0.001]
saga_models = {}

for C in C_values:
    model = LogisticRegression(C=C, solver="saga", max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    saga_models[C] = model

    pred = model.predict(X_test)
    correct = (pred == y_test).sum()
    print(f"\n  C={C}：测试样本数 {len(y_test)}，预测正确数 {correct}，"
          f"准确率 {model.score(X_test, y_test):.4f}")
    print(f"  权重：{np.round(model.coef_[0], 4)}")
    print(classification_report(y_test, pred,
                                 target_names=["Benign(0)", "Malignant(1)"],
                                 zero_division=0))

# ============================================================
# 步骤8：十折交叉验证
# ============================================================
print("\n" + "=" * 65)
print("步骤8. 十折交叉验证（C=1，SAGA）")
print("=" * 65)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_model = LogisticRegression(C=1, solver="saga", max_iter=5000, random_state=RANDOM_STATE)
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=skf, scoring="accuracy")

print(f"\n10折准确率：{np.round(cv_scores, 4)}")
print(f"均值：{cv_scores.mean():.4f}，标准差：{cv_scores.std():.4f}")

# ============================================================
# 步骤9：绘制步骤7的4条 P-R 曲线 和 4条 ROC 曲线
# ============================================================
print("\n" + "=" * 65)
print("步骤9. 绘制 P-R 曲线和 ROC 曲线")
print("=" * 65)

colors = ["steelblue", "darkorange", "green", "red"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for C, color in zip(C_values, colors):
    prob = saga_models[C].predict_proba(X_test)[:, 1]  # 恶性概率

    # P-R 曲线
    precision, recall, _ = precision_recall_curve(y_test, prob)
    pr_auc = auc(recall, precision)
    axes[0].plot(recall, precision, color=color, label=f"C={C} (AUC={pr_auc:.3f})")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=color, label=f"C={C} (AUC={roc_auc:.3f})")

# P-R 曲线设置
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("P-R Curves (SAGA, C = 1 / 0.1 / 0.01 / 0.001)")
axes[0].legend(loc="lower left")
axes[0].grid(alpha=0.3)

# ROC 曲线设置
axes[1].plot([0, 1], [0, 1], "k--", label="Random")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves (SAGA, C = 1 / 0.1 / 0.01 / 0.001)")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
out_path = OUTPUT_DIR / "pr_roc_curves.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"P-R 和 ROC 曲线已保存：{out_path}")

print("\n实验完成！")
