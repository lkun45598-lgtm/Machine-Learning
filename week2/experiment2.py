import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_FILE = '（加表头）breast-cancer-wisconsin.csv'

# ============================================================
# 1. 加载数据，显示前5行
# ============================================================
print("=" * 60)
print("1. 加载数据，显示前5行")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
print(df.head())

# ============================================================
# 2. info() 查看数据信息及缺失值
# ============================================================
print("\n" + "=" * 60)
print("2. info() 数据信息")
print("=" * 60)

# Bare Nuclei 列含 "?" 缺失标记，转为 NaN
df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'], errors='coerce')
df.info()
print(f"\n缺失值统计：\n{df.isnull().sum()}")

# ============================================================
# 3. describe() 统计变量
# ============================================================
print("\n" + "=" * 60)
print("3. describe() 统计变量")
print("=" * 60)
print(df.describe())

# ============================================================
# 4. 类别分布
# ============================================================
print("\n" + "=" * 60)
print("4. 类别分布（2=良性，4=恶性）")
print("=" * 60)
print(df['Class'].value_counts())

# ============================================================
# 5. 特征分布直方图
# ============================================================
print("\n" + "=" * 60)
print("5. 绘制特征分布直方图")
print("=" * 60)

feature_cols = [c for c in df.columns if c not in ('Sample code number', 'Class')]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='white')
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
plt.suptitle('Feature Distributions - Breast Cancer Wisconsin', fontsize=14)
plt.tight_layout()
plt.savefig('histograms.png', dpi=150)
plt.close()
print("直方图已保存为 histograms.png")

# ============================================================
# 数据预处理：切分 → 填充缺失值 → 标准化
# ============================================================
X = df[feature_cols].copy()
y = df['Class'].map({2: 0, 4: 1})  # 2=良性→0，4=恶性→1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 只用训练集中位数填充（避免数据泄漏）
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test  = X_test.fillna(train_median)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ============================================================
# 6 & 7. 建模、评估、参数调优
# ============================================================
print("\n" + "=" * 60)
print("6 & 7. 建模与评估（Accuracy / Precision / Recall / F1 / AUC）")
print("=" * 60)


def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None
    auc = roc_auc_score(y_te, prob) if prob is not None else float('nan')
    return {
        'Model': name,
        'Accuracy':  round(accuracy_score(y_te, pred), 4),
        'Precision': round(precision_score(y_te, pred), 4),
        'Recall':    round(recall_score(y_te, pred), 4),
        'F1':        round(f1_score(y_te, pred), 4),
        'AUC':       round(auc, 4),
        '_model':    model,
        '_prob':     prob,
    }


results = []

results.append(evaluate('LogisticRegression(C=1)',   LogisticRegression(C=1,  max_iter=1000), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('LogisticRegression(C=0.1)', LogisticRegression(C=0.1,max_iter=1000), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('LogisticRegression(C=10)',  LogisticRegression(C=10, max_iter=1000), X_train_s, X_test_s, y_train, y_test))

results.append(evaluate('KNN(k=3)',  KNeighborsClassifier(n_neighbors=3),  X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('KNN(k=5)',  KNeighborsClassifier(n_neighbors=5),  X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('KNN(k=11)', KNeighborsClassifier(n_neighbors=11), X_train_s, X_test_s, y_train, y_test))

results.append(evaluate('SVM(C=1,rbf)',   SVC(C=1,  kernel='rbf',    probability=True), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('SVM(C=10,rbf)',  SVC(C=10, kernel='rbf',    probability=True), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('SVM(C=1,linear)',SVC(C=1,  kernel='linear', probability=True), X_train_s, X_test_s, y_train, y_test))

results.append(evaluate('DecisionTree(depth=3)',  DecisionTreeClassifier(max_depth=3,  random_state=42), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('DecisionTree(depth=5)',  DecisionTreeClassifier(max_depth=5,  random_state=42), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('DecisionTree(depth=None)',DecisionTreeClassifier(max_depth=None,random_state=42),X_train_s,X_test_s,y_train,y_test))

results.append(evaluate('RandomForest(n=50)',  RandomForestClassifier(n_estimators=50,  random_state=42), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('RandomForest(n=100)', RandomForestClassifier(n_estimators=100, random_state=42), X_train_s, X_test_s, y_train, y_test))
results.append(evaluate('RandomForest(n=200)', RandomForestClassifier(n_estimators=200, random_state=42), X_train_s, X_test_s, y_train, y_test))

results.append(evaluate('GaussianNB', GaussianNB(), X_train_s, X_test_s, y_train, y_test))

display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
result_df = pd.DataFrame(results)[display_cols]
print(result_df.to_string(index=False))

# ============================================================
# 8. 最优模型混淆矩阵 & ROC 曲线
# ============================================================
print("\n" + "=" * 60)
print("8. 最优模型可视化（按 F1 排序）")
print("=" * 60)

best = max(results, key=lambda r: r['F1'])
print(f"最优模型：{best['Model']}  F1={best['F1']}  AUC={best['AUC']}")

best_model = best['_model']
best_pred  = best_model.predict(X_test_s)
best_prob  = best['_prob']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 混淆矩阵
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign(0)', 'Malignant(1)'])
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title(f'Confusion Matrix\n{best["Model"]}')

# ROC 曲线
if best_prob is not None:
    fpr, tpr, _ = roc_curve(y_test, best_prob)
    axes[1].plot(fpr, tpr, label=f'AUC = {best["AUC"]}', color='steelblue')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve\n{best["Model"]}')
    axes[1].legend()

plt.tight_layout()
plt.savefig('confusion_roc.png', dpi=150)
plt.close()
print("混淆矩阵和ROC曲线已保存为 confusion_roc.png")

print("\n实验完成！")
