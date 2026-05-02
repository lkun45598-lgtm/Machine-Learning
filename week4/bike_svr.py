# -*- coding: utf-8 -*-
"""
实验4：基于支持向量机回归（SVR）的共享单车日骑行量预测
数据集：FE_day（已做独热编码，731 条样本，35 列）
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

matplotlib.rcParams["axes.unicode_minus"] = False

DATA_FILE = "FE_day_数据说明.xlsx"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 步骤 1：加载数据 ----------
df = pd.read_excel(DATA_FILE, sheet_name="FE_day")
print("=" * 60)
print("步骤1：数据加载")
print("=" * 60)
print(f"样本数量: {df.shape[0]}")
print(f"原始维度（含 instant 与 cnt）: {df.shape[1]}")

X_full = df.drop(columns=["instant", "cnt"])
y = df["cnt"].values
print(f"特征维度: {X_full.shape[1]}")
print(f"标签数量（样本对应的 cnt 个数）: {len(y)}")
print("\n前 5 个样本：")
print(df.head().to_string())


# ---------- 步骤 2：挑选 20 个特征 ----------
print("\n" + "=" * 60)
print("步骤2：挑选 20 个特征用于建模")
print("=" * 60)
selector = SelectKBest(score_func=f_regression, k=20)
selector.fit(X_full, y)
mask = selector.get_support()
selected_features = X_full.columns[mask].tolist()
X = X_full[selected_features].copy()
print(f"通过 SelectKBest(f_regression) 选出的 20 个特征：")
for i, name in enumerate(selected_features, 1):
    print(f"  {i:2d}. {name}  (F={selector.scores_[X_full.columns.get_loc(name)]:.2f})")


# ---------- 步骤 3：拆分 + 标准化 + 网格搜索 ----------
print("\n" + "=" * 60)
print("步骤3：划分数据集 + RBF 核 SVR 网格搜索")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

param_grid = {
    "C": [0.1, 1, 10, 50, 100],
    "gamma": [0.001, 0.01, 0.05, 0.1, 0.5],
    "epsilon": [0.01, 0.1, 0.2],
}

grid = GridSearchCV(
    estimator=SVR(kernel="rbf"),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    return_train_score=True,
)
grid.fit(X_train_s, y_train_s)

best = grid.best_params_
print(f"最优参数: {best}")
print(f"交叉验证最优 R²: {grid.best_score_:.4f}")


# ---------- 步骤 4：评估 MSE / R² ----------
print("\n" + "=" * 60)
print("步骤4：训练集 / 测试集 评估")
print("=" * 60)

best_model = grid.best_estimator_
y_train_pred_s = best_model.predict(X_train_s)
y_test_pred_s = best_model.predict(X_test_s)

y_train_pred = y_scaler.inverse_transform(y_train_pred_s.reshape(-1, 1)).ravel()
y_test_pred = y_scaler.inverse_transform(y_test_pred_s.reshape(-1, 1)).ravel()

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

result_table = pd.DataFrame(
    {
        "数据集": ["训练集", "测试集"],
        "MSE": [train_mse, test_mse],
        "RMSE": [np.sqrt(train_mse), np.sqrt(test_mse)],
        "R²": [train_r2, test_r2],
    }
)
print(result_table.to_string(index=False))
result_table.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False, encoding="utf-8-sig")


# ---------- 步骤 5：gamma × C 热力图 ----------
print("\n" + "=" * 60)
print("步骤5：gamma × C 组合预测精度热力图")
print("=" * 60)

cv_results = pd.DataFrame(grid.cv_results_)
best_eps = best["epsilon"]
sub = cv_results[cv_results["param_epsilon"] == best_eps]

heat = sub.pivot_table(
    index="param_gamma", columns="param_C", values="mean_test_score"
)
heat = heat.astype(float).sort_index().sort_index(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heat.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)
ax.set_xlabel("C")
ax.set_ylabel("gamma")
ax.set_title(f"RBF SVR 5-fold CV R^2 Heatmap (epsilon={best_eps})")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                color="white" if v < heat.values.mean() else "black", fontsize=9)
fig.colorbar(im, ax=ax, label="mean CV R²")
plt.tight_layout()
heat_path = os.path.join(OUT_DIR, "gamma_C_heatmap.png")
plt.savefig(heat_path, dpi=150)
plt.close()
print(f"已保存热力图: {heat_path}")

# 预测对比图
fig, ax = plt.subplots(figsize=(10, 5))
order = np.argsort(y_test)
ax.plot(y_test[order], label="Ground truth", linewidth=1.5)
ax.plot(y_test_pred[order], label="Prediction", linewidth=1.0, alpha=0.8)
ax.set_xlabel("Test samples (sorted by ground truth)")
ax.set_ylabel("cnt")
ax.set_title(f"Test set: prediction vs ground truth  (R^2={test_r2:.3f})")
ax.legend()
plt.tight_layout()
pred_path = os.path.join(OUT_DIR, "test_prediction.png")
plt.savefig(pred_path, dpi=150)
plt.close()
print(f"已保存预测对比图: {pred_path}")

print("\n全部完成。")
