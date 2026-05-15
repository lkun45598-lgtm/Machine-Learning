# -*- coding: utf-8 -*-
"""
实验六：基于随机森林、XGBoost 的糖尿病预测（按 7 步流水线写法）

执行：conda run -n pytorch python experiment6.py
"""

from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
CLASS_NAMES = ["non-diabetic", "diabetic"]


def cm_df(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES],
    )


def print_eval(name, model, X_tr, y_tr, X_te, y_te):
    tr_pred = model.predict(X_tr)
    te_pred = model.predict(X_te)
    tr_acc = accuracy_score(y_tr, tr_pred)
    te_acc = accuracy_score(y_te, te_pred)
    print(f"\n[{name}] train acc = {tr_acc:.4f}, test acc = {te_acc:.4f}")
    print(f"\n[{name}] train confusion matrix:")
    print(cm_df(y_tr, tr_pred, "tr").to_string())
    print(f"\n[{name}] test confusion matrix:")
    print(cm_df(y_te, te_pred, "te").to_string())
    print(f"\n[{name}] train classification report:")
    print(classification_report(y_tr, tr_pred, target_names=CLASS_NAMES, digits=4))
    print(f"[{name}] test classification report:")
    print(classification_report(y_te, te_pred, target_names=CLASS_NAMES, digits=4))
    return tr_acc, te_acc


# =============================================================================
# Step 1: 查看数据集结构、前 5 行、缺失值、数据类型
# =============================================================================
print("\n" + "=" * 70 + "\nStep 1: load diabetes dataset\n" + "=" * 70)

df = pd.read_csv(DATA_PATH)
print("数据集形状:", df.shape)
print("\n前 5 行:")
print(df.head().to_string())
print("\n数据类型:")
print(df.dtypes)
print("\n各列 NaN 缺失值数量:")
print(df.isna().sum())
print("\n各列等于 0 的样本数（仅 ZERO_AS_NAN_COLS 是生理上不可能的 0）:")
zero_counts = (df[FEATURE_NAMES] == 0).sum()
print(zero_counts.to_string())
print("\nOutcome 分布:")
print(df["Outcome"].value_counts().to_string())


# =============================================================================
# Step 2: 处理 “0” 值（仅对 5 列生理不可能为 0 的特征），划分训练集 / 测试集
# =============================================================================
print("\n" + "=" * 70 + "\nStep 2: replace impossible 0 with NaN, split, impute (median)\n" + "=" * 70)

df_rf = df.copy()
df_rf[ZERO_AS_NAN_COLS] = df_rf[ZERO_AS_NAN_COLS].replace(0, np.nan)
print(f"将 {ZERO_AS_NAN_COLS} 中的 0 替换为 NaN 后，缺失值统计:")
print(df_rf[FEATURE_NAMES].isna().sum().to_string())

X_all = df_rf[FEATURE_NAMES].to_numpy(dtype=float)
y_all = df_rf["Outcome"].to_numpy(dtype=int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all,
)
print(f"\n训练集大小: {X_train_raw.shape}, 测试集大小: {X_test_raw.shape}")

# 中位数填充：仅用训练集 fit，对训练 / 测试集都 transform，避免数据泄漏
median_imp = SimpleImputer(strategy="median")
X_train = median_imp.fit_transform(X_train_raw)
X_test = median_imp.transform(X_test_raw)
print("中位数填充完成（仅用训练集统计量）。")


# =============================================================================
# Step 3: 默认参数随机森林 + 评估
# =============================================================================
print("\n" + "=" * 70 + "\nStep 3: default RandomForestClassifier\n" + "=" * 70)

rf_default = RandomForestClassifier(random_state=RANDOM_STATE)
rf_default.fit(X_train, y_train)
print("\n[rf_default] 模型参数:")
print(rf_default.get_params())
rf_default_tr_acc, rf_default_te_acc = print_eval(
    "rf_default", rf_default, X_train, y_train, X_test, y_test
)


# =============================================================================
# Step 4: 调参随机森林（n_estimators × max_depth × min_samples_split）
# =============================================================================
print("\n" + "=" * 70 + "\nStep 4: tuned RandomForest grid\n" + "=" * 70)

RF_N_ESTIMATORS = [100, 200, 400]
RF_MAX_DEPTHS = [3, 5, 8, None]
RF_MIN_SPLITS = [2, 5, 10]

rf_rows = []
best_rf_acc = -1.0
best_rf = None
best_rf_params = None

for n_est, depth, msp in product(RF_N_ESTIMATORS, RF_MAX_DEPTHS, RF_MIN_SPLITS):
    clf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, min_samples_split=msp,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    tr_acc = accuracy_score(y_train, clf.predict(X_train))
    te_acc = accuracy_score(y_test, clf.predict(X_test))
    rf_rows.append({
        "n_estimators": n_est,
        "max_depth": "None" if depth is None else depth,
        "min_samples_split": msp,
        "train_acc": round(tr_acc, 4),
        "test_acc": round(te_acc, 4),
    })
    if te_acc > best_rf_acc:
        best_rf_acc = te_acc
        best_rf = clf
        best_rf_params = (n_est, depth, msp)

rf_table = pd.DataFrame(rf_rows).sort_values(
    by=["test_acc", "train_acc"], ascending=[False, False]
).reset_index(drop=True)
print("\n[rf_grid] train/test accuracy across grid (top 10 by test_acc):")
print(rf_table.head(10).to_string(index=False))
rf_table.to_csv(OUT_DIR / "step4_rf_grid.csv", index=False)

print(
    f"\n[rf_best] n_estimators={best_rf_params[0]}, "
    f"max_depth={best_rf_params[1]}, "
    f"min_samples_split={best_rf_params[2]}, "
    f"test_acc={best_rf_acc:.4f}"
)
print("\n[rf_best] 模型参数:")
print(best_rf.get_params())
print_eval("rf_best", best_rf, X_train, y_train, X_test, y_test)


# =============================================================================
# Step 5: 原数据集 "0" 设为 NaN，采用 XGBoost（默认参数），重复步骤 3
# =============================================================================
print("\n" + "=" * 70 + "\nStep 5: XGBoost with 0->NaN (no imputation; XGB handles NaN)\n" + "=" * 70)

df_xgb = df.copy()
df_xgb[ZERO_AS_NAN_COLS] = df_xgb[ZERO_AS_NAN_COLS].replace(0, np.nan)

X_xgb_all = df_xgb[FEATURE_NAMES].to_numpy(dtype=float)
y_xgb_all = df_xgb["Outcome"].to_numpy(dtype=int)

X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(
    X_xgb_all, y_xgb_all,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_xgb_all,
)
print(f"\n训练集大小: {X_xgb_train.shape}, 测试集大小: {X_xgb_test.shape}")
print("训练集各列 NaN 数:")
print(pd.DataFrame(X_xgb_train, columns=FEATURE_NAMES).isna().sum().to_string())

xgb_default = XGBClassifier(
    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_default.fit(X_xgb_train, y_xgb_train)
print("\n[xgb_default] 模型参数:")
print(xgb_default.get_params())
xgb_default_tr_acc, xgb_default_te_acc = print_eval(
    "xgb_default", xgb_default, X_xgb_train, y_xgb_train, X_xgb_test, y_xgb_test
)


# =============================================================================
# Step 6: XGBoost 9 维网格搜索
# =============================================================================
print("\n" + "=" * 70 + "\nStep 6: XGBoost grid search (9 hyper-parameters)\n" + "=" * 70)

param_grid = {
    "learning_rate":    [0.05, 0.1],
    "n_estimators":     [100, 200],
    "max_depth":        [3, 5],
    "min_child_weight": [1, 3],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma":            [0, 0.1],
    "reg_alpha":        [0, 0.1],
    "reg_lambda":       [1.0, 1.5],
}
n_combos = int(np.prod([len(v) for v in param_grid.values()]))
print(f"参数组合总数: {n_combos}")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(
    estimator=XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=1, nthread=1, tree_method="hist",
    ),
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=8,
    verbose=0,
    refit=True,
)
gs.fit(X_xgb_train, y_xgb_train)

print(f"\n[xgb_grid] best cv acc = {gs.best_score_:.4f}")
print(f"[xgb_grid] best params  = {gs.best_params_}")

cv_results = pd.DataFrame(gs.cv_results_)[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score").reset_index(drop=True)
cv_results.to_csv(OUT_DIR / "step6_xgb_grid.csv", index=False)
print("\n[xgb_grid] top 10 by mean cv acc:")
print(cv_results.head(10).to_string())

xgb_best = gs.best_estimator_
print("\n[xgb_best] 模型参数:")
print(xgb_best.get_params())
xgb_best_tr_acc, xgb_best_te_acc = print_eval(
    "xgb_best", xgb_best, X_xgb_train, y_xgb_train, X_xgb_test, y_xgb_test
)


# =============================================================================
# Step 7: 特征重要性对比（RF best vs XGB best）柱状图
# =============================================================================
print("\n" + "=" * 70 + "\nStep 7: feature importance comparison (RF best vs XGB best)\n" + "=" * 70)

rf_imp = best_rf.feature_importances_
xgb_imp = xgb_best.feature_importances_

imp_df = pd.DataFrame({
    "feature": FEATURE_NAMES,
    "rf_best_importance":  rf_imp,
    "xgb_best_importance": xgb_imp,
})
imp_df["rf_rank"]  = imp_df["rf_best_importance"].rank(ascending=False).astype(int)
imp_df["xgb_rank"] = imp_df["xgb_best_importance"].rank(ascending=False).astype(int)
print("\n[importance] 特征重要性对比表:")
print(imp_df.to_string(index=False))
imp_df.to_csv(OUT_DIR / "step7_feature_importance.csv", index=False)

# 按 RF 重要性排序，便于读图
order = np.argsort(rf_imp)[::-1]
feat_sorted = [FEATURE_NAMES[i] for i in order]
rf_sorted   = rf_imp[order]
xgb_sorted  = xgb_imp[order]

x = np.arange(len(FEATURE_NAMES))
width = 0.38
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - width / 2, rf_sorted,  width, label="Random Forest (best)", color="#3b82f6")
b2 = ax.bar(x + width / 2, xgb_sorted, width, label="XGBoost (best)",       color="#f97316")
ax.set_xticks(x)
ax.set_xticklabels(feat_sorted, rotation=30, ha="right")
ax.set_ylabel("Feature importance")
ax.set_title("Feature importance: Random Forest vs XGBoost (best models)")
ax.legend()
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved: feature_importance.png")


# =============================================================================
# 汇总
# =============================================================================
print("\n" + "=" * 70 + "\nSummary\n" + "=" * 70)
summary = pd.DataFrame([
    {"model": "RF default",  "train_acc": rf_default_tr_acc,  "test_acc": rf_default_te_acc},
    {"model": "RF best",     "train_acc": accuracy_score(y_train, best_rf.predict(X_train)),
                              "test_acc": best_rf_acc},
    {"model": "XGB default", "train_acc": xgb_default_tr_acc, "test_acc": xgb_default_te_acc},
    {"model": "XGB best",    "train_acc": xgb_best_tr_acc,    "test_acc": xgb_best_te_acc},
])
summary["train_acc"] = summary["train_acc"].round(4)
summary["test_acc"]  = summary["test_acc"].round(4)
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "summary.csv", index=False)

print("\nAll outputs written to:", OUT_DIR)
