import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据，显示前5行
# ============================================================
print("=" * 60)
print("1. 加载数据，显示前5行")
print("=" * 60)

df = pd.read_csv('HousingData.csv')
print(df.head())

# ============================================================
# 2. info() 查看数据信息及缺失值
# ============================================================
print("\n" + "=" * 60)
print("2. info() 数据信息")
print("=" * 60)
df.info()

# ============================================================
# 3. describe() 统计变量
# ============================================================
print("\n" + "=" * 60)
print("3. describe() 统计变量")
print("=" * 60)
print(df.describe())

# ============================================================
# 4. corr() 相关系数
# ============================================================
print("\n" + "=" * 60)
print("4. corr() 相关系数")
print("=" * 60)
print(df.corr())

# ============================================================
# 5. 前8列分布直方图
# ============================================================
print("\n" + "=" * 60)
print("5. 绘制前8列分布直方图")
print("=" * 60)

cols8 = df.columns[:8]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(cols8):
    axes[i].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='white')
    axes[i].set_title(col)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
plt.suptitle('Distribution Histograms (First 8 Columns)', fontsize=14)
plt.tight_layout()
plt.savefig('histograms.png', dpi=150)
plt.close()
print("直方图已保存为 histograms.png")

# ============================================================
# 数据预处理：填充缺失值 + 划分数据集
# ============================================================
df_clean = df.fillna(df.median())

X = df_clean.drop('MEDV', axis=1)
y = df_clean['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ============================================================
# 6 & 7. 建模、评估、参数调优
# ============================================================
print("\n" + "=" * 60)
print("6 & 7. 建模与评估（MSE / MAE / R²）")
print("=" * 60)

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    r2  = r2_score(y_te, pred)
    return {'Model': name, 'MSE': round(mse, 4), 'MAE': round(mae, 4), 'R²': round(r2, 4)}

results = []

# LinearRegression
results.append(evaluate('LinearRegression', LinearRegression(),
                         X_train_s, X_test_s, y_train, y_test))

# Ridge — alpha = 1, 10, 20
for alpha in [1, 10, 20]:
    results.append(evaluate(f'Ridge(alpha={alpha})', Ridge(alpha=alpha),
                             X_train_s, X_test_s, y_train, y_test))

# Lasso — alpha = 1, 10, 20
for alpha in [1, 10, 20]:
    results.append(evaluate(f'Lasso(alpha={alpha})', Lasso(alpha=alpha, max_iter=10000),
                             X_train_s, X_test_s, y_train, y_test))

# ElasticNet — alpha × l1_ratio 组合
for alpha in [1, 10, 20]:
    for l1 in [0.2, 0.8]:
        results.append(evaluate(
            f'ElasticNet(alpha={alpha},l1={l1})',
            ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000),
            X_train_s, X_test_s, y_train, y_test
        ))

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

print("\n实验完成！直方图已保存至 histograms.png")
