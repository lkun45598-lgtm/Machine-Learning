# Week 4：基于支持向量机回归（SVR）的共享单车骑行量预测

## 实验目的
1. 掌握支持向量机回归算法
2. 熟悉 sklearn 中 SVR 相关 API
3. 进一步理解核函数与核技巧
4. 掌握 SVR 调参（C / gamma / epsilon）
5. 巩固独热编码

## 数据集
- 文件：`FE_day_数据说明.xlsx`，工作表 `FE_day`
- 样本数：731；原始列数：35（已对 season、mnth、weathersit、weekday 做独热编码）
- 标签：`cnt`（每日骑行总量）
- 去除 `instant`（记录号）与 `cnt` 后可用特征 33 个

## 实验流程
1. **加载数据**：读取 731 条样本，输出维度与前 5 行
2. **特征筛选**：用 `SelectKBest(f_regression, k=20)` 选出与 `cnt` 关联最强的 20 个特征
3. **拆分 + 标准化**：`test_size=0.2, random_state=42`；`StandardScaler` 仅在训练集上 `fit`，测试集只 `transform`；标签也做标准化以利于 SVR 收敛
4. **RBF 核网格搜索**：5 折 CV，搜索空间
   - `C ∈ {0.1, 1, 10, 50, 100}`
   - `gamma ∈ {0.001, 0.01, 0.05, 0.1, 0.5}`
   - `epsilon ∈ {0.01, 0.1, 0.2}`
5. **评估**：在训练集与测试集上计算 MSE / RMSE / R²
6. **可视化**：在最优 epsilon 下绘制 gamma × C 的 5 折 CV R² 热力图，并画测试集预测 vs 真实曲线

## 关键结果
- 最优参数：`C=50, gamma=0.01, epsilon=0.2`
- 交叉验证最优 R²：**0.8599**

| 数据集 | MSE | RMSE | R² |
|--------|-----|------|----|
| 训练集 | 326095.37 | 571.05 | 0.9110 |
| 测试集 | 514051.26 | 716.97 | 0.8718 |

## 输出文件
- `outputs/metrics.csv`：训练 / 测试评估指标
- `outputs/gamma_C_heatmap.png`：gamma × C 组合的 CV R² 热力图
- `outputs/test_prediction.png`：测试集预测值与真实值对比

## 运行方式
```bash
conda run -n pytorch python bike_svr.py
```

## 结果分析（要点）
- **特征选择**：`temp`、`atemp`、`yr`、`season_1`、`mnth_1` 与 `cnt` 的 F 值最高，符合常识——温度与年份对骑行量影响最显著
- **gamma**：过小（0.001）模型欠拟合，过大（0.5）严重过拟合；最优值在 0.01 附近
- **C**：当 gamma 较小（0.01）时，C 越大越好（C=50 最佳）；当 gamma 较大时，C 增大反而过拟合
- **epsilon**：本实验对 epsilon 不敏感，0.2 略优——允许更宽的不敏感带可减少支持向量数量、提升泛化
- **训练 / 测试差距**（R² 0.911 → 0.872）说明仍存在轻微过拟合，可考虑进一步增大 epsilon 或减小 C
