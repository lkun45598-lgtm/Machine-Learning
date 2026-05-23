# 实验七：基于改进 LeNet 的 SVHN 街景数字识别

## 数据集

SVHN 街景数字（Street View House Numbers）的 32×32 RGB 子集，共 **8000** 张图片，类别 0–9。

- 文件：`data_svhn_8k.7z`，解压后 `svhn_8k/{class}_{id}.jpg`（标签由文件名首字符解析）
- 类别分布（不均衡，1 最多 / 9 最少）：

| class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 样本数 | 563 | 1527 | 1137 | 928 | 858 | 757 | 607 | 566 | 556 | 501 |

- 划分：`train_test_split` 8:2 (`stratify=y`)，训练集 6400 张 / 测试集 1600 张
- 预处理：
  - 训练集：随机旋转 ±10° + 随机平移 ±10% + 色彩抖动 (亮度/对比度) → Resize 32×32 → 归一化到 `[-1, 1]`
  - 测试集：仅 Resize + 归一化，不做增强

## 模型结构（改进 LeNet）

```
input (3×32×32)
 ├─ C1: Conv 5×5,  3→6   → ReLU → S2: MaxPool 2×2   (6@28×28 → 6@14×14)
 ├─ C3: Conv 5×5,  6→16  → ReLU → S4: MaxPool 2×2   (16@10×10 → 16@5×5)
 ├─ C5: FC 400→120  → ReLU
 ├─ F6: FC 120→84   → ReLU
 └─ out: FC 84→10   → CrossEntropy
```

参数量 62006；损失 `CrossEntropyLoss`，优化器 **Adam**（文档给定的 `lr=1e-3 / 1e-4` 是 Adam 的典型量级，SGD 在 10 个 epoch 内会停在退化解 ~19%，预测全部 → 多数类 `1`）。

## 实验流程

1. 扫描 `svhn_8k/`，按文件名解析标签，分层切分 80/20
2. 训练 / 测试 transform 差异化，每类各取 1 张可视化
3. **基线**：`lr=1e-3, batch=64, epochs=10`，记录 train_loss / train_acc / test_acc 曲线
4. 测试集评估：混淆矩阵 + 分类报告 + 准确率
5. **学习率扫描**：固定 `batch=64`，`lr ∈ {1e-4, 1e-3}`
6. **批量扫描**：固定 `lr=1e-3`，`batch ∈ {32, 64, 128}`

## 主要结果

### 基线（lr=1e-3, bs=64, 10 epoch）

| epoch | train_loss | train_acc | test_acc |
|---|---|---|---|
| 1 | 2.2373 | 0.1873 | 0.1912 |
| 5 | 1.2163 | 0.6142 | 0.7406 |
| 10 | **0.8947** | **0.7178** | **0.8263** |

测试集准确率 **82.63%**，混淆矩阵主对角线占主导；分类报告 `macro-F1 = 0.8097`，`weighted-F1 = 0.8247`。类别 8 最难（F1=0.6346），主要错分到 3 / 6 / 5；类别 7 最好（F1=0.8670）。

> 注：训练时模型在做数据增强（旋转 / 平移 / 色彩抖动），所以 train_acc 显著低于 test_acc 是正常现象——并非过拟合，而是「训练阶段看到的图比测试阶段更难」。

### 学习率扫描（batch=64）

| lr | 第 10 epoch train_acc | 第 10 epoch test_acc | 备注 |
|---|---|---|---|
| 1e-4 | 0.3528 | 0.3988 | 学习率过小，10 个 epoch 还在缓慢爬升、未收敛 |
| **1e-3** | **0.7084** | **0.8188** | 在 10 个 epoch 内已经基本收敛 |

`lr=1e-4` 前 4 个 epoch 仍卡在 19% 的多数类退化解，第 5 epoch 后才开始真正学到特征——这种 lr 在 10 epoch 的预算下严重欠拟合。`lr=1e-3` 在第 2 epoch 就跳出退化解，最终测试准确率比 `1e-4` 高 **约 42 个百分点**。

### 批量扫描（lr=1e-3）

| batch | 第 10 epoch train_acc | 第 10 epoch test_acc |
|---|---|---|
| 32  | **0.7569** | **0.8319** |
| 64  | 0.7088 | 0.8169 |
| 128 | 0.6338 | 0.7462 |

固定 epoch 数时，**batch 越小，每个 epoch 的参数更新次数越多**（bs=32 是 bs=128 的 4 倍），所以小 batch 在前几个 epoch 收敛更快、最终准确率更高。bs=128 在 10 个 epoch 里完成的优化步数等价于 bs=32 跑 2.5 epoch，自然落后；如果允许更多 epoch，三者会逐渐接近。

## 输出文件（`outputs/`）

| 文件 | 内容 |
|---|---|
| `samples_per_class.png` | 训练集每类 1 张样本 |
| `step3_baseline_curves.png` | 基线 train_loss / train_acc / test_acc 随 epoch 曲线 |
| `step3_baseline_curve.csv` | 基线学习曲线数据 |
| `step4_baseline_test_confusion.png` | 基线测试集混淆矩阵热力图 |
| `step4_baseline_test_confusion.csv` | 基线测试集混淆矩阵数据 |
| `step5_lr_sweep.png` | lr ∈ {1e-4, 1e-3} 训练准确率折线图 |
| `step5_lr_sweep.csv` | lr 扫描每个 epoch 的 train_acc / test_acc |
| `step6_bs_sweep.png` | batch ∈ {32, 64, 128} 训练准确率折线图 |
| `step6_bs_sweep.csv` | batch 扫描每个 epoch 的 train_acc / test_acc |
| `summary.csv` | 全部配置的 final_train_acc / final_test_acc |
| `run_log.txt` | 完整运行日志（含模型结构、混淆矩阵、分类报告） |

## 运行方式

```bash
cd week7
# 数据需先解压（仅首次）：
#   conda run -n pytorch python -c "import py7zr; py7zr.SevenZipFile('data_svhn_8k.7z','r').extractall('./')"
conda run -n pytorch python experiment7.py
```

> 依赖：`torch / torchvision / pandas / numpy / matplotlib / scikit-learn / Pillow / py7zr`
> GPU：在 RTX 4090 上单次基线训练 ~25s，全部 6 次训练（基线 + 2 个 lr + 3 个 bs）共 ~2 分钟。
