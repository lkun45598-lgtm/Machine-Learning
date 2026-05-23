# -*- coding: utf-8 -*-
"""
实验七：基于改进 LeNet 的 SVHN 街景数字识别（按流水线写法）

执行：conda run -n pytorch python experiment7.py
"""

from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "svhn_8k"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CLASSES = 10
CLASS_NAMES = [str(i) for i in range(N_CLASSES)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练默认超参（步骤 3：基线）
BASE_LR = 1e-3
BASE_BS = 64
BASE_EPOCHS = 10


# =============================================================================
# 数据集类：从文件名 "{class}_{id}.jpg" 解析标签
# =============================================================================
class SVHNFromFiles(Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img, int(self.labels[idx])


# =============================================================================
# 改进 LeNet（按实验文档结构）
# 32×32 → C1: 6@28×28 → S2: 6@14×14 → C3: 16@10×10 → S4: 16@5×5
#       → C5(FC): 120 → F6(FC): 84 → out: 10
# 激活：ReLU，下采样：MaxPool2d，损失：CrossEntropy（这里只算 logits）
# =============================================================================
class ImprovedLeNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)    # 32 -> 28
        self.pool1 = nn.MaxPool2d(2, 2)                # 28 -> 14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 14 -> 10
        self.pool2 = nn.MaxPool2d(2, 2)                # 10 -> 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)          # C5
        self.fc2 = nn.Linear(120, 84)                  # F6
        self.fc3 = nn.Linear(84, n_classes)            # output

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# 公共训练循环：返回每个 epoch 的 train_loss / train_acc / test_acc
# =============================================================================
def train_model(model, train_loader, test_loader, lr, epochs, tag=""):
    model = model.to(DEVICE)
    # 文档给的 lr=1e-3 / 1e-4 是 Adam 的典型量级，SGD 在 10 个 epoch 内学不动
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        # train
        model.train()
        n, loss_sum, correct = 0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            bs = yb.size(0)
            n += bs
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == yb).sum().item()
        tr_loss = loss_sum / n
        tr_acc = correct / n

        # eval on test
        te_acc = evaluate(model, test_loader)[0]

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        print(
            f"  [{tag}] epoch {ep:02d}/{epochs} "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}"
        )
    return hist


def evaluate(model, loader):
    """返回 (acc, y_true, y_pred)。"""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return accuracy_score(y_true, y_pred), y_true, y_pred


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Step 1: 扫描数据集，按文件名解析标签，分层切分 80/20
# =============================================================================
print("\n" + "=" * 70 + "\nStep 1: scan svhn_8k, parse labels from filename, split 80/20\n" + "=" * 70)
print("device:", DEVICE)

files = sorted(DATA_DIR.glob("*.jpg"))
pat = re.compile(r"^(\d)_\d+\.jpg$")
labels = []
keep = []
for p in files:
    m = pat.match(p.name)
    if m:
        labels.append(int(m.group(1)))
        keep.append(p)
files = keep
labels = np.asarray(labels)
print(f"共扫描到 {len(files)} 张图片")
print("类别分布:")
for c in range(N_CLASSES):
    print(f"  class {c}: {(labels == c).sum()}")

train_files, test_files, y_train, y_test = train_test_split(
    files, labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels,
)
print(f"\n训练集: {len(train_files)} 张，测试集: {len(test_files)} 张")


# =============================================================================
# Step 2: 数据预处理与增强
#   训练集：随机旋转(±10°) + 随机平移(±10%) + 色彩抖动 → 32×32 → [-1, 1]
#   测试集：仅 32×32 + 归一化
# =============================================================================
print("\n" + "=" * 70 + "\nStep 2: transforms (train aug + test plain) and DataLoader\n" + "=" * 70)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)  # 归一化到 [-1, 1]

train_tf = T.Compose([
    T.Resize((32, 32)),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])
test_tf = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(NORM_MEAN, NORM_STD),
])

train_ds = SVHNFromFiles(train_files, y_train, train_tf)
test_ds = SVHNFromFiles(test_files, y_test, test_tf)
print(f"train_ds: {len(train_ds)}, test_ds: {len(test_ds)}")

# 样本可视化：每类各取一张
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for c in range(N_CLASSES):
    idx = np.where(y_train == c)[0][0]
    img = Image.open(train_files[idx]).convert("RGB")
    ax = axes[c // 5, c % 5]
    ax.imshow(img)
    ax.set_title(f"class {c}")
    ax.axis("off")
fig.suptitle("SVHN-8k samples (one per class, train split)")
fig.tight_layout()
fig.savefig(OUT_DIR / "samples_per_class.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved: samples_per_class.png")


# =============================================================================
# Step 3: 基线模型——lr=1e-3, batch=64, epochs=10
# =============================================================================
print("\n" + "=" * 70 + "\nStep 3: baseline ImprovedLeNet (lr=1e-3, batch=64, epochs=10)\n" + "=" * 70)
set_seed(RANDOM_STATE)

base_train_loader = DataLoader(train_ds, batch_size=BASE_BS, shuffle=True,
                               num_workers=2, pin_memory=True)
base_test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

base_model = ImprovedLeNet(n_classes=N_CLASSES)
print(base_model)
n_params = sum(p.numel() for p in base_model.parameters())
print(f"模型参数量: {n_params}")

t0 = time.time()
base_hist = train_model(
    base_model, base_train_loader, base_test_loader,
    lr=BASE_LR, epochs=BASE_EPOCHS, tag="baseline",
)
print(f"baseline 训练耗时 {time.time() - t0:.1f}s")

# 保存学习曲线 CSV
base_curve_df = pd.DataFrame({
    "epoch": np.arange(1, BASE_EPOCHS + 1),
    "train_loss": base_hist["train_loss"],
    "train_acc": base_hist["train_acc"],
    "test_acc": base_hist["test_acc"],
})
base_curve_df.to_csv(OUT_DIR / "step3_baseline_curve.csv", index=False)
print(base_curve_df.to_string(index=False))

# 训练误差 / 准确率随 epoch 的曲线（双图：loss 一张、acc 一张）
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(base_curve_df["epoch"], base_curve_df["train_loss"], "o-", color="#ef4444")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("train loss")
axes[0].set_title("Baseline: training loss vs epoch")
axes[0].grid(alpha=0.3)

axes[1].plot(base_curve_df["epoch"], base_curve_df["train_acc"], "o-",
             color="#3b82f6", label="train acc")
axes[1].plot(base_curve_df["epoch"], base_curve_df["test_acc"], "s--",
             color="#10b981", label="test acc")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")
axes[1].set_title("Baseline: accuracy vs epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "step3_baseline_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved: step3_baseline_curves.png")


# =============================================================================
# Step 4: 在测试集上评估基线模型——混淆矩阵 + 分类报告 + 准确率
# =============================================================================
print("\n" + "=" * 70 + "\nStep 4: evaluate baseline on test set (CM + report + acc)\n" + "=" * 70)

te_acc, y_true, y_pred = evaluate(base_model, base_test_loader)
print(f"[baseline] test accuracy = {te_acc:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in CLASS_NAMES],
    columns=[f"pred_{c}" for c in CLASS_NAMES],
)
print("\n[baseline] test confusion matrix:")
print(cm_df.to_string())
cm_df.to_csv(OUT_DIR / "step4_baseline_test_confusion.csv")

print("\n[baseline] test classification report:")
print(classification_report(y_true, y_pred,
                            labels=list(range(N_CLASSES)),
                            target_names=CLASS_NAMES, digits=4))

# 混淆矩阵热力图
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"Baseline test confusion matrix (acc={te_acc:.4f})")
vmax = cm.max()
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        v = cm[i, j]
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > vmax * 0.55 else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT_DIR / "step4_baseline_test_confusion.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved: step4_baseline_test_confusion.png")


# =============================================================================
# Step 5: 固定 batch=64，比较 lr ∈ {1e-4, 1e-3} 对训练准确率的影响
# =============================================================================
print("\n" + "=" * 70 + "\nStep 5: LR sweep — lr ∈ {1e-4, 1e-3}, batch=64\n" + "=" * 70)

lr_train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                             num_workers=2, pin_memory=True)
lr_test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True)

LR_LIST = [1e-4, 1e-3]
lr_hists = {}
for lr in LR_LIST:
    print(f"\n--- 训练 lr={lr} ---")
    set_seed(RANDOM_STATE)
    m = ImprovedLeNet(n_classes=N_CLASSES)
    h = train_model(m, lr_train_loader, lr_test_loader,
                    lr=lr, epochs=BASE_EPOCHS, tag=f"lr={lr}")
    lr_hists[lr] = h

lr_df = pd.DataFrame({
    "epoch": np.arange(1, BASE_EPOCHS + 1),
    **{f"train_acc_lr{lr}": lr_hists[lr]["train_acc"] for lr in LR_LIST},
    **{f"test_acc_lr{lr}":  lr_hists[lr]["test_acc"]  for lr in LR_LIST},
})
lr_df.to_csv(OUT_DIR / "step5_lr_sweep.csv", index=False)
print("\n[lr_sweep] 每个 epoch 的训练 / 测试准确率:")
print(lr_df.round(4).to_string(index=False))

# 训练准确率折线图
fig, ax = plt.subplots(figsize=(8, 5))
colors = {1e-4: "#f59e0b", 1e-3: "#3b82f6"}
for lr in LR_LIST:
    ax.plot(np.arange(1, BASE_EPOCHS + 1), lr_hists[lr]["train_acc"], "o-",
            color=colors[lr], label=f"train acc, lr={lr}")
    ax.plot(np.arange(1, BASE_EPOCHS + 1), lr_hists[lr]["test_acc"], "s--",
            color=colors[lr], alpha=0.55, label=f"test  acc, lr={lr}")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_title("LR sweep: training accuracy vs epoch (batch=64)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "step5_lr_sweep.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved: step5_lr_sweep.png")


# =============================================================================
# Step 6: 固定 lr=1e-3，比较 batch ∈ {32, 64, 128} 对训练准确率的影响
# =============================================================================
print("\n" + "=" * 70 + "\nStep 6: batch sweep — batch ∈ {32, 64, 128}, lr=1e-3\n" + "=" * 70)

BS_LIST = [32, 64, 128]
bs_hists = {}
for bs in BS_LIST:
    print(f"\n--- 训练 batch={bs} ---")
    set_seed(RANDOM_STATE)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    m = ImprovedLeNet(n_classes=N_CLASSES)
    h = train_model(m, train_loader, test_loader,
                    lr=BASE_LR, epochs=BASE_EPOCHS, tag=f"bs={bs}")
    bs_hists[bs] = h

bs_df = pd.DataFrame({
    "epoch": np.arange(1, BASE_EPOCHS + 1),
    **{f"train_acc_bs{bs}": bs_hists[bs]["train_acc"] for bs in BS_LIST},
    **{f"test_acc_bs{bs}":  bs_hists[bs]["test_acc"]  for bs in BS_LIST},
})
bs_df.to_csv(OUT_DIR / "step6_bs_sweep.csv", index=False)
print("\n[bs_sweep] 每个 epoch 的训练 / 测试准确率:")
print(bs_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
bs_colors = {32: "#10b981", 64: "#3b82f6", 128: "#ef4444"}
for bs in BS_LIST:
    ax.plot(np.arange(1, BASE_EPOCHS + 1), bs_hists[bs]["train_acc"], "o-",
            color=bs_colors[bs], label=f"train acc, bs={bs}")
    ax.plot(np.arange(1, BASE_EPOCHS + 1), bs_hists[bs]["test_acc"], "s--",
            color=bs_colors[bs], alpha=0.55, label=f"test  acc, bs={bs}")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_title("Batch sweep: training accuracy vs epoch (lr=1e-3)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "step6_bs_sweep.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("saved: step6_bs_sweep.png")


# =============================================================================
# 汇总
# =============================================================================
print("\n" + "=" * 70 + "\nSummary\n" + "=" * 70)

summary_rows = [{
    "config": "baseline (lr=1e-3, bs=64)",
    "final_train_acc": base_hist["train_acc"][-1],
    "final_test_acc":  base_hist["test_acc"][-1],
}]
for lr in LR_LIST:
    summary_rows.append({
        "config": f"lr={lr},  bs=64",
        "final_train_acc": lr_hists[lr]["train_acc"][-1],
        "final_test_acc":  lr_hists[lr]["test_acc"][-1],
    })
for bs in BS_LIST:
    summary_rows.append({
        "config": f"lr=1e-3, bs={bs}",
        "final_train_acc": bs_hists[bs]["train_acc"][-1],
        "final_test_acc":  bs_hists[bs]["test_acc"][-1],
    })
summary = pd.DataFrame(summary_rows)
summary["final_train_acc"] = summary["final_train_acc"].round(4)
summary["final_test_acc"]  = summary["final_test_acc"].round(4)
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "summary.csv", index=False)

print("\nAll outputs written to:", OUT_DIR)
