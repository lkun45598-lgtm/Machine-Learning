# -*- coding: utf-8 -*-
"""
实验8：朴素贝叶斯实现垃圾邮件（短信）分类
数据集：SMSSpamCollection（5574 条英文短信，标签 ham / spam）

流水线式直写代码，按实验文档 5 个步骤顺序展开：
  Step 1  加载数据，展示前 5 条
  Step 2  WordCloud 绘制词云图
  Step 3  CountVectorizer 文本特征提取，输出第 100 个样本的词向量与长度
  Step 4  统计词频，top-20 单词柱状图
  Step 5  划分训练 / 测试集 (test=20%)，MultinomialNB 建模，输出正确率与混淆矩阵

运行：conda run -n pytorch python experiment8.py
约定：matplotlib 图一律用英文标签，避免 Linux 缺中文字体乱码。
"""
import os
import openpyxl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# ----------------------------------------------------------------------------
# Step 1  加载数据，展示前 5 条
# ----------------------------------------------------------------------------
# xlsx 由原始 tab 分隔文本转存而来，含逗号的消息被拆到了多列；
# 还原方式：每行所有非空单元格用 "," 拼回整行，再按第一个制表符切分 label / text。
log("=" * 70)
log("Step 1  加载数据，展示前 5 条")
log("=" * 70)

wb = openpyxl.load_workbook(os.path.join(HERE, "SMSSpamCollection.xlsx"), read_only=True)
ws = wb.active
labels, texts = [], []
for row in ws.iter_rows(values_only=True):
    cells = [str(c) for c in row if c is not None]
    if not cells:
        continue
    line = ",".join(cells)          # 把被逗号拆开的消息拼回去
    label, _, text = line.partition("\t")
    labels.append(label.strip())
    texts.append(text.strip())
wb.close()

df = pd.DataFrame({"label": labels, "text": texts})
df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)

log(f"样本总数：{len(df)}")
log(f"类别分布：\n{df['label'].value_counts().to_string()}")
spam_ratio = (df['label'] == 'spam').mean()
log(f"spam 占比：{spam_ratio:.4f}")
log("\n前 5 条数据：")
for i in range(5):
    log(f"  [{df.loc[i, 'label']:>4}] {df.loc[i, 'text'][:80]}")

# 标签数值化：ham=0, spam=1
y = (df["label"] == "spam").astype(int).values

# ----------------------------------------------------------------------------
# Step 2  WordCloud 绘制词云图
# ----------------------------------------------------------------------------
log("\n" + "=" * 70)
log("Step 2  WordCloud 绘制词云图")
log("=" * 70)

stop = set(STOPWORDS)
all_text = " ".join(df["text"].tolist())
ham_text = " ".join(df.loc[df["label"] == "ham", "text"].tolist())
spam_text = " ".join(df.loc[df["label"] == "spam", "text"].tolist())

def make_wc(text):
    return WordCloud(width=800, height=400, background_color="white",
                     stopwords=stop, collocations=False,
                     random_state=RANDOM_STATE).generate(text)

# 整体词云
plt.figure(figsize=(10, 5))
plt.imshow(make_wc(all_text), interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - All SMS")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "wordcloud_all.png"), dpi=120)
plt.close()

# ham vs spam 对比词云
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].imshow(make_wc(ham_text), interpolation="bilinear")
axes[0].set_title("Word Cloud - HAM (normal)")
axes[0].axis("off")
axes[1].imshow(make_wc(spam_text), interpolation="bilinear")
axes[1].set_title("Word Cloud - SPAM")
axes[1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "wordcloud_ham_spam.png"), dpi=120)
plt.close()
log("已保存：wordcloud_all.png, wordcloud_ham_spam.png")

# ----------------------------------------------------------------------------
# Step 3  CountVectorizer 文本特征提取
# ----------------------------------------------------------------------------
log("\n" + "=" * 70)
log("Step 3  CountVectorizer 文本特征提取（构建词汇表 + 词频矩阵）")
log("=" * 70)

# 探索用向量器：在全量语料上 fit，仅用于展示词汇表 / 第 100 个样本词向量 / 词频统计
vec_explore = CountVectorizer()
X_all = vec_explore.fit_transform(df["text"])     # 稀疏词频矩阵
vocab = vec_explore.get_feature_names_out()
log(f"词频矩阵形状（样本数 × 词汇表大小）：{X_all.shape}")
log(f"词汇表长度：{len(vocab)}")

idx = 100  # 第 100 个样本（0-based 索引 100，即第 101 条；按下标 100 取）
vec_100 = X_all[idx].toarray().ravel()
log(f"\n第 {idx} 个样本的词向量长度：{len(vec_100)}")
log(f"第 {idx} 个样本词向量非零元素个数：{int((vec_100 > 0).sum())}")
nz = np.nonzero(vec_100)[0]
log("第 {} 个样本非零词（word: count）：".format(idx))
log("  " + ", ".join(f"{vocab[j]}:{vec_100[j]}" for j in nz))
log(f"\n第 {idx} 个样本原始文本：")
log(f"  [{df.loc[idx, 'label']}] {df.loc[idx, 'text']}")

# ----------------------------------------------------------------------------
# Step 4  词频统计 + top-20 柱状图
# ----------------------------------------------------------------------------
log("\n" + "=" * 70)
log("Step 4  词汇表词频统计，top-20 柱状图")
log("=" * 70)

word_counts = np.asarray(X_all.sum(axis=0)).ravel()
order = np.argsort(word_counts)[::-1]
top20_idx = order[:20]
top20_words = vocab[top20_idx]
top20_counts = word_counts[top20_idx]

log("Top-20 高频词：")
for w, c in zip(top20_words, top20_counts):
    log(f"  {w:<10} {int(c)}")

plt.figure(figsize=(12, 6))
plt.bar(range(20), top20_counts, color="#4C72B0")
plt.xticks(range(20), top20_words, rotation=45, ha="right")
plt.ylabel("Frequency")
plt.xlabel("Word")
plt.title("Top-20 Most Frequent Words")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "top20_words.png"), dpi=120)
plt.close()
log("已保存：top20_words.png")

# 同时保存 top-20 词频表
pd.DataFrame({"word": top20_words, "count": top20_counts.astype(int)}).to_csv(
    os.path.join(OUT, "top20_words.csv"), index=False)

# ----------------------------------------------------------------------------
# Step 5  划分训练 / 测试集，MultinomialNB 建模
# ----------------------------------------------------------------------------
log("\n" + "=" * 70)
log("Step 5  划分训练/测试集 (test=20%)，MultinomialNB 建模")
log("=" * 70)

# 先切分原始文本，再仅在训练集上 fit 向量器 —— 避免测试集词汇泄漏（遵循项目规范）
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    df["text"], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

vec = CountVectorizer()
X_train = vec.fit_transform(X_train_txt)   # 仅用训练集构建词汇表
X_test = vec.transform(X_test_txt)         # 测试集只 transform
log(f"训练集：{X_train.shape[0]} 条，测试集：{X_test.shape[0]} 条")
log(f"建模词汇表大小（仅训练集）：{len(vec.get_feature_names_out())}")

clf = MultinomialNB()
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
log(f"\n训练集正确率：{train_acc:.4f}")
log(f"测试集正确率：{test_acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
log("\n测试集混淆矩阵（行=真实，列=预测；0=ham, 1=spam）：")
log(f"          pred_ham  pred_spam")
log(f"true_ham   {cm[0,0]:>7d}  {cm[0,1]:>9d}")
log(f"true_spam  {cm[1,0]:>7d}  {cm[1,1]:>9d}")
log("\n分类报告：")
log(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))

# 混淆矩阵可视化
plt.figure(figsize=(5, 4.5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["ham", "spam"])
plt.yticks([0, 1], ["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"MultinomialNB Confusion Matrix (test acc={test_acc:.4f})")
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120)
plt.close()
log("已保存：confusion_matrix.png")

# ----------------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------------
pd.DataFrame([{
    "n_samples": len(df),
    "spam_ratio": round(spam_ratio, 4),
    "vocab_size_full": len(vocab),
    "vocab_size_train": len(vec.get_feature_names_out()),
    "train_acc": round(train_acc, 4),
    "test_acc": round(test_acc, 4),
}]).to_csv(os.path.join(OUT, "summary.csv"), index=False)

with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log("\n全部完成，输出已写入 outputs/")
