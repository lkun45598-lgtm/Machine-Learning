# 实验八：朴素贝叶斯实现垃圾短信分类

## 数据集

SMSSpamCollection —— 英文短信垃圾分类数据集，共 **5574** 条，每条为 `label \t text`。

- 文件：`SMSSpamCollection.xlsx`（原始 tab 分隔文本转存为 xlsx，含逗号的消息被拆到多列，代码中按 `","` 拼回整行再按制表符切分）
- 类别分布（不均衡）：

| label | ham（正常） | spam（垃圾） |
|---|---|---|
| 样本数 | 4827 | 747 |
| 占比 | 86.6% | 13.4% |

## 实验流程（对应实验文档 5 个步骤）

1. **加载数据**：还原 `label / text`，展示前 5 条，统计类别分布
2. **词云图**：`WordCloud` 绘制整体词云，并对比 ham / spam 两类词云
3. **文本特征提取**：`CountVectorizer` 分词构建词汇表、生成词频矩阵，输出第 100 个样本的词向量、向量长度及对应原文
4. **词频统计**：统计词汇表每个单词出现总次数，降序排序，top-20 柱状图
5. **建模**：`train_test_split`（test=20%, `stratify`）→ `MultinomialNB` → 训练 / 测试正确率 + 混淆矩阵

> 防数据泄漏：建模阶段的 `CountVectorizer` **仅在训练集上 `fit`**，测试集只 `transform`（第 2–4 步的全量向量化只用于可视化探索）。

## 运行

```bash
conda run -n pytorch python experiment8.py
```

依赖：`pandas / numpy / matplotlib / scikit-learn / openpyxl / wordcloud`

## 主要结果

| 指标 | 数值 |
|---|---|
| 样本总数 | 5574 |
| 全量词汇表大小 | 8920 |
| 训练集词汇表大小 | 7716 |
| 训练集正确率 | **0.9942** |
| 测试集正确率 | **0.9848** |

第 100 个样本（`ham`）原文：`Please don't text me anymore. I have nothing else to say.`
其词向量长度 = 全量词汇表大小 8920，非零词 10 个（`anymore, don, else, have, me, nothing, please, say, text, to`，各出现 1 次）。

测试集混淆矩阵（0=ham, 1=spam）：

| | pred ham | pred spam |
|---|---|---|
| **true ham** | 963 | 3 |
| **true spam** | 14 | 135 |

- spam 精确率 0.978 / 召回率 0.906；ham 精确率 0.986 / 召回率 0.997
- 仅 3 条正常短信被误判为垃圾（误杀少），14 条垃圾短信漏判（漏报略多，受类别不均衡影响）

## 结果分析

- **MultinomialNB 非常适合该任务**：词频（计数）特征与多项式分布假设天然契合，几行代码即可达到 98.5% 测试正确率。
- **词云已能直观区分两类**：spam 高频词为 `FREE / call / txt / claim / prize / mobile / Nokia / STOP`（营销诱导 + 短码回复），ham 则是 `now / ok / will / u / call / love` 等日常口语词。
- **top-20 高频词以英语停用词为主**（`to / you / the / and ...`），但 `call / now` 等带营销倾向的词已进前列，说明垃圾短信对整体词频有可观影响。
- **误差主要来自类别不均衡**：spam 仅占 13%，模型对少数类召回（0.906）低于多数类。若需进一步提升，可尝试 TF-IDF 特征、去停用词、调整 `MultinomialNB` 的 `class_prior` 或对 spam 过采样。

## 实验总结

本实验完整走通了「文本 → 特征 → 朴素贝叶斯分类」的流水线：用 `CountVectorizer` 把短信转为词频向量，用 `MultinomialNB` 完成二分类。掌握了三种朴素贝叶斯（GaussianNB 连续特征 / MultinomialNB 计数特征 / BernoulliNB 二值特征）的适用场景——文本词频场景应选 **MultinomialNB**。同时实践了词云、词频柱状图等文本可视化手段，并通过「先切分后仅在训练集 fit 向量器」避免了测试集词汇泄漏。

## 输出文件（`outputs/`）

| 文件 | 说明 |
|---|---|
| `wordcloud_all.png` | 整体词云 |
| `wordcloud_ham_spam.png` | ham / spam 对比词云 |
| `top20_words.png` / `top20_words.csv` | top-20 高频词柱状图与表 |
| `confusion_matrix.png` | 测试集混淆矩阵 |
| `summary.csv` | 关键指标汇总 |
| `run_log.txt` | 完整运行日志 |
