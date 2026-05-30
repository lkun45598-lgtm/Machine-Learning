# Machine Learning Experiments

华南农业大学 机器学习课程实验代码

---

## 目录结构

```
Machine-Learning/
├── week1/          # 实验一：线性回归实现波士顿房价预测
├── week2/          # 实验二：分类算法实现乳腺癌良恶性预测
├── week3/          # 实验三：基于支持向量机的手写数字识别
├── week4/          # 实验四：基于支持向量机回归的共享单车骑行量预测
├── week5/          # 实验五：基于决策树的鸢尾花分类
├── week6/          # 实验六：随机森林、XGBoost 预测糖尿病
├── week7/          # 实验七：基于改进 LeNet 的 SVHN 街景数字识别
├── week8/          # 实验八：朴素贝叶斯实现垃圾短信分类
└── ...
```

## 各周实验

| 周次 | 实验内容 | 主要方法 |
|---|---|---|
| Week 1 | 波士顿房价预测 | LinearRegression / Ridge / Lasso / ElasticNet |
| Week 2 | 乳腺癌良恶性分类 | LogisticRegression / KNN / SVM / DecisionTree / RandomForest / GaussianNB |
| Week 3 | 手写数字识别 | SVM (linear / poly / rbf / sigmoid) + GridSearchCV |
| Week 4 | 共享单车骑行量预测 | SVR (RBF) + SelectKBest + GridSearchCV (C / gamma / epsilon) |
| Week 5 | 鸢尾花分类 | DecisionTreeClassifier (entropy / gini) + 缺失值处理（丢弃 / 均值 / KNN / 标记） |
| Week 6 | 糖尿病预测 | RandomForest + XGBoost + GridSearchCV (9 维超参) + 特征重要性对比 |
| Week 7 | SVHN 街景数字识别 | 改进 LeNet (PyTorch) + Adam + 数据增强 + lr / batch 超参扫描 |
| Week 8 | 垃圾短信分类 | CountVectorizer + MultinomialNB + WordCloud 词云 / top-20 词频可视化 |

---

> 运行环境：Python 3.x，conda 环境 `pytorch`
> 依赖：pandas / numpy / matplotlib / scikit-learn / xgboost / torch / torchvision / Pillow / py7zr
