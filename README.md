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
└── ...
```

## 各周实验

| 周次 | 实验内容 | 主要方法 |
|---|---|---|
| Week 1 | 波士顿房价预测 | LinearRegression / Ridge / Lasso / ElasticNet |
| Week 2 | 乳腺癌良恶性分类 | LogisticRegression / KNN / SVM / DecisionTree / RandomForest / GaussianNB |
| Week 3 | 手写数字识别 | SVM (linear / poly / rbf / sigmoid) + GridSearchCV |
| Week 4 | 共享单车骑行量预测 | SVR (RBF) + SelectKBest + GridSearchCV (C / gamma / epsilon) |

---

> 运行环境：Python 3.x，conda 环境 `pytorch`
> 依赖：pandas / numpy / matplotlib / scikit-learn
