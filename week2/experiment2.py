"""
实验二：威斯康星乳腺癌数据集分类实验

实验流程：
1. 加载并校验数据集
2. 清洗 Bare Nuclei 列中的 '?' 异常值
3. 使用无数据泄漏的 Pipeline：SimpleImputer -> StandardScaler -> 分类器
4. 仅在训练集上做交叉验证，选出最优模型
5. 在 hold-out 测试集上评估一次最优模型
6. 保存图表和结果表格

运行方式（在 week2 目录或项目根目录均可）：
    python experiment2.py

数据文件名（放在脚本同目录下任意一个即可）：
    breast_cancer_wisconsin.csv
或原始文件名：
    （加表头）breast-cancer-wisconsin.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # 兼容较旧版本的 scikit-learn
    StratifiedGroupKFold = None


# ============================================================
# 全局配置
# ============================================================

RANDOM_STATE = 42       # 随机种子，保证结果可复现
TEST_SIZE = 0.20        # 测试集比例
N_JOBS = 1              # 并行数，设为 -1 可使用全部 CPU

# 为 True 时优先使用按患者 ID 分组的切分，防止同一患者同时出现在训练集和测试集
# 若课程要求普通随机切分，改为 False 即可
USE_GROUP_SPLIT_WHEN_POSSIBLE = True

# 脚本所在目录（无论从哪里运行都能正确找到数据文件）
BASE_DIR = Path(__file__).resolve().parent
# 依次尝试两个可能的数据文件名，找到第一个存在的就用
DATA_CANDIDATES = [
    BASE_DIR / "breast_cancer_wisconsin.csv",
    BASE_DIR / "（加表头）breast-cancer-wisconsin.csv",
]
# 所有输出图表和结果表格统一存放到 outputs/ 子目录
OUTPUT_DIR = BASE_DIR / "outputs"

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# 数据加载与校验
# ============================================================


def find_data_file(candidates: Iterable[Path]) -> Path:
    """从候选路径列表中返回第一个存在的文件，都不存在则报错。"""
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "找不到数据文件，请将 CSV 重命名为 breast_cancer_wisconsin.csv "
        f"并放在脚本同目录下。已搜索路径：\n{searched}"
    )


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """读取 CSV，清洗数值列，返回 df、X、y、groups、feature_cols。"""
    data_file = find_data_file(DATA_CANDIDATES)
    print(f"数据文件：{data_file}")

    df = pd.read_csv(data_file)

    # 校验必需列是否都存在
    required_cols = {
        "Sample code number",
        "Clump Thickness",
        "Uniformity of Cell Size",
        "Uniformity of Cell Shape",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "Bare Nuclei",
        "Bland Chromatin",
        "Normal Nucleoli",
        "Mitoses",
        "Class",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"数据集缺少必需列：{missing_cols}")

    # 排除编号列和标签列，其余均为特征列
    feature_cols = [c for c in df.columns if c not in ["Sample code number", "Class"]]

    # 原始 UCI 文件中 Bare Nuclei 含 '?'，用 errors="coerce" 将其转为 NaN
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 将标签从 {2, 4} 映射为 {0, 1}：2（良性）→ 0，4（恶性）→ 1
    y = df["Class"].map({2: 0, 4: 1})
    if y.isna().any():
        bad_labels = sorted(df.loc[y.isna(), "Class"].unique())
        raise ValueError(f"发现意外的类别标签：{bad_labels}，仅支持 2 和 4。")
    y = y.astype(int)

    X = df[feature_cols].copy()
    # 保存患者编号，用于后续分组切分（防止同一患者跨训练/测试集）
    groups = df["Sample code number"].copy()

    return df, X, y, groups, feature_cols


def print_data_report(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
    """打印基础探索性数据分析（EDA）和数据质量检查结果。"""
    print("\n" + "=" * 80)
    print("1. 数据集前5行")
    print("=" * 80)
    print(df.head())

    print("\n" + "=" * 80)
    print("2. 数据集基本信息与缺失值")
    print("=" * 80)
    df.info()
    print("\n各列缺失值数量：")
    print(df.isna().sum())

    print("\n" + "=" * 80)
    print("3. 描述性统计")
    print("=" * 80)
    print(df.describe())

    print("\n" + "=" * 80)
    print("4. 类别分布")
    print("=" * 80)
    class_counts = df["Class"].value_counts().sort_index()
    print(class_counts.rename(index={2: "良性 / 2", 4: "恶性 / 4"}))

    # keep=False：将所有重复项都标记为 True（不只是第二次出现的）
    duplicate_id_count = int(groups.duplicated(keep=False).sum())
    full_duplicate_count = int(df.duplicated().sum())
    print("\n数据质量说明：")
    print(f"- 总行数：{len(df)}")
    print(f"- 特征数：{X.shape[1]}")
    print(f"- 共享同一患者编号的行数：{duplicate_id_count}")
    print(f"- 完全重复的行数：{full_duplicate_count}")


# ============================================================
# 数据切分与建模
# ============================================================


def make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, str]:
    """创建 hold-out 切分，优先使用按患者分组的切分方式。"""
    # 同时满足三个条件才启用分组切分：
    # 1. 配置开启  2. sklearn 版本支持  3. 数据集中确实存在重复患者编号
    can_group_split = (
        USE_GROUP_SPLIT_WHEN_POSSIBLE
        and StratifiedGroupKFold is not None
        and groups.duplicated(keep=False).any()
    )

    if can_group_split:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # next() 只取第一折作为测试集（约 20%），其余 80% 作训练集
        train_idx, test_idx = next(splitter.split(X, y, groups))
        split_name = "StratifiedGroupKFold 第一折（按患者分组的安全切分）"

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        groups_train = groups.iloc[train_idx].copy()
        groups_test = groups.iloc[test_idx].copy()
    else:
        # 普通分层随机切分，stratify=y 保证训练/测试集类别比例一致
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X,
            y,
            groups,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        split_name = "train_test_split（stratify=y）"

    # 验证训练集和测试集的患者编号确实没有重叠
    overlap = set(groups_train).intersection(set(groups_test))
    print("\n" + "=" * 80)
    print("5. Hold-out 切分")
    print("=" * 80)
    print(f"切分方式：{split_name}")
    print(f"训练集大小：{len(X_train)} | 测试集大小：{len(X_test)}")
    print("训练集类别分布：")
    print(y_train.value_counts(normalize=True).sort_index().rename(index={0: "良性", 1: "恶性"}))
    print("测试集类别分布：")
    print(y_test.value_counts(normalize=True).sort_index().rename(index={0: "良性", 1: "恶性"}))
    print(f"训练集与测试集重叠的患者编号数：{len(overlap)}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test, split_name


def build_models() -> Dict[str, Pipeline]:
    """构建所有候选模型，每个模型都封装为无数据泄漏的 Pipeline。"""
    def pipe(model) -> Pipeline:
        # Pipeline 保证 imputer 和 scaler 只在训练折上 fit，测试折只做 transform
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),  # 用中位数填充缺失值
                ("scaler", StandardScaler()),                    # 标准化（均值0方差1）
                ("model", model),                               # 分类器
            ]
        )

    return {
        "LogisticRegression(C=0.1)": pipe(LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE)),
        "LogisticRegression(C=1)":   pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        "LogisticRegression(C=10)":  pipe(LogisticRegression(C=10.0, max_iter=2000, random_state=RANDOM_STATE)),
        "KNN(k=3)":                  pipe(KNeighborsClassifier(n_neighbors=3)),
        "KNN(k=5)":                  pipe(KNeighborsClassifier(n_neighbors=5)),
        "KNN(k=11)":                 pipe(KNeighborsClassifier(n_neighbors=11)),
        "SVM(C=1, rbf)":             pipe(SVC(C=1.0, kernel="rbf")),
        "SVM(C=10, rbf)":            pipe(SVC(C=10.0, kernel="rbf")),
        "SVM(C=1, linear)":          pipe(SVC(C=1.0, kernel="linear")),
        "DecisionTree(depth=3)":     pipe(DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)),
        "DecisionTree(depth=5)":     pipe(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        "DecisionTree(depth=None)":  pipe(DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE)),
        "RandomForest(n=50)":        pipe(RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
        "RandomForest(n=100)":       pipe(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        "RandomForest(n=200)":       pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        "GaussianNB":                pipe(GaussianNB()),
    }


def get_cv_strategy(y_train: pd.Series, groups_train: pd.Series):
    """根据数据情况返回合适的交叉验证对象和对应的分组数组。"""
    if USE_GROUP_SPLIT_WHEN_POSSIBLE and StratifiedGroupKFold is not None and groups_train.duplicated(keep=False).any():
        # 有重复患者编号时，用分组分层 K 折，防止患者信息泄漏
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), groups_train

    # 无需分组控制时，用重复分层 K 折（5折×10次=50个评估结果），估计更稳定
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE), None


def cross_validate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: Optional[pd.Series],
) -> pd.DataFrame:
    """仅在训练集上对所有候选模型做交叉验证，返回汇总结果表。"""
    # 同时评估多个指标
    scoring = {
        "accuracy":          "accuracy",
        "precision":         "precision",
        "recall":            "recall",
        "f1":                "f1",
        "roc_auc":           "roc_auc",
        "average_precision": "average_precision",
    }

    cv, cv_groups = get_cv_strategy(y_train, groups_train)
    print("\n" + "=" * 80)
    print("6. 仅在训练集上做交叉验证")
    print("=" * 80)
    print(f"交叉验证策略：{cv.__class__.__name__}")

    rows = []
    for name, model in models.items():
        # 有分组时传入 groups 参数，否则不传
        kwargs = {"groups": cv_groups} if cv_groups is not None else {}
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
            error_score="raise",  # 出错时直接报错而非静默跳过
            **kwargs,
        )

        row = {"Model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()  # 各折均值
            row[f"{metric}_std"] = values.std()    # 各折标准差
        rows.append(row)

    # 按 F1 均值降序排列，F1 相同再看召回率，再看 AUC
    result = pd.DataFrame(rows).sort_values(
        by=["f1_mean", "recall_mean", "roc_auc_mean"], ascending=False
    )
    return result


def get_positive_class_score(fitted_model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    返回正类（恶性）的连续预测得分。
    ROC 曲线和 PR 曲线需要概率值，不能只用 0/1 硬预测。
    优先用 predict_proba，没有则用 decision_function，再没有才用 predict。
    """
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]   # 取正类概率
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X)      # SVM 默认无概率输出，用决策函数值代替
    return fitted_model.predict(X)


def evaluate_on_test(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, float], np.ndarray, np.ndarray]:
    """在全量训练集上重新训练最优模型，并在测试集上评估一次。"""
    # clone 复制模型结构和超参数，但清空训练状态，确保从头训练
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    y_pred = fitted_model.predict(X_test)
    y_score = get_positive_class_score(fitted_model, X_test)

    metrics = {
        "Accuracy":          accuracy_score(y_test, y_pred),
        "Precision":         precision_score(y_test, y_pred, zero_division=0),
        "Recall":            recall_score(y_test, y_pred, zero_division=0),
        "F1":                f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC":           roc_auc_score(y_test, y_score),
        "Average_Precision": average_precision_score(y_test, y_score),
    }

    print("\n" + "=" * 80)
    print("7. 最终测试集评估（仅此一次）")
    print("=" * 80)
    print(f"交叉验证选出的最优模型（按 F1）：{model_name}")
    print(pd.Series(metrics).round(4).to_string())
    print("\n分类报告：")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["良性(Benign)", "恶性(Malignant)"],
            digits=4,
            zero_division=0,
        )
    )

    return fitted_model, metrics, y_pred, y_score


# ============================================================
# 可视化
# ============================================================


def plot_feature_distributions_by_class(df: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    """为每个特征绘制按类别归一化的分布柱状图（类内百分比）。"""
    class_map = {2: "良性(Benign)", 4: "恶性(Malignant)"}
    x_values = np.arange(1, 11)  # 特征取值范围 1~10
    bar_width = 0.38

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        for class_value, offset in [(2, -bar_width / 2), (4, bar_width / 2)]:
            # 取当前类别的该列数值，转数值型并去掉 NaN
            values = pd.to_numeric(df.loc[df["Class"] == class_value, col], errors="coerce").dropna()
            # 统计各分值出现次数，reindex 补全未出现的分值为 0
            counts = values.astype(int).value_counts().reindex(x_values, fill_value=0)
            # 转为类内百分比，消除良恶性样本数量不同的干扰
            percentages = counts / counts.sum() * 100 if counts.sum() > 0 else counts
            # 两类柱子左右错开 bar_width/2 并排显示
            ax.bar(x_values + offset, percentages, width=bar_width, label=class_map[class_value])

        ax.set_title(col)
        ax.set_xticks(x_values)
        ax.set_xlabel("评分")
        ax.set_ylabel("类内占比 (%)")
        ax.grid(axis="y", alpha=0.25)

    # 从第一个子图取图例句柄，在整张图顶部统一显示
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("各特征按良恶性分类的分布", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_diagnostics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """绘制混淆矩阵（含行百分比）、ROC 曲线、PR 曲线，三图合一保存。"""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    row_totals = cm.sum(axis=1, keepdims=True)
    # 行归一化：每格除以该行总数，where 防止除以零
    cm_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # —— 混淆矩阵 ——
    im = axes[0].imshow(cm)
    axes[0].set_title(f"混淆矩阵\n{model_name}")
    axes[0].set_xlabel("预测标签")
    axes[0].set_ylabel("真实标签")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["良性", "恶性"])
    axes[0].set_yticklabels(["良性", "恶性"])

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # viridis 色图：低值为深紫色→用白字；高值为亮黄色→用黑字
            text_color = "black" if cm[i, j] > threshold else "white"
            axes[0].text(
                j, i,
                f"{cm[i, j]}\n{cm_pct[i, j] * 100:.1f}%",
                ha="center", va="center", color=text_color,
            )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # —— ROC 曲线 ——
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", label="随机分类器")
    axes[1].set_title("ROC 曲线")
    axes[1].set_xlabel("假正率 (FPR)")
    axes[1].set_ylabel("真正率 / 召回率 (TPR)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)

    # —— PR 曲线 ——
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    avg_precision = average_precision_score(y_test, y_score)
    axes[2].plot(recall, precision, label=f"AP = {avg_precision:.4f}")
    axes[2].set_title("Precision-Recall 曲线")
    axes[2].set_xlabel("召回率 (Recall)")
    axes[2].set_ylabel("精准率 (Precision)")
    axes[2].legend(loc="lower left")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================


def main() -> None:
    # exist_ok=True：目录已存在时不报错
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. 加载数据
    df, X, y, groups, feature_cols = load_dataset()
    print_data_report(df, X, y, groups)

    # 2. 绘制特征分布图（使用原始 df，切分前）
    feature_plot_path = OUTPUT_DIR / "feature_distributions_by_class.png"
    plot_feature_distributions_by_class(df, feature_cols, feature_plot_path)

    # 3. 切分训练集和测试集
    X_train, X_test, y_train, y_test, groups_train, groups_test, split_name = make_holdout_split(X, y, groups)

    # 4. 构建候选模型，在训练集上做交叉验证选出最优模型
    models = build_models()
    cv_results = cross_validate_models(models, X_train, y_train, groups_train)

    # 保存交叉验证结果
    cv_output = OUTPUT_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output, index=False)

    display_cols = [
        "Model", "accuracy_mean", "precision_mean", "recall_mean",
        "f1_mean", "f1_std", "roc_auc_mean", "average_precision_mean",
    ]
    print("\n交叉验证汇总（按 F1 降序）：")
    print(cv_results[display_cols].round(4).to_string(index=False))

    # 取 CV F1 最高的模型
    best_model_name = cv_results.iloc[0]["Model"]
    best_model = models[best_model_name]

    # 5. 在全量训练集上重新训练，在测试集上评估一次
    fitted_model, test_metrics, y_pred, y_score = evaluate_on_test(
        best_model_name, best_model, X_train, y_train, X_test, y_test,
    )

    # 保存测试集结果
    test_output = OUTPUT_DIR / "test_results.csv"
    pd.DataFrame([{"Model": best_model_name, **test_metrics}]).to_csv(test_output, index=False)

    # 6. 绘制诊断图（混淆矩阵 + ROC + PR）
    diagnostics_plot_path = OUTPUT_DIR / "confusion_roc_pr.png"
    plot_final_diagnostics(y_test, y_pred, y_score, best_model_name, diagnostics_plot_path)

    print("\n" + "=" * 80)
    print("8. 输出文件")
    print("=" * 80)
    print(f"特征分布图：  {feature_plot_path}")
    print(f"诊断图：      {diagnostics_plot_path}")
    print(f"交叉验证结果：{cv_output}")
    print(f"测试集结果：  {test_output}")
    print("\n完成。")


if __name__ == "__main__":
    main()
