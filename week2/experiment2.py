"""
Week 2: Breast Cancer Wisconsin classification experiment

What this script does:
1. Loads and validates the dataset.
2. Cleans the '?' values in Bare Nuclei.
3. Uses a leakage-safe Pipeline: SimpleImputer -> StandardScaler -> model.
4. Selects the model by cross-validation on the training data only.
5. Evaluates the selected model once on the hold-out test set.
6. Saves standard plots and result tables.

Run from either the week2 folder or the project root:
    python experiment2_complete.py

Expected data file names in the same folder as this script:
    breast_cancer_wisconsin.csv
or the original file name:
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
except ImportError:  # for very old scikit-learn versions
    StratifiedGroupKFold = None


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_JOBS = 1

# True is stricter because the same Sample code number will not appear in both
# train and test. If your course requires the original random split, set it False.
USE_GROUP_SPLIT_WHEN_POSSIBLE = True

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "breast_cancer_wisconsin.csv",
    BASE_DIR / "（加表头）breast-cancer-wisconsin.csv",
]
OUTPUT_DIR = BASE_DIR / "outputs"

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# Data loading and validation
# ============================================================


def find_data_file(candidates: Iterable[Path]) -> Path:
    """Return the first existing data file from candidates."""
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Cannot find the dataset. Rename your CSV to breast_cancer_wisconsin.csv "
        f"or place it next to this script. Searched:\n{searched}"
    )


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load the CSV, clean numeric columns, and return df, X, y, groups, feature_cols."""
    data_file = find_data_file(DATA_CANDIDATES)
    print(f"Data file: {data_file}")

    df = pd.read_csv(data_file)

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
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    feature_cols = [c for c in df.columns if c not in ["Sample code number", "Class"]]

    # Bare Nuclei contains '?' in the original UCI file. Coerce it to NaN.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    y = df["Class"].map({2: 0, 4: 1})
    if y.isna().any():
        bad_labels = sorted(df.loc[y.isna(), "Class"].unique())
        raise ValueError(f"Unexpected class labels found: {bad_labels}. Expected only 2 and 4.")
    y = y.astype(int)

    X = df[feature_cols].copy()
    groups = df["Sample code number"].copy()

    return df, X, y, groups, feature_cols


def print_data_report(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
    """Print basic EDA and data quality checks."""
    print("\n" + "=" * 80)
    print("1. First 5 rows")
    print("=" * 80)
    print(df.head())

    print("\n" + "=" * 80)
    print("2. Dataset info and missing values")
    print("=" * 80)
    df.info()
    print("\nMissing values:")
    print(df.isna().sum())

    print("\n" + "=" * 80)
    print("3. Descriptive statistics")
    print("=" * 80)
    print(df.describe())

    print("\n" + "=" * 80)
    print("4. Class distribution")
    print("=" * 80)
    class_counts = df["Class"].value_counts().sort_index()
    print(class_counts.rename(index={2: "Benign / 2", 4: "Malignant / 4"}))

    duplicate_id_count = int(groups.duplicated(keep=False).sum())
    full_duplicate_count = int(df.duplicated().sum())
    print("\nData quality notes:")
    print(f"- Rows: {len(df)}")
    print(f"- Features used: {X.shape[1]}")
    print(f"- Rows sharing duplicate Sample code number: {duplicate_id_count}")
    print(f"- Fully duplicated rows: {full_duplicate_count}")


# ============================================================
# Splitting and modeling
# ============================================================


def make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, str]:
    """Create a hold-out split. Prefer group-safe split when possible."""
    can_group_split = (
        USE_GROUP_SPLIT_WHEN_POSSIBLE
        and StratifiedGroupKFold is not None
        and groups.duplicated(keep=False).any()
    )

    if can_group_split:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        split_name = "StratifiedGroupKFold first fold, group-safe hold-out split"

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        groups_train = groups.iloc[train_idx].copy()
        groups_test = groups.iloc[test_idx].copy()
    else:
        X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X,
            y,
            groups,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        split_name = "train_test_split with stratify=y"

    overlap = set(groups_train).intersection(set(groups_test))
    print("\n" + "=" * 80)
    print("5. Hold-out split")
    print("=" * 80)
    print(f"Split method: {split_name}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("Train class distribution:")
    print(y_train.value_counts(normalize=True).sort_index().rename(index={0: "Benign", 1: "Malignant"}))
    print("Test class distribution:")
    print(y_test.value_counts(normalize=True).sort_index().rename(index={0: "Benign", 1: "Malignant"}))
    print(f"Overlapping Sample code numbers between train and test: {len(overlap)}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test, split_name


def build_models() -> Dict[str, Pipeline]:
    """Build all candidate models as leakage-safe sklearn Pipelines."""
    def pipe(model) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    return {
        "LogisticRegression(C=0.1)": pipe(LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE)),
        "LogisticRegression(C=1)": pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        "LogisticRegression(C=10)": pipe(LogisticRegression(C=10.0, max_iter=2000, random_state=RANDOM_STATE)),
        "KNN(k=3)": pipe(KNeighborsClassifier(n_neighbors=3)),
        "KNN(k=5)": pipe(KNeighborsClassifier(n_neighbors=5)),
        "KNN(k=11)": pipe(KNeighborsClassifier(n_neighbors=11)),
        "SVM(C=1, rbf)": pipe(SVC(C=1.0, kernel="rbf")),
        "SVM(C=10, rbf)": pipe(SVC(C=10.0, kernel="rbf")),
        "SVM(C=1, linear)": pipe(SVC(C=1.0, kernel="linear")),
        "DecisionTree(depth=3)": pipe(DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)),
        "DecisionTree(depth=5)": pipe(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        "DecisionTree(depth=None)": pipe(DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE)),
        "RandomForest(n=50)": pipe(RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)),
        "RandomForest(n=100)": pipe(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        "RandomForest(n=200)": pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        "GaussianNB": pipe(GaussianNB()),
    }


def get_cv_strategy(y_train: pd.Series, groups_train: pd.Series):
    """Return the appropriate CV object and group array."""
    if USE_GROUP_SPLIT_WHEN_POSSIBLE and StratifiedGroupKFold is not None and groups_train.duplicated(keep=False).any():
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), groups_train

    # Repeated CV gives a more stable estimate when group control is not needed.
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE), None


def cross_validate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: Optional[pd.Series],
) -> pd.DataFrame:
    """Evaluate candidate models by cross-validation on training data only."""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    cv, cv_groups = get_cv_strategy(y_train, groups_train)
    print("\n" + "=" * 80)
    print("6. Cross-validation on training data only")
    print("=" * 80)
    print(f"CV strategy: {cv.__class__.__name__}")

    rows = []
    for name, model in models.items():
        kwargs = {"groups": cv_groups} if cv_groups is not None else {}
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=N_JOBS,
            error_score="raise",
            **kwargs,
        )

        row = {"Model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        by=["f1_mean", "recall_mean", "roc_auc_mean"], ascending=False
    )
    return result


def get_positive_class_score(fitted_model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Return a continuous score for the positive class.
    Prefer predict_proba when available; otherwise use decision_function.
    ROC AUC and PR curves need scores, not just hard class labels.
    """
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X)
    return fitted_model.predict(X)


def evaluate_on_test(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, float], np.ndarray, np.ndarray]:
    """Fit selected model on all training data and evaluate once on test data."""
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    y_pred = fitted_model.predict(X_test)
    y_score = get_positive_class_score(fitted_model, X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_score),
        "Average_Precision": average_precision_score(y_test, y_score),
    }

    print("\n" + "=" * 80)
    print("7. Final hold-out test evaluation")
    print("=" * 80)
    print(f"Selected model by CV F1: {model_name}")
    print(pd.Series(metrics).round(4).to_string())
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0,
        )
    )

    return fitted_model, metrics, y_pred, y_score


# ============================================================
# Visualization
# ============================================================


def plot_feature_distributions_by_class(df: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    """Plot class-normalized distributions for each discrete 1-10 feature."""
    class_map = {2: "Benign", 4: "Malignant"}
    x_values = np.arange(1, 11)
    bar_width = 0.38

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        for class_value, offset in [(2, -bar_width / 2), (4, bar_width / 2)]:
            values = pd.to_numeric(df.loc[df["Class"] == class_value, col], errors="coerce").dropna()
            counts = values.astype(int).value_counts().reindex(x_values, fill_value=0)
            percentages = counts / counts.sum() * 100 if counts.sum() > 0 else counts
            ax.bar(x_values + offset, percentages, width=bar_width, label=class_map[class_value])

        ax.set_title(col)
        ax.set_xticks(x_values)
        ax.set_xlabel("Score")
        ax.set_ylabel("% within class")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Feature Distributions by Class", fontsize=14, y=0.98)
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
    """Plot confusion matrix with row percentages, ROC curve, and PR curve."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix: counts + row percentage.
    im = axes[0].imshow(cm)
    axes[0].set_title(f"Confusion Matrix\n{model_name}")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Benign", "Malignant"])
    axes[0].set_yticklabels(["Benign", "Malignant"])

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # viridis: low values are dark purple → need white text; high values are bright → need black text
            text_color = "black" if cm[i, j] > threshold else "white"
            axes[0].text(
                j,
                i,
                f"{cm[i, j]}\n{cm_pct[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=text_color,
            )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", label="Random")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate / Recall")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.25)

    # Precision-Recall curve.
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    avg_precision = average_precision_score(y_test, y_score)
    axes[2].plot(recall, precision, label=f"AP = {avg_precision:.4f}")
    axes[2].set_title("Precision-Recall Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df, X, y, groups, feature_cols = load_dataset()
    print_data_report(df, X, y, groups)

    feature_plot_path = OUTPUT_DIR / "feature_distributions_by_class.png"
    plot_feature_distributions_by_class(df, feature_cols, feature_plot_path)

    X_train, X_test, y_train, y_test, groups_train, groups_test, split_name = make_holdout_split(X, y, groups)

    models = build_models()
    cv_results = cross_validate_models(models, X_train, y_train, groups_train)

    cv_output = OUTPUT_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output, index=False)

    display_cols = [
        "Model",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "average_precision_mean",
    ]
    print("\nCross-validation summary, sorted by F1:")
    print(cv_results[display_cols].round(4).to_string(index=False))

    best_model_name = cv_results.iloc[0]["Model"]
    best_model = models[best_model_name]

    fitted_model, test_metrics, y_pred, y_score = evaluate_on_test(
        best_model_name,
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    test_output = OUTPUT_DIR / "test_results.csv"
    pd.DataFrame([{ "Model": best_model_name, **test_metrics }]).to_csv(test_output, index=False)

    diagnostics_plot_path = OUTPUT_DIR / "confusion_roc_pr.png"
    plot_final_diagnostics(y_test, y_pred, y_score, best_model_name, diagnostics_plot_path)

    print("\n" + "=" * 80)
    print("8. Saved outputs")
    print("=" * 80)
    print(f"Feature distribution plot: {feature_plot_path}")
    print(f"Final diagnostic plot:      {diagnostics_plot_path}")
    print(f"CV results table:          {cv_output}")
    print(f"Test results table:        {test_output}")
    print("\nDone.")


if __name__ == "__main__":
    main()
