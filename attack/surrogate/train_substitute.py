"""
Substitute Model Training Pipeline
===================================
Trains a surrogate (substitute) classifier on labels extracted from the
black-box prediction API (stolen_dataset.csv), then compares its behavior
against the original deployed model.

Academic context
----------------
In model stealing / model extraction attacks, the adversary trains a
substitute model f'(x) to mimic the victim model f(x) using only query-
response pairs. High agreement between f and f' indicates a successful
extraction attack.

Usage
-----
    python train_substitute.py
    python train_substitute.py --stolen-data ../stolen_dataset.csv

Outputs
-------
    substitute_model.joblib
    outputs/substitute_metrics.csv
    outputs/model_comparison.csv
    outputs/confusion_matrix_substitute.png
    outputs/confusion_matrix_original.png
    outputs/agreement_analysis.png
    outputs/roc_comparison.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    ORIGINAL_MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    STOLEN_DATASET_PATH,
    SUBSTITUTE_MODEL_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
    TRAINING_DATASET_PATH,
)

warnings.filterwarnings("ignore")

# XGBoost is optional at import time; required for full pipeline
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# SECTION 1: LOAD STOLEN DATASET
# =============================================================================
# Explanation:
# The stolen dataset contains (input features, API prediction labels) pairs
# collected during the black-box attack. Only successful API responses are used.


def load_stolen_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the stolen query-response dataset."""
    print(f"\n{'=' * 70}")
    print("SECTION 1: LOAD STOLEN DATASET")
    print(f"{'=' * 70}")

    if not path.exists():
        raise FileNotFoundError(
            f"Stolen dataset not found: {path}\n"
            "Run attack/collect_responses.py first."
        )

    df = pd.read_csv(path)
    print(f"Raw records loaded: {len(df):,}")

    if "success" in df.columns:
        df = df[df["success"] == True].copy()  # noqa: E712
        print(f"Successful API responses: {len(df):,}")

    if df.empty:
        raise ValueError("No successful records in stolen dataset.")

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    if len(df) < 20:
        print(
            f"WARNING: Only {len(df)} samples available. "
            "Metrics may be unstable. Consider collecting more API queries (3000+)."
        )

    print(f"\nClass distribution (teacher labels):")
    print(df[TARGET_COLUMN].value_counts().to_string())

    return df


# =============================================================================
# SECTION 2: PREPROCESSING (IDENTICAL TO ORIGINAL MODEL)
# =============================================================================
# Explanation:
# The substitute model uses the same preprocessing strategy as the victim:
#   - StandardScaler on numeric features
#   - OneHotEncoder on categorical features
# This ensures fair architectural parity during model extraction analysis.


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and encoded target y from stolen data."""
    print(f"\n{'=' * 70}")
    print("SECTION 2: PREPROCESSING")
    print(f"{'=' * 70}")

    data = df.copy()
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].map(TARGET_MAPPING)

    if y.isnull().any():
        unknown = data.loc[y.isnull(), TARGET_COLUMN].unique()
        raise ValueError(f"Unrecognized target labels: {unknown}")

    # Handle missing values in features (if any)
    for col in NUMERIC_FEATURES:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    for col in CATEGORICAL_FEATURES:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    print(f"Features: {len(FEATURE_COLUMNS)} ({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    print("Preprocessing: StandardScaler (numeric) + OneHotEncoder (categorical)")
    print("Target encoding: Leave=1, Stay=0 (teacher API labels)")

    return X, y.astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline matching the original victim model."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


# =============================================================================
# SECTION 3: TRAIN-TEST SPLIT
# =============================================================================


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80:20 train-test split."""
    print(f"\n{'=' * 70}")
    print("SECTION 3: TRAIN-TEST SPLIT (80:20, Stratified)")
    print(f"{'=' * 70}")

    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        raise ValueError("Need at least 2 samples per class for stratified split.")

    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    print(f"Training set: {len(X_train):,} samples (Leave rate: {y_train.mean()*100:.1f}%)")
    print(f"Testing set:  {len(X_test):,} samples (Leave rate: {y_test.mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 4: TRAIN SUBSTITUTE MODELS
# =============================================================================


def get_model_candidates(preprocessor: ColumnTransformer) -> dict:
    """Define substitute model candidates with compact hyperparameter grids."""
    candidates = {
        "Logistic Regression": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
    }

    if XGBOOST_AVAILABLE:
        candidates["XGBoost"] = {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
            },
        }
    else:
        print("WARNING: xgboost not installed. Skipping XGBoost candidate.")

    return candidates


def train_and_evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """Train all substitute candidates and return results."""
    print(f"\n{'=' * 70}")
    print("SECTION 4: TRAIN SUBSTITUTE MODELS")
    print(f"{'=' * 70}")

    preprocessor = build_preprocessor()
    candidates = get_model_candidates(preprocessor)

    n_splits = min(CV_FOLDS, y_train.value_counts().min())
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    tuned_models: dict = {}
    metrics_rows: list[dict] = []

    for name, config in candidates.items():
        print(f"\nTraining {name}...")

        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        tuned_models[name] = grid

        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(y_test, y_pred, y_proba, model_name=name)
        metrics["best_cv_f1"] = round(grid.best_score_, 4)
        metrics["best_params"] = str(grid.best_params_)
        metrics_rows.append(metrics)

        print(f"  Best CV F1: {grid.best_score_:.4f}")
        print(f"  Test F1:    {metrics['F1-Score']:.4f}")
        print(f"  Test AUC:   {metrics['ROC-AUC']:.4f}")

    comparison_df = pd.DataFrame(metrics_rows)
    return tuned_models, comparison_df


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> dict:
    """Compute standard classification metrics."""
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_proba), 4),
    }


# =============================================================================
# SECTION 5: SELECT BEST SUBSTITUTE MODEL
# =============================================================================


def select_best_model(comparison_df: pd.DataFrame, tuned_models: dict):
    """Select best substitute by F1-Score, then ROC-AUC."""
    print(f"\n{'=' * 70}")
    print("SECTION 5: SELECT BEST SUBSTITUTE MODEL")
    print(f"{'=' * 70}")

    ranked = comparison_df.sort_values(["F1-Score", "ROC-AUC"], ascending=False)
    print("\nSubstitute Model Comparison (Test Set vs Teacher Labels):")
    print(ranked[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))

    best_name = ranked.iloc[0]["Model"]
    best_model = tuned_models[best_name]
    print(f"\nBest Substitute Model: {best_name}")

    return best_name, best_model, ranked


# =============================================================================
# SECTION 6: COMPARE WITH ORIGINAL (VICTIM) MODEL
# =============================================================================
# Explanation:
# The original model requires 24 features; the API fills missing fields with
# defaults. We replicate that logic so victim predictions are comparable on
# the same stolen-data test inputs.


def compute_original_feature_defaults(feature_columns: list[str]) -> dict:
    """Compute default values for features not in the stolen 10-feature set."""
    defaults: dict = {}

    if TRAINING_DATASET_PATH.exists():
        df = pd.read_csv(TRAINING_DATASET_PATH)
        df = df.drop(columns=["Employee_ID", "Attrition"], errors="ignore")
        for col in feature_columns:
            if col not in FEATURE_COLUMNS and col in df.columns:
                if df[col].dtype == "object":
                    defaults[col] = df[col].mode()[0]
                else:
                    defaults[col] = int(df[col].median())
    else:
        defaults = {
            "Marital_Status": "Married",
            "Job_Level": 3,
            "Hourly_Rate": 50,
            "Years_in_Current_Role": 5,
            "Years_Since_Last_Promotion": 2,
            "Performance_Rating": 2,
            "Training_Hours_Last_Year": 50,
            "Project_Count": 5,
            "Average_Hours_Worked_Per_Week": 45,
            "Absenteeism": 7,
            "Work_Environment_Satisfaction": 3,
            "Relationship_with_Manager": 3,
            "Job_Involvement": 3,
            "Distance_From_Home": 15,
        }

    return defaults


def get_original_predictions(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the victim model on test inputs.

    Returns (hard_labels, leave_probabilities).
    """
    if not ORIGINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Original model not found: {ORIGINAL_MODEL_PATH}")

    artifact = joblib.load(ORIGINAL_MODEL_PATH)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    defaults = compute_original_feature_defaults(feature_columns)

    full_records = []
    for _, row in X.iterrows():
        record = {**defaults, **row[FEATURE_COLUMNS].to_dict()}
        full_records.append(record)

    input_df = pd.DataFrame(full_records)[feature_columns]
    preds = pipeline.predict(input_df).astype(int)
    probas = pipeline.predict_proba(input_df)[:, 1]

    return preds, probas


def compare_substitute_vs_original(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_name: str,
) -> dict:
    """Compare substitute and victim model predictions on the same test set."""
    print(f"\n{'=' * 70}")
    print("SECTION 6: SUBSTITUTE vs ORIGINAL MODEL COMPARISON")
    print(f"{'=' * 70}")

    y_substitute = best_model.predict(X_test)
    y_substitute_proba = best_model.predict_proba(X_test)[:, 1]

    y_original, y_original_proba = get_original_predictions(X_test)

    agreement = (y_substitute == y_original).mean()
    label_agreement_df = pd.DataFrame(
        {
            "substitute": y_substitute,
            "original": y_original,
            "teacher_label": y_test.values,
        }
    )

    print(f"\nAgreement Rate (Substitute vs Original): {agreement * 100:.2f}%")
    print(f"  Matching predictions: {(y_substitute == y_original).sum()} / {len(y_test)}")

    # Metrics vs teacher labels (API responses used as ground truth for attack)
    sub_metrics = evaluate_predictions(y_test, y_substitute, y_substitute_proba, f"Substitute ({best_name})")
    orig_metrics = evaluate_predictions(y_test, y_original, y_original_proba, "Original (Victim)")

    comparison = pd.DataFrame([sub_metrics, orig_metrics])
    print("\nPerformance vs Teacher Labels (stolen API responses):")
    print(comparison.to_string(index=False))

    print("\nClassification Report - Substitute Model:")
    print(classification_report(y_test, y_substitute, target_names=["Stay (0)", "Leave (1)"]))

    return {
        "agreement_rate": round(agreement * 100, 2),
        "y_substitute": y_substitute,
        "y_substitute_proba": y_substitute_proba,
        "y_original": y_original,
        "y_original_proba": y_original_proba,
        "comparison_metrics": comparison,
        "label_agreement_df": label_agreement_df,
    }


# =============================================================================
# SECTION 7: VISUALIZATIONS
# =============================================================================


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay (0)", "Leave (1)"],
        yticklabels=["Stay (0)", "Leave (1)"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_agreement_analysis(
    y_substitute: np.ndarray,
    y_original: np.ndarray,
    agreement_rate: float,
    output_path: Path,
) -> None:
    """Visualize agreement and disagreement between substitute and victim."""
    agree_mask = y_substitute == y_original
    categories = ["Agree", "Disagree"]
    counts = [agree_mask.sum(), (~agree_mask).sum()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#2563eb", "#dc2626"]
    axes[0].bar(categories, counts, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title(f"Prediction Agreement\n(Agreement Rate: {agreement_rate:.1f}%)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.1, str(v), ha="center", fontweight="bold")

    cross_cm = confusion_matrix(y_original, y_substitute)
    sns.heatmap(
        cross_cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=["Sub: Stay", "Sub: Leave"],
        yticklabels=["Orig: Stay", "Orig: Leave"],
        ax=axes[1],
    )
    axes[1].set_title("Substitute vs Original Predictions")
    axes[1].set_xlabel("Substitute Prediction")
    axes[1].set_ylabel("Original Prediction")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_comparison(
    y_true: pd.Series,
    y_substitute_proba: np.ndarray,
    y_original_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Plot ROC curves for substitute and original models."""
    plt.figure(figsize=(7, 6))

    for probas, label, color in [
        (y_substitute_proba, "Substitute Model", "#2563eb"),
        (y_original_proba, "Original (Victim) Model", "#059669"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probas)
        auc = roc_auc_score(y_true, probas)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison: Substitute vs Original")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_model_metrics_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart comparing metrics across all trained substitute candidates."""
    plot_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="Blues")
    plt.title("Substitute Model Candidates - Test Set Metrics")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_visualizations(
    y_test: pd.Series,
    comparison_result: dict,
    substitute_metrics_df: pd.DataFrame,
    best_name: str,
    output_dir: Path,
) -> None:
    """Generate all comparison plots."""
    print(f"\n{'=' * 70}")
    print("SECTION 7: GENERATE VISUALIZATIONS")
    print(f"{'=' * 70}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_test,
        comparison_result["y_substitute"],
        f"Confusion Matrix - Substitute ({best_name})",
        output_dir / "confusion_matrix_substitute.png",
    )

    plot_confusion_matrix(
        y_test,
        comparison_result["y_original"],
        "Confusion Matrix - Original (Victim) Model",
        output_dir / "confusion_matrix_original.png",
    )

    plot_agreement_analysis(
        comparison_result["y_substitute"],
        comparison_result["y_original"],
        comparison_result["agreement_rate"],
        output_dir / "agreement_analysis.png",
    )

    plot_roc_comparison(
        y_test,
        comparison_result["y_substitute_proba"],
        comparison_result["y_original_proba"],
        output_dir / "roc_comparison.png",
    )

    plot_model_metrics_comparison(
        substitute_metrics_df,
        output_dir / "substitute_candidates_metrics.png",
    )

    print(f"Plots saved to: {output_dir}")


# =============================================================================
# SECTION 8: SAVE BEST SUBSTITUTE MODEL
# =============================================================================


def save_substitute_model(
    best_name: str,
    best_model,
    agreement_rate: float,
    output_path: Path,
) -> None:
    """Persist the best substitute model and metadata."""
    print(f"\n{'=' * 70}")
    print("SECTION 8: SAVE SUBSTITUTE MODEL")
    print(f"{'=' * 70}")

    artifact = {
        "model_name": best_name,
        "pipeline": best_model.best_estimator_,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_mapping": TARGET_MAPPING,
        "agreement_with_victim_pct": agreement_rate,
        "model_type": "substitute_surrogate",
    }

    joblib.dump(artifact, output_path)
    print(f"Substitute model saved to: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train substitute model on stolen API dataset.")
    parser.add_argument(
        "--stolen-data",
        type=Path,
        default=STOLEN_DATASET_PATH,
        help="Path to stolen_dataset.csv",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=SUBSTITUTE_MODEL_PATH,
        help="Path to save substitute_model.joblib",
    )
    return parser.parse_args()


def run_training_pipeline(
    stolen_data_path: Path = STOLEN_DATASET_PATH,
    output_model_path: Path = SUBSTITUTE_MODEL_PATH,
) -> dict:
    """
    Execute full substitute training, evaluation, and chart generation.
    Callable programmatically from the attack pipeline.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stolen_df = load_stolen_dataset(stolen_data_path)
    X, y = encode_target(stolen_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    tuned_models, substitute_metrics_df = train_and_evaluate_candidates(
        X_train, y_train, X_test, y_test
    )
    best_name, best_model, _ = select_best_model(substitute_metrics_df, tuned_models)
    comparison_result = compare_substitute_vs_original(best_model, X_test, y_test, best_name)

    generate_visualizations(
        y_test, comparison_result, substitute_metrics_df, best_name, OUTPUT_DIR
    )

    save_substitute_model(
        best_name, best_model, comparison_result["agreement_rate"], output_model_path
    )

    substitute_metrics_df.to_csv(OUTPUT_DIR / "substitute_metrics.csv", index=False)
    comparison_result["comparison_metrics"].to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    summary = {
        "best_substitute_model": best_name,
        "agreement_rate_pct": comparison_result["agreement_rate"],
        "agreement_rate": comparison_result["agreement_rate"],
        "test_samples": len(y_test),
        "train_samples": len(y_train),
        "stolen_dataset_size": len(stolen_df),
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "attack_summary.csv", index=False)

    return summary


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print(" SUBSTITUTE MODEL TRAINING PIPELINE")
    print(" Model Extraction Attack - Surrogate Training Phase")
    print("=" * 70)

    summary = run_training_pipeline(args.stolen_data, args.output_model)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'=' * 70}")
    print(f"Best substitute model : {summary['best_substitute_model']}")
    print(f"Agreement with victim : {summary['agreement_rate']}%")
    print(f"Model saved           : {args.output_model}")
    print(f"Reports & plots       : {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
