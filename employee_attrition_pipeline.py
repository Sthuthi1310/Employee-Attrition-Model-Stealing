"""
Employee Attrition Prediction - Complete Machine Learning Pipeline
==================================================================
This script implements an end-to-end ML workflow for predicting whether
an employee is likely to leave the organization (attrition).

Designed for academic projects: each section includes explanations
suitable for project reports and viva voce discussions.
"""

# =============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# =============================================================================
# Explanation:
# We import libraries for data handling (pandas, numpy), visualization
# (matplotlib, seaborn), machine learning (scikit-learn), and model persistence
# (joblib). Configuration constants keep paths and hyperparameters centralized.

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from utils.helper import API_FEATURE_COLUMNS, load_dataset
from utils.metrics import evaluate_model, print_evaluation_report
from utils.paths import (
    METRICS_PATH,
    RESULTS_DIR,
    TARGET_MODEL_PATH,
    TRAINING_DATASET_PATH,
    VICTIM_CONFUSION_MATRIX_PATH,
    VICTIM_EVALUATION_PATH,
)
from utils.preprocessing import build_preprocessor, identify_target_column

warnings.filterwarnings("ignore")

# --- Configuration ---
DATASET_PATH = TRAINING_DATASET_PATH
OUTPUT_DIR = RESULTS_DIR
MODEL_PATH = TARGET_MODEL_PATH
RANDOM_STATE = 42
TEST_SIZE = 0.20


# =============================================================================
# SECTION 2: LOAD DATASET
# =============================================================================
# Explanation:
# The dataset is loaded using pandas.read_csv(). Pandas provides efficient
# tabular data structures (DataFrame) for exploration and preprocessing.


def load_and_report_dataset(file_path):
    """Load dataset and print loading summary."""
    df = load_dataset(file_path)
    print(f"\n{'=' * 70}")
    print("SECTION 2: DATA LOADING")
    print(f"{'=' * 70}")
    print(f"Dataset loaded successfully from:\n  {file_path}")
    return df


# =============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# Explanation:
# EDA helps us understand data quality, feature types, and class imbalance
# before modeling. This step guides preprocessing choices and evaluation metrics.


def perform_eda(df: pd.DataFrame, target_col: str) -> None:
    """Display shape, dtypes, missing values, and target distribution."""
    print(f"\n{'=' * 70}")
    print("SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print(f"{'=' * 70}")

    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

    print("\n--- Column Names and Data Types ---")
    dtype_summary = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values})
    print(dtype_summary.to_string(index=False))

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing (%)": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    if missing_df.empty:
        print("No missing values found in the dataset.")
    else:
        print(missing_df.to_string())

    print(f"\n--- Target Variable Distribution: '{target_col}' ---")
    class_counts = df[target_col].value_counts()
    class_pct = (df[target_col].value_counts(normalize=True) * 100).round(2)
    distribution = pd.DataFrame({"Count": class_counts, "Percentage (%)": class_pct})
    print(distribution.to_string())

    # Visualize class imbalance (useful for report)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, palette="Set2")
    plt.title("Class Distribution of Employee Attrition")
    plt.xlabel("Attrition")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"\nClass distribution plot saved to: {OUTPUT_DIR / 'class_distribution.png'}")


def report_target_column(target_col: str) -> None:
    """Print target column identification summary."""
    print(f"\n{'=' * 70}")
    print("SECTION 3: TARGET COLUMN IDENTIFICATION")
    print(f"{'=' * 70}")
    print(f"Target column identified: '{target_col}'")


def report_preprocessing(X, numeric_features, categorical_features) -> None:
    """Print preprocessing summary."""
    print(f"\n{'=' * 70}")
    print("SECTION 5: DATA PREPROCESSING")
    print(f"{'=' * 70}")
    print(f"Features used for modeling: {X.shape[1]}")
    print(f"  - Numerical features ({len(numeric_features)}): {numeric_features}")
    print(f"  - Categorical features ({len(categorical_features)}): {categorical_features}")
    print("Target encoding: Yes/1 = Employee likely to leave | No/0 = Employee likely to stay")


# =============================================================================
# SECTION 6: TRAIN-TEST SPLIT
# =============================================================================
# Explanation:
# An 80:20 stratified split preserves the class ratio in both sets.
# Stratification is important because attrition datasets are often imbalanced.


def split_data(X, y):
    """Split data into training (80%) and testing (20%) sets."""
    print(f"\n{'=' * 70}")
    print("SECTION 6: TRAIN-TEST SPLIT (80:20, Stratified)")
    print(f"{'=' * 70}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Testing set:  {X_test.shape[0]:,} samples")
    print(f"Training attrition rate: {y_train.mean() * 100:.2f}%")
    print(f"Testing attrition rate:  {y_test.mean() * 100:.2f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 7: MODEL TRAINING WITH HYPERPARAMETER TUNING
# =============================================================================
# Explanation:
# We train Logistic Regression and Random Forest using GridSearchCV with
# StratifiedKFold cross-validation. Cross-validation estimates generalization
# performance on unseen folds, reducing overfitting risk.
#
# class_weight='balanced' helps handle class imbalance without leaking labels.


def train_and_tune_models(preprocessor, X_train, y_train):
    """Tune the Logistic Regression target model (Algorithm 1) using Stratified CV."""
    print(f"\n{'=' * 70}")
    print("SECTION 7: MODEL TRAINING & HYPERPARAMETER TUNING (ALGORITHM 1: LOGISTIC REGRESSION)")
    print(f"{'=' * 70}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    search = GridSearchCV(
        estimator=lr_pipeline,
        param_grid={
            "classifier__C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear"],
            "classifier__class_weight": [None, "balanced"],
        },
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"\nBest Logistic Regression F1 (CV): {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    return {"Logistic Regression": search}


# =============================================================================
# SECTION 8: MODEL EVALUATION
# =============================================================================
# Explanation:
# Multiple metrics are reported because accuracy alone can be misleading on
# imbalanced data. Precision, Recall, F1, ROC-AUC, and Confusion Matrix
# provide a comprehensive view of model behavior.


def plot_roc_curves(y_test, probas_dict: dict) -> None:
    """Plot ROC curves for all models (report-ready visualization)."""
    plt.figure(figsize=(7, 5))
    for name, y_proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=150)
    plt.close()


# =============================================================================
# SECTION 9: MODEL COMPARISON AND BEST MODEL SELECTION
# =============================================================================
# Explanation:
# The best model is selected primarily by F1-Score (balances precision and
# recall), with ROC-AUC as a tie-breaker for ranking imbalanced classification.


def select_best_model(comparison_df: pd.DataFrame) -> str:
    """Select best model using F1-Score, then ROC-AUC as tie-breaker."""
    ranked = comparison_df.sort_values(
        by=["F1-Score", "ROC-AUC"], ascending=[False, False]
    )
    best_model_name = ranked.iloc[0]["Model"]

    print(f"\n{'=' * 70}")
    print("SECTION 9: MODEL COMPARISON & BEST MODEL SELECTION")
    print(f"{'=' * 70}")
    print("\nModel Comparison (Test Set):")
    print(comparison_df.sort_values("F1-Score", ascending=False).to_string(index=False))
    print(f"\nBest Model Selected: {best_model_name}")
    print("Selection criteria: Highest F1-Score, then highest ROC-AUC.")

    return best_model_name


# =============================================================================
# SECTION 10: FEATURE IMPORTANCE (RANDOM FOREST)
# =============================================================================
# Explanation:
# Random Forest provides feature_importances_ after training. Because our
# model uses one-hot encoded categories, we map importances back to the
# transformed feature names for interpretability.


def display_feature_importance(model, top_n: int = 10) -> pd.DataFrame:
    """Extract and visualize top feature sensitivities/importances (PPT Slide 15: Overtime #1, Monthly Income #2, Work-Life Balance #3)."""
    import numpy as np
    print(f"\n{'=' * 70}")
    print("SECTION 10: FEATURE SENSITIVITY / IMPORTANCE ANALYSIS")
    print(f"{'=' * 70}")

    best_pipeline = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    classifier = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        coef = classifier.coef_
        importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        importances = np.zeros(len(feature_names))

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} Important Features:")
    print(importance_df.head(top_n).to_string(index=False))

    plt.figure(figsize=(8, 6))
    top_features = importance_df.head(top_n)
    sns.barplot(data=top_features, y="Feature", x="Importance", palette="viridis")
    plt.title(f"Top {top_n} Feature Sensitivities - Target Model (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"\nFeature importance plot saved to: {OUTPUT_DIR / 'feature_importance.png'}")

    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    return importance_df


# =============================================================================
# SECTION 11: SAVE BEST MODEL
# =============================================================================
# Explanation:
# joblib is efficient for persisting scikit-learn models. We save the full
# pipeline (preprocessing + classifier) so predictions work on raw employee data.


def save_best_model(model, model_name: str, feature_columns: list) -> None:
    """Persist the best model and metadata using joblib."""
    print(f"\n{'=' * 70}")
    print("SECTION 11: SAVE BEST MODEL")
    print(f"{'=' * 70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_name": model_name,
        "pipeline": model.best_estimator_,
        "feature_columns": feature_columns,
        "target_mapping": {"Yes": 1, "No": 0},
        "prediction_labels": {0: "Stay", 1: "Leave"},
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Best model ({model_name}) saved to:\n  {MODEL_PATH}")


def save_victim_evaluation(y_test, y_pred, metrics: dict) -> None:
    """Persist victim metrics and confusion matrix from the same holdout predictions."""
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    evaluation = {
        "model_name": "Logistic Regression",
        "dataset_path": str(DATASET_PATH),
        "target_column": "Attrition",
        "feature_columns": API_FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "test_samples": int(len(y_test)),
        "metrics": {key: float(value) for key, value in metrics.items() if key != "Model"},
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["Stay", "Leave"],
    }
    VICTIM_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VICTIM_EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay (0)", "Leave (1)"],
        yticklabels=["Stay (0)", "Leave (1)"],
    )
    plt.title("Victim Logistic Regression - Independent Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(VICTIM_CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()


# =============================================================================
# SECTION 12: PREDICTION FUNCTION
# =============================================================================
# Explanation:
# This function accepts employee details as a dictionary or pandas Series
# and returns attrition prediction with probability. It uses the saved pipeline
# to ensure consistent preprocessing at inference time.


def predict_employee_attrition(
    employee_details: dict,
    model_path: Path = MODEL_PATH,
) -> dict:
    """
    Predict whether an employee is likely to leave or stay.

    Parameters
    ----------
    employee_details : dict
        Dictionary containing employee feature values (same columns as training,
        excluding Employee_ID and Attrition).

    Returns
    -------
    dict
        Prediction result with label, probability, and human-readable message.
    """
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    labels = artifact["prediction_labels"]

    # Build single-row DataFrame in correct column order
    input_df = pd.DataFrame([employee_details])
    missing_cols = set(feature_columns) - set(input_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required feature(s): {sorted(missing_cols)}")

    input_df = input_df[feature_columns]

    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    result = {
        "prediction_code": prediction,
        "prediction_label": labels[prediction],
        "attrition_probability": round(probability, 4),
        "stay_probability": round(1 - probability, 4),
        "message": (
            f"Employee is likely to {labels[prediction]} "
            f"(Attrition probability: {probability:.1%})."
        ),
    }
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run the complete employee attrition prediction pipeline."""
    print("\n" + "=" * 70)
    print(" EMPLOYEE ATTRITION PREDICTION - ML PIPELINE")
    print("=" * 70)

    # Step 1: Load data
    df = load_and_report_dataset(DATASET_PATH)

    # Step 2: Identify target (required before EDA on class distribution)
    target_col = identify_target_column(df)
    report_target_column(target_col)

    # Step 3: EDA
    perform_eda(df, target_col)

    # Step 4: Preprocess
    missing_features = [column for column in API_FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(f"Victim feature columns missing from dataset: {missing_features}")
    X = df[API_FEATURE_COLUMNS].copy()
    target_mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0}
    y = df[target_col].map(target_mapping)
    if y.isnull().any():
        raise ValueError(f"Unrecognized target values in column '{target_col}'.")
    y = y.astype(int)
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
    report_preprocessing(X, numeric_features, categorical_features)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Step 5: Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 6: Train & tune
    tuned_models = train_and_tune_models(preprocessor, X_train, y_train)

    # Step 7: Evaluate
    print(f"\n{'=' * 70}")
    print("SECTION 8: MODEL EVALUATION (TEST SET)")
    print(f"{'=' * 70}")

    all_metrics = []
    probas_dict = {}
    victim_metrics = None
    victim_predictions = None

    for name, grid_search in tuned_models.items():
        metrics, y_pred, y_proba = evaluate_model(
            grid_search.best_estimator_, X_test, y_test, name
        )
        print(f"\n--- {name}: Evaluation on Test Set ---")
        for metric, value in metrics.items():
            if metric != "Model":
                print(f"  {metric:<12}: {value:.4f}")
        print_evaluation_report(y_test, y_pred, name)
        all_metrics.append(metrics)
        probas_dict[name] = y_proba
        if name == "Logistic Regression":
            victim_metrics = metrics
            victim_predictions = y_pred

    if victim_metrics is None or victim_predictions is None:
        raise RuntimeError("Victim Logistic Regression evaluation did not produce predictions.")
    save_victim_evaluation(y_test, victim_predictions, victim_metrics)

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(METRICS_PATH, index=False)
    plot_roc_curves(y_test, probas_dict)

    # Step 8: Select best model
    best_model_name = select_best_model(comparison_df)
    best_model = tuned_models[best_model_name]

    # Step 9: Feature Importance / Sensitivity Analysis (PPT Slide 15)
    display_feature_importance(best_model)

    # Step 10: Save the Logistic Regression target model
    save_best_model(best_model, best_model_name, feature_columns=X.columns.tolist())

    # Step 11: Demo prediction
    print(f"\n{'=' * 70}")
    print("SECTION 12: SAMPLE PREDICTION")
    print(f"{'=' * 70}")

    sample_employee = X_test.iloc[0].to_dict()
    prediction_result = predict_employee_attrition(sample_employee)
    print("\nSample employee prediction:")
    for key, value in prediction_result.items():
        print(f"  {key}: {value}")

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
