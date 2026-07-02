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
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --- Configuration ---
DATASET_PATH = Path(
    r"C:\Users\Sthuthi Sheela\Downloads\Employee-Attrition\employee_attrition_dataset_10000.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_attrition_model.joblib"
METRICS_PATH = OUTPUT_DIR / "model_comparison.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Possible target column names (auto-detection)
TARGET_CANDIDATES = ["attrition", "employee attrition", "left", "churn", "turnover"]

# Columns that should never be used as predictive features
ID_COLUMNS = ["employee_id", "emp_id", "id"]


# =============================================================================
# SECTION 2: LOAD DATASET
# =============================================================================
# Explanation:
# The dataset is loaded using pandas.read_csv(). Pandas provides efficient
# tabular data structures (DataFrame) for exploration and preprocessing.


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the employee attrition dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
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


# =============================================================================
# SECTION 3: AUTOMATIC TARGET COLUMN IDENTIFICATION
# =============================================================================
# Explanation:
# Real-world datasets may use different names for the target variable.
# We search column names against known attrition-related keywords.


def identify_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the attrition target column."""
    print(f"\n{'=' * 70}")
    print("SECTION 3: TARGET COLUMN IDENTIFICATION")
    print(f"{'=' * 70}")

    normalized_cols = {col: col.strip().lower().replace(" ", "_") for col in df.columns}

    for original, normalized in normalized_cols.items():
        if normalized in TARGET_CANDIDATES or "attrition" in normalized:
            print(f"Target column identified: '{original}'")
            return original

    raise ValueError(
        "Could not automatically identify target column. "
        f"Expected one of: {TARGET_CANDIDATES}"
    )


# =============================================================================
# SECTION 5: DATA PREPROCESSING
# =============================================================================
# Explanation:
# Preprocessing steps:
# 1. Remove identifier columns (Employee_ID) to prevent meaningless patterns.
# 2. Encode target labels (Yes/No -> 1/0).
# 3. Separate numerical and categorical features.
# 4. Build a ColumnTransformer that scales numeric features and one-hot
#    encodes categorical features.
#
# IMPORTANT (Data Leakage Prevention):
# All transformers are wrapped inside sklearn Pipeline and fitted ONLY on
# training data during model training. Test data is never used for fitting.


def preprocess_features(df: pd.DataFrame, target_col: str):
    """
    Prepare feature matrix X and encoded target vector y.
    Returns feature names list and column type groups for the pipeline.
    """
    print(f"\n{'=' * 70}")
    print("SECTION 5: DATA PREPROCESSING")
    print(f"{'=' * 70}")

    data = df.copy()

    # Handle missing values (if any appear in future data)
    missing_before = data.isnull().sum().sum()
    if missing_before > 0:
        print(f"Handling {missing_before} missing values...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            if col != target_col and data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)

        for col in categorical_cols:
            if col != target_col and data[col].isnull().any():
                data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        print("No missing values to impute.")

    # Encode target: Yes -> 1 (leave), No -> 0 (stay)
    target_mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0}
    y = data[target_col].map(target_mapping)

    if y.isnull().any():
        raise ValueError(f"Unrecognized target values in column '{target_col}'.")

    # Drop ID and target columns from features
    drop_cols = [target_col]
    for col in data.columns:
        if col.strip().lower().replace(" ", "_") in ID_COLUMNS:
            drop_cols.append(col)

    X = data.drop(columns=drop_cols)

    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Features used for modeling: {X.shape[1]}")
    print(f"  - Numerical features ({len(numeric_features)}): {numeric_features}")
    print(f"  - Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Target encoding: Yes/1 = Employee likely to leave | No/0 = Employee likely to stay")

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Create preprocessing pipeline: scale numeric, encode categorical."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


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
    """
    Train Logistic Regression (GridSearchCV) and Random Forest (RandomizedSearchCV).

    Logistic Regression uses a compact grid; Random Forest uses randomized search
    to explore a wider hyperparameter space efficiently without overfitting the
    validation strategy itself.
    """
    print(f"\n{'=' * 70}")
    print("SECTION 7: MODEL TRAINING & HYPERPARAMETER TUNING")
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
                    class_weight="balanced",
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    search_configs = [
        {
            "name": "Logistic Regression",
            "search": GridSearchCV(
                estimator=lr_pipeline,
                param_grid={
                    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "classifier__penalty": ["l2"],
                    "classifier__solver": ["lbfgs", "liblinear"],
                },
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                refit=True,
            ),
        },
        {
            "name": "Random Forest",
            "search": RandomizedSearchCV(
                estimator=rf_pipeline,
                param_distributions={
                    "classifier__n_estimators": [100, 200, 300, 500],
                    "classifier__max_depth": [None, 8, 12, 16, 24, 32],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                    "classifier__max_features": ["sqrt", "log2", None],
                },
                n_iter=40,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                refit=True,
            ),
        },
    ]

    tuned_models = {}

    for config in search_configs:
        name = config["name"]
        search = config["search"]
        search_type = type(search).__name__
        print(f"\nTuning {name} using {search_type}...")
        search.fit(X_train, y_train)
        tuned_models[name] = search

        print(f"  Best F1 (CV): {search.best_score_:.4f}")
        print(f"  Best parameters: {search.best_params_}")

    return tuned_models


# =============================================================================
# SECTION 8: MODEL EVALUATION
# =============================================================================
# Explanation:
# Multiple metrics are reported because accuracy alone can be misleading on
# imbalanced data. Precision, Recall, F1, ROC-AUC, and Confusion Matrix
# provide a comprehensive view of model behavior.


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute classification metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {model_name}: Evaluation on Test Set ---")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"  {metric:<12}: {value:.4f}")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {cm}")
    print(f"  (Rows: Actual | Columns: Predicted | [0=Stay, 1=Leave])")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stay (0)", "Leave (1)"]))

    return metrics, y_pred, y_proba


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


def display_feature_importance(rf_model, top_n: int = 15) -> pd.DataFrame:
    """Extract and visualize top feature importances from Random Forest."""
    print(f"\n{'=' * 70}")
    print("SECTION 10: FEATURE IMPORTANCE (RANDOM FOREST)")
    print(f"{'=' * 70}")

    best_pipeline = rf_model.best_estimator_
    classifier = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

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
    plt.title(f"Top {top_n} Feature Importances - Random Forest")
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
    df = load_dataset(DATASET_PATH)

    # Step 2: Identify target (required before EDA on class distribution)
    target_col = identify_target_column(df)

    # Step 3: EDA
    perform_eda(df, target_col)

    # Step 4: Preprocess
    X, y, numeric_features, categorical_features = preprocess_features(df, target_col)
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

    for name, grid_search in tuned_models.items():
        metrics, _, y_proba = evaluate_model(
            grid_search.best_estimator_, X_test, y_test, name
        )
        all_metrics.append(metrics)
        probas_dict[name] = y_proba

    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(METRICS_PATH, index=False)
    plot_roc_curves(y_test, probas_dict)

    # Step 8: Select best model
    best_model_name = select_best_model(comparison_df)
    best_model = tuned_models[best_model_name]

    # Step 9: Feature importance (Random Forest)
    if "Random Forest" in tuned_models:
        display_feature_importance(tuned_models["Random Forest"])

    # Step 10: Save best model
    save_best_model(best_model, best_model_name, feature_columns=X.columns.tolist())

    # Step 11: Demo prediction
    print(f"\n{'=' * 70}")
    print("SECTION 12: SAMPLE PREDICTION")
    print(f"{'=' * 70}")

    sample_employee = X.iloc[0].to_dict()
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
