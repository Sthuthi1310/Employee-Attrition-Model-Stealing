"""Classification metrics utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def evaluate_model(model, X_test, y_test, model_name: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute classification metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_pred, y_proba, model_name)
    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_test, y_pred, zero_division=0)
    metrics["F1-Score"] = f1_score(y_test, y_pred, zero_division=0)
    metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)

    return metrics, y_pred, y_proba


def print_evaluation_report(y_test, y_pred: np.ndarray, model_name: str) -> None:
    """Print confusion matrix and classification report."""
    print(f"\n--- {model_name}: Evaluation on Test Set ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n  {cm}")
    print(f"  (Rows: Actual | Columns: Predicted | [0=Stay, 1=Leave])")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stay (0)", "Leave (1)"]))
