"""
Analysis Data Service
=====================
Aggregates model stealing experiment results from CSV/JSON files and
computes dynamic metrics for the Model Stealing Analysis Dashboard.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DIR.parent

ORIGINAL_MODEL_PATH = PROJECT_DIR / "outputs" / "best_attrition_model.joblib"
ORIGINAL_FEATURE_IMPORTANCE = PROJECT_DIR / "outputs" / "feature_importance.csv"
SUBSTITUTE_MODEL_PATH = PROJECT_DIR / "attack" / "surrogate" / "substitute_model.joblib"
SURROGATE_OUTPUT_DIR = PROJECT_DIR / "attack" / "surrogate" / "outputs"
STOLEN_DATASET_PATH = PROJECT_DIR / "attack" / "stolen_dataset.csv"
ATTACK_SUMMARY_JSON = PROJECT_DIR / "attack" / "attack_summary.json"
ATTACK_LOGS_DIR = PROJECT_DIR / "attack" / "logs"
TRAINING_DATASET_PATH = Path(
    r"C:\Users\Sthuthi Sheela\Downloads\Employee-Attrition\employee_attrition_dataset_10000.csv"
)

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Department",
    "Job_Role",
    "Monthly_Income",
    "Years_at_Company",
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Overtime",
    "Number_of_Companies_Worked",
]

TARGET_MAPPING = {"Leave": 1, "Stay": 0}


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read CSV if it exists, else return None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _get_original_model_name() -> str:
    if ORIGINAL_MODEL_PATH.exists():
        artifact = joblib.load(ORIGINAL_MODEL_PATH)
        return artifact.get("model_name", "Unknown")
    return "Not Available"


def _get_substitute_model_name() -> str:
    summary = _read_csv_safe(SURROGATE_OUTPUT_DIR / "attack_summary.csv")
    if summary is not None and not summary.empty:
        return summary.iloc[0].get("best_substitute_model", "Unknown")
    if SUBSTITUTE_MODEL_PATH.exists():
        artifact = joblib.load(SUBSTITUTE_MODEL_PATH)
        return artifact.get("model_name", "Unknown")
    return "Not Available"


def _compute_original_feature_defaults(feature_columns: list[str]) -> dict:
    """Default values for features not exposed via the API."""
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
    return defaults


def _build_full_features(row: pd.Series, feature_columns: list[str], defaults: dict) -> pd.DataFrame:
    """Build full feature row for the original victim model."""
    record = {**defaults, **{col: row[col] for col in FEATURE_COLUMNS}}
    return pd.DataFrame([record])[feature_columns]


def _parse_attack_log_stats() -> dict:
    """Parse the most recent attack log for query statistics."""
    stats = {
        "total_queries_sent": 0,
        "successful_queries": 0,
        "failed_queries": 0,
        "elapsed_seconds": 0.0,
        "average_response_time_ms": 0.0,
    }

    if not ATTACK_LOGS_DIR.exists():
        return stats

    log_files = sorted(ATTACK_LOGS_DIR.glob("attack_*.log"), key=lambda p: p.stat().st_mtime)
    if not log_files:
        return stats

    content = log_files[-1].read_text(encoding="utf-8")

    patterns = {
        "total_queries_sent": r"Total queries sent\s*:\s*([\d,]+)",
        "successful_queries": r"Successful responses\s*:\s*([\d,]+)",
        "failed_queries": r"Failed responses\s*:\s*([\d,]+)",
        "elapsed_seconds": r"Elapsed time\s*:\s*([\d.]+)\s*seconds",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1).replace(",", "")
            stats[key] = float(value) if "." in value else int(value)

    if stats["total_queries_sent"] > 0 and stats["elapsed_seconds"] > 0:
        stats["average_response_time_ms"] = round(
            (stats["elapsed_seconds"] / stats["total_queries_sent"]) * 1000, 2
        )

    return stats


def _compute_response_time_from_dataset(df: pd.DataFrame) -> float:
    """Estimate avg response time from consecutive query timestamps."""
    if "timestamp" not in df.columns or len(df) < 2:
        return 0.0

    try:
        times = pd.to_datetime(df["timestamp"], utc=True)
        diffs = times.diff().dropna().dt.total_seconds()
        diffs = diffs[(diffs > 0) & (diffs < 30)]  # Filter outliers
        return round(float(diffs.mean()) * 1000, 2) if len(diffs) else 0.0
    except Exception:
        return 0.0


def _extract_substitute_feature_importance() -> list[dict]:
    """Extract top feature importances from the substitute model pipeline."""
    if not SUBSTITUTE_MODEL_PATH.exists():
        print("[feature_importance] Substitute model artifact not found.")
        return []

    artifact = joblib.load(SUBSTITUTE_MODEL_PATH)
    if "pipeline" not in artifact:
        print("[feature_importance] Substitute model artifact missing pipeline.")
        return []

    pipeline = artifact["pipeline"]
    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if classifier is None or preprocessor is None:
        print("[feature_importance] Substitute pipeline missing classifier or preprocessor.")
        return []

    model_type = type(classifier).__name__
    print(f"[feature_importance] Substitute model type: {model_type}")

    try:
        names = preprocessor.get_feature_names_out()
    except Exception as exc:
        print(f"[feature_importance] Failed to extract feature names: {exc}")
        return []

    importances = None
    importance_source = None

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_)
        importance_source = "feature_importances_"
    elif hasattr(classifier, "coef_"):
        coef = np.asarray(classifier.coef_)
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
        importance_source = "coef_"
    else:
        df = _read_csv_safe(STOLEN_DATASET_PATH)
        if df is None or df.empty:
            print("[feature_importance] No stolen dataset available for permutation importance fallback.")
            importances = np.zeros(len(names), dtype=float)
        else:
            if "prediction_label" not in df.columns:
                print("[feature_importance] Stolen dataset missing 'prediction_label' column for permutation importance.")
                importances = np.zeros(len(names), dtype=float)
            else:
                try:
                    X = df[FEATURE_COLUMNS]
                    y = df["prediction_label"].map(TARGET_MAPPING)
                    if y.isnull().any():
                        missing = df.loc[y.isnull(), "prediction_label"].unique().tolist()
                        print(
                            f"[feature_importance] Unknown prediction labels in stolen dataset: {missing}"
                        )
                        importances = np.zeros(len(names), dtype=float)
                    else:
                        y = y.astype(int)
                        perm_result = permutation_importance(
                            pipeline,
                            X,
                            y,
                            n_repeats=10,
                            random_state=42,
                            n_jobs=-1,
                        )
                        importances = perm_result.importances_mean
                        importance_source = "permutation_importance"
                except Exception as exc:
                    print(f"[feature_importance] Permutation importance failed: {exc}")
                    importances = np.zeros(len(names), dtype=float)

    if importances is None:
        print("[feature_importance] Unable to compute substitute importances.")
        return []

    if len(names) != len(importances):
        print(
            f"[feature_importance] Feature count mismatch: {len(names)} names vs {len(importances)} importance values"
        )
        min_len = min(len(names), len(importances))
        names = names[:min_len]
        importances = importances[:min_len]

    print(f"[feature_importance] Feature names count: {len(names)}")
    print(f"[feature_importance] Importance values count: {len(importances)}")
    print(f"[feature_importance] Importance source: {importance_source}")
    print(f"[feature_importance] Importance values: {importances.tolist()}")

    df = (
        pd.DataFrame({"Feature": names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    return [
        {"feature": _clean_feature_name(row["Feature"]), "importance": round(float(row["Importance"]), 4)}
        for _, row in df.iterrows()
    ]


def _clean_feature_name(name: str) -> str:
    """Convert encoded feature names to readable labels."""
    name = name.replace("num__", "").replace("cat__", "")
    return name.replace("_", " ")


def _load_original_feature_importance() -> list[dict]:
    df = _read_csv_safe(ORIGINAL_FEATURE_IMPORTANCE)
    if df is None:
        return []

    df = df.head(10)
    return [
        {
            "feature": _clean_feature_name(row["Feature"]),
            "importance": round(float(row["Importance"]), 4),
        }
        for _, row in df.iterrows()
    ]


def _compute_live_agreement(df: pd.DataFrame) -> dict:
    """
    Compute agreement metrics by running both models on stolen inputs.
    Used when detailed per-query comparison is needed.
    """
    if df.empty or not SUBSTITUTE_MODEL_PATH.exists() or not ORIGINAL_MODEL_PATH.exists():
        return {
            "agreement_rate": 0.0,
            "prediction_similarity": 0.0,
            "matching_predictions": 0,
            "different_predictions": 0,
            "query_progression": [],
        }

    sub_artifact = joblib.load(SUBSTITUTE_MODEL_PATH)
    orig_artifact = joblib.load(ORIGINAL_MODEL_PATH)
    sub_pipeline = sub_artifact["pipeline"]
    orig_pipeline = orig_artifact["pipeline"]
    orig_features = orig_artifact["feature_columns"]
    defaults = _compute_original_feature_defaults(orig_features)

    X = df[FEATURE_COLUMNS]
    sub_preds = sub_pipeline.predict(X)
    sub_proba = sub_pipeline.predict_proba(X)[:, 1]

    orig_preds = []
    orig_proba = []
    for _, row in X.iterrows():
        full_df = _build_full_features(row, orig_features, defaults)
        orig_preds.append(int(orig_pipeline.predict(full_df)[0]))
        orig_proba.append(float(orig_pipeline.predict_proba(full_df)[0, 1]))

    orig_preds = np.array(orig_preds)
    orig_proba = np.array(orig_proba)

    matches = sub_preds == orig_preds
    matching = int(matches.sum())
    different = int(len(matches) - matching)
    agreement_rate = round(float(matches.mean()) * 100, 2)

    # Probability similarity: 100% = identical probabilities
    similarity = round(float(1 - np.mean(np.abs(sub_proba - orig_proba))) * 100, 2)

    # Cumulative agreement progression for line chart
    progression = []
    for n in range(1, len(df) + 1):
        cum_agree = float((sub_preds[:n] == orig_preds[:n]).mean()) * 100
        progression.append({"queries": n, "agreement_rate": round(cum_agree, 2)})

    return {
        "agreement_rate": agreement_rate,
        "prediction_similarity": similarity,
        "matching_predictions": matching,
        "different_predictions": different,
        "query_progression": progression,
    }


def _generate_conclusions(
    agreement_rate: float,
    similarity: float,
    original_metrics: dict,
    substitute_metrics: dict,
) -> dict:
    """Generate automated research conclusions from experiment results."""
    acc_diff = abs(original_metrics.get("Accuracy", 0) - substitute_metrics.get("Accuracy", 0))

    if agreement_rate >= 85 and similarity >= 80:
        status = "Successful"
        status_class = "success"
        interpretation = (
            f"The substitute model achieved {agreement_rate}% agreement and {similarity}% "
            "prediction similarity with the victim model, indicating a highly effective "
            "black-box model extraction attack."
        )
    elif agreement_rate >= 65:
        status = "Partially Successful"
        status_class = "warning"
        interpretation = (
            f"With {agreement_rate}% agreement, the adversary partially replicated victim "
            "model behavior. Additional queries would likely improve surrogate fidelity."
        )
    else:
        status = "Limited Success"
        status_class = "danger"
        interpretation = (
            f"Agreement rate of {agreement_rate}% suggests limited model extraction. "
            "The attack captured some decision patterns but failed to fully clone the victim."
        )

    performance_gap = (
        f"The substitute model trails the original by {acc_diff:.2%} accuracy on stolen "
        f"teacher labels, with F1 gap of "
        f"{abs(original_metrics.get('F1-Score', 0) - substitute_metrics.get('F1-Score', 0)):.2%}."
    )

    return {
        "stealing_status": status,
        "status_class": status_class,
        "agreement_interpretation": interpretation,
        "performance_difference": performance_gap,
        "recommendation": (
            "Deploy API rate limiting, query monitoring, and prediction perturbation "
            "to mitigate model stealing attacks on production ML endpoints."
        ),
    }


def _load_attack_summary_json() -> dict | None:
    """Load attack_summary.json written by collect_responses.py."""
    if not ATTACK_SUMMARY_JSON.exists():
        return None
    try:
        return json.loads(ATTACK_SUMMARY_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_analysis_payload() -> dict:
    """
    Build the complete JSON payload for the Model Stealing Analysis Dashboard.
    Reads from CSV files and computes dynamic metrics from saved models.
    """
    # --- Load CSV artifacts ---
    attack_summary = _read_csv_safe(SURROGATE_OUTPUT_DIR / "attack_summary.csv")
    model_comparison = _read_csv_safe(SURROGATE_OUTPUT_DIR / "model_comparison.csv")
    substitute_metrics = _read_csv_safe(SURROGATE_OUTPUT_DIR / "substitute_metrics.csv")
    stolen_df = _read_csv_safe(STOLEN_DATASET_PATH)

    if stolen_df is not None and "success" in stolen_df.columns:
        stolen_df = stolen_df[stolen_df["success"] == True].copy()  # noqa: E712

    # --- Overview ---
    original_name = _get_original_model_name()
    substitute_name = _get_substitute_model_name()
    stolen_size = len(stolen_df) if stolen_df is not None else 0

    attack_json = _load_attack_summary_json()
    attack_stats = _parse_attack_log_stats()

    # Prefer attack_summary.json (most accurate for large runs)
    if attack_json:
        attack_stats["total_queries_sent"] = attack_json.get("total_queries", 0)
        attack_stats["successful_queries"] = attack_json.get("successful_queries", 0)
        attack_stats["failed_queries"] = attack_json.get("failed_queries", 0)
        attack_stats["average_response_time_ms"] = attack_json.get("average_response_time_ms", 0)

    if stolen_df is not None and "response_time_ms" in stolen_df.columns:
        rt_vals = stolen_df["response_time_ms"].dropna()
        if len(rt_vals) > 0:
            attack_stats["average_response_time_ms"] = round(float(rt_vals.mean()), 2)

    if stolen_size > 0 and attack_stats["total_queries_sent"] == 0:
        attack_stats["total_queries_sent"] = stolen_size
        attack_stats["successful_queries"] = stolen_size

    avg_rt = attack_stats.get("average_response_time_ms", 0)
    if avg_rt == 0 and stolen_df is not None:
        avg_rt = _compute_response_time_from_dataset(stolen_df)

    overview = {
        "original_model_name": original_name,
        "substitute_model_name": substitute_name,
        "total_attack_queries": attack_stats["total_queries_sent"] or stolen_size,
        "stolen_dataset_size": stolen_size,
    }

    # --- Performance metrics ---
    original_perf = {"Model": "Original"}
    substitute_perf = {"Model": "Substitute"}

    if model_comparison is not None and not model_comparison.empty:
        for _, row in model_comparison.iterrows():
            metrics = {
                "Accuracy": float(row["Accuracy"]),
                "Precision": float(row["Precision"]),
                "Recall": float(row["Recall"]),
                "F1-Score": float(row["F1-Score"]),
                "ROC-AUC": float(row["ROC-AUC"]),
            }
            if "Substitute" in str(row["Model"]):
                substitute_perf.update(metrics)
            elif "Original" in str(row["Model"]):
                original_perf.update(metrics)

    # --- Agreement analysis ---
    agreement_data = _compute_live_agreement(stolen_df) if stolen_df is not None else {}

    if attack_summary is not None and not attack_summary.empty:
        csv_agreement = float(attack_summary.iloc[0].get("agreement_rate_pct", 0))
        if agreement_data.get("agreement_rate", 0) == 0 and csv_agreement > 0:
            agreement_data["agreement_rate"] = csv_agreement

    agreement = {
        "agreement_rate": agreement_data.get("agreement_rate", 0),
        "prediction_similarity": agreement_data.get("prediction_similarity", 0),
        "matching_predictions": agreement_data.get("matching_predictions", 0),
        "different_predictions": agreement_data.get("different_predictions", 0),
        "query_progression": agreement_data.get("query_progression", []),
    }

    # --- Class distribution ---
    class_distribution = {"original_labels": {}, "substitute_labels": {}}
    if stolen_df is not None and "prediction_label" in stolen_df.columns:
        counts = stolen_df["prediction_label"].value_counts()
        class_distribution["original_labels"] = {
            "Stay": int(counts.get("Stay", 0)),
            "Leave": int(counts.get("Leave", 0)),
        }

    if stolen_df is not None and SUBSTITUTE_MODEL_PATH.exists():
        sub_artifact = joblib.load(SUBSTITUTE_MODEL_PATH)
        preds = sub_artifact["pipeline"].predict(stolen_df[FEATURE_COLUMNS])
        labels = sub_artifact.get("target_mapping", {0: "Stay", 1: "Leave"})
        inv_map = {v: k for k, v in labels.items()} if isinstance(list(labels.keys())[0], int) else {}
        stay = int((preds == 0).sum())
        leave = int((preds == 1).sum())
        class_distribution["substitute_labels"] = {"Stay": stay, "Leave": leave}

    # --- Feature importance ---
    substitute_feature_importance = _extract_substitute_feature_importance()
    feature_importance = {
        "original": _load_original_feature_importance(),
        "substitute": substitute_feature_importance,
    }

    substitute_feature_names = [item["feature"] for item in substitute_feature_importance]
    substitute_feature_values = [item["importance"] for item in substitute_feature_importance]

    # Cache-bust image URLs so dashboard refreshes after retraining
    cache_bust = int(datetime.utcnow().timestamp())
    confusion_matrices = {
        "original": f"/analysis/assets/confusion_matrix_original.png?v={cache_bust}",
        "substitute": f"/analysis/assets/confusion_matrix_substitute.png?v={cache_bust}",
    }

    # --- Attack analytics ---
    attack_analytics = {
        "total_api_queries": attack_stats["total_queries_sent"] or stolen_size,
        "successful_queries": attack_stats["successful_queries"] or stolen_size,
        "failed_queries": attack_stats["failed_queries"],
        "average_response_time_ms": avg_rt,
        "success_rate_pct": round(
            (attack_stats["successful_queries"] / attack_stats["total_queries_sent"]) * 100, 2
        )
        if attack_stats["total_queries_sent"] > 0
        else 100.0,
    }

    # --- Conclusions ---
    conclusions = _generate_conclusions(
        agreement["agreement_rate"],
        agreement["prediction_similarity"],
        original_perf,
        substitute_perf,
    )

    # --- Substitute candidates table ---
    candidates = []
    if substitute_metrics is not None:
        candidates = substitute_metrics.to_dict(orient="records")

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "overview": overview,
        "performance": {
            "original": original_perf,
            "substitute": substitute_perf,
        },
        "agreement": agreement,
        "class_distribution": class_distribution,
        "feature_importance": feature_importance,
        "substitute_features": substitute_feature_names,
        "substitute_importances": substitute_feature_values,
        "confusion_matrices": confusion_matrices,
        "attack_analytics": attack_analytics,
        "conclusions": conclusions,
        "substitute_candidates": candidates,
    }

    return payload


def export_analysis_json(output_path: Path | None = None) -> Path:
    """Export analysis payload to JSON file for caching."""
    payload = build_analysis_payload()
    path = output_path or (SURROGATE_OUTPUT_DIR / "analysis_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
