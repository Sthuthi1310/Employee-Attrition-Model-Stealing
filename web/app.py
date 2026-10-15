"""
Employee Attrition Predictor - Flask Backend API
================================================
Serves the web dashboard and handles ML predictions using the saved pipeline.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from utils.helper import API_FEATURE_COLUMNS
from utils.paths import (
    ATTACK_RESULTS_DIR,
    PREDICTIONS_FILE,
    RESULTS_DIR,
    TARGET_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = TARGET_MODEL_PATH
SURROGATE_OUTPUT_DIR = ATTACK_RESULTS_DIR

# Form fields exposed in the UI (subset of model features)
FORM_FIELDS = list(API_FEATURE_COLUMNS)

VALID_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Department": ["Finance", "HR", "IT", "Marketing", "Sales"],
    "Job_Role": ["Analyst", "Assistant", "Executive", "Manager"],
    "Overtime": ["No", "Yes"],
}

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
artifact: dict | None = None
feature_defaults: dict = {}


def load_model() -> dict:
    """Load the trained model artifact from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run employee_attrition_pipeline.py first."
        )
    return joblib.load(MODEL_PATH)


def ensure_predictions_store() -> None:
    """Create predictions JSON store if it does not exist."""
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS_FILE.exists():
        PREDICTIONS_FILE.write_text("[]", encoding="utf-8")


def read_predictions() -> list[dict]:
    """Read all stored predictions from JSON file."""
    ensure_predictions_store()
    try:
        return json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_prediction(record: dict) -> None:
    """Append a prediction record to persistent storage."""
    predictions = read_predictions()
    predictions.insert(0, record)
    predictions = predictions[:100]  # Keep last 100 for dashboard
    PREDICTIONS_FILE.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def validate_payload(data: dict) -> tuple[dict | None, str | None]:
    """
    Validate incoming form data.
    Returns (cleaned_data, error_message).
    """
    if not data:
        return None, "Request body is required."

    cleaned: dict = {}

    # Age
    try:
        age = int(data.get("Age"))
        if age < 18 or age > 70:
            return None, "Age must be between 18 and 70."
        cleaned["Age"] = age
    except (TypeError, ValueError):
        return None, "Age must be a valid number."

    # Categorical fields
    for field, options in VALID_OPTIONS.items():
        value = data.get(field, "").strip() if isinstance(data.get(field), str) else data.get(field)
        if value not in options:
            return None, f"{field.replace('_', ' ')} must be one of: {', '.join(options)}."
        cleaned[field] = value

    # Monthly Income
    try:
        income = int(data.get("Monthly_Income"))
        if income < 1000 or income > 25000:
            return None, "Monthly Income must be between 1,000 and 25,000."
        cleaned["Monthly_Income"] = income
    except (TypeError, ValueError):
        return None, "Monthly Income must be a valid number."

    # Years at Company
    try:
        years = int(data.get("Years_at_Company"))
        if years < 0 or years > 40:
            return None, "Years at Company must be between 0 and 40."
        cleaned["Years_at_Company"] = years
    except (TypeError, ValueError):
        return None, "Years at Company must be a valid number."

    # Rating domains match the values present in the training dataset.
    for field in ["Job_Satisfaction", "Work_Life_Balance"]:
        try:
            rating = int(data.get(field))
            if rating < 1 or rating > 5:
                return None, f"{field.replace('_', ' ')} must be between 1 and 5."
            cleaned[field] = rating
        except (TypeError, ValueError):
            return None, f"{field.replace('_', ' ')} must be a valid number."

    # Number of Companies Worked
    try:
        companies = int(data.get("Number_of_Companies_Worked"))
        if companies < 0 or companies > 10:
            return None, "Number of Companies Worked must be between 0 and 10."
        cleaned["Number_of_Companies_Worked"] = companies
    except (TypeError, ValueError):
        return None, "Number of Companies Worked must be a valid number."

    return cleaned, None


def build_feature_vector(form_data: dict) -> pd.DataFrame:
    """Build an input row using the exact victim/API feature schema."""
    global artifact
    feature_columns = artifact["feature_columns"]
    missing = set(feature_columns) - set(form_data)
    if missing:
        raise ValueError(f"Missing required feature(s): {sorted(missing)}")
    return pd.DataFrame([form_data])[feature_columns]


def run_prediction(form_data: dict) -> dict:
    """Execute prediction using the saved sklearn pipeline."""
    global artifact

    pipeline = artifact["pipeline"]
    labels = artifact["prediction_labels"]
    input_df = build_feature_vector(form_data)

    prediction_code = int(pipeline.predict(input_df)[0])
    label = labels[prediction_code]
    classes = pipeline.named_steps["classifier"].classes_
    probabilities = pipeline.predict_proba(input_df)[0]
    class_probabilities = {
        labels[int(class_code)]: round(float(probability), 6)
        for class_code, probability in zip(classes, probabilities)
    }
    attrition_prob = class_probabilities["Leave"]
    stay_prob = class_probabilities["Stay"]
    confidence = attrition_prob if prediction_code == 1 else stay_prob

    model_name = artifact.get("model_name", "Logistic Regression")
    attrition_binary = "Yes" if prediction_code == 1 else "No"

    return {
        "attrition": attrition_binary,
        "prediction_label": label,
        "class_probabilities": class_probabilities,
        "attrition_status": "Likely to Leave" if label == "Leave" else "Likely to Stay",
        "attrition_probability": round(attrition_prob * 100, 2),
        "stay_probability": round(stay_prob * 100, 2),
        "prediction_confidence": round(confidence * 100, 2),
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": artifact is not None})


@app.route("/api/options", methods=["GET"])
def get_options():
    """Return valid categorical options for form dropdowns."""
    return jsonify(VALID_OPTIONS)


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept employee details via POST JSON and return attrition prediction.
    Exposed at both /predict and /api/predict.
    """
    if artifact is None:
        return jsonify({"error": "Model is not loaded."}), 503

    data = request.get_json(silent=True)
    cleaned, error = validate_payload(data or {})

    if error:
        return jsonify({"error": error}), 400

    try:
        result = run_prediction(cleaned)

        record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "department": cleaned["Department"],
            "job_role": cleaned["Job_Role"],
            "attrition_status": result["attrition_status"],
            "attrition_risk": result["attrition_probability"],
            "confidence": result["prediction_confidence"],
        }
        save_prediction(record)
        result["record_id"] = record["id"]

        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Return dashboard statistics:
    - Total predictions
    - Average attrition risk
    - Recent predictions table data
    """
    predictions = read_predictions()
    total = len(predictions)
    avg_risk = (
        round(sum(p["attrition_risk"] for p in predictions) / total, 2) if total else 0.0
    )

    recent = [
        {
            "id": p["id"],
            "timestamp": p["timestamp"],
            "department": p["department"],
            "job_role": p["job_role"],
            "status": p["attrition_status"],
            "risk": p["attrition_risk"],
            "confidence": p["confidence"],
        }
        for p in predictions[:10]
    ]

    return jsonify(
        {
            "total_predictions": total,
            "average_attrition_risk": avg_risk,
            "recent_predictions": recent,
        }
    )


@app.route("/analysis")
def analysis_dashboard():
    """Serve the Model Stealing Analysis Dashboard."""
    return render_template("analysis.html")


@app.route("/api/analysis", methods=["GET"])
def analysis_api():
    """
    Return aggregated model stealing experiment metrics as JSON.
    Data is read dynamically from CSV files and computed from saved models.
    """
    try:
        try:
            from analysis_data import build_analysis_payload
        except ModuleNotFoundError:
            from web.analysis_data import build_analysis_payload

        payload = build_analysis_payload()
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": f"Failed to build analysis data: {str(exc)}"}), 500


@app.route("/analysis/assets/<path:filename>")
def analysis_assets(filename):
    """Serve confusion matrix and other analysis images from surrogate outputs."""
    if filename == "victim_confusion_matrix.png":
        return send_from_directory(RESULTS_DIR, filename)
    return send_from_directory(SURROGATE_OUTPUT_DIR, filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def init_app():
    """Initialize the trained victim model on startup."""
    global artifact
    artifact = load_model()
    if artifact.get("feature_columns") != API_FEATURE_COLUMNS:
        raise ValueError("Victim model feature schema does not match the prediction API schema.")
    ensure_predictions_store()
    print("Victim model loaded.")
    print(f"Features: {len(artifact['feature_columns'])}")


# Initialize model when the module loads (works with Flask dev server)
with app.app_context():
    try:
        init_app()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
