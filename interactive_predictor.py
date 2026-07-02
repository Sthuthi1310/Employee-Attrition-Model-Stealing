"""
Interactive Employee Attrition Prediction System
================================================
Console-based application that loads the trained model and collects
employee details from the user to predict attrition risk.

Usage:
    python interactive_predictor.py

Prerequisites:
    Run employee_attrition_pipeline.py first to generate:
        outputs/best_attrition_model.joblib
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "best_attrition_model.joblib"

# Human-readable descriptions shown during data entry (for project demo)
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "Age": "Employee age in years",
    "Gender": "Employee gender",
    "Marital_Status": "Marital status",
    "Department": "Department name",
    "Job_Role": "Job role / designation",
    "Job_Level": "Job level (typically 1-5)",
    "Monthly_Income": "Monthly salary income",
    "Hourly_Rate": "Hourly pay rate",
    "Years_at_Company": "Total years with the company",
    "Years_in_Current_Role": "Years in the current role",
    "Years_Since_Last_Promotion": "Years since last promotion",
    "Work_Life_Balance": "Work-life balance rating (1=Low, 5=High)",
    "Job_Satisfaction": "Job satisfaction rating (1=Low, 5=High)",
    "Performance_Rating": "Performance rating (1-4)",
    "Training_Hours_Last_Year": "Training hours completed last year",
    "Overtime": "Whether the employee works overtime (Yes/No)",
    "Project_Count": "Number of active projects",
    "Average_Hours_Worked_Per_Week": "Average weekly working hours",
    "Absenteeism": "Number of absent days (annual)",
    "Work_Environment_Satisfaction": "Work environment rating (1=Low, 5=High)",
    "Relationship_with_Manager": "Manager relationship rating (1=Low, 5=High)",
    "Job_Involvement": "Job involvement rating (1=Low, 5=High)",
    "Distance_From_Home": "Distance from home to office (km/miles)",
    "Number_of_Companies_Worked": "Number of previous companies worked at",
}


# =============================================================================
# MODULE 1: LOAD SAVED MODEL
# =============================================================================
# Explanation:
# The trained pipeline (preprocessing + classifier) is loaded from disk.
# Loading the full pipeline guarantees that inference uses the exact same
# scaling and encoding steps applied during model training.


def load_saved_model(model_path: Path = MODEL_PATH) -> dict:
    """
    Load the persisted model artifact.

    Returns
    -------
    dict
        Contains 'pipeline', 'feature_columns', 'model_name', and label maps.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at:\n  {model_path}\n\n"
            "Please run 'employee_attrition_pipeline.py' first to train and save the model."
        )

    artifact = joblib.load(model_path)
    required_keys = {"pipeline", "feature_columns", "prediction_labels"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(f"Invalid model artifact. Missing keys: {sorted(missing)}")

    return artifact


# =============================================================================
# MODULE 2: EXTRACT FEATURE SCHEMA FROM TRAINED PIPELINE
# =============================================================================
# Explanation:
# Categorical valid options are read directly from the fitted OneHotEncoder
# inside the saved pipeline. This avoids mismatch between training and inference.


def extract_categorical_options(artifact: dict) -> dict[str, list[str]]:
    """Read valid categorical values from the fitted preprocessing pipeline."""
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    categorical_features = preprocessor.transformers_[1][2]

    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categories = encoder.categories_

    return {
        feature: [str(option) for option in options]
        for feature, options in zip(categorical_features, categories)
    }


def build_feature_schema(artifact: dict) -> dict[str, dict[str, Any]]:
    """
    Build input schema combining numeric and categorical feature rules.

    Numeric ranges are practical validation bounds suitable for demo input.
    """
    categorical_options = extract_categorical_options(artifact)
    numeric_rules = {
        "Age": {"type": "int", "min": 18, "max": 70},
        "Job_Level": {"type": "int", "min": 1, "max": 5},
        "Monthly_Income": {"type": "int", "min": 1000, "max": 25000},
        "Hourly_Rate": {"type": "int", "min": 10, "max": 100},
        "Years_at_Company": {"type": "int", "min": 0, "max": 40},
        "Years_in_Current_Role": {"type": "int", "min": 0, "max": 40},
        "Years_Since_Last_Promotion": {"type": "int", "min": 0, "max": 20},
        "Work_Life_Balance": {"type": "int", "min": 1, "max": 5},
        "Job_Satisfaction": {"type": "int", "min": 1, "max": 5},
        "Performance_Rating": {"type": "int", "min": 1, "max": 4},
        "Training_Hours_Last_Year": {"type": "int", "min": 0, "max": 100},
        "Project_Count": {"type": "int", "min": 0, "max": 10},
        "Average_Hours_Worked_Per_Week": {"type": "int", "min": 20, "max": 70},
        "Absenteeism": {"type": "int", "min": 0, "max": 30},
        "Work_Environment_Satisfaction": {"type": "int", "min": 1, "max": 5},
        "Relationship_with_Manager": {"type": "int", "min": 1, "max": 5},
        "Job_Involvement": {"type": "int", "min": 1, "max": 5},
        "Distance_From_Home": {"type": "int", "min": 1, "max": 50},
        "Number_of_Companies_Worked": {"type": "int", "min": 0, "max": 10},
    }

    schema: dict[str, dict[str, Any]] = {}
    for feature in artifact["feature_columns"]:
        if feature in categorical_options:
            schema[feature] = {
                "kind": "categorical",
                "options": categorical_options[feature],
                "description": FEATURE_DESCRIPTIONS.get(feature, feature),
            }
        else:
            rules = numeric_rules.get(feature, {"type": "int", "min": 0, "max": 99999})
            schema[feature] = {
                "kind": "numeric",
                "value_type": rules["type"],
                "min": rules["min"],
                "max": rules["max"],
                "description": FEATURE_DESCRIPTIONS.get(feature, feature),
            }

    return schema


# =============================================================================
# MODULE 3: USER INPUT HANDLING
# =============================================================================
# Explanation:
# Separate prompt functions for numeric and categorical inputs keep the code
# modular and make validation/error handling easier to maintain.


def prompt_numeric_feature(feature: str, rules: dict[str, Any]) -> int:
    """Prompt user for a numeric feature with validation and retry logic."""
    description = rules["description"]
    min_val = rules["min"]
    max_val = rules["max"]

    while True:
        raw_value = input(
            f"\n  Enter {feature} ({description}) [{min_val}-{max_val}]: "
        ).strip()

        if not raw_value:
            print("  [Error] Input cannot be empty. Please try again.")
            continue

        try:
            value = int(raw_value)
        except ValueError:
            print("  [Error] Please enter a valid whole number.")
            continue

        if value < min_val or value > max_val:
            print(f"  [Error] Value must be between {min_val} and {max_val}.")
            continue

        return value


def prompt_categorical_feature(feature: str, rules: dict[str, Any]) -> str:
    """Prompt user to select a categorical feature from valid trained options."""
    options = rules["options"]
    description = rules["description"]

    print(f"\n  Select {feature} ({description}):")
    for index, option in enumerate(options, start=1):
        print(f"    {index}. {option}")

    while True:
        raw_choice = input(f"  Enter option number (1-{len(options)}): ").strip()

        if not raw_choice:
            print("  [Error] Input cannot be empty. Please try again.")
            continue

        try:
            choice = int(raw_choice)
        except ValueError:
            print("  [Error] Please enter a valid option number.")
            continue

        if choice < 1 or choice > len(options):
            print(f"  [Error] Please choose a number between 1 and {len(options)}.")
            continue

        return options[choice - 1]


def collect_employee_details(schema: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Interactively collect all employee feature values from the console.

    Returns a dictionary aligned with the model's expected feature columns.
    """
    print("\n" + "-" * 70)
    print(" EMPLOYEE DETAILS INPUT")
    print("-" * 70)
    print(" Please provide the following information.")
    print(" For categorical fields, choose from the displayed valid options.")
    print("-" * 70)

    employee_data: dict[str, Any] = {}

    for feature, rules in schema.items():
        if rules["kind"] == "categorical":
            employee_data[feature] = prompt_categorical_feature(feature, rules)
        else:
            employee_data[feature] = prompt_numeric_feature(feature, rules)

    return employee_data


# =============================================================================
# MODULE 4: PREDICTION (USES TRAINING PREPROCESSING PIPELINE)
# =============================================================================
# Explanation:
# User input is converted to a single-row DataFrame with columns in the
# exact order used during training. The saved sklearn Pipeline handles
# scaling, encoding, and classification automatically.


def format_input_for_model(
    employee_details: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Convert user input dictionary into model-ready DataFrame format."""
    missing = set(feature_columns) - set(employee_details.keys())
    if missing:
        raise ValueError(f"Missing required feature(s): {sorted(missing)}")

    return pd.DataFrame([employee_details])[feature_columns]


def predict_attrition(artifact: dict, employee_details: dict[str, Any]) -> dict[str, Any]:
    """
    Run attrition prediction using the saved preprocessing + model pipeline.

    Returns prediction label and probability scores.
    """
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    labels = artifact["prediction_labels"]

    input_df = format_input_for_model(employee_details, feature_columns)

    prediction_code = int(pipeline.predict(input_df)[0])
    attrition_probability = float(pipeline.predict_proba(input_df)[0][1])
    stay_probability = 1.0 - attrition_probability

    return {
        "model_name": artifact.get("model_name", "Unknown"),
        "prediction_code": prediction_code,
        "prediction_label": labels[prediction_code],
        "attrition_probability": attrition_probability,
        "stay_probability": stay_probability,
    }


# =============================================================================
# MODULE 5: PROFESSIONAL RESULT DISPLAY
# =============================================================================
# Explanation:
# A structured, report-style output is useful for college project demos and viva.


def print_welcome_banner() -> None:
    """Display application header."""
    print("\n" + "=" * 70)
    print("   EMPLOYEE ATTRITION PREDICTION SYSTEM")
    print("   Interactive Console-Based Prediction Module")
    print("=" * 70)
    print(" This tool predicts whether an employee is likely to:")
    print("   - Leave the company")
    print("   - Stay in the company")
    print("=" * 70)


def print_input_summary(employee_details: dict[str, Any]) -> None:
    """Display entered employee details before prediction."""
    print("\n" + "-" * 70)
    print(" ENTERED EMPLOYEE PROFILE")
    print("-" * 70)
    for feature, value in employee_details.items():
        print(f"  {feature:<32}: {value}")
    print("-" * 70)


def print_prediction_report(result: dict[str, Any]) -> None:
    """Display prediction output in a professional demo-friendly format."""
    attrition_pct = result["attrition_probability"] * 100
    stay_pct = result["stay_probability"] * 100
    label = result["prediction_label"]

    if label == "Leave":
        outcome_text = "The employee is LIKELY TO LEAVE the company."
        risk_level = "High Attrition Risk" if attrition_pct >= 50 else "Moderate Attrition Risk"
    else:
        outcome_text = "The employee is LIKELY TO STAY in the company."
        risk_level = "Low Attrition Risk" if attrition_pct < 30 else "Moderate Retention Risk"

    print("\n" + "=" * 70)
    print(" PREDICTION RESULT")
    print("=" * 70)
    print(f" Model Used              : {result['model_name']}")
    print(f" Prediction Label        : {label}")
    print(f" Interpretation          : {outcome_text}")
    print(f" Risk Assessment         : {risk_level}")
    print("-" * 70)
    print(f" Attrition Probability   : {attrition_pct:6.2f}%")
    print(f" Stay Probability        : {stay_pct:6.2f}%")
    print("=" * 70)

    # Simple visual probability bar for demonstration (ASCII-safe for Windows console)
    bar_length = 40
    leave_blocks = int(round((attrition_pct / 100) * bar_length))
    stay_blocks = bar_length - leave_blocks
    print("\n Probability Distribution")
    print(f" Leave [{'#' * leave_blocks}{'.' * stay_blocks}] {attrition_pct:.1f}%")
    print(f" Stay  [{'#' * stay_blocks}{'.' * leave_blocks}] {stay_pct:.1f}%")
    print("=" * 70 + "\n")


# =============================================================================
# MODULE 6: MAIN APPLICATION LOOP
# =============================================================================


def ask_yes_no(prompt: str) -> bool:
    """Return True for yes, False for no, with validation."""
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            print("\nInput stream closed.")
            return False

        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  [Error] Please enter 'yes' or 'no'.")


def run_interactive_session() -> None:
    """Main interactive workflow: load model, collect input, predict, repeat."""
    print_welcome_banner()

    try:
        artifact = load_saved_model()
        schema = build_feature_schema(artifact)
    except (FileNotFoundError, ValueError) as error:
        print(f"\n[Startup Error] {error}")
        sys.exit(1)

    print(f"\n Model loaded successfully from:\n   {MODEL_PATH}")
    print(f" Trained model type: {artifact.get('model_name', 'Unknown')}")
    print(f" Total input features: {len(artifact['feature_columns'])}")

    while True:
        try:
            employee_details = collect_employee_details(schema)
            print_input_summary(employee_details)

            if not ask_yes_no("\nProceed with prediction? (yes/no): "):
                print("\nPrediction cancelled. Returning to main menu...")
            else:
                result = predict_attrition(artifact, employee_details)
                print_prediction_report(result)

        except KeyboardInterrupt:
            print("\n\nSession ended by user. Goodbye!")
            break
        except EOFError:
            print("\n\nInput ended unexpectedly. Goodbye!")
            break
        except Exception as error:
            print(f"\n[Unexpected Error] {error}")
            print("Please try again.")

        if not ask_yes_no("\nPredict for another employee? (yes/no): "):
            print("\nThank you for using the Employee Attrition Prediction System.")
            break


if __name__ == "__main__":
    run_interactive_session()
