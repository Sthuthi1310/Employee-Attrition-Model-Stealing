"""
Substitute Model Training - Configuration
=========================================
Paths and hyperparameters for training a surrogate model on stolen API labels.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------

SURROGATE_DIR = Path(__file__).resolve().parent
ATTACK_DIR = SURROGATE_DIR.parent
PROJECT_DIR = ATTACK_DIR.parent

STOLEN_DATASET_PATH = ATTACK_DIR / "stolen_dataset.csv"
ORIGINAL_MODEL_PATH = PROJECT_DIR / "outputs" / "best_attrition_model.joblib"
SUBSTITUTE_MODEL_PATH = SURROGATE_DIR / "substitute_model.joblib"
OUTPUT_DIR = SURROGATE_DIR / "outputs"

TRAINING_DATASET_PATH = Path(
    r"C:\Users\Sthuthi Sheela\Downloads\Employee-Attrition\employee_attrition_dataset_10000.csv"
)

# ---------------------------------------------------------------------------
# Feature schema (matches attack / API exposed features)
# ---------------------------------------------------------------------------

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

NUMERIC_FEATURES = [
    "Age",
    "Monthly_Income",
    "Years_at_Company",
    "Job_Satisfaction",
    "Work_Life_Balance",
    "Number_of_Companies_Worked",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Department",
    "Job_Role",
    "Overtime",
]

TARGET_COLUMN = "prediction_label"
TARGET_MAPPING = {"Leave": 1, "Stay": 0, "Yes": 1, "No": 0}

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
