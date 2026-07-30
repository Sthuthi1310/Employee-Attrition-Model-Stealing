"""
Substitute Model Training - Configuration
=========================================
Paths and hyperparameters for training a surrogate model on stolen API labels.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.helper import API_FEATURE_COLUMNS
from utils.paths import (
    ATTACK_RESULTS_DIR,
    ATTACK_DIR,
    PROJECT_DIR,
    STOLEN_DATASET_PATH,
    SUBSTITUTE_MODEL_PATH,
    TARGET_MODEL_PATH,
    TRAINING_DATASET_PATH,
)

SURROGATE_DIR = PROJECT_DIR / "attack" / "surrogate"

ORIGINAL_MODEL_PATH = TARGET_MODEL_PATH
OUTPUT_DIR = ATTACK_RESULTS_DIR

# ---------------------------------------------------------------------------
# Feature schema (matches attack / API exposed features)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = API_FEATURE_COLUMNS

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
