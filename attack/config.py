"""
Attack Module - Shared Configuration
=====================================
Central configuration for the model-stealing attack simulation.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.helper import API_FEATURE_COLUMNS
from utils.paths import (
    ATTACK_DIR,
    ATTACK_LOGS_DIR,
    ATTACK_SUMMARY_JSON,
    QUERIES_CACHE_PATH,
    STOLEN_DATASET_PATH,
    SURROGATE_DIR,
    TRAINING_DATASET_PATH,
)

LOGS_DIR = ATTACK_LOGS_DIR

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:5000"
PREDICT_ENDPOINT = "/api/predict"
PUBLIC_PREDICT_ENDPOINT = "/predict"
HEALTH_ENDPOINT = "/api/health"

# Attack scale (CLI can override up to MAX_QUERIES: 500 - 5000)
DEFAULT_NUM_QUERIES = 500
MAX_QUERIES = 5000

# HTTP client settings
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5
DELAY_BETWEEN_REQUESTS_SECONDS = 0.005
CHECKPOINT_EVERY = 50  # Save CSV every N queries (success or fail)

# ---------------------------------------------------------------------------
# Bounded Synthetic Query Distribution Rules (PPT Slide 10)
# ---------------------------------------------------------------------------

CATEGORICAL_OPTIONS = {
    "Gender": ["Female", "Male"],
    "Department": ["Finance", "HR", "IT", "Marketing", "Sales"],
    "Job_Role": ["Analyst", "Assistant", "Executive", "Manager"],
    "Overtime": ["No", "Yes"],
}

NUMERIC_RANGES = {
    "Age": (18, 65),                          # Uniform (18, 65)
    "Monthly_Income": (1500, 25000),          # Uniform (1500, 25000)
    "Years_at_Company": (0, 40),              # Uniform (0, 40); <= Age - 18
    "Job_Satisfaction": (1, 4),               # Discrete Uniform (1, 4)
    "Work_Life_Balance": (1, 4),              # Discrete Uniform (1, 4)
    "Number_of_Companies_Worked": (0, 10),    # Discrete Uniform (0, 10)
}

OVERTIME_BERNOULLI_PROBABILITY = 0.30         # Bernoulli (p = 0.30)

FEATURE_COLUMNS = API_FEATURE_COLUMNS

OUTPUT_COLUMNS = [
    "query_number",
    "timestamp",
    *FEATURE_COLUMNS,
    "attrition",            # Binary Yes/No captured label (hard label)
    "prediction_label",
    "attrition_status",
    "attrition_probability",
    "stay_probability",
    "prediction_confidence",
    "response_time_ms",
    "model_name",
    "record_id",
    "http_status",
    "success",
    "error_message",
]

