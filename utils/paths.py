"""Central path configuration for the project."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATASET_PATH = RAW_DATA_DIR / "employee_attrition_dataset_10000.csv"

# Models
MODELS_DIR = PROJECT_DIR / "models"
TARGET_MODEL_PATH = MODELS_DIR / "target_model.pkl"
SUBSTITUTE_MODEL_PATH = MODELS_DIR / "substitute_model.pkl"

# Experiment results
RESULTS_DIR = PROJECT_DIR / "results"
ATTACK_RESULTS_DIR = RESULTS_DIR / "attack"

# Attack module
ATTACK_DIR = PROJECT_DIR / "attack"
STOLEN_DATASET_PATH = ATTACK_DIR / "stolen_dataset.csv"
QUERIES_CACHE_PATH = ATTACK_DIR / "queries_cache.csv"
ATTACK_SUMMARY_JSON = ATTACK_DIR / "attack_summary.json"
ATTACK_LOGS_DIR = ATTACK_DIR / "logs"
SURROGATE_DIR = ATTACK_DIR / "surrogate"

# Web app
WEB_DIR = PROJECT_DIR / "web"
PREDICTIONS_FILE = WEB_DIR / "data" / "predictions.json"

# Screenshots
SCREENSHOTS_DIR = PROJECT_DIR / "screenshots"

# Training pipeline artifacts
METRICS_PATH = RESULTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
VICTIM_EVALUATION_PATH = RESULTS_DIR / "victim_evaluation.json"
VICTIM_CONFUSION_MATRIX_PATH = RESULTS_DIR / "victim_confusion_matrix.png"

# Legacy aliases (for gradual migration)
ORIGINAL_MODEL_PATH = TARGET_MODEL_PATH
