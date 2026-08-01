"""General helper utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.paths import TRAINING_DATASET_PATH

# API-exposed feature columns (subset of full training schema)
API_FEATURE_COLUMNS = [
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


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the employee attrition dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Place employee_attrition_dataset_10000.csv in data/raw/"
        )
    return pd.read_csv(file_path)


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read CSV if it exists, else return None."""
    if path.exists():
        return pd.read_csv(path)
    return None


from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype

def compute_feature_defaults(
    feature_columns: list[str],
    api_columns: list[str] | None = None,
    dataset_path: Path | None = None,
) -> dict:
    """
    Compute median/mode defaults for features not collected in the web form.
    Ensures the full feature vector matches the training schema.
    """
    api_columns = api_columns or API_FEATURE_COLUMNS
    dataset_path = dataset_path or TRAINING_DATASET_PATH
    defaults: dict = {}

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)

        # Remove columns not used for prediction
        df = df.drop(columns=["Employee_ID", "Attrition"], errors="ignore")

        for col in feature_columns:
            if col not in api_columns and col in df.columns:

                # Numeric columns -> median
                if is_numeric_dtype(df[col]):
                    defaults[col] = int(df[col].median())

                # Categorical/String columns -> mode
                else:
                    mode = df[col].mode(dropna=True)
                    defaults[col] = mode.iloc[0] if not mode.empty else ""

    else:
        defaults = {
            "Marital_Status": "Married",
            "Job_Level": 3,
            "Hourly_Rate": 50,
            "Years_in_Current_Role": 5,
            "Years_Since_Last_Promotion": 2,
            "Performance_Rating": 2,
            "Training_Hours_Last_Year": 50,
            "Project_Count": 5,
            "Average_Hours_Worked_Per_Week": 45,
            "Absenteeism": 7,
            "Work_Environment_Satisfaction": 3,
            "Relationship_with_Manager": 3,
            "Job_Involvement": 3,
            "Distance_From_Home": 15,
        }

    return defaults
