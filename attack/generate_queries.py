"""
Generate Queries - Synthetic Employee Record Generator
======================================================
Produces realistic synthetic employee records for black-box model
stealing attack simulation.

Strategies
----------
1. distribution  : Sample from training data marginals + light perturbation (default)
2. random        : Uniform random sampling within API-valid bounds
3. grid          : Structured coverage across categorical combinations

Academic context
----------------
In a model extraction attack, the adversary crafts diverse inputs to probe
the target model's decision boundary. Realistic queries improve surrogate
model fidelity compared to purely random noise.

Usage
-----
    python generate_queries.py --count 3000 --strategy distribution
    python generate_queries.py --count 1000 --output queries_cache.csv
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_OPTIONS,
    FEATURE_COLUMNS,
    NUMERIC_RANGES,
    QUERIES_CACHE_PATH,
    TRAINING_DATASET_PATH,
)

# ---------------------------------------------------------------------------
# Random record generation
# ---------------------------------------------------------------------------


def _load_training_reference() -> pd.DataFrame | None:
    """Load training CSV and extract API-visible feature columns."""
    if not TRAINING_DATASET_PATH.exists():
        return None

    df = pd.read_csv(TRAINING_DATASET_PATH)
    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    if len(available) != len(FEATURE_COLUMNS):
        return None
    return df[FEATURE_COLUMNS].copy()


def _clip_numeric_features(record: dict) -> dict:
    """Ensure numeric values respect API validation bounds."""
    for field, (low, high) in NUMERIC_RANGES.items():
        value = int(round(record[field]))
        record[field] = int(np.clip(value, low, high))
    return record


def _perturb_record(record: dict, rng: np.random.Generator) -> dict:
    """Apply small random perturbations to a base record for diversity."""
    out = record.copy()

    # Numeric jitter
    out["Age"] += rng.integers(-3, 4)
    out["Monthly_Income"] += rng.integers(-1500, 1501)
    out["Years_at_Company"] = max(0, out["Years_at_Company"] + rng.integers(-2, 3))
    out["Job_Satisfaction"] += rng.integers(-1, 2)
    out["Work_Life_Balance"] += rng.integers(-1, 2)
    out["Number_of_Companies_Worked"] += rng.integers(-1, 2)

    # Occasional categorical flip
    if rng.random() < 0.15:
        field = rng.choice(list(CATEGORICAL_OPTIONS.keys()))
        out[field] = rng.choice(CATEGORICAL_OPTIONS[field])

    if rng.random() < 0.10:
        out["Overtime"] = rng.choice(CATEGORICAL_OPTIONS["Overtime"])

    return _clip_numeric_features(out)


def generate_random_record(rng: np.random.Generator) -> dict:
    """Generate one record by uniform random sampling within valid bounds."""
    record: dict = {}

    for field, (low, high) in NUMERIC_RANGES.items():
        record[field] = int(rng.integers(low, high + 1))

    for field, options in CATEGORICAL_OPTIONS.items():
        record[field] = str(rng.choice(options))

    return record


def generate_from_distribution(
    n: int,
    rng: np.random.Generator,
    reference_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Generate records by resampling training data with perturbation.

    Falls back to random generation if training data is unavailable.
    """
    if reference_df is None or reference_df.empty:
        return [generate_random_record(rng) for _ in range(n)]

    records: list[dict] = []
    base_indices = rng.integers(0, len(reference_df), size=n)

    for idx in base_indices:
        base = reference_df.iloc[int(idx)].to_dict()
        # Ensure native Python types for JSON serialization
        base = {k: (int(v) if k in NUMERIC_RANGES else str(v)) for k, v in base.items()}
        records.append(_perturb_record(base, rng))

    return records


def generate_grid_records(max_records: int, rng: np.random.Generator) -> list[dict]:
    """
    Generate structured records covering categorical combinations.

    Numeric features are sampled around mid-range values per combination.
    """
    cat_fields = list(CATEGORICAL_OPTIONS.keys())
    option_lists = [CATEGORICAL_OPTIONS[f] for f in cat_fields]
    combinations = list(itertools.product(*option_lists))

    records: list[dict] = []
    for combo in combinations:
        record = {}
        for field, value in zip(cat_fields, combo):
            record[field] = value

        for field, (low, high) in NUMERIC_RANGES.items():
            mid = (low + high) // 2
            spread = (high - low) // 4
            record[field] = int(np.clip(mid + rng.integers(-spread, spread + 1), low, high))

        records.append(record)

        if len(records) >= max_records:
            break

    # Fill remainder with distribution-based records if grid is smaller than n
    if len(records) < max_records:
        extra = generate_from_distribution(max_records - len(records), rng)
        records.extend(extra)

    return records[:max_records]


def generate_synthetic_records(
    n: int,
    strategy: str = "distribution",
    seed: int = 42,
) -> list[dict]:
    """
    Main entry point: generate n synthetic employee records.

    Parameters
    ----------
    n : int
        Number of records to generate.
    strategy : str
        One of 'distribution', 'random', or 'grid'.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        List of API-ready employee feature dictionaries.
    """
    rng = np.random.default_rng(seed)
    reference_df = _load_training_reference()

    if strategy == "random":
        return [generate_random_record(rng) for _ in range(n)]
    if strategy == "grid":
        return generate_grid_records(n, rng)
    if strategy == "distribution":
        return generate_from_distribution(n, rng, reference_df)

    raise ValueError(f"Unknown strategy: {strategy}. Use distribution, random, or grid.")


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert list of records to a pandas DataFrame with stable column order."""
    return pd.DataFrame(records)[FEATURE_COLUMNS]


def save_queries(records: list[dict], output_path: Path) -> None:
    """Persist generated queries to CSV for inspection or reuse."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_to_dataframe(records).to_csv(output_path, index=False)
    print(f"Saved {len(records):,} queries to: {output_path}")


def load_queries(input_path: Path) -> list[dict]:
    """Load previously generated queries from CSV."""
    df = pd.read_csv(input_path)
    return df[FEATURE_COLUMNS].to_dict(orient="records")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic employee records for model stealing simulation."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3000,
        help="Number of synthetic records to generate (default: 3000).",
    )
    parser.add_argument(
        "--strategy",
        choices=["distribution", "random", "grid"],
        default="distribution",
        help="Query generation strategy (default: distribution).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=QUERIES_CACHE_PATH,
        help=f"Output CSV path (default: {QUERIES_CACHE_PATH.name}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Generating {args.count:,} synthetic records (strategy={args.strategy})...")

    records = generate_synthetic_records(
        n=args.count,
        strategy=args.strategy,
        seed=args.seed,
    )
    save_queries(records, args.output)

    print("\nSample record:")
    for key, value in records[0].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
