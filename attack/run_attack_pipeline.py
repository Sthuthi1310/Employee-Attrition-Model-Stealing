"""
Full Attack Pipeline
====================
End-to-end model stealing workflow:
  1. Collect 500+ API query-response pairs
  2. Retrain substitute model
  3. Regenerate evaluation metrics and charts
  4. Export dashboard-ready JSON

Usage
-----
    # Ensure Flask is running: python web/app.py
    python run_attack_pipeline.py
    python run_attack_pipeline.py --count 1000
    python run_attack_pipeline.py --skip-attack
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from collect_responses import run_attack, setup_logging
from config import (
    ATTACK_DIR,
    DEFAULT_NUM_QUERIES,
    DELAY_BETWEEN_REQUESTS_SECONDS,
    LOGS_DIR,
    MAX_QUERIES,
    STOLEN_DATASET_PATH,
)
from generate_queries import generate_synthetic_records

SURROGATE_DIR = ATTACK_DIR / "surrogate"


def retrain_and_evaluate(logger: logging.Logger | None = None) -> None:
    """Retrain substitute model, regenerate charts, export dashboard JSON."""
    log = logger or logging.getLogger("model_stealing_attack")

    log.info("=" * 70)
    log.info("SUBSTITUTE MODEL RETRAINING & EVALUATION")
    log.info("=" * 70)

    train_script = SURROGATE_DIR / "train_substitute.py"
    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(SURROGATE_DIR),
        check=False,
    )
    if result.returncode != 0:
        log.error("Substitute training failed (exit code %s)", result.returncode)
        sys.exit(result.returncode)

    log.info("Substitute training complete.")

    web_dir = ATTACK_DIR.parent / "web"
    export_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from analysis_data import export_analysis_json; print(export_analysis_json())",
        ],
        cwd=str(web_dir),
        capture_output=True,
        text=True,
    )
    if export_result.returncode == 0:
        log.info("Dashboard JSON exported: %s", export_result.stdout.strip())
    else:
        log.warning("Dashboard JSON export skipped: %s", export_result.stderr.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full model stealing attack pipeline.")
    parser.add_argument("--count", type=int, default=DEFAULT_NUM_QUERIES)
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--strategy", choices=["distribution", "random", "grid"], default="distribution")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_REQUESTS_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-attack", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.count < 1 or args.count > MAX_QUERIES:
        print(f"Error: --count must be between 1 and {MAX_QUERIES}")
        sys.exit(1)

    logger, _ = setup_logging(LOGS_DIR)

    if not args.skip_attack:
        logger.info("Generating %s synthetic queries...", f"{args.count:,}")
        queries = generate_synthetic_records(n=args.count, strategy=args.strategy, seed=args.seed)
        run_attack(
            queries=queries,
            api_url=args.api_url,
            output_path=STOLEN_DATASET_PATH,
            delay=args.delay,
            checkpoint_every=50,
            logger=logger,
        )

    if not args.skip_train:
        retrain_and_evaluate(logger)

    print("\nPipeline finished. Open http://localhost:5000/analysis to view results.")


if __name__ == "__main__":
    main()
