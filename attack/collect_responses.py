"""
Collect Responses - Model Stealing Attack Orchestrator
======================================================
Sends synthetic employee queries to the Flask /api/predict endpoint,
collects query-response pairs, and saves them to stolen_dataset.csv.

Usage
-----
    python collect_responses.py --count 500
    python collect_responses.py --count 1000 --auto-train
    python run_attack_pipeline.py --count 500
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    API_BASE_URL,
    ATTACK_SUMMARY_JSON,
    CHECKPOINT_EVERY,
    DEFAULT_NUM_QUERIES,
    DELAY_BETWEEN_REQUESTS_SECONDS,
    FEATURE_COLUMNS,
    HEALTH_ENDPOINT,
    LOGS_DIR,
    MAX_QUERIES,
    MAX_RETRIES,
    OUTPUT_COLUMNS,
    PREDICT_ENDPOINT,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    STOLEN_DATASET_PATH,
)
from generate_queries import generate_synthetic_records, load_queries


def setup_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    """Configure file and console logging for the attack run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"attack_{timestamp}.log"

    logger = logging.getLogger("model_stealing_attack")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Log file: %s", log_file)
    return logger, log_file


def create_http_session() -> requests.Session:
    """Create a requests session with retry logic for transient failures."""
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AttritionModelStealingResearch/2.0",
        }
    )

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_SECONDS,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def check_api_health(session: requests.Session, api_url: str, logger: logging.Logger) -> bool:
    """Verify the target API is reachable before starting the attack."""
    health_url = f"{api_url.rstrip('/')}{HEALTH_ENDPOINT}"
    try:
        response = session.get(health_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            logger.error("Health check failed: HTTP %s", response.status_code)
            return False
        data = response.json()
        if not data.get("model_loaded"):
            logger.error("API is up but model is not loaded.")
            return False
        logger.info("API health check passed: %s", data)
        return True
    except requests.RequestException as exc:
        logger.error("Health check failed: %s", exc)
        return False


def query_prediction_api(
    session: requests.Session,
    api_url: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, int, str | None, float]:
    """
    Send a single prediction request to the black-box API.

    Returns (response_json, http_status, error_message, response_time_ms).
    """
    predict_url = f"{api_url.rstrip('/')}{PREDICT_ENDPOINT}"
    start = time.perf_counter()

    try:
        response = session.post(predict_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        status = response.status_code

        try:
            body = response.json()
        except json.JSONDecodeError:
            return None, status, "Invalid JSON response from API.", elapsed_ms

        if status == 200:
            return body, status, None, elapsed_ms
        return None, status, body.get("error", f"HTTP {status}"), elapsed_ms

    except requests.Timeout:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return None, 0, "Request timed out.", elapsed_ms
    except requests.ConnectionError:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return None, 0, "Connection error - is the Flask server running?", elapsed_ms
    except requests.RequestException as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return None, 0, str(exc), elapsed_ms


class AttackProgress:
    """Track and display live attack progress."""

    def __init__(self, total: int, logger: logging.Logger):
        self.total = total
        self.logger = logger
        self.sent = 0
        self.success = 0
        self.failed = 0
        self.response_times: list[float] = []
        self.start_time = time.time()

    def update(self, success: bool, response_time_ms: float = 0.0) -> None:
        self.sent += 1
        if success:
            self.success += 1
            if response_time_ms > 0:
                self.response_times.append(response_time_ms)
        else:
            self.failed += 1

        # Live console update every 10 queries and at milestones
        if self.sent % 10 == 0 or self.sent == self.total or self.sent % 50 == 0:
            self._print_progress()

    def _print_progress(self) -> None:
        elapsed = time.time() - self.start_time
        rate = self.sent / elapsed if elapsed > 0 else 0
        pct = (self.sent / self.total) * 100
        avg_rt = (
            round(sum(self.response_times) / len(self.response_times), 2)
            if self.response_times
            else 0.0
        )

        self.logger.info(
            "[LIVE] Queries: %s/%s (%.1f%%) | Success: %s | Failed: %s | "
            "Dataset: %s rows | Avg RT: %s ms | Rate: %.1f req/s",
            f"{self.sent:,}",
            f"{self.total:,}",
            pct,
            f"{self.success:,}",
            f"{self.failed:,}",
            f"{self.success:,}",
            avg_rt,
            rate,
        )

    def summary(self) -> dict[str, Any]:
        elapsed = time.time() - self.start_time
        avg_rt = (
            round(sum(self.response_times) / len(self.response_times), 2)
            if self.response_times
            else 0.0
        )
        return {
            "total_queries": self.total,
            "sent": self.sent,
            "successful": self.success,
            "failed": self.failed,
            "elapsed_seconds": round(elapsed, 2),
            "success_rate_pct": round((self.success / self.sent) * 100, 2) if self.sent else 0,
            "average_response_time_ms": avg_rt,
            "total_records_collected": self.success,
        }


def build_result_row(
    query_number: int,
    payload: dict[str, Any],
    response: dict[str, Any] | None,
    http_status: int,
    error_message: str | None,
    response_time_ms: float,
) -> dict[str, Any]:
    """Merge query features and API response into one CSV row."""
    row: dict[str, Any] = {
        "query_number": query_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for field in FEATURE_COLUMNS:
        row[field] = payload.get(field)

    if response:
        row.update(
            {
                "prediction_label": response.get("prediction_label"),
                "attrition_status": response.get("attrition_status"),
                "attrition_probability": response.get("attrition_probability"),
                "stay_probability": response.get("stay_probability"),
                "prediction_confidence": response.get("prediction_confidence"),
                "response_time_ms": response_time_ms,
                "model_name": response.get("model_name"),
                "record_id": response.get("record_id"),
                "http_status": http_status,
                "success": True,
                "error_message": None,
            }
        )
    else:
        row.update(
            {
                "prediction_label": None,
                "attrition_status": None,
                "attrition_probability": None,
                "stay_probability": None,
                "prediction_confidence": None,
                "response_time_ms": response_time_ms,
                "model_name": None,
                "record_id": None,
                "http_status": http_status,
                "success": False,
                "error_message": error_message,
            }
        )

    return row


def save_dataset(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write collected query-response pairs to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df[OUTPUT_COLUMNS].to_csv(output_path, index=False)


def save_attack_summary(summary: dict[str, Any], path: Path) -> None:
    """Persist attack_summary.json for the dashboard."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_queries": summary["sent"],
        "successful_queries": summary["successful"],
        "failed_queries": summary["failed"],
        "success_rate_pct": summary["success_rate_pct"],
        "average_response_time_ms": summary["average_response_time_ms"],
        "total_records_collected": summary["total_records_collected"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "stolen_dataset_path": str(STOLEN_DATASET_PATH),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_attack(
    queries: list[dict[str, Any]],
    api_url: str,
    output_path: Path,
    delay: float,
    checkpoint_every: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Execute the model stealing data collection attack."""
    session = create_http_session()
    progress = AttackProgress(total=len(queries), logger=logger)
    collected_rows: list[dict[str, Any]] = []

    logger.info("=" * 70)
    logger.info("MODEL STEALING ATTACK - DATA COLLECTION")
    logger.info("=" * 70)
    logger.info("Target API       : %s%s", api_url, PREDICT_ENDPOINT)
    logger.info("Queries planned  : %s", f"{len(queries):,}")
    logger.info("Checkpoint every : %s queries", checkpoint_every)
    logger.info("Output file      : %s", output_path)

    if not check_api_health(session, api_url, logger):
        logger.error("Aborting attack: API is not available.")
        sys.exit(1)

    logger.info("-" * 70)

    for index, payload in enumerate(queries, start=1):
        response, http_status, error, response_time_ms = query_prediction_api(
            session, api_url, payload
        )
        success = response is not None

        row = build_result_row(index, payload, response, http_status, error, response_time_ms)
        collected_rows.append(row)
        progress.update(success, response_time_ms)

        if not success:
            logger.warning("Query %s FAILED | HTTP %s | %s", index, http_status, error)

        # Checkpoint every N queries (includes failed)
        if index % checkpoint_every == 0:
            save_dataset(collected_rows, output_path)
            logger.info(">> Checkpoint saved: %s rows -> %s", len(collected_rows), output_path)

        if delay > 0 and index < len(queries):
            time.sleep(delay)

    save_dataset(collected_rows, output_path)
    summary = progress.summary()
    save_attack_summary(summary, ATTACK_SUMMARY_JSON)

    logger.info("=" * 70)
    logger.info("ATTACK COMPLETE")
    logger.info("Total sent           : %s", f"{summary['sent']:,}")
    logger.info("Successful           : %s", f"{summary['successful']:,}")
    logger.info("Failed               : %s", f"{summary['failed']:,}")
    logger.info("Success rate         : %s%%", summary["success_rate_pct"])
    logger.info("Avg response time    : %s ms", summary["average_response_time_ms"])
    logger.info("Records collected    : %s", f"{summary['total_records_collected']:,}")
    logger.info("Stolen dataset       : %s", output_path)
    logger.info("Attack summary JSON  : %s", ATTACK_SUMMARY_JSON)
    logger.info("=" * 70)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Black-box model stealing attack collector.")
    parser.add_argument("--count", type=int, default=DEFAULT_NUM_QUERIES, help="Number of queries.")
    parser.add_argument("--api-url", type=str, default=API_BASE_URL)
    parser.add_argument("--strategy", choices=["distribution", "random", "grid"], default="distribution")
    parser.add_argument("--queries-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=STOLEN_DATASET_PATH)
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_REQUESTS_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--auto-train",
        action="store_true",
        help="Automatically retrain substitute model after attack completes.",
    )
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()

    if args.count < 1 or args.count > MAX_QUERIES:
        print(f"Error: --count must be between 1 and {MAX_QUERIES}")
        sys.exit(1)

    logger, _ = setup_logging(LOGS_DIR)

    if args.queries_file:
        logger.info("Loading queries from: %s", args.queries_file)
        queries = load_queries(args.queries_file)
    else:
        logger.info("Generating %s synthetic queries (strategy=%s)...", f"{args.count:,}", args.strategy)
        queries = generate_synthetic_records(n=args.count, strategy=args.strategy, seed=args.seed)

    if not queries:
        logger.error("No queries to send.")
        sys.exit(1)

    summary = run_attack(
        queries=queries,
        api_url=args.api_url,
        output_path=args.output,
        delay=args.delay,
        checkpoint_every=args.checkpoint_every,
        logger=logger,
    )

    if args.auto_train:
        logger.info("Starting automatic substitute model retraining...")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "surrogate" / "train_substitute.py")],
            cwd=str(Path(__file__).resolve().parent / "surrogate"),
            check=True,
        )
        web_dir = Path(__file__).resolve().parent.parent / "web"
        subprocess.run(
            [sys.executable, "-c", "from analysis_data import export_analysis_json; export_analysis_json()"],
            cwd=str(web_dir),
            check=False,
        )

    return summary


if __name__ == "__main__":
    main()
