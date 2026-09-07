"""Replay real bookings against a running API, then reconcile their true segments.

A drift monitor with no traffic monitors nothing, so this exists to produce a
prediction log that is worth analysing. What it deliberately does *not* do is
invent data: it replays the actual dataset, sliced by arrival date, and posts
each booking's real `market_segment` back as ground truth.

That matters. The obvious way to demonstrate a drift monitor is to perturb a
feature with random noise until the statistic moves, which proves only that the
statistic responds to noise. This dataset spans July 2015 to August 2017 and
contains a genuine shift over that period - the booking mix moves towards online
travel agencies as the window advances - so replaying 2015-2016 and then 2017
exercises the monitor against drift that actually happened, at the magnitude it
actually had. A baseline trained on the early window and evaluated on the late
one is a backtest, not a mock.

    # baseline period - what the model was trained on
    python scripts/replay_bookings.py --year 2016 --limit 3000

    # later period - where the channel mix has moved
    python scripts/replay_bookings.py --year 2017 --months 5 6 7 --limit 3000

`--label-fraction` models the reconciliation delay: in production only part of a
window is labelled by the time the monitor runs, and the monitor has to behave
correctly when that fraction is small (it reports SKIPPED rather than guessing).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_config  # noqa: E402
from src.data_processing import load_and_clean_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("replay")

# Columns the API's Pydantic schema accepts. Anything else in the processed
# frame (the target itself, engineered leftovers) is dropped rather than sent:
# posting the true segment alongside the features would make the prediction
# meaningless and the resulting "no drift" verdict a lie.
TARGET_COLUMN = "market_segment"


def _payload(row: pd.Series, text_columns: set[str]) -> dict:
    """One booking as the API expects it, with missing values made explicit.

    The substitute for a missing value is chosen from the *column's* dtype, not
    from the value's: a NaN is a float whatever column it sits in, so deciding
    per-value sends 0.0 for a missing `country` and the request is rejected with
    a 422. Losing rows that way would silently bias the replayed window towards
    complete records - drift measured on a filtered sample is not drift.
    """
    record = row.drop(labels=[TARGET_COLUMN], errors="ignore").to_dict()
    cleaned: dict = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = "Unknown" if key in text_columns else 0.0
        elif isinstance(value, (int, float)) and float(value).is_integer():
            cleaned[key] = int(value)
        else:
            cleaned[key] = value
    return cleaned


def replay(
    api_url: str,
    frame: pd.DataFrame,
    label_fraction: float,
    batch_size: int,
    timeout: float,
) -> tuple[int, int]:
    """POST each booking to /predict, then reconcile the true segments.

    Labels are posted *after* the whole replay rather than alongside it. That
    ordering is not cosmetic: /predict persists asynchronously, so posting a
    label microseconds after the prediction that produced it races the flush -
    the API reports those rows as `skipped` and the caller has to re-post them.
    Reconciling at the end both mirrors how a real reconciliation job behaves
    and keeps the retry loop below to a single pass in the normal case.
    """
    session = requests.Session()
    outstanding: list[dict] = []
    served = 0
    text_columns = {str(c) for c in frame.columns if frame[c].dtype == object}

    for position, (_, row) in enumerate(frame.iterrows()):
        try:
            response = session.post(
                f"{api_url}/predict", json=_payload(row, text_columns), timeout=timeout
            )
        except requests.RequestException as e:
            logger.warning("predict request failed: %s", e)
            continue

        if response.status_code != 200:
            logger.warning("predict returned %s: %s", response.status_code, response.text[:200])
            continue

        served += 1
        body = response.json()

        # Only a fraction gets a label, deterministically by position rather
        # than at random, so a given --label-fraction always reconciles the same
        # rows and two runs of this script are comparable.
        if label_fraction > 0 and (position % 100) < round(label_fraction * 100):
            outstanding.append(
                {
                    "prediction_id": body["prediction_id"],
                    "actual_market_segment": str(row[TARGET_COLUMN]),
                    "label_source": "replay-reconciliation",
                }
            )

        if served % 500 == 0:
            logger.info("served %s predictions", served)

    labelled = _reconcile(session, api_url, outstanding, batch_size, timeout)
    return served, labelled


def _reconcile(
    session: requests.Session,
    api_url: str,
    labels: list[dict],
    batch_size: int,
    timeout: float,
    max_attempts: int = 4,
) -> int:
    """Post labels in batches, re-posting any the API could not match yet.

    Re-posting a whole batch is safe by design - /feedback is an idempotent
    upsert - so this is the same at-least-once loop a production reconciliation
    job would run, rather than a workaround for the test harness.
    """
    accepted_total = 0
    for start in range(0, len(labels), batch_size):
        batch = labels[start : start + batch_size]
        for attempt in range(1, max_attempts + 1):
            accepted, skipped = _post_labels(session, api_url, batch, timeout)
            if skipped == 0:
                accepted_total += accepted
                break
            logger.info(
                "reconciliation attempt %s/%s: %s accepted, %s not durable yet",
                attempt,
                max_attempts,
                accepted,
                skipped,
            )
            if attempt == max_attempts:
                accepted_total += accepted
            else:
                # The API flushes its prediction queue on an interval; give it
                # one before re-posting rather than spinning.
                time.sleep(6)
    return accepted_total


def _post_labels(
    session: requests.Session, api_url: str, batch: list[dict], timeout: float
) -> tuple[int, int]:
    try:
        response = session.post(f"{api_url}/feedback", json=batch, timeout=timeout)
    except requests.RequestException as e:
        logger.warning("feedback request failed: %s", e)
        return 0, len(batch)
    if response.status_code >= 300:
        logger.warning("feedback returned %s: %s", response.status_code, response.text[:200])
        return 0, len(batch)
    body = response.json()
    return int(body.get("accepted", 0)), int(body.get("skipped", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--year", type=int, default=None, help="arrival_date_year to replay (e.g. 2017)"
    )
    parser.add_argument(
        "--months", type=int, nargs="*", default=None, help="arrival months to keep, e.g. 5 6 7"
    )
    parser.add_argument("--limit", type=int, default=2000, help="maximum bookings to replay")
    parser.add_argument(
        "--label-fraction",
        type=float,
        default=0.8,
        help="share of served predictions to reconcile with ground truth (0-1). "
        "Lower it to simulate a window the reconciliation job has not caught up with.",
    )
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument(
        "--seed", type=int, default=42, help="sampling seed, for reproducible replays"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--toy", action="store_true", help="replay from the toy dataset instead of the full one"
    )
    args = parser.parse_args()

    config = load_config()
    raw_path = config["data"]["toy_raw_data_path"] if args.toy else config["data"]["raw_data_path"]
    # Cleaned through the same function training uses, so the replayed rows are
    # drawn from exactly the distribution the model was fitted on - otherwise
    # any drift measured would just be the difference between two cleaning
    # implementations.
    frame = load_and_clean_data(raw_path, config["data"]["valid_classes_min_count"])

    if args.year is not None:
        frame = frame[frame["arrival_date_year"] == args.year]
    if args.months:
        frame = frame[frame["month"].isin(args.months)]
    if frame.empty:
        logger.error("no rows match the requested slice")
        return 1

    # A seeded sample, not `.head()`: the raw file is grouped by hotel rather
    # than ordered by date, so the first N rows of a year are almost all one
    # property in one season. Replaying that slice measures the bias of the
    # file's row order and reports it as drift. Seeded so two runs of this
    # script replay the same bookings and their reports are comparable.
    if len(frame) > args.limit:
        frame = frame.sample(n=args.limit, random_state=args.seed)
    logger.info(
        "replaying %s bookings (year=%s months=%s) against %s",
        len(frame),
        args.year,
        args.months,
        args.api_url,
    )

    served, labelled = replay(
        args.api_url, frame, args.label_fraction, args.batch_size, args.timeout
    )
    logger.info("done: %s predictions served, %s reconciled with ground truth", served, labelled)
    return 0 if served else 1


if __name__ == "__main__":
    sys.exit(main())
