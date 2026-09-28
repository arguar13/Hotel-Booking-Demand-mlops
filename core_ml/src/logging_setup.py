"""One structured-logging setup for every core_ml entrypoint.

Same approach as api/main.py: one log line per event, as a JSON object, ready
for CloudWatch/Elasticsearch - not prose meant for a human tailing a terminal.
Called from each entrypoint (train, promote_model, drift_check) so a
CronJob's output and a training run's output can be parsed the same way.
"""

import logging
import sys

import structlog


def configure_logging() -> None:
    # MLflow >=3 prints run/model links decorated with emoji. A Windows console
    # defaults to cp1252, which cannot encode them, so the process dies with
    # UnicodeEncodeError *after* the work is done - a non-zero exit for a run
    # that actually succeeded. Force UTF-8 on the standard streams; a no-op on
    # Linux and in CI, where they already are.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
