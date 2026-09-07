"""Drift monitoring: the read half of the production feedback loop.

`api/monitoring.py` writes what production actually did (predictions, and the
ground truth that later arrives for them); this package reads that back on a
schedule and decides whether the model still fits the world it is deployed in.

    profile.py     the immutable training baseline, logged with the model
    statistics.py  PSI, Jensen-Shannon, bootstrap intervals
    store.py       windowed, read-only access to the prediction log
    drift_monitor.py  the CronJob entrypoint that ties the three together
    report.py      the human-readable artifact an operator opens
"""
