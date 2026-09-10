"""Drift monitoring: the read half of the production feedback loop.

`api/monitoring.py` writes what production actually did (predictions, and the
ground truth that later arrives for them); `drift_check.py` reads a recent
batch of that back and compares it against the training-time reference
profile with a simple per-feature z-test. See that module's docstring for
the full picture.
"""
