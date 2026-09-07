"""The reference baseline a drift report is measured against.

Drift is not a property of a dataset, it is a *comparison*. Every number this
system reports therefore needs a reference that is (a) precisely identified,
(b) immutable, and (c) reachable from wherever the monitor happens to run. The
obvious candidate - "just re-read the training CSV" - fails all three: the file
in S3 can be re-pushed, the monitor would need DVC and bucket credentials to
fetch tens of megabytes it will immediately throw away, and nothing would stop
the baseline from silently changing under a model that never retrained.

So the baseline is computed once, at training time, from the exact DataFrame the
model was fitted on, and logged as an artifact of that model's own MLflow run:

    <run>/monitoring/reference_profile.json

That makes it immutable (an MLflow artifact is write-once), precisely
identified (it is reachable only through the model version that produced it),
and cheap (kilobytes of histograms, not the dataset). It is the same pattern
SageMaker Model Monitor's baselining job and Vertex AI's skew detection use,
for the same reasons.

What it carries, and which question each part answers:

    numeric / categorical histograms  -> has P(X) moved?      (data drift)
    target_frequencies                -> has P(y-hat) moved?  (prediction drift)
    performance.mean_confidence       -> is the model less    (confidence drift,
    performance.mean_margin              sure than it was?     the unsupervised
                                                               proxy)
    performance.f1_weighted           -> is the model still   (CONCEPT drift, the
                                         *right*?              only supervised one)

`f1_weighted` is the single most important field in the file: concept drift is
measured as the live, label-joined F1 falling below this number by more than
the configured tolerance. Everything else is early warning; this is the actual
finding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Equal-frequency (quantile) bins, not equal-width. The features here are
# heavily right-skewed - lead_time, adr, days_in_waiting_list all have long
# thin tails - and equal-width binning would drop ~95% of the mass into the
# first bucket, leaving a PSI that cannot move no matter what the tail does.
DEFAULT_N_BINS = 10

# Frequencies are compared as ratios, so an empty bin would make PSI infinite
# (log of zero) on a single unseen value. Floor every proportion at this
# instead: with 10 bins it is three orders of magnitude below the smallest
# meaningful share, so it bounds the statistic without distorting it.
EPSILON = 1e-6

# A categorical feature with thousands of levels (country: ~180, agent: ~330)
# would make the profile larger than the model. Keep the head that carries the
# mass and fold the rest into one bucket, which is also what makes the
# "unseen category" signal meaningful rather than noise.
MAX_CATEGORY_LEVELS = 50
OTHER_CATEGORY = "__other__"


@dataclass
class NumericBaseline:
    """Quantile histogram of one numeric feature, plus enough to describe it."""

    bin_edges: list[float]
    frequencies: list[float]
    mean: float
    std: float
    p01: float
    p50: float
    p99: float
    missing_rate: float


@dataclass
class CategoricalBaseline:
    """Level frequencies of one categorical feature, truncated to the head."""

    frequencies: dict[str, float]
    missing_rate: float


@dataclass
class PerformanceBaseline:
    """How good, and how sure, the model was on data it had not seen.

    Measured on the held-out test split rather than on training data, because
    the whole point is to be the number live performance is compared to - a
    baseline taken in-sample would be optimistic by exactly the amount that
    makes every live window look like concept drift.
    """

    f1_weighted: float
    accuracy: float
    mean_confidence: float | None
    mean_margin: float | None
    n_eval: int


@dataclass
class ReferenceProfile:
    created_at: str
    n_rows: int
    target_column: str
    n_bins: int
    numeric: dict[str, NumericBaseline] = field(default_factory=dict)
    categorical: dict[str, CategoricalBaseline] = field(default_factory=dict)
    target_frequencies: dict[str, float] = field(default_factory=dict)
    performance: PerformanceBaseline | None = None

    # -- serialization -----------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReferenceProfile:
        performance = payload.get("performance")
        return cls(
            created_at=payload["created_at"],
            n_rows=int(payload["n_rows"]),
            target_column=payload["target_column"],
            n_bins=int(payload.get("n_bins", DEFAULT_N_BINS)),
            numeric={k: NumericBaseline(**v) for k, v in payload.get("numeric", {}).items()},
            categorical={
                k: CategoricalBaseline(**v) for k, v in payload.get("categorical", {}).items()
            },
            target_frequencies=payload.get("target_frequencies", {}),
            performance=PerformanceBaseline(**performance) if performance else None,
        )

    @classmethod
    def read(cls, path: str | Path) -> ReferenceProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def feature_names(self) -> list[str]:
        return sorted([*self.numeric.keys(), *self.categorical.keys()])


def _quantile_edges(values: np.ndarray, n_bins: int) -> list[float]:
    """Bin edges from the reference's own quantiles, open at both ends.

    Duplicate edges are collapsed: a feature where more than 1/n_bins of the
    mass sits on a single value (is_repeated_guest, babies, most of the
    `previous_*` counters) produces repeated quantiles, and np.histogram
    rejects a non-monotonic edge array. Collapsing yields fewer, wider bins,
    which is the honest representation of a spiky distribution.

    The outer edges are infinite so that a live value beyond anything seen in
    training still lands in a bin instead of being silently dropped - an
    out-of-range value is precisely the kind of drift worth catching.
    """
    raw = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(raw).astype(float)
    if edges.size < 2:
        # Constant feature: one bin covering everything. PSI will be 0 unless
        # the live data stops being constant, which is itself the finding.
        edges = np.array([float(values[0]) - 1.0, float(values[0]) + 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(e) for e in edges]


def histogram(values: np.ndarray, bin_edges: list[float]) -> list[float]:
    """Proportion of `values` falling in each bin defined by `bin_edges`."""
    counts, _ = np.histogram(values, bins=np.asarray(bin_edges, dtype=float))
    total = counts.sum()
    if total == 0:
        return [0.0] * len(counts)
    return [float(c) / float(total) for c in counts]


def _top_levels(series: pd.Series, max_levels: int) -> dict[str, float]:
    counts = series.astype(str).value_counts()
    if len(counts) > max_levels:
        head = counts.iloc[:max_levels]
        counts = pd.concat([head, pd.Series({OTHER_CATEGORY: counts.iloc[max_levels:].sum()})])
    total = float(counts.sum())
    if total == 0:
        return {}
    return {str(k): float(v) / total for k, v in counts.items()}


def build_reference_profile(
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    performance: PerformanceBaseline | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> ReferenceProfile:
    """Summarise the training distribution into a compact, immutable baseline.

    `features` must be the columns the model actually consumes. Profiling
    columns the model never sees would report drift that cannot possibly affect
    a prediction, which is the fastest way to teach a team to ignore the
    monitor.
    """
    numeric: dict[str, NumericBaseline] = {}
    categorical: dict[str, CategoricalBaseline] = {}

    for column in features.columns:
        series = features[column]
        missing_rate = float(series.isna().mean())
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna().to_numpy(dtype=float)
            if clean.size == 0:
                continue
            edges = _quantile_edges(clean, n_bins)
            numeric[str(column)] = NumericBaseline(
                bin_edges=edges,
                frequencies=histogram(clean, edges),
                mean=float(np.mean(clean)),
                std=float(np.std(clean)),
                p01=float(np.quantile(clean, 0.01)),
                p50=float(np.quantile(clean, 0.50)),
                p99=float(np.quantile(clean, 0.99)),
                missing_rate=missing_rate,
            )
        else:
            levels = _top_levels(series.dropna(), MAX_CATEGORY_LEVELS)
            if not levels:
                continue
            categorical[str(column)] = CategoricalBaseline(
                frequencies=levels, missing_rate=missing_rate
            )

    target_counts = target.astype(str).value_counts()
    target_total = float(target_counts.sum())

    return ReferenceProfile(
        created_at=datetime.now(UTC).isoformat(),
        n_rows=int(len(features)),
        target_column=target_column,
        n_bins=n_bins,
        numeric=numeric,
        categorical=categorical,
        target_frequencies={str(k): float(v) / target_total for k, v in target_counts.items()},
        performance=performance,
    )
