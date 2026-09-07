"""The statistics behind every number this monitor reports.

Deliberately implemented here rather than pulled from a drift library. Three
reasons, in order of weight:

1. **Auditability.** A drift alert is an operational claim about a production
   model. When someone asks six months later why version 7 was pulled, the
   answer has to be a formula and a threshold that were both under version
   control at the time - not "the library said so", where the library's binning
   strategy and default thresholds may have changed across a minor release.
   This is the same reason model-risk regimes (SR 11-7 and its equivalents)
   require drift statistics to be reproducible from documented inputs.
2. **Dependency surface.** `make security` fails the build on any HIGH or
   CRITICAL finding in the dependency tree. A drift framework drags in a large
   transitive graph - and its rendering stack - for four formulas that are
   forty lines of numpy. Every one of those packages would become this
   project's problem on the next CVE.
3. **It is genuinely small.** The whole file is below.

**Effect sizes, not p-values.** There is no hypothesis test here, and that is a
choice rather than an omission. With a window of tens of thousands of
predictions, a chi-square or KS test rejects the null on differences far too
small to change a single decision - the classic failure mode that makes a
p-value-driven monitor cry wolf every night until it is switched off. PSI and
Jensen-Shannon are *effect sizes*: they answer "how much has this moved", which
is the question an operator can act on, and they are compared against fixed,
documented thresholds.

The one place a confidence interval does appear is the concept-drift verdict
(`bootstrap_f1_ci`), because there the question genuinely is inferential: a
labelled window is a sample, and "live F1 is 0.03 below baseline" means nothing
until you know whether a sample that size could produce that gap by chance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.monitoring.profile import EPSILON, OTHER_CATEGORY

# The Population Stability Index convention used across credit-risk and model-
# risk practice, and the reason PSI is the primary statistic here: the bands are
# an industry-standard vocabulary, so "PSI 0.31 on lead_time" needs no local
# explanation to be actionable.
#
#   PSI < 0.10   no meaningful shift
#   0.10 - 0.25  moderate shift, worth watching
#   PSI > 0.25   significant shift, investigate
PSI_WARN_THRESHOLD = 0.10
PSI_ALERT_THRESHOLD = 0.25

# Bin proportions arrive either as a plain list (straight off a reference
# profile's JSON) or as a numpy array (straight out of align_categorical), and
# both are normalised on entry, so the signatures accept both rather than
# forcing every call site to convert.
Distribution = Sequence[float] | np.ndarray


@dataclass
class FeatureDrift:
    """Per-feature verdict. `psi` decides; the rest explains."""

    feature: str
    kind: str  # "numeric" | "categorical"
    psi: float
    jensen_shannon: float
    status: str  # "OK" | "WARN" | "ALERT"
    n_current: int
    new_categories: list[str]
    detail: str


def _normalise(counts: Distribution) -> np.ndarray:
    array = np.asarray(counts, dtype=float)
    total = array.sum()
    if total <= 0:
        return np.full(array.shape, 1.0 / max(array.size, 1))
    return array / total


def _smooth(proportions: np.ndarray) -> np.ndarray:
    """Floor every proportion at EPSILON and renormalise.

    Without this a single empty bin sends PSI to infinity, so one unseen value
    would drown out the actual shape of the shift. Flooring bounds the
    contribution of an empty bin instead of letting it dominate.
    """
    floored = np.clip(proportions, EPSILON, None)
    return floored / floored.sum()


def population_stability_index(reference: Distribution, current: Distribution) -> float:
    """PSI between two discrete distributions over the same bins.

        PSI = sum_i (c_i - r_i) * ln(c_i / r_i)

    Symmetric in the sense that it penalises mass moving in either direction,
    and unbounded above, which is what makes the 0.25 band meaningful.
    """
    ref = _smooth(_normalise(reference))
    cur = _smooth(_normalise(current))
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def jensen_shannon_distance(reference: Distribution, current: Distribution) -> float:
    """Bounded companion to PSI, in [0, 1].

    PSI is unbounded, which makes it excellent for thresholding and useless for
    comparing one feature against another - a PSI of 4.0 and a PSI of 0.8 are
    both "very drifted" but the ratio means nothing. JS distance is a metric on
    the probability simplex, so it ranks features honestly. Reported alongside
    PSI, never used for the verdict.
    """
    ref = _smooth(_normalise(reference))
    cur = _smooth(_normalise(current))
    mean = 0.5 * (ref + cur)
    divergence = 0.5 * np.sum(ref * np.log(ref / mean)) + 0.5 * np.sum(cur * np.log(cur / mean))
    # Clip: floating point can push a divergence a hair below zero when the two
    # distributions are identical, and sqrt of that is nan.
    return float(np.sqrt(max(divergence, 0.0)))


def classify(psi: float) -> str:
    if psi >= PSI_ALERT_THRESHOLD:
        return "ALERT"
    if psi >= PSI_WARN_THRESHOLD:
        return "WARN"
    return "OK"


def align_categorical(
    reference_frequencies: dict[str, float], current_values: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Put two categorical distributions on the same support.

    Levels the baseline never saw are folded into the same `__other__` bucket
    the baseline itself uses for its tail, and reported separately. Giving each
    unseen level its own bin would make PSI scale with the *number* of new
    levels rather than with the mass they carry, so a handful of one-off typos
    would outrank a genuine channel shift.
    """
    known = [level for level in reference_frequencies if level != OTHER_CATEGORY]
    support = [*known, OTHER_CATEGORY]
    index = {level: position for position, level in enumerate(support)}

    reference = np.array([reference_frequencies.get(level, 0.0) for level in support], dtype=float)

    current = np.zeros(len(support), dtype=float)
    unseen: set[str] = set()
    for value in current_values:
        level = str(value)
        position = index.get(level)
        if position is None:
            unseen.add(level)
            position = index[OTHER_CATEGORY]
        current[position] += 1.0

    return reference, current, sorted(unseen)


@dataclass
class PerformanceInterval:
    """A point estimate with the uncertainty that makes it decidable."""

    estimate: float
    lower: float
    upper: float
    n: int


def bootstrap_metric_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PerformanceInterval:
    """Percentile bootstrap confidence interval for any classification metric.

    Weighted F1 has no usable closed-form standard error - it is a ratio of
    sums over per-class quantities whose weights are themselves estimated from
    the same sample - so resampling is the honest way to get an interval. With
    a few thousand labelled rows and 1000 resamples this costs well under a
    second, which is nothing next to the cost of pulling a healthy model out of
    production on a difference that was noise.

    Seeded, because a monitoring verdict has to be reproducible: re-running the
    job over the same window must reach the same conclusion.
    """
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    n = truth.size
    estimate = float(metric(truth, predictions))
    if n == 0:
        return PerformanceInterval(estimate=estimate, lower=estimate, upper=estimate, n=0)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        # A resample can miss a rare class entirely; the metric is still defined
        # (weighted average over the classes present), which is the behaviour we
        # want - it widens the interval exactly when the sample is too thin to
        # support a confident verdict.
        samples[i] = metric(truth[idx], predictions[idx])

    tail = (1.0 - confidence) / 2.0
    return PerformanceInterval(
        estimate=estimate,
        lower=float(np.quantile(samples, tail)),
        upper=float(np.quantile(samples, 1.0 - tail)),
        n=int(n),
    )
