"""Promotion: the canary/shadow gate between a registered model version and
the alias production actually serves.

train.py's quality gate is absolute - min_f1_threshold - and deliberately
stops there (see its own comment): clearing a floor says nothing about
whether a version is better or worse than the one already aliased. A
retrain triggered by mitigation.py after a confirmed ALERT is exactly the
case where that gap matters most - a challenger trained on the same
drifted data the monitor just flagged could clear the floor and still be
worse than what it would replace. drift_monitor.py's original design
explicitly refused to let any retrain reach production unattended; this
module is the thing that now stands between "a version exists" and "a
version serves traffic", for every promotion, not only automated ones -
a manual retrain that regresses deserves the same check.

**What "canary/shadow" means here, and the trade-off in that.** The most
rigorous version of this check would re-score the challenger and the
champion against a replayed, held-out slice of *current* production
traffic - a true shadow deployment. This project does not build that; it
would need a labelled slice kept back from both models and a second
inference path running unpromoted challengers against live requests, real
infrastructure this project's traffic volume does not justify building
(the same class of trade-off already made explicitly for Kafka in
api/monitoring.py). Instead, both figures come from each version's own
`monitoring/reference_profile.json` - `performance.f1_weighted`, already
logged by train.py against that run's own held-out test split. That is a
weaker guarantee than a live comparison: the two splits are not the same
rows if the underlying dataset changed between the two training runs. But
it is the exact standard drift_monitor.py's own concept drift check
already holds a *serving* model to - comparing live F1 against a
baseline_f1 read from this same field, never re-scoring the champion live
- so a challenger is measured by the same yardstick production already is,
rather than a second, invented one.

**The gate.** challenger.f1_weighted must be no more than `canary_tolerance`
below champion.f1_weighted (config.yaml's `model.canary_tolerance` - a
small negative number, not zero; see that file's comment for why requiring
a strict improvement on every retrain is the wrong bar). No version
currently aliased (first deployment) always promotes - there is nothing to
compare against.

**Rollback.** Every promotion tags the new version with
`promoted_from_version`, the version it replaced ("none" for a bootstrap
promotion). `--rollback` reads that tag off whatever is currently aliased
and moves the alias back exactly one step. Deliberately one step, not a
full history walk: automatically unwinding more than one promotion without
a human choosing which earlier version is actually safe is exactly the
kind of unattended authority mitigation.py's own docstring argues against
granting this system.
"""

from __future__ import annotations

import argparse
import sys

import mlflow
import structlog
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config_loader import load_config
from src.monitoring.drift_monitor import load_reference_profile

log = structlog.get_logger("hotel_mlops.promote_model")


class PromotionError(Exception):
    """A candidate exists but was rejected, or there is nothing to roll back to."""


def _f1_for_run(run_id: str) -> float:
    profile = load_reference_profile(run_id)
    if profile.performance is None:
        raise PromotionError(
            f"run {run_id} has no monitoring/reference_profile.json performance baseline - "
            "cannot canary-compare it. Every run produced by train.py logs one; a run "
            "missing it predates that artifact and cannot be promoted through this path."
        )
    return float(profile.performance.f1_weighted)


def _promote_version(
    client: MlflowClient,
    registry_name: str,
    registry_alias: str,
    version: str,
    previous_version: str | None,
) -> None:
    client.set_model_version_tag(
        registry_name, version, "promoted_from_version", previous_version or "none"
    )
    client.set_registered_model_alias(registry_name, registry_alias, version)
    log.info(
        "model_version_aliased",
        registry_name=registry_name,
        alias=registry_alias,
        version=version,
        previous_version=previous_version or "none",
    )


def promote(run_id: str, config: dict | None = None) -> str:
    """Promote the model version produced by `run_id`, if it clears the canary bar.

    Returns the promoted version's registry version number. Raises
    PromotionError on rejection - the caller (CI, a human, or mitigation.py's
    own pipeline) decides what a rejection means; this function's job is
    only to decide the verdict and record it.
    """
    config = config or load_config()
    model_config = config["model"]
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    run = client.get_run(run_id)
    if run.data.tags.get("quality_gate") != "passed":
        raise PromotionError(
            f"run {run_id} does not carry quality_gate=passed - either it failed "
            "train.py's absolute floor or this is not a training run. Refusing to promote it."
        )

    [candidate] = client.search_model_versions(f"run_id='{run_id}'")
    registry_name = model_config["registry_name"]
    registry_alias = model_config["registry_alias"]

    try:
        champion = client.get_model_version_by_alias(registry_name, registry_alias)
    except MlflowException:
        champion = None

    if champion is None:
        log.info(
            "promotion_bootstrap",
            registry_name=registry_name,
            version=candidate.version,
            reason="no version currently aliased - nothing to compare against",
        )
        _promote_version(client, registry_name, registry_alias, candidate.version, None)
        return str(candidate.version)

    if champion.run_id is None:
        raise PromotionError(
            f"version {champion.version}, currently aliased '{registry_alias}', has no "
            "run_id on record - cannot look up its performance baseline to canary-compare "
            "against. This should be unreachable for any version promoted through this module."
        )

    challenger_f1 = _f1_for_run(run_id)
    champion_f1 = _f1_for_run(champion.run_id)
    tolerance = float(model_config.get("canary_tolerance", 0.0))
    gap = challenger_f1 - champion_f1

    if gap < tolerance:
        client.set_tag(run_id, "canary_verdict", "rejected")
        reason = (
            f"challenger f1_weighted={challenger_f1:.4f} is {abs(gap):.4f} below champion "
            f"f1_weighted={champion_f1:.4f} (version {champion.version}), beyond the "
            f"canary_tolerance of {tolerance} - refusing to promote"
        )
        log.error(
            "promotion_rejected",
            registry_name=registry_name,
            candidate_version=candidate.version,
            champion_version=champion.version,
            challenger_f1=challenger_f1,
            champion_f1=champion_f1,
        )
        raise PromotionError(reason)

    client.set_tag(run_id, "canary_verdict", "accepted")
    log.info(
        "promotion_accepted",
        registry_name=registry_name,
        candidate_version=candidate.version,
        champion_version=champion.version,
        challenger_f1=challenger_f1,
        champion_f1=champion_f1,
        gap=gap,
    )
    _promote_version(client, registry_name, registry_alias, candidate.version, champion.version)
    return str(candidate.version)


def rollback(config: dict | None = None) -> str:
    """Move the alias back exactly one promotion, using the lineage promote() recorded."""
    config = config or load_config()
    model_config = config["model"]
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    client = MlflowClient()

    registry_name = model_config["registry_name"]
    registry_alias = model_config["registry_alias"]

    current = client.get_model_version_by_alias(registry_name, registry_alias)
    previous_version = current.tags.get("promoted_from_version")
    if not previous_version or previous_version == "none":
        raise PromotionError(
            f"version {current.version} (currently '{registry_alias}') has no recorded "
            "previous version - either it was never promoted through promote_model.py, "
            "or it is the first version ever promoted. Nothing to roll back to."
        )

    client.set_registered_model_alias(registry_name, registry_alias, previous_version)
    log.info(
        "rollback_completed",
        registry_name=registry_name,
        alias=registry_alias,
        from_version=current.version,
        to_version=previous_version,
    )
    return previous_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="MLflow run id of the candidate to promote")
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Move the alias back to the previously promoted version, ignoring --run-id",
    )
    args = parser.parse_args()

    try:
        if args.rollback:
            rollback()
        else:
            if not args.run_id:
                parser.error("--run-id is required unless --rollback is passed")
            promote(args.run_id)
    except PromotionError as e:
        log.error("promote_model_failed", error=str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
