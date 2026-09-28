"""Feature definition and the model pipeline built from it.

Which columns the model may see is declared once, in config.yaml's
`features:` block, and everything else (train.py, the drift profile, the
tests that guard against leakage) reads it from there.

The derived features are computed *inside* the scikit-learn pipeline by
`BookingFeatureBuilder`, never in the data-cleaning step. The API therefore
posts the raw booking fields and the logged model does the same
transformation at serving time that it did at training time - there is no
second implementation that could drift out of sync with this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class FeatureSpec:
    """The model's input contract, as declared in config.yaml."""

    numeric: list[str]
    categorical: list[str]
    # column -> period, e.g. {"month": 12}. Encoded as sin/cos so that
    # December and January end up next to each other instead of 11 apart.
    cyclical: dict[str, int] = field(default_factory=dict)
    # Identifier columns (travel agent id, company id) where only the
    # *presence* of a value carries meaning. NaN means "booked without an
    # agent/company", so these become has_<column> flags instead of being
    # median-imputed and scaled like a quantity.
    presence_flags: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FeatureSpec:
        features = config["features"]
        return cls(
            numeric=list(features.get("numeric", [])),
            categorical=list(features.get("categorical", [])),
            cyclical={str(k): int(v) for k, v in (features.get("cyclical") or {}).items()},
            presence_flags=list(features.get("presence_flags", [])),
        )

    @property
    def input_columns(self) -> list[str]:
        """Raw columns the pipeline needs from a booking, in a stable order."""
        return [*self.numeric, *self.cyclical, *self.presence_flags, *self.categorical]

    @property
    def monitored_numeric_columns(self) -> list[str]:
        """Raw numeric inputs whose mean is meaningful to track for drift.

        Presence-flag sources are identifiers, so their mean says nothing.
        """
        return [*self.numeric, *self.cyclical]


def cyclical_names(column: str) -> list[str]:
    return [f"{column}_sin", f"{column}_cos"]


def presence_name(column: str) -> str:
    return f"has_{column}"


class BookingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds the derived columns (sin/cos pairs, has_<id> flags) to a booking frame.

    Stateless: nothing is learned in `fit`, so it cannot leak information
    from the validation or test split into training.
    """

    def __init__(
        self,
        cyclical: dict[str, int] | None = None,
        presence_flags: list[str] | None = None,
    ) -> None:
        self.cyclical = cyclical
        self.presence_flags = presence_flags

    def fit(self, X: pd.DataFrame, y: Any = None) -> BookingFeatureBuilder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for column, period in (self.cyclical or {}).items():
            angle = 2 * np.pi * pd.to_numeric(out[column], errors="coerce") / period
            sin_name, cos_name = cyclical_names(column)
            out[sin_name] = np.sin(angle)
            out[cos_name] = np.cos(angle)
        for column in self.presence_flags or []:
            # IDs start at 1 in the source data. Treating <= 0 as "absent"
            # too keeps a client that sends 0 instead of null from being
            # counted as a booking made through agent/company number 0.
            values = pd.to_numeric(out[column], errors="coerce")
            out[presence_name(column)] = (values.notna() & (values > 0)).astype(int)
        return out


def build_preprocessor(spec: FeatureSpec) -> Pipeline:
    derived_numeric = [name for column in spec.cyclical for name in cyclical_names(column)]
    flags = [presence_name(column) for column in spec.presence_flags]

    columns = ColumnTransformer(
        transformers=[
            # No scaler: tree ensembles are invariant to monotonic rescaling.
            ("num", SimpleImputer(strategy="median"), [*spec.numeric, *derived_numeric]),
            ("flags", "passthrough", flags),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        # Rare categories (most of the ~180 countries) are
                        # grouped instead of getting a column each, and a
                        # category never seen in training maps to that same
                        # group at serving time instead of raising.
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
                        ),
                    ]
                ),
                spec.categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            (
                "derive",
                BookingFeatureBuilder(cyclical=spec.cyclical, presence_flags=spec.presence_flags),
            ),
            ("columns", columns),
        ]
    )


def build_model_pipeline(spec: FeatureSpec, params: dict[str, Any], random_state: int) -> Pipeline:
    """Preprocessing + a class-weighted RandomForest.

    Class imbalance is handled with `class_weight="balanced_subsample"` and
    nothing else. SMOTE was dropped: it interpolated between one-hot rows
    (producing "0.4 of a country"), slowed every trial down, and stacking it
    with class weights would correct the imbalance twice.
    """
    classifier = RandomForestClassifier(
        **params,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", build_preprocessor(spec)), ("classifier", classifier)])
