"""Helpers for imputing missing values in numeric and categorical columns."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def numerical_missing(
    frame: pd.DataFrame,
    estimator,
    column: str,
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Replace missing or zero entries in ``column`` using a provided estimator.

    ``estimator`` can be a scalar, Series, or DataFrame. When ``group_columns``
    is provided, the estimator is looked up using the row values for those
    columns, mirroring the notebook's conditional logic.
    """

    data = frame.copy()
    mask_missing = data[column].isna() | (data[column] == 0)

    if group_columns is None:
        group_columns = []

    for idx, row in data.loc[mask_missing].iterrows():
        if not group_columns:
            value = estimator
        elif len(group_columns) == 1:
            value = estimator.loc[row[group_columns[0]]]
        else:
            key = tuple(row[g] for g in group_columns)
            value = estimator.loc[key]
        data.at[idx, column] = value

    return data


def categorical_missing(
    frame: pd.DataFrame,
    column: str,
    probabilities: Iterable[float] | None = None,
    values: Sequence | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Impute missing categorical values by drawing from a multinomial.

    ``probabilities`` and ``values`` allow deterministic overrides. When not
    supplied, the function estimates class probabilities from the non-missing
    values in ``column`` and uses those labels as the candidate values.
    """

    data = frame.copy()
    if rng is None:
        rng = np.random.default_rng()

    observed = data[column].dropna()
    if values is None:
        values = observed.unique()
    if probabilities is None:
        counts = observed.value_counts(normalize=True)
        probabilities = counts.loc[values].to_numpy()

    missing_idx = np.where(data[column].isna())[0]
    if len(missing_idx) == 0:
        return data

    draws = rng.multinomial(1, probabilities, size=len(missing_idx))
    sampled = [values[row.argmax()] for row in draws]
    data.loc[missing_idx, column] = sampled
    return data
