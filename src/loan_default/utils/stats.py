"""Statistical utilities derived from the notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy as spy


def modified_zscore(data: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    Compute modified z-scores column-wise and flag values beyond ``threshold``.

    Returns a boolean DataFrame where ``True`` indicates an outlier per column.
    The first column is skipped to mirror the original notebook behavior.
    """

    med = data.apply(lambda x: abs(x - x.median()))
    mad = med.apply(lambda x: x.median())
    flags = []
    for idx in range(1, len(mad)):
        if mad.iloc[idx] == 0:
            continue
        score = abs(0.6745 * (data.iloc[:, idx] - data.iloc[:, idx].median()) / mad.iloc[idx])
        flags.append(score > threshold)
    if not flags:
        return pd.DataFrame(index=data.index)
    return pd.concat(flags, axis=1).reset_index(drop=True)


def mahalanobis_distance(data: pd.DataFrame) -> np.ndarray:
    """
    Compute Mahalanobis distance for each observation.

    Falls back to returning an empty array if covariance inversion fails.
    """

    centered = data - data.mean()
    cov = np.cov(centered.values.T)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.array([])
    left = centered.to_numpy() @ inv_cov
    distances = np.sqrt(np.einsum("ij,ij->i", left, centered.to_numpy()))
    return distances


def normality_test(feature: pd.Series) -> dict:
    """Run multiple normality tests and return their statistics."""

    return {
        "d_agostino": spy.stats.normaltest(feature),
        "anderson": spy.stats.anderson(feature),
        "kstest": spy.stats.kstest(feature, np.random.normal(feature.mean(), feature.std(), len(feature))),
    }
