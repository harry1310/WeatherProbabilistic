"""Frequentist logistic regression baseline.

The point of this baseline is to give us a "known-working" reference for
the Bayesian model in `src/models/phase1.py`. Weakly informative priors
on a dataset of 10k+ rows should produce nearly identical results to
maximum-likelihood logistic regression - if they don't, something has
gone wrong in the Bayesian implementation.

Features are standardised (z-scored) on the training set and the same
transform is applied to the test set. Standardisation also matters for
the Bayesian model: with raw un-scaled features the posterior has very
different scales across parameters and NUTS samples poorly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineResult:
    model: LogisticRegression
    scaler: StandardScaler
    feature_names: list[str]
    coefficients: pd.Series  # indexed by feature name
    intercept: float
    test_brier: float
    test_log_loss: float
    test_proba: np.ndarray


def fit_frequentist_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> BaselineResult:
    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values)
    X_test_s = scaler.transform(X_test.values)

    # No regularisation penalty (C very large) so we approximate plain MLE.
    # Bayesian model uses N(0, 1) priors which are much weaker than typical
    # sklearn defaults, so the closest frequentist analogue is "no penalty".
    model = LogisticRegression(
        penalty=None, solver="lbfgs", max_iter=2000
    )
    model.fit(X_train_s, y_train.values)

    proba = model.predict_proba(X_test_s)[:, 1]
    coefs = pd.Series(model.coef_[0], index=feature_names, name="coef")

    return BaselineResult(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        coefficients=coefs,
        intercept=float(model.intercept_[0]),
        test_brier=float(brier_score_loss(y_test.values, proba)),
        test_log_loss=float(log_loss(y_test.values, proba)),
        test_proba=proba,
    )


if __name__ == "__main__":
    from src.data import prepare_phase1_dataset

    ds = prepare_phase1_dataset()
    res = fit_frequentist_baseline(
        ds.X_train, ds.y_train, ds.X_test, ds.y_test, ds.feature_names
    )
    print(f"\nFrequentist baseline:")
    print(f"  intercept   {res.intercept:+.4f}")
    print(f"  test Brier  {res.test_brier:.4f}")
    print(f"  test logloss {res.test_log_loss:.4f}")
    print("\nCoefficients (standardised features):")
    print(res.coefficients.to_string())
