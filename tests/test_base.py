"""Tests for BaseNaiveBayes, input validation, and log-sum-exp normalization."""

from __future__ import annotations

import numpy as np
import pytest

from naive_bayes.base import check_is_fitted, log_sum_exp
from naive_bayes.gaussian import GaussianNB


class DummyEstimator:
    """Mock estimator to test check_is_fitted helper."""

    def __init__(self) -> None:
        self.fitted_attr = None


def test_check_is_fitted() -> None:
    dummy = DummyEstimator()
    with pytest.raises(RuntimeError, match="is not fitted"):
        check_is_fitted(dummy, ["fitted_attr"])

    dummy.fitted_attr = "ready"
    check_is_fitted(dummy, ["fitted_attr"])


def test_log_sum_exp_numerical_stability() -> None:
    # Extreme values where naive np.exp would overflow or underflow
    log_vals = np.array([[-1000.0, -1001.0], [1000.0, 1001.0]])
    res = log_sum_exp(log_vals, axis=1)

    assert not np.isnan(res[0])
    assert not np.isnan(res[1])
    assert np.isclose(res[1], 1001.0 + np.log(1.0 + np.exp(-1.0)))

    # Pure -inf row
    neginf_row = np.array([[-np.inf, -np.inf]])
    res_neginf = log_sum_exp(neginf_row, axis=1)
    assert np.isneginf(res_neginf[0])


def test_unfitted_model_errors() -> None:
    gnb = GaussianNB()
    X = np.array([[1.0, 2.0]])
    with pytest.raises(RuntimeError, match="is not fitted"):
        gnb.predict(X)
    with pytest.raises(RuntimeError, match="is not fitted"):
        gnb.predict_proba(X)
    with pytest.raises(RuntimeError, match="is not fitted"):
        gnb.predict_log_proba(X)


def test_input_validation_errors() -> None:
    gnb = GaussianNB()
    with pytest.raises(ValueError, match="Expected 2D array"):
        gnb.fit(np.array([1.0, 2.0]), np.array([0, 1]))

    with pytest.raises(ValueError, match="zero samples"):
        gnb.fit(np.empty((0, 2)), np.empty(0))

    with pytest.raises(ValueError, match="Sample count mismatch"):
        gnb.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0]))

    with pytest.raises(ValueError, match="Expected 1D target array"):
        gnb.fit(np.array([[1.0, 2.0]]), np.array([[0]]))


def test_user_supplied_priors_validation() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="Number of priors"):
        GaussianNB(priors=np.array([0.5, 0.3, 0.2])).fit(X, y)

    with pytest.raises(ValueError, match="non-negative"):
        GaussianNB(priors=np.array([-0.2, 1.2])).fit(X, y)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        GaussianNB(priors=np.array([0.4, 0.4])).fit(X, y)


def test_feature_mismatch_on_predict() -> None:
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_train = np.array([0, 1])
    gnb = GaussianNB().fit(X_train, y_train)

    X_test = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Feature count mismatch"):
        gnb.predict(X_test)
