"""Tests for Gaussian Naive Bayes implementation and scikit-learn parity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

from naive_bayes.gaussian import GaussianNB


def test_gaussian_hand_calculated_parameters() -> None:
    X = np.array([
        [1.0, 10.0],
        [3.0, 12.0],
        [5.0, 20.0],
        [7.0, 22.0],
    ])
    y = np.array([0, 0, 1, 1])

    gnb = GaussianNB(var_smoothing=0.0).fit(X, y)

    # Class 0: mean = [2.0, 11.0], var = [1.0, 1.0]
    # Class 1: mean = [6.0, 21.0], var = [1.0, 1.0]
    assert gnb.theta_ is not None
    assert gnb.var_ is not None
    assert np.allclose(gnb.theta_[0], [2.0, 11.0])
    assert np.allclose(gnb.theta_[1], [6.0, 21.0])
    assert np.allclose(gnb.var_[0], [1.0, 1.0])
    assert np.allclose(gnb.var_[1], [1.0, 1.0])


def test_gaussian_string_labels() -> None:
    X = np.array([[1.0, 2.0], [1.2, 1.8], [8.0, 9.0], [8.2, 8.8]])
    y = np.array(["cat", "cat", "dog", "dog"])

    gnb = GaussianNB().fit(X, y)
    preds = gnb.predict(X)
    assert np.array_equal(preds, y)


def test_gaussian_rejects_sparse() -> None:
    X_sparse = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    gnb = GaussianNB()

    with pytest.raises(TypeError, match="does not support sparse"):
        gnb.fit(X_sparse, y)


def test_gaussian_constant_features_no_nan() -> None:
    # Feature 1 has constant value 5.0 (variance 0.0)
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 8.0], [5.0, 9.0]])
    y = np.array([0, 0, 1, 1])

    gnb = GaussianNB(var_smoothing=1e-9).fit(X, y)
    probs = gnb.predict_proba(X)

    assert not np.isnan(probs).any()
    assert np.allclose(np.sum(probs, axis=1), 1.0)


@pytest.mark.parametrize("dataset_loader", [load_iris, load_wine, load_breast_cancer])
def test_gaussian_parity_with_sklearn(dataset_loader: object) -> None:
    data = dataset_loader()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.75 * len(X))
    X_train, X_test = X[indices[:split]], X[indices[split:]]
    y_train = y[indices[:split]]

    var_smoothing = 1e-9
    custom = GaussianNB(var_smoothing=var_smoothing).fit(X_train, y_train)
    ref = SklearnGaussianNB(var_smoothing=var_smoothing).fit(X_train, y_train)

    custom_pred = custom.predict(X_test)
    ref_pred = ref.predict(X_test)

    agreement = np.mean(custom_pred == ref_pred)
    assert agreement >= 0.98

    custom_proba = custom.predict_proba(X_test)
    ref_proba = ref.predict_proba(X_test)
    assert np.allclose(custom_proba, ref_proba, atol=1e-4)


def test_gaussian_single_sample_prediction() -> None:
    X = np.array([[1.0, 2.0], [5.0, 6.0]])
    y = np.array([0, 1])
    gnb = GaussianNB().fit(X, y)

    single_x = np.array([[1.1, 2.1]])
    pred = gnb.predict(single_x)
    proba = gnb.predict_proba(single_x)

    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
    assert np.isclose(np.sum(proba), 1.0)
