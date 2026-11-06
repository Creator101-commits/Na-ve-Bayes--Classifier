"""Tests for Bernoulli Naive Bayes implementation and absent-feature behavior."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB

from naive_bayes.bernoulli import BernoulliNB


def test_bernoulli_absent_feature_distinction() -> None:
    # Key property of Bernoulli NB: absence of a feature carries information!
    # Feature 0 is present in all class 0, absent in class 1.
    # Feature 1 is present in class 1, absent in class 0.
    X = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    y = np.array([0, 0, 1, 1])

    bnb = BernoulliNB(alpha=1.0, binarize=None).fit(X, y)

    # For sample [0, 0], absence of feature 0 hurts class 0,
    # and absence of feature 1 hurts class 1 equally
    p_empty = bnb.predict_proba(np.array([[0, 0]]))
    assert np.isclose(p_empty[0, 0], 0.5)

    # For sample [1, 0], feature 0 is present AND feature 1 is absent -> strong class 0
    p_c0 = bnb.predict_proba(np.array([[1, 0]]))
    assert p_c0[0, 0] > 0.8


def test_bernoulli_binarize_threshold() -> None:
    # Continuous data binarized at threshold 0.5
    X = np.array([[0.2, 0.8], [0.9, 0.1], [0.1, 0.7]])
    y = np.array([0, 1, 0])

    bnb = BernoulliNB(alpha=1.0, binarize=0.5).fit(X, y)
    assert bnb.feature_count_ is not None

    # Class 0: samples [0.2, 0.8] -> [0, 1], [0.1, 0.7] -> [0, 1]. Counts = [0, 2]
    # Class 1: sample [0.9, 0.1] -> [1, 0]. Counts = [1, 0]
    assert np.allclose(bnb.feature_count_[0], [0, 2])
    assert np.allclose(bnb.feature_count_[1], [1, 0])


def test_bernoulli_nonbinary_input_error_when_no_binarize() -> None:
    X = np.array([[2.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    bnb = BernoulliNB(binarize=None)

    with pytest.raises(ValueError, match="Expected binary data"):
        bnb.fit(X, y)


def test_bernoulli_parity_with_sklearn() -> None:
    np.random.seed(42)
    # Generate binary matrix: 50 samples, 10 features, 2 classes
    X = np.random.binomial(n=1, p=0.3, size=(50, 10))
    y = np.random.choice([0, 1], size=50)

    X_train, X_test = X[:35], X[35:]
    y_train = y[:35]

    for alpha in [0.1, 0.5, 1.0, 2.0]:
        custom = BernoulliNB(alpha=alpha, binarize=None).fit(X_train, y_train)
        ref = SklearnBernoulliNB(alpha=alpha, binarize=None).fit(X_train, y_train)

        custom_preds = custom.predict(X_test)
        ref_preds = ref.predict(X_test)
        assert np.array_equal(custom_preds, ref_preds)

        custom_proba = custom.predict_proba(X_test)
        ref_proba = ref.predict_proba(X_test)
        assert np.allclose(custom_proba, ref_proba, atol=1e-5)


def test_bernoulli_sparse_support() -> None:
    np.random.seed(42)
    X_dense = np.random.binomial(n=1, p=0.2, size=(30, 8))
    y = np.random.choice([0, 1], size=30)
    X_sparse = sparse.csr_matrix(X_dense)

    model_dense = BernoulliNB(alpha=1.0, binarize=None).fit(X_dense, y)
    model_sparse = BernoulliNB(alpha=1.0, binarize=None).fit(X_sparse, y)

    prob_dense = model_dense.predict_proba(X_dense)
    prob_sparse = model_sparse.predict_proba(X_sparse)

    assert np.allclose(prob_dense, prob_sparse, atol=1e-7)
