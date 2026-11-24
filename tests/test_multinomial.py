"""Tests for Multinomial Naive Bayes implementation and scikit-learn parity."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

from naive_bayes.multinomial import MultinomialNB


def test_multinomial_negative_features_rejected() -> None:
    X = np.array([[1.0, -2.0], [3.0, 4.0]])
    y = np.array([0, 1])
    mnb = MultinomialNB()

    with pytest.raises(ValueError, match="Negative values in X are not allowed"):
        mnb.fit(X, y)

    X_sp = sparse.csr_matrix([[1.0, -2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Negative values in X are not allowed"):
        mnb.fit(X_sp, y)


def test_multinomial_hand_calculated_probabilities() -> None:
    # 2 classes, 3 words
    # class 0: doc1 = [2, 1, 0], doc2 = [1, 2, 0] -> sum = [3, 3, 0]
    # class 1: doc3 = [0, 1, 3], doc4 = [0, 0, 2] -> sum = [0, 1, 5]
    X = np.array([
        [2, 1, 0],
        [1, 2, 0],
        [0, 1, 3],
        [0, 0, 2],
    ])
    y = np.array([0, 0, 1, 1])

    # alpha = 1.0 (Laplace smoothing)
    # class 0: count_smoothed = [4, 4, 1], total = 9 -> p = [4/9, 4/9, 1/9]
    # class 1: count_smoothed = [1, 2, 6], total = 9 -> p = [1/9, 2/9, 6/9]
    mnb = MultinomialNB(alpha=1.0).fit(X, y)

    assert mnb.feature_log_prob_ is not None
    p0 = np.exp(mnb.feature_log_prob_[0])
    p1 = np.exp(mnb.feature_log_prob_[1])

    assert np.allclose(p0, [4 / 9, 4 / 9, 1 / 9])
    assert np.allclose(p1, [1 / 9, 2 / 9, 6 / 9])


def test_multinomial_parity_with_sklearn() -> None:
    np.random.seed(42)
    X = np.random.poisson(lam=2.0, size=(60, 20))
    y = np.random.choice([0, 1, 2], size=60)

    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]

    for alpha in [0.1, 0.5, 1.0, 2.0]:
        custom = MultinomialNB(alpha=alpha).fit(X_train, y_train)
        ref = SklearnMultinomialNB(alpha=alpha).fit(X_train, y_train)

        custom_preds = custom.predict(X_test)
        ref_preds = ref.predict(X_test)
        assert np.array_equal(custom_preds, ref_preds)

        custom_proba = custom.predict_proba(X_test)
        ref_proba = ref.predict_proba(X_test)
        assert np.allclose(custom_proba, ref_proba, atol=1e-5)


def test_multinomial_sparse_dense_equivalence() -> None:
    np.random.seed(42)
    X_dense = np.random.poisson(lam=1.5, size=(30, 15))
    y = np.random.choice([0, 1], size=30)
    X_sparse = sparse.csr_matrix(X_dense)

    model_dense = MultinomialNB(alpha=1.0).fit(X_dense, y)
    model_sparse = MultinomialNB(alpha=1.0).fit(X_sparse, y)

    pred_dense = model_dense.predict(X_dense)
    pred_sparse = model_sparse.predict(X_sparse)

    assert np.array_equal(pred_dense, pred_sparse)

    prob_dense = model_dense.predict_proba(X_dense)
    prob_sparse = model_sparse.predict_proba(X_sparse)

    assert np.allclose(prob_dense, prob_sparse, atol=1e-7)


def test_multinomial_zero_smoothing() -> None:
    X = np.array([[2, 1, 0], [0, 1, 3]])
    y = np.array([0, 1])

    # With alpha=0, unseen features have log_prob = -inf
    mnb = MultinomialNB(alpha=0.0).fit(X, y)
    assert mnb.feature_log_prob_ is not None
    assert np.isneginf(mnb.feature_log_prob_[0, 2])
    assert np.isneginf(mnb.feature_log_prob_[1, 0])
