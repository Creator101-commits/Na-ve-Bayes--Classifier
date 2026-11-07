"""Regression checks for numerical and split defects found in review."""

import numpy as np
import pytest
from scipy import sparse

from naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from naive_bayes.data import split_text_indices
from naive_bayes.evaluation import compute_expected_calibration_error


@pytest.mark.parametrize("as_sparse", [False, True])
def test_zero_smoothing_uses_impossible_events_without_nan(as_sparse):
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    features = sparse.csr_matrix(x) if as_sparse else x
    for cls in [MultinomialNB, BernoulliNB]:
        model = cls(alpha=0).fit(features, y)
        np.testing.assert_allclose(model.predict_proba(features), [[1, 0], [1, 0], [0, 1], [0, 1]])


@pytest.mark.parametrize("cls", [GaussianNB, MultinomialNB, BernoulliNB])
def test_nonfinite_features_are_rejected(cls):
    with pytest.raises(ValueError, match="finite"):
        cls().fit(np.array([[0.0, np.nan], [1.0, 2.0]]), np.array([0, 1]))


def test_multinomial_rejects_negative_prediction_features():
    model = MultinomialNB().fit(np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([0, 1]))
    with pytest.raises(ValueError, match="Negative"):
        model.predict(np.array([[-1.0, 1.0]]))


def test_gaussian_rejects_degenerate_likelihood():
    with pytest.raises(ValueError, match="variance"):
        GaussianNB().fit(np.ones((4, 2)), np.array([0, 0, 1, 1]))


def test_calibration_respects_model_class_columns_when_test_class_is_absent():
    ece = compute_expected_calibration_error(
        np.array([1, 1]), np.array([[0.1, 0.9], [0.2, 0.8]]), classes=np.array([0, 1])
    )
    assert ece == pytest.approx(0.15)


def test_normalized_duplicates_never_cross_partitions():
    texts = [f"message {i}" for i in range(30)]
    texts += [f"  MESSAGE {i}  " for i in range(30)]
    labels = ["a" if i < 15 else "b" for i in range(30)] * 2
    splits = split_text_indices(texts, labels)
    normalized = [{" ".join(texts[i].casefold().split()) for i in split} for split in splits]
    assert all(not normalized[i] & normalized[j] for i in range(3) for j in range(i))
