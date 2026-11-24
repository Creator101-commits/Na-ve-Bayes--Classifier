"""Reproducible benchmark experiments and mathematical verification."""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import (  # type: ignore[import-untyped]
    BernoulliNB as SklearnBernoulliNB,
)
from sklearn.naive_bayes import (
    GaussianNB as SklearnGaussianNB,
)
from sklearn.naive_bayes import (
    MultinomialNB as SklearnMultinomialNB,
)

from naive_bayes.bernoulli import BernoulliNB
from naive_bayes.data import (
    build_text_pipeline,
    load_cancer_dataset,
    load_iris_dataset,
    load_sms_fixture,
    load_wine_dataset,
    split_text_indices,
    train_val_test_split,
)
from naive_bayes.evaluation import (
    ReferenceComparisonResult,
    compare_with_reference,
)
from naive_bayes.gaussian import GaussianNB
from naive_bayes.multinomial import MultinomialNB


def verify_mathematics() -> dict[str, bool]:
    """Run hand-calculated checks confirming mathematical correctness of estimators."""
    results: dict[str, bool] = {}

    # 1. Gaussian hand check
    # Class 0: [1.0, 2.0], [3.0, 4.0] -> mean=[2.0, 3.0], var=[1.0, 1.0]
    # Class 1: [5.0, 6.0], [7.0, 8.0] -> mean=[6.0, 7.0], var=[1.0, 1.0]
    X_g = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_g = np.array([0, 0, 1, 1])
    gnb = GaussianNB(var_smoothing=0.0)
    gnb.fit(X_g, y_g)

    expected_means = np.array([[2.0, 3.0], [6.0, 7.0]])
    expected_vars = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert gnb.theta_ is not None and gnb.var_ is not None
    means_match = bool(np.allclose(gnb.theta_, expected_means))
    vars_match = bool(np.allclose(gnb.var_, expected_vars))
    results["gaussian_moments"] = means_match and vars_match

    # Predict test point [2.0, 3.0] -> exact mean of class 0
    # Class 0 posterior should be higher than Class 1
    prob = gnb.predict_proba(np.array([[2.0, 3.0]]))
    results["gaussian_decision"] = bool(prob[0, 0] > prob[0, 1] and np.isclose(np.sum(prob), 1.0))

    # 2. Multinomial hand check
    # 2 classes, 3 features
    # Class 0: [2, 1, 0], [1, 2, 0] -> sum counts = [3, 3, 0], total words = 6
    # With alpha=1.0: smoothed counts = [4, 4, 1], total smoothed = 9
    # P(f|c=0) = [4/9, 4/9, 1/9]
    X_m = np.array([[2, 1, 0], [1, 2, 0], [0, 1, 3], [0, 0, 2]])
    y_m = np.array([0, 0, 1, 1])
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_m, y_m)

    expected_p0 = np.array([4 / 9, 4 / 9, 1 / 9])
    assert mnb.feature_log_prob_ is not None
    actual_p0 = np.exp(mnb.feature_log_prob_[0])
    results["multinomial_smoothed_likelihood"] = bool(np.allclose(actual_p0, expected_p0))

    # 3. Bernoulli hand check
    # Class 0: [1, 0], [1, 1] (2 samples) -> presence counts = [2, 1]
    # alpha=1.0 -> denominator = 2 + 2*1 = 4
    # P(f1=1|c=0) = (2+1)/4 = 3/4 = 0.75
    # P(f2=1|c=0) = (1+1)/4 = 2/4 = 0.50
    X_b = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    y_b = np.array([0, 0, 1, 1])
    bnb = BernoulliNB(alpha=1.0, binarize=None)
    bnb.fit(X_b, y_b)

    assert bnb.feature_log_prob_ is not None
    actual_pb0 = np.exp(bnb.feature_log_prob_[0])
    expected_pb0 = np.array([0.75, 0.50])
    results["bernoulli_smoothed_likelihood"] = bool(np.allclose(actual_pb0, expected_pb0))

    return results


def run_continuous_benchmarks() -> dict[str, ReferenceComparisonResult]:
    """Run Gaussian Naive Bayes benchmarks on Iris, Wine, and Breast Cancer datasets."""
    datasets = [
        load_iris_dataset(),
        load_wine_dataset(),
        load_cancer_dataset(),
    ]
    results: dict[str, ReferenceComparisonResult] = {}

    for ds in datasets:
        X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
            ds.X, ds.y, val_size=0.15, test_size=0.15, random_state=42
        )
        custom_gnb = GaussianNB(var_smoothing=1e-9)
        ref_gnb = SklearnGaussianNB(var_smoothing=1e-9)

        comp = compare_with_reference(custom_gnb, ref_gnb, X_tr, y_tr, X_te, y_te)
        results[ds.name] = comp

    return results


def run_text_benchmarks() -> dict[str, dict[str, float]]:
    """Run text classification benchmarks comparing Multinomial and Bernoulli NB."""
    texts, labels = load_sms_fixture()
    y = np.array(labels)
    train_idx, _, test_idx = split_text_indices(texts, labels)

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Fit vectorizers on training data only
    vec_count, X_train_count = build_text_pipeline(train_texts, binary=False)
    X_test_count = vec_count.transform(test_texts)

    vec_bin, X_train_bin = build_text_pipeline(train_texts, binary=True)
    X_test_bin = vec_bin.transform(test_texts)

    results: dict[str, dict[str, float]] = {}

    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0]:
        c_mnb = MultinomialNB(alpha=alpha).fit(X_train_count, y_train)
        c_bnb = BernoulliNB(alpha=alpha, binarize=None).fit(X_train_bin, y_train)

        r_mnb = SklearnMultinomialNB(alpha=alpha).fit(X_train_count, y_train)
        r_bnb = SklearnBernoulliNB(alpha=alpha, binarize=None).fit(X_train_bin, y_train)

        mnb_agree = float(np.mean(c_mnb.predict(X_test_count) == r_mnb.predict(X_test_count)))
        bnb_agree = float(np.mean(c_bnb.predict(X_test_bin) == r_bnb.predict(X_test_bin)))

        mnb_acc = float(np.mean(c_mnb.predict(X_test_count) == y_test))
        bnb_acc = float(np.mean(c_bnb.predict(X_test_bin) == y_test))

        results[f"multinomial_alpha_{alpha}"] = {
            "accuracy": mnb_acc,
            "agreement_with_sklearn": mnb_agree,
        }
        results[f"bernoulli_alpha_{alpha}"] = {
            "accuracy": bnb_acc,
            "agreement_with_sklearn": bnb_agree,
        }

    return results
