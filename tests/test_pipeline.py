"""Integration tests for dataset loaders, splits, metrics, and comparisons."""

from __future__ import annotations

import numpy as np

from naive_bayes.data import (
    build_text_pipeline,
    load_cancer_dataset,
    load_iris_dataset,
    load_sms_spam_dataset,
    load_wine_dataset,
    train_val_test_split,
)
from naive_bayes.evaluation import (
    compute_classification_metrics,
)
from naive_bayes.experiments import (
    run_continuous_benchmarks,
    run_text_benchmarks,
    verify_mathematics,
)
from naive_bayes.reporting import (
    generate_research_report,
)


def test_dataset_bundles() -> None:
    iris = load_iris_dataset()
    assert iris.name == "iris"
    assert iris.X.shape == (150, 4)
    assert len(iris.y) == 150

    wine = load_wine_dataset()
    assert wine.name == "wine"
    assert wine.X.shape == (178, 13)

    cancer = load_cancer_dataset()
    assert cancer.name == "breast_cancer"
    assert cancer.X.shape == (569, 30)


def test_sms_spam_dataset_and_vectorizer() -> None:
    messages, labels = load_sms_spam_dataset()
    assert len(messages) == len(labels)
    assert "spam" in labels and "ham" in labels

    vec, X_counts = build_text_pipeline(messages[:20])
    assert X_counts.shape[0] == 20
    # Vectorizer vocabulary fitted on training messages
    assert len(vec.vocabulary_) > 10


def test_train_val_test_split() -> None:
    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 25 + [1] * 25)

    X_tr, X_val, X_te, y_tr, y_val, y_te = train_val_test_split(
        X, y, val_size=0.2, test_size=0.2, random_state=42
    )

    assert len(X_tr) == 30
    assert len(X_val) == 10
    assert len(X_te) == 10
    # Stratified balance check
    assert sum(y_tr == 0) == 15
    assert sum(y_val == 0) == 5
    assert sum(y_te == 0) == 5


def test_classification_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.6, 0.4],
    ])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert metrics.accuracy == 0.75
    assert metrics.brier_score is not None
    assert metrics.expected_calibration_error >= 0.0


def test_verify_mathematics_and_benchmarks() -> None:
    math_res = verify_mathematics()
    assert all(math_res.values())

    cont_res = run_continuous_benchmarks()
    assert "iris" in cont_res
    assert "wine" in cont_res
    assert "breast_cancer" in cont_res
    for res in cont_res.values():
        assert res.agreement_rate >= 0.95

    text_res = run_text_benchmarks()
    assert len(text_res) > 0


def test_reporting_formatting() -> None:
    math_res = verify_mathematics()
    cont_res = run_continuous_benchmarks()
    text_res = run_text_benchmarks()

    rep = generate_research_report(math_res, cont_res, text_res)
    assert "Mathematical Verification" in rep
    assert "Gaussian Naive Bayes Benchmark" in rep
