"""Evaluation metrics, calibration analysis, and reference model comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard classification and probability metrics."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    log_loss: float
    expected_calibration_error: float
    brier_score: float | None
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class ReferenceComparisonResult:
    """Agreement and divergence metrics comparing custom model against scikit-learn."""

    agreement_rate: float
    max_prob_diff: float
    mean_prob_diff: float
    custom_accuracy: float
    ref_accuracy: float
    custom_fit_seconds: float
    ref_fit_seconds: float


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    classes: np.ndarray | None = None,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins.

    For multiclass or binary classification, bin highest predicted probabilities
    and measure gap between confidence and actual accuracy in each bin.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)

    # Encode y_true to indices if needed
    classes = np.unique(y_true) if classes is None else np.asarray(classes)
    if len(classes) != y_prob.shape[1]:
        raise ValueError("Provide model classes matching probability columns.")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[y] for y in y_true])

    accuracies = predictions == y_indices
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i > 0:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Calculate comprehensive evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    ll = float(log_loss(y_true, y_prob, labels=classes))

    ece = compute_expected_calibration_error(y_true, y_prob, classes=classes)

    unique_classes = np.unique(y_true) if classes is None else classes
    brier: float | None = None
    if len(unique_classes) == 2:
        try:
            # For binary classification, use probability of positive class (index 1)
            y_binary = (y_true == unique_classes[1]).astype(int)
            brier = float(brier_score_loss(y_binary, y_prob[:, 1]))
        except Exception:
            brier = None

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return ClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        log_loss=ll,
        expected_calibration_error=ece,
        brier_score=brier,
        confusion_matrix=cm,
    )


def compare_with_reference(
    custom_model: Any,
    reference_model: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
) -> ReferenceComparisonResult:
    """Fit both models on identical train data and compare test predictions and probabilities."""
    t0 = time.perf_counter()
    custom_model.fit(X_train, y_train)
    custom_fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    reference_model.fit(X_train, y_train)
    ref_fit_time = time.perf_counter() - t1

    custom_preds = custom_model.predict(X_test)
    ref_preds = reference_model.predict(X_test)

    custom_prob = custom_model.predict_proba(X_test)
    ref_prob = reference_model.predict_proba(X_test)

    agreement = float(np.mean(custom_preds == ref_preds))
    prob_diff = np.abs(custom_prob - ref_prob)
    max_diff = float(np.max(prob_diff))
    mean_diff = float(np.mean(prob_diff))

    custom_acc = float(np.mean(custom_preds == y_test))
    ref_acc = float(np.mean(ref_preds == y_test))

    return ReferenceComparisonResult(
        agreement_rate=agreement,
        max_prob_diff=max_diff,
        mean_prob_diff=mean_diff,
        custom_accuracy=custom_acc,
        ref_accuracy=ref_acc,
        custom_fit_seconds=custom_fit_time,
        ref_fit_seconds=ref_fit_time,
    )
