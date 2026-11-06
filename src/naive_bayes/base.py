"""Base estimator for from-scratch Naive Bayes implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import scipy.sparse as sparse  # type: ignore[import-untyped]


def check_is_fitted(estimator: Any, attributes: list[str]) -> None:
    """Check if the estimator has been fitted by verifying required attributes exist."""
    for attr in attributes:
        if not hasattr(estimator, attr) or getattr(estimator, attr) is None:
            raise RuntimeError(
                f"Estimator {estimator.__class__.__name__} is not fitted. "
                f"Call 'fit' before using this method."
            )


def log_sum_exp(log_matrix: np.ndarray, axis: int = 1, keepdims: bool = False) -> np.ndarray:
    """Compute log(sum(exp(x))) in a numerically stable way."""
    max_val = np.max(log_matrix, axis=axis, keepdims=True)
    # If all values in a row are -inf, max_val is -inf; preserve it safely
    safe_max = np.where(np.isneginf(max_val), 0.0, max_val)
    sum_exp = np.sum(np.exp(log_matrix - safe_max), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = np.where(np.isneginf(max_val), -np.inf, np.log(sum_exp) + safe_max)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


class BaseNaiveBayes(ABC):
    """Abstract base class for Naive Bayes classifiers."""

    def __init__(self, priors: np.ndarray | None = None) -> None:
        self.priors = priors
        self.classes_: np.ndarray | None = None
        self.class_count_: np.ndarray | None = None
        self.class_prior_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def _validate_data(
        self,
        X: np.ndarray | sparse.spmatrix,
        y: np.ndarray | None = None,
        reset: bool = True,
    ) -> tuple[np.ndarray | sparse.spmatrix, np.ndarray | None]:
        """Validate shapes and types for input features and target labels."""
        if sparse.issparse(X):
            if X.ndim != 2:
                raise ValueError(f"Expected 2D sparse matrix, got shape {X.shape}")
        else:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError(f"Expected 2D array for X, got shape {X.shape}")

        n_samples, n_features = X.shape
        if n_samples == 0:
            raise ValueError("Input feature matrix X contains zero samples.")
        if n_features == 0:
            raise ValueError("Input feature matrix X contains zero features.")

        if reset:
            self.n_features_in_ = n_features
        elif self.n_features_in_ is not None and n_features != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: model was fitted with {self.n_features_in_} "
                f"features, but received {n_features} features."
            )

        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.ndim != 1:
                raise ValueError(f"Expected 1D target array y, got shape {y_arr.shape}")
            if len(y_arr) != n_samples:
                raise ValueError(
                    f"Sample count mismatch: X has {n_samples} samples, "
                    f"but y has {len(y_arr)} labels."
                )
            return X, y_arr

        return X, None

    def _setup_classes_and_priors(
        self, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode labels to indices 0..C-1 and compute class priors."""
        classes, y_indices = np.unique(y, return_inverse=True)
        self.classes_ = classes
        n_classes = len(classes)

        class_counts = np.bincount(y_indices, minlength=n_classes)
        self.class_count_ = class_counts.astype(np.float64)

        if self.priors is not None:
            priors_arr = np.asarray(self.priors, dtype=np.float64)
            if priors_arr.shape != (n_classes,):
                raise ValueError(
                    f"Number of priors ({len(priors_arr)}) does not match "
                    f"number of classes ({n_classes})."
                )
            if np.any(priors_arr < 0.0):
                raise ValueError("Class priors must be non-negative.")
            if not np.isclose(np.sum(priors_arr), 1.0, atol=1e-5):
                raise ValueError("Class priors must sum to 1.0.")
            self.class_prior_ = priors_arr
        else:
            total_samples = len(y)
            self.class_prior_ = self.class_count_ / total_samples

        self.class_log_prior_ = np.log(self.class_prior_)
        return classes, y_indices

    @abstractmethod
    def _joint_log_likelihood(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Compute joint log likelihood P(C, X) = log P(C) + log P(X | C)."""

    def predict_log_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Calculate log posterior probabilities P(C | X)."""
        check_is_fitted(self, ["classes_", "class_log_prior_"])
        X_validated, _ = self._validate_data(X, reset=False)
        jll = self._joint_log_likelihood(X_validated)
        log_prob_x = log_sum_exp(jll, axis=1, keepdims=True)
        return cast(np.ndarray, jll - log_prob_x)

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Calculate posterior probabilities P(C | X) summing to 1 across classes."""
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)

    def predict(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Predict the class label with the highest posterior probability."""
        check_is_fitted(self, ["classes_", "class_log_prior_"])
        X_validated, _ = self._validate_data(X, reset=False)
        jll = self._joint_log_likelihood(X_validated)
        best_indices = np.argmax(jll, axis=1)
        assert self.classes_ is not None
        return self.classes_[best_indices]

    def score(self, X: np.ndarray | sparse.spmatrix, y: np.ndarray) -> float:
        """Compute mean classification accuracy."""
        y_pred = self.predict(X)
        y_true = np.asarray(y)
        return float(np.mean(y_pred == y_true))
