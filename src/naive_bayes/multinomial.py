"""Multinomial Naive Bayes classifier for count and frequency data."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sparse  # type: ignore[import-untyped]

from naive_bayes.base import BaseNaiveBayes, check_is_fitted


class MultinomialNB(BaseNaiveBayes):
    """Multinomial Naive Bayes classifier from scratch.

    Suitable for discrete count features such as word occurrences.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified, priors are not adjusted
        from data.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        priors: np.ndarray | None = None,
    ) -> None:
        super().__init__(priors=priors)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"Smoothing parameter alpha must be non-negative, got {alpha}")
        self.alpha = float(alpha)
        self.feature_count_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray | sparse.spmatrix,
        y: np.ndarray,
    ) -> MultinomialNB:
        """Fit Multinomial Naive Bayes model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training vectors containing non-negative counts.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MultinomialNB
            Fitted estimator.
        """
        X_validated, y_arr = self._validate_data(X, y, reset=True)
        assert y_arr is not None

        # Check for negative values
        if sparse.issparse(X_validated):
            sp_data = np.asarray(cast(Any, X_validated).data)
            if sp_data.size > 0 and np.any(sp_data < 0):
                raise ValueError("Negative values in X are not allowed for MultinomialNB.")
        else:
            if np.any(X_validated < 0):
                raise ValueError("Negative values in X are not allowed for MultinomialNB.")

        classes, y_indices = self._setup_classes_and_priors(y_arr)
        n_classes = len(classes)
        assert self.n_features_in_ is not None
        n_features = self.n_features_in_

        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        if sparse.issparse(X_validated):
            X_csr = cast(Any, X_validated).tocsr()
            for i in range(n_classes):
                mask = y_indices == i
                feature_count[i, :] = np.asarray(X_csr[mask].sum(axis=0)).ravel()
        else:
            X_dense = np.asarray(X_validated, dtype=np.float64)
            for i in range(n_classes):
                feature_count[i, :] = np.sum(X_dense[y_indices == i], axis=0)

        self.feature_count_ = feature_count

        # Compute smoothed log probabilities:
        # P(x_j | c) = (count_jc + alpha) / (sum_c + alpha * n_features)
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = np.sum(smoothed_fc, axis=1, keepdims=True)

        if self.alpha == 0.0:
            with np.errstate(divide="ignore"):
                self.feature_log_prob_ = np.where(
                    smoothed_fc > 0,
                    np.log(smoothed_fc) - np.log(smoothed_cc),
                    -np.inf,
                )
        else:
            self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        return self

    def _joint_log_likelihood(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Compute joint log likelihood P(C, X) = log P(C) + log P(X | C)."""
        check_is_fitted(self, ["feature_log_prob_", "class_log_prior_"])
        assert self.feature_log_prob_ is not None
        assert self.class_log_prior_ is not None

        values = np.asarray(cast(Any, X).data) if sparse.issparse(X) else np.asarray(X)
        if np.any(values < 0):
            raise ValueError("Negative values in X are not allowed for MultinomialNB.")

        if self.alpha == 0.0:
            features = X
            columns = []
            for log_prob, prior in zip(self.feature_log_prob_, self.class_log_prior_, strict=True):
                finite = np.isfinite(log_prob)
                scores = np.asarray(features @ np.where(finite, log_prob, 0.0)).ravel() + prior
                impossible = np.asarray(features @ (~finite).astype(float)).ravel() > 0
                scores[impossible] = -np.inf
                columns.append(scores)
            return np.column_stack(columns)

        if sparse.issparse(X):
            # Sparse matrix multiplication with (D, C)
            jll_raw = X.dot(self.feature_log_prob_.T)
            jll = jll_raw.toarray() if hasattr(jll_raw, "toarray") else np.asarray(jll_raw)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            jll = np.dot(X_arr, self.feature_log_prob_.T)

        out: np.ndarray = np.asarray(jll, dtype=np.float64) + self.class_log_prior_
        return out
