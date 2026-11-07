"""Bernoulli Naive Bayes classifier for binary feature data."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import scipy.sparse as sparse  # type: ignore[import-untyped]

from naive_bayes.base import BaseNaiveBayes, check_is_fitted


class BernoulliNB(BaseNaiveBayes):
    """Bernoulli Naive Bayes classifier from scratch.

    Models binary feature presence and absence.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
    binarize : float or None, default=0.0
        Threshold for binarizing features. If None, input is assumed to already
        consist of binary values (0 or 1).
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        binarize: float | None = 0.0,
        priors: np.ndarray | None = None,
    ) -> None:
        super().__init__(priors=priors)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"Smoothing parameter alpha must be non-negative, got {alpha}")
        self.alpha = float(alpha)
        if binarize is not None and (not np.isfinite(binarize) or binarize < 0):
            raise ValueError("binarize must be finite and non-negative, or None.")
        self.binarize = binarize
        self.feature_count_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None
        self.neg_feature_log_prob_: np.ndarray | None = None

    def _binarize_data(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.spmatrix:
        """Binarize feature matrix according to self.binarize."""
        if self.binarize is not None:
            if sparse.issparse(X):
                X_bin = cast(Any, X).copy()
                X_bin.data = np.asarray(X_bin.data > self.binarize, dtype=np.float64)
                X_bin.eliminate_zeros()
                return cast(sparse.spmatrix, X_bin)
            return np.asarray(self.binarize < X, dtype=np.float64)

        # If binarize is None, verify values are binary
        if sparse.issparse(X):
            sp_data = np.asarray(cast(Any, X).data)
            if sp_data.size > 0 and not np.all(np.isin(sp_data, [0.0, 1.0])):
                raise ValueError("Expected binary data ({0, 1}) when binarize=None.")
            return X
        X_arr = np.asarray(X, dtype=np.float64)
        if not np.all(np.isin(X_arr, [0.0, 1.0])):
            raise ValueError("Expected binary data ({0, 1}) when binarize=None.")
        return X_arr

    def fit(
        self,
        X: np.ndarray | sparse.spmatrix,
        y: np.ndarray,
    ) -> BernoulliNB:
        """Fit Bernoulli Naive Bayes model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BernoulliNB
            Fitted estimator.
        """
        X_validated, y_arr = self._validate_data(X, y, reset=True)
        assert y_arr is not None

        X_bin = self._binarize_data(X_validated)
        classes, y_indices = self._setup_classes_and_priors(y_arr)
        n_classes = len(classes)
        assert self.n_features_in_ is not None
        n_features = self.n_features_in_
        assert self.class_count_ is not None

        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        if sparse.issparse(X_bin):
            X_csr = cast(Any, X_bin).tocsr()
            for i in range(n_classes):
                mask = y_indices == i
                feature_count[i, :] = np.asarray(X_csr[mask].sum(axis=0)).ravel()
        else:
            X_dense = np.asarray(X_bin, dtype=np.float64)
            for i in range(n_classes):
                feature_count[i, :] = np.sum(X_dense[y_indices == i], axis=0)

        self.feature_count_ = feature_count

        # Compute smoothed probabilities:
        # P(x_j = 1 | c) = (feature_count + alpha) / (class_count + 2 * alpha)
        # P(x_j = 0 | c) = (class_count - feature_count + alpha) / (class_count + 2 * alpha)
        smoothed_cc = self.class_count_[:, np.newaxis] + self.alpha * 2.0
        smoothed_fc_pos = feature_count + self.alpha
        smoothed_fc_neg = (self.class_count_[:, np.newaxis] - feature_count) + self.alpha

        if self.alpha == 0.0:
            with np.errstate(divide="ignore"):
                self.feature_log_prob_ = np.where(
                    smoothed_fc_pos > 0,
                    np.log(smoothed_fc_pos) - np.log(smoothed_cc),
                    -np.inf,
                )
                self.neg_feature_log_prob_ = np.where(
                    smoothed_fc_neg > 0,
                    np.log(smoothed_fc_neg) - np.log(smoothed_cc),
                    -np.inf,
                )
        else:
            self.feature_log_prob_ = np.log(smoothed_fc_pos) - np.log(smoothed_cc)
            self.neg_feature_log_prob_ = np.log(smoothed_fc_neg) - np.log(smoothed_cc)

        return self

    def _joint_log_likelihood(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Compute joint log likelihood P(C, X) = log P(C) + log P(X | C)."""
        check_is_fitted(
            self,
            ["feature_log_prob_", "neg_feature_log_prob_", "class_log_prior_"],
        )
        assert self.feature_log_prob_ is not None
        assert self.neg_feature_log_prob_ is not None
        assert self.class_log_prior_ is not None

        X_bin = self._binarize_data(X)

        if self.alpha == 0.0:
            # Keep sparse text inputs sparse and avoid multiplying zero by infinity.
            columns = []
            for positive, negative, prior in zip(
                self.feature_log_prob_,
                self.neg_feature_log_prob_,
                self.class_log_prior_,
                strict=True,
            ):
                finite_pos = np.isfinite(positive)
                finite_neg = np.isfinite(negative)
                positive_weights = np.where(finite_pos, positive, 0.0)
                negative_weights = np.where(finite_neg, negative, 0.0)
                scores = np.asarray(X_bin @ (positive_weights - negative_weights)).ravel()
                scores += negative_weights.sum() + prior
                if sparse.issparse(X_bin):
                    present_impossible = np.asarray(X_bin @ (~finite_pos).astype(float)).ravel() > 0
                    missing_impossible = np.asarray(
                        X_bin @ (~finite_neg).astype(float)
                    ).ravel() < int(np.sum(~finite_neg))
                else:
                    dense = np.asarray(X_bin)
                    present_impossible = np.any(dense[:, ~finite_pos] > 0, axis=1)
                    missing_impossible = np.any(dense[:, ~finite_neg] == 0, axis=1)
                scores[present_impossible | missing_impossible] = -np.inf
                columns.append(scores)
            return np.column_stack(columns)

        # log P(x | c) = sum [ x_j * (log P(1|c) - log P(0|c)) ] + sum [ log P(0|c) ]
        diff_log_prob = self.feature_log_prob_ - self.neg_feature_log_prob_
        absent_log_prob_sum = np.sum(self.neg_feature_log_prob_, axis=1)

        if sparse.issparse(X_bin):
            jll = X_bin.dot(diff_log_prob.T)
            jll = jll.toarray() if hasattr(jll, "toarray") else np.asarray(jll)
        else:
            X_arr = np.asarray(X_bin, dtype=np.float64)
            jll = np.dot(X_arr, diff_log_prob.T)

        jll_arr = np.asarray(jll, dtype=np.float64)
        out: np.ndarray = jll_arr + absent_log_prob_sum + self.class_log_prior_
        return out
