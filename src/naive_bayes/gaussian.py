"""Gaussian Naive Bayes classifier for continuous features."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse  # type: ignore[import-untyped]

from naive_bayes.base import BaseNaiveBayes, check_is_fitted


class GaussianNB(BaseNaiveBayes):
    """Gaussian Naive Bayes (GaussianNB) implementation from scratch.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    priors : array-like of shape (n_classes,), default=None
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    """

    def __init__(
        self,
        var_smoothing: float = 1e-9,
        priors: np.ndarray | None = None,
    ) -> None:
        super().__init__(priors=priors)
        if not np.isfinite(var_smoothing) or var_smoothing < 0:
            raise ValueError(f"var_smoothing must be non-negative, got {var_smoothing}")
        self.var_smoothing = float(var_smoothing)
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.epsilon_: float | None = None
        self.var_smoothed_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray | sparse.spmatrix,
        y: np.ndarray,
    ) -> GaussianNB:
        """Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GaussianNB
            Fitted estimator.
        """
        if sparse.issparse(X):
            raise TypeError("GaussianNB does not support sparse input matrices.")

        X_arr, y_arr = self._validate_data(X, y, reset=True)
        assert isinstance(X_arr, np.ndarray)
        assert y_arr is not None

        # Convert to float64 for stable statistics
        X_arr = np.asarray(X_arr, dtype=np.float64)
        classes, y_indices = self._setup_classes_and_priors(y_arr)
        n_classes = len(classes)
        assert self.n_features_in_ is not None
        n_features = self.n_features_in_

        theta = np.zeros((n_classes, n_features), dtype=np.float64)
        var = np.zeros((n_classes, n_features), dtype=np.float64)

        for i in range(n_classes):
            X_c = X_arr[y_indices == i]
            theta[i, :] = np.mean(X_c, axis=0)
            # Use ddof=0 matching standard population variance
            var[i, :] = np.var(X_c, axis=0, ddof=0)

        self.theta_ = theta
        self.var_ = var

        # Variance smoothing: var_smoothing * max(variance of features over all samples)
        global_var = np.var(X_arr, axis=0)
        max_var = float(np.max(global_var)) if global_var.size > 0 else 0.0
        self.epsilon_ = self.var_smoothing * max_var
        self.var_smoothed_ = var + self.epsilon_
        if np.any(self.var_smoothed_ <= 0):
            raise ValueError(
                "Gaussian likelihood requires positive variance; use variable data "
                "and positive var_smoothing."
            )

        return self

    def _joint_log_likelihood(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        """Compute joint log likelihood P(C, X) = log P(C) + log P(X | C)."""
        check_is_fitted(self, ["theta_", "var_smoothed_", "class_log_prior_"])
        assert self.theta_ is not None
        assert self.var_smoothed_ is not None
        assert self.class_log_prior_ is not None

        if sparse.issparse(X):
            raise TypeError("GaussianNB does not support sparse input matrices.")

        X_arr = np.asarray(X, dtype=np.float64)

        # Vectorized normal log density
        # X: (N, 1, D), theta: (1, C, D), var_smoothed: (1, C, D)
        diff = X_arr[:, np.newaxis, :] - self.theta_[np.newaxis, :, :]
        scaled_sq_diff = (diff**2) / self.var_smoothed_[np.newaxis, :, :]
        sum_sq_diff = np.sum(scaled_sq_diff, axis=2)  # shape (N, C)

        log_2pi_var = np.sum(np.log(2.0 * np.pi * self.var_smoothed_), axis=1)  # shape (C,)

        # joint log likelihood = log P(c) - 0.5 * sum [ log(2*pi*var) + (x-mu)^2/var ]
        jll = self.class_log_prior_ - 0.5 * (log_2pi_var + sum_sq_diff)
        return jll
