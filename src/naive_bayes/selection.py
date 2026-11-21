"""Validation-only hyperparameter selection for text Naive Bayes models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sparse  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import-untyped]
from sklearn.metrics import f1_score, log_loss  # type: ignore[import-untyped]

from naive_bayes.bernoulli import BernoulliNB
from naive_bayes.multinomial import MultinomialNB

ALPHAS: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
MIN_DFS: tuple[int, ...] = (1, 2, 5)
NGRAM_RANGES: tuple[tuple[int, int], ...] = ((1, 1), (1, 2))
MODEL_KINDS: tuple[str, ...] = ("multinomial", "bernoulli")


@dataclass(frozen=True)
class TextConfig:
    """One cell of the text-model selection grid."""

    model: str
    alpha: float
    min_df: int
    ngram_range: tuple[int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "alpha": self.alpha,
            "min_df": self.min_df,
            "ngram_range": list(self.ngram_range),
        }


def binarize_counts(X: sparse.spmatrix) -> sparse.spmatrix:
    """Binarize a count matrix in place-safe fashion (presence = 1)."""
    Xb = X.copy().tocsr()
    Xb.data = np.ones_like(Xb.data, dtype=np.float64)
    return Xb


# CountVectorizer raises either of these when min_df pruning empties the vocab.
_EMPTY_VOCAB_MARKERS = ("empty vocabulary", "no terms remain")


def _is_empty_vocab_error(exc: ValueError) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _EMPTY_VOCAB_MARKERS)


def _fit_predict(
    config: TextConfig,
    vec: CountVectorizer,
    X_train: sparse.spmatrix,
    y_train: np.ndarray,
    X_val: sparse.spmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the configured model on train counts and predict validation."""
    model: MultinomialNB | BernoulliNB
    if config.model == "multinomial":
        model = MultinomialNB(alpha=config.alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)
        classes = model.classes_
    elif config.model == "bernoulli":
        model = BernoulliNB(alpha=config.alpha, binarize=None)
        model.fit(binarize_counts(X_train), y_train)
        pred = model.predict(binarize_counts(X_val))
        proba = model.predict_proba(binarize_counts(X_val))
        classes = model.classes_
    else:
        raise ValueError(f"unknown model kind {config.model!r}")
    return pred, proba, np.asarray(classes)


def selection_key(row: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    """Selection ordering: F1 desc, log loss asc, then alpha, min_df, ngram."""
    return (
        -row["macro_f1"],
        row["log_loss"],
        row["alpha"],
        row["min_df"],
        row["ngram_min"],
        row["ngram_max"],
    )


def select_text_config(
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    model: str,
) -> tuple[TextConfig, list[dict[str, Any]]]:
    """Grid-search one (dataset, model) on validation; return best + table.

    Vectorizers are fit on the train partition only. Selection uses
    validation macro-F1, ties broken by lower validation log loss, smaller
    alpha, smaller min_df, then the narrower ngram range. Cells whose
    vectorizer yields an empty vocabulary are recorded with ``skipped`` and
    excluded from selection.
    """
    if model not in MODEL_KINDS:
        raise ValueError(f"model must be one of {MODEL_KINDS}")
    table: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for min_df in MIN_DFS:
            for ngram in NGRAM_RANGES:
                vec = CountVectorizer(min_df=min_df, ngram_range=ngram)
                try:
                    vec.fit(train_texts)
                except ValueError as exc:
                    if _is_empty_vocab_error(exc):
                        table.append(
                            {
                                "model": model,
                                "alpha": float(alpha),
                                "min_df": float(min_df),
                                "ngram_min": float(ngram[0]),
                                "ngram_max": float(ngram[1]),
                                "skipped": True,
                                "reason": "empty vocabulary",
                            }
                        )
                        continue
                    raise
                X_train = vec.transform(train_texts)
                X_val = vec.transform(val_texts)
                config = TextConfig(model=model, alpha=alpha, min_df=min_df, ngram_range=ngram)
                pred, proba, classes = _fit_predict(config, vec, X_train, y_train, X_val)
                macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
                # Align probabilities to the union of classes for log loss.
                all_classes = np.unique(np.concatenate([y_train, y_val]))
                proba_aligned = np.zeros((len(y_val), len(all_classes)))
                for ci, c in enumerate(classes):
                    proba_aligned[:, int(np.nonzero(all_classes == c)[0][0])] = proba[:, ci]
                ll = float(log_loss(y_val, proba_aligned, labels=all_classes))
                table.append(
                    {
                        "model": model,
                        "alpha": float(alpha),
                        "min_df": float(min_df),
                        "ngram_min": float(ngram[0]),
                        "ngram_max": float(ngram[1]),
                        "macro_f1": macro_f1,
                        "log_loss": ll,
                        "skipped": False,
                        "reason": "",
                    }
                )
    valid = [r for r in table if not r.get("skipped")]
    if not valid:
        raise ValueError("no selection cell produced a non-empty vocabulary")
    best_row = min(valid, key=selection_key)
    best = TextConfig(
        model=model,
        alpha=float(best_row["alpha"]),
        min_df=int(best_row["min_df"]),
        ngram_range=(int(best_row["ngram_min"]), int(best_row["ngram_max"])),
    )
    return best, table


def vectorize_for_config(
    config: TextConfig, train_texts: list[str], texts: list[str]
) -> tuple[CountVectorizer, sparse.spmatrix]:
    """Fit the vectorizer on train only and transform ``texts``."""
    vec = CountVectorizer(min_df=config.min_df, ngram_range=config.ngram_range)
    X_train = vec.fit_transform(train_texts)
    return vec, X_train


def transform_texts(vec: CountVectorizer, config: TextConfig, texts: list[str]) -> sparse.spmatrix:
    """Transform texts, binarizing for the Bernoulli model."""
    X = vec.transform(texts)
    if config.model == "bernoulli":
        return binarize_counts(X)
    return X
