"""End-to-end benchmark: selection, frozen test, metrics, controlled studies."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    CountVectorizer,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import f1_score, log_loss  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore[import-untyped]
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
    DatasetBundle,
    TextDataset,
    load_cancer_dataset,
    load_iris_dataset,
    load_text_dataset,
    load_wine_dataset,
)
from naive_bayes.evaluation import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    compute_expected_calibration_error,
    paired_bootstrap_f1_diff,
    per_class_metrics,
    t_interval,
)
from naive_bayes.gaussian import GaussianNB
from naive_bayes.multinomial import MultinomialNB
from naive_bayes.selection import (
    ALPHAS,
    MIN_DFS,
    NGRAM_RANGES,
    TextConfig,
    binarize_counts,
    select_text_config,
)
from naive_bayes.textprep import (
    NEAR_DUP_THRESHOLD,
    audit_duplicates,
    group_majority_label,
    group_stratified_split,
    near_duplicate_pairs,
    normalize_text,
    stratified_group_subset,
)

SEED = 42
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
ECE_BINS = 15
GAUSSIAN_SPLITS = 10
GAUSSIAN_VAR_SMOOTHING = 1e-9
TRAINING_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
TRAINING_SEEDS = tuple(range(5))
CORR_KS = (0, 1, 2, 4, 8)
CORR_N = 5000
CORR_SEEDS = tuple(range(5))
RUNTIME_REPEATS = 5
MODEL_KINDS = ("multinomial", "bernoulli")


@dataclass
class PreparedData:
    """Datasets produced by ``prepare-data`` and consumed by ``benchmark``."""

    sms: TextDataset
    news_train: TextDataset
    news_test: TextDataset
    manifest: dict[str, Any] = field(default_factory=dict)


def load_prepared(data_dir: Path) -> PreparedData:
    """Load datasets and manifest written by ``prepare-data``."""
    data_dir = Path(data_dir)
    sms = load_text_dataset(data_dir / "sms.pkl")
    news_train = load_text_dataset(data_dir / "news_train.pkl")
    news_test = load_text_dataset(data_dir / "news_test.pkl")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return PreparedData(sms=sms, news_train=news_train, news_test=news_test, manifest=manifest)


@dataclass
class TextSplit:
    """Train/validation/test index partitions for one text dataset."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    group_ids: np.ndarray
    n_groups: int
    dup_stats: dict[str, int]


@dataclass
class TextSplitData:
    """All partitions and texts for one text dataset in the benchmark."""

    name: str
    train_texts: list[str]
    y_train: np.ndarray
    val_texts: list[str]
    y_val: np.ndarray
    test_texts: list[str]
    y_test: np.ndarray
    test_texts_nodup: list[str] | None
    y_test_nodup: np.ndarray | None
    split: TextSplit
    source: TextDataset


def _indices_for(ds: TextDataset, idx: np.ndarray) -> tuple[list[str], np.ndarray]:
    return [ds.texts[int(i)] for i in idx], np.array([ds.labels[int(i)] for i in idx])


def _dup_stats(audit: Any) -> dict[str, int]:
    return {
        "exact_dup_groups": int(audit.exact_dup_groups),
        "exact_dup_rows": int(audit.exact_dup_rows),
        "near_dup_pairs": int(len(audit.near_pairs)),
        "near_dup_pairs_nonexact": int(audit.near_pairs_nonexact),
        "multi_member_groups": int(audit.multi_member_groups),
        "multi_member_rows": int(audit.multi_member_rows),
    }


def prepare_sms_split(ds: TextDataset) -> tuple[TextSplit, dict[str, int]]:
    """Group-level stratified 70/15/15 split over exact+near duplicates."""
    audit = audit_duplicates(ds.texts, NEAR_DUP_THRESHOLD)
    split = group_stratified_split(audit.groups, ds.labels, 0.15, 0.15, seed=SEED)
    ts = TextSplit(
        train=split.train,
        val=split.val,
        test=split.test,
        group_ids=split.group_ids,
        n_groups=len(audit.groups),
        dup_stats=_dup_stats(audit),
    )
    return ts, ts.dup_stats


def prepare_news_split(
    train: TextDataset, test: TextDataset
) -> tuple[TextSplit, TextDataset, dict[str, Any]]:
    """Official 20NG split; grouped 15% validation carved from train.

    Returns the split, a secondary test dataset with test documents that are
    exact duplicates or near-duplicates of the official train partition
    removed, and duplicate-audit statistics.
    """
    # Groups within official train so duplicates stay on one side of val.
    audit = audit_duplicates(train.texts, NEAR_DUP_THRESHOLD)
    groups = audit.groups
    n_groups = len(groups)
    group_labels = np.array(
        [group_majority_label(g, train.labels) for g in groups],
        dtype=object,
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_g, va_g = next(sss.split(np.zeros((n_groups, 1)), group_labels))

    def expand(indices: np.ndarray) -> np.ndarray:
        flat = [i for gi in indices.ravel() for i in groups[int(gi)]]
        return np.array(sorted(flat), dtype=np.int64)

    tr_idx = expand(tr_g)
    va_idx = expand(va_g)
    group_ids = np.empty(len(train.texts), dtype=np.int64)
    for gi, g in enumerate(groups):
        for i in g:
            group_ids[i] = gi

    # Audit test vs the full official train partition (train + val carve).
    train_texts = list(train.texts)
    train_keys = {normalize_text(t) for t in train_texts}
    exact_dup_test = {i for i, t in enumerate(test.texts) if normalize_text(t) in train_keys}
    near_pairs = near_duplicate_pairs(test.texts, train_texts, threshold=NEAR_DUP_THRESHOLD)
    near_dup_test = {i for i, _, _ in near_pairs}
    drop = exact_dup_test | near_dup_test
    overlap = exact_dup_test & near_dup_test
    keep = [i for i in range(len(test.texts)) if i not in drop]
    secondary = TextDataset(
        name=test.name + "_nodup",
        texts=[test.texts[i] for i in keep],
        labels=[test.labels[i] for i in keep],
        source=test.source,
    )

    split = TextSplit(
        train=tr_idx,
        val=va_idx,
        test=np.arange(len(test.texts)),
        group_ids=group_ids,
        n_groups=n_groups,
        dup_stats=_dup_stats(audit),
    )
    audit_stats = {
        **_dup_stats(audit),
        "test_exact_dup": int(len(exact_dup_test)),
        "test_near_dup": int(len(near_dup_test)),
        "test_dup_overlap": int(len(overlap)),
        "test_dropped_dup": int(len(drop)),
    }
    return split, secondary, audit_stats


def _eval_text_model(
    config: TextConfig,
    train_texts: list[str],
    y_train: np.ndarray,
    eval_texts: list[str],
    y_eval: np.ndarray,
) -> dict[str, Any]:
    """Fit on train only, predict eval, return predictions and probabilities."""
    vec = CountVectorizer(min_df=config.min_df, ngram_range=config.ngram_range)
    vec.fit(train_texts)
    X_train = vec.transform(train_texts)
    X_eval = vec.transform(eval_texts)
    if config.model == "bernoulli":
        X_train = binarize_counts(X_train)
        X_eval = binarize_counts(X_eval)
        model: MultinomialNB | BernoulliNB = BernoulliNB(alpha=config.alpha, binarize=None)
    else:
        model = MultinomialNB(alpha=config.alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    proba = model.predict_proba(X_eval)
    return {"pred": pred, "proba": proba, "classes": model.classes_, "model": model}


def _sklearn_text_model(
    config: TextConfig,
    train_texts: list[str],
    y_train: np.ndarray,
    eval_texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Scikit-learn reference at the same selected config."""
    vec = CountVectorizer(min_df=config.min_df, ngram_range=config.ngram_range)
    vec.fit(train_texts)
    X_train = vec.transform(train_texts)
    X_eval = vec.transform(eval_texts)
    if config.model == "bernoulli":
        X_train = binarize_counts(X_train)
        X_eval = binarize_counts(X_eval)
        ref = SklearnBernoulliNB(alpha=config.alpha, binarize=None)
    else:
        ref = SklearnMultinomialNB(alpha=config.alpha)
    ref.fit(X_train, y_train)
    return ref.predict(X_eval), ref.predict_proba(X_eval)


def _proba_aligned(proba: np.ndarray, classes: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(all_classes)))
    for ci, c in enumerate(classes):
        out[:, int(np.nonzero(all_classes == c)[0][0])] = proba[:, ci]
    return out


def _token_counts(texts: list[str]) -> np.ndarray:
    return np.array([len(t.split()) for t in texts], dtype=np.float64)


def _unigram_vocab(texts: list[str]) -> set[str]:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(t.casefold().split())
    return vocab


def _oov_rate(text: str, vocab: set[str]) -> float:
    toks = text.casefold().split()
    if not toks:
        return 0.0
    return float(sum(1 for tok in toks if tok not in vocab) / len(toks))


def _slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Compact metrics for one slice of test predictions (same single read)."""
    if len(y_true) == 0:
        return {"n": 0}
    m = compute_classification_metrics(y_true, y_pred, y_prob, classes=classes)
    return {
        "n": int(len(y_true)),
        "accuracy": m.accuracy,
        "balanced_accuracy": m.balanced_accuracy,
        "macro_f1": m.macro_f1,
        "log_loss": m.log_loss,
        "ece": m.expected_calibration_error,
    }


def _tertile_labels(values: np.ndarray, cut_points: np.ndarray) -> np.ndarray:
    return np.asarray(np.digitize(values, cut_points))


def _confidence_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = ECE_BINS,
) -> list[dict[str, Any]]:
    """Per equal-width confidence bin: count, accuracy, mean confidence."""
    conf = np.max(proba, axis=1)
    correct = (y_pred == y_true).astype(np.float64)
    bounds = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        lo, hi = bounds[b], bounds[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        n = int(np.sum(mask))
        if n == 0:
            out.append({"bin": b, "n": 0})
            continue
        out.append(
            {
                "bin": b,
                "n": n,
                "accuracy": float(np.mean(correct[mask])),
                "mean_confidence": float(np.mean(conf[mask])),
            }
        )
    return out


def _most_confused_pairs(
    confusion: list[list[int]], classes: np.ndarray, top: int = 5
) -> list[dict[str, Any]]:
    """Top off-diagonal (true, predicted) confusion pairs."""
    pairs: list[dict[str, Any]] = []
    cm = np.asarray(confusion)
    for ti in range(len(classes)):
        for pi in range(len(classes)):
            if ti != pi and cm[ti, pi] > 0:
                pairs.append(
                    {
                        "true": str(classes[ti]),
                        "predicted": str(classes[pi]),
                        "count": int(cm[ti, pi]),
                    }
                )
    pairs.sort(key=lambda r: -r["count"])
    return pairs[:top]


def _failure_analysis(
    train_texts: list[str],
    val_texts: list[str],
    test_texts: list[str],
    y_test: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    confusion: list[list[int]],
    dataset_name: str,
) -> dict[str, Any]:
    """Slice the same single test read by length, OOV rate, and confidence."""
    # Document-length tertiles: cut points from TRAIN token counts.
    train_len = _token_counts(train_texts)
    len_cuts = np.percentile(train_len, [100 / 3, 200 / 3])
    test_len = _token_counts(test_texts)
    len_tertile = _tertile_labels(test_len, len_cuts)
    by_length = [
        _slice_metrics(
            y_test[len_tertile == t], pred[len_tertile == t], proba[len_tertile == t], classes
        )
        for t in range(3)
    ]

    # OOV-rate tertiles: cut points from VALIDATION OOV rates vs train vocab.
    vocab = _unigram_vocab(train_texts)
    val_oov = np.array([_oov_rate(t, vocab) for t in val_texts])
    oov_cuts = np.percentile(val_oov, [100 / 3, 200 / 3])
    test_oov = np.array([_oov_rate(t, vocab) for t in test_texts])
    oov_tertile = _tertile_labels(test_oov, oov_cuts)
    by_oov = [
        _slice_metrics(
            y_test[oov_tertile == t], pred[oov_tertile == t], proba[oov_tertile == t], classes
        )
        for t in range(3)
    ]

    out: dict[str, Any] = {
        "by_length_tertile": {
            "cut_points": [float(c) for c in len_cuts],
            "tertiles": by_length,
        },
        "by_oov_tertile": {
            "cut_points": [float(c) for c in oov_cuts],
            "tertiles": by_oov,
        },
        "by_confidence_bin": _confidence_bins(y_test, pred, proba, classes),
    }
    if dataset_name == "20newsgroups":
        out["most_confused_pairs"] = _most_confused_pairs(confusion, classes)
    return out


def _text_test_metrics(
    ds: TextSplitData,
    model: str,
    config: TextConfig,
    all_classes: np.ndarray,
) -> dict[str, Any]:
    """Frozen single test read for one (dataset, model); returns metrics."""
    res = _eval_text_model(config, ds.train_texts, ds.y_train, ds.test_texts, ds.y_test)
    pred = res["pred"]
    proba = _proba_aligned(res["proba"], res["classes"], all_classes)
    classes = np.asarray(res["classes"])
    label = ds.name

    m = compute_classification_metrics(ds.y_test, pred, proba, classes=classes)
    metrics: dict[str, Any] = {
        "dataset": label,
        "model": model,
        "config": config.as_dict(),
        "n_test": int(len(ds.y_test)),
        "accuracy": m.accuracy,
        "balanced_accuracy": m.balanced_accuracy,
        "macro_f1": m.macro_f1,
        "log_loss": m.log_loss,
        "brier_score": m.brier_score,
        "ece": m.expected_calibration_error,
        "per_class": per_class_metrics(ds.y_test, pred, classes=classes),
        "confusion_matrix": m.confusion_matrix,
        "classes": [str(c) for c in classes],
    }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "ece"):
        metrics[f"{metric}_ci"] = bootstrap_metric_ci(
            ds.y_test, pred, proba, metric, n_resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED, classes=classes,
        )
    # Parity vs sklearn at the same config.
    ref_pred, ref_proba = _sklearn_text_model(config, ds.train_texts, ds.y_train, ds.test_texts)
    ref_proba_aligned = _proba_aligned(ref_proba, classes, all_classes)
    metrics["parity"] = {
        "agreement_rate": float(np.mean(pred == ref_pred)),
        "max_prob_diff": float(np.max(np.abs(proba - ref_proba_aligned))),
        "mean_prob_diff": float(np.mean(np.abs(proba - ref_proba_aligned))),
    }
    metrics["failure_analysis"] = _failure_analysis(
        ds.train_texts, ds.val_texts, ds.test_texts, ds.y_test, pred, proba,
        classes, m.confusion_matrix, ds.name,
    )
    return {"metrics": metrics, "pred": pred, "proba": proba, "classes": classes}


def _gaussian_tabular() -> dict[str, Any]:
    """GaussianNB on iris/wine/breast_cancer: 10 seeded splits + parity."""
    datasets: list[DatasetBundle] = [
        load_iris_dataset(),
        load_wine_dataset(),
        load_cancer_dataset(),
    ]
    results: dict[str, Any] = {}
    for ds in datasets:
        accs: list[float] = []
        f1s: list[float] = []
        agreements: list[float] = []
        max_diffs: list[float] = []
        mean_diffs: list[float] = []
        for seed in range(GAUSSIAN_SPLITS):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            tr, te = next(sss.split(ds.X, ds.y))
            custom = GaussianNB(var_smoothing=GAUSSIAN_VAR_SMOOTHING).fit(ds.X[tr], ds.y[tr])
            ref = SklearnGaussianNB(var_smoothing=GAUSSIAN_VAR_SMOOTHING).fit(ds.X[tr], ds.y[tr])
            pred = custom.predict(ds.X[te])
            proba = custom.predict_proba(ds.X[te])
            ref_proba = ref.predict_proba(ds.X[te])
            accs.append(float(np.mean(pred == ds.y[te])))
            f1s.append(float(f1_score(ds.y[te], pred, average="macro", zero_division=0)))
            agreements.append(float(np.mean(pred == ref.predict(ds.X[te]))))
            d = np.abs(proba - ref_proba)
            max_diffs.append(float(np.max(d)))
            mean_diffs.append(float(np.mean(d)))
        results[ds.name] = {
            "accuracy": t_interval(accs),
            "macro_f1": t_interval(f1s),
            "parity": {
                "agreement_rate": float(np.mean(agreements)),
                "max_prob_diff": float(np.max(max_diffs)),
                "mean_prob_diff": float(np.mean(mean_diffs)),
            },
            "var_smoothing": GAUSSIAN_VAR_SMOOTHING,
            "n_splits": GAUSSIAN_SPLITS,
        }
    return results


def _correlated_feature_stress() -> dict[str, Any]:
    """Synthetic binary task: 2 informative features + k copies of feature 1."""

    def gen(seed: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, CORR_N)
        x1 = rng.normal(0.0, 1.0, CORR_N) + y * 1.5
        x2 = rng.normal(0.0, 1.0, CORR_N) + y * 1.0
        cols = [x1, x2] + [x1.copy() for _ in range(k)]
        return np.column_stack(cols), y

    rows: list[dict[str, float]] = []
    for k in CORR_KS:
        gnb_ece: list[float] = []
        gnb_ll: list[float] = []
        gnb_maxp: list[float] = []
        lr_ece: list[float] = []
        lr_ll: list[float] = []
        lr_maxp: list[float] = []
        for seed in CORR_SEEDS:
            X, y = gen(seed, k)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
            tr, te = next(sss.split(X, y))
            gnb = GaussianNB(var_smoothing=GAUSSIAN_VAR_SMOOTHING).fit(X[tr], y[tr])
            gp = gnb.predict_proba(X[te])
            gnb_ece.append(
                compute_expected_calibration_error(y[te], gp, n_bins=ECE_BINS, classes=gnb.classes_)
            )
            gnb_ll.append(float(log_loss(y[te], gp, labels=gnb.classes_)))
            gnb_maxp.append(float(np.mean(np.max(gp, axis=1))))
            lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            lp = lr.predict_proba(X[te])
            lr_ece.append(
                compute_expected_calibration_error(y[te], lp, n_bins=ECE_BINS, classes=lr.classes_)
            )
            lr_ll.append(float(log_loss(y[te], lp, labels=lr.classes_)))
            lr_maxp.append(float(np.mean(np.max(lp, axis=1))))
        rows.append(
            {
                "k": float(k),
                "gnb_ece": float(np.mean(gnb_ece)),
                "gnb_log_loss": float(np.mean(gnb_ll)),
                "gnb_mean_max_prob": float(np.mean(gnb_maxp)),
                "lr_ece": float(np.mean(lr_ece)),
                "lr_log_loss": float(np.mean(lr_ll)),
                "lr_mean_max_prob": float(np.mean(lr_maxp)),
            }
        )
    return {"synthetic": True, "n": CORR_N, "seeds": list(CORR_SEEDS), "rows": rows}


def _assert_subset_ok(
    sub: np.ndarray, frac: float, split: TextSplit, ds: TextDataset
) -> None:
    """Guard: subset non-empty and within 1 majority-labelled group of target."""
    assert len(sub) > 0, f"empty training subset at fraction {frac}"
    # Group train docs by group_id and label each group by its majority class,
    # matching stratified_group_subset's accounting exactly.
    members_by_group: dict[int, list[int]] = {}
    for i in split.train:
        members_by_group.setdefault(int(split.group_ids[int(i)]), []).append(int(i))
    group_label = {
        gid: group_majority_label(members, ds.labels)
        for gid, members in members_by_group.items()
    }
    train_groups_by_class: dict[str, set[int]] = {}
    for gid, lab in group_label.items():
        train_groups_by_class.setdefault(lab, set()).add(gid)
    per_class_actual: dict[str, set[int]] = {}
    for i in sub:
        gid = int(split.group_ids[int(i)])
        per_class_actual.setdefault(group_label[gid], set()).add(gid)
    for cls, actual in per_class_actual.items():
        n_cls = len(train_groups_by_class.get(cls, set()))
        target = max(1, int(round(n_cls * frac)))
        assert abs(len(actual) - target) <= 1, (
            f"subset group count {len(actual)} for class {cls} not within 1 of target {target}"
        )


def _training_size_study(
    split: TextSplit,
    ds: TextDataset,
    configs: dict[str, TextConfig],
    val_texts: list[str],
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Training-size curves on validation for both text models."""
    out: dict[str, Any] = {}
    for model, config in configs.items():
        rows: list[dict[str, float]] = []
        for frac in TRAINING_FRACTIONS:
            f1s: list[float] = []
            lls: list[float] = []
            for seed in TRAINING_SEEDS:
                sub = stratified_group_subset(
                    split.train, ds.labels, split.group_ids, frac, seed=seed * 1000 + SEED
                )
                _assert_subset_ok(sub, frac, split, ds)
                tr_texts, y_tr = _indices_for(ds, sub)
                res = _eval_text_model(config, tr_texts, y_tr, val_texts, y_val)
                pred, proba, classes = res["pred"], res["proba"], res["classes"]
                all_classes = np.unique(np.concatenate([y_tr, y_val]))
                proba_a = _proba_aligned(proba, classes, all_classes)
                f1s.append(float(f1_score(y_val, pred, average="macro", zero_division=0)))
                lls.append(float(log_loss(y_val, proba_a, labels=all_classes)))
            rows.append(
                {
                    "fraction": frac,
                    "macro_f1_mean": float(np.mean(f1s)),
                    "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                    "log_loss_mean": float(np.mean(lls)),
                    "log_loss_std": float(np.std(lls, ddof=1)) if len(lls) > 1 else 0.0,
                }
            )
        out[model] = rows
    return out


def _class_prior_study(
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    configs: dict[str, TextConfig],
) -> dict[str, Any]:
    """Learned vs uniform class priors on SMS validation (spam metrics)."""
    out: dict[str, Any] = {}
    for model, config in configs.items():
        for prior_name, priors in (("learned", None), ("uniform", np.array([0.5, 0.5]))):
            vec = CountVectorizer(min_df=config.min_df, ngram_range=config.ngram_range)
            vec.fit(train_texts)
            Xtr = vec.transform(train_texts)
            Xva = vec.transform(val_texts)
            m: BernoulliNB | MultinomialNB
            if config.model == "bernoulli":
                Xtr = binarize_counts(Xtr)
                Xva = binarize_counts(Xva)
                m = BernoulliNB(alpha=config.alpha, binarize=None, priors=priors)
            else:
                m = MultinomialNB(alpha=config.alpha, priors=priors)
            m.fit(Xtr, y_train)
            pred = m.predict(Xva)
            proba = m.predict_proba(Xva)
            classes = np.asarray(m.classes_)
            spam_mask = y_val == "spam"
            tp = int(np.sum((pred == "spam") & spam_mask))
            fp = int(np.sum((pred == "spam") & ~spam_mask))
            fn = int(np.sum((pred != "spam") & spam_mask))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            all_classes = np.unique(np.concatenate([y_train, y_val]))
            proba_a = _proba_aligned(proba, classes, all_classes)
            ece = compute_expected_calibration_error(
                y_val, proba_a, n_bins=ECE_BINS, classes=all_classes
            )
            out[f"{model}_{prior_name}"] = {
                "spam_precision": precision,
                "spam_recall": recall,
                "ece": ece,
            }
    return out


def _runtime_study(
    train_texts: list[str],
    y_train: np.ndarray,
    eval_texts: list[str],
    configs: dict[str, TextConfig],
) -> dict[str, Any]:
    """Fit+predict wall time, custom vs sklearn, median of 5 repeats (20NG val)."""
    out: dict[str, Any] = {}
    for model, config in configs.items():
        vec = CountVectorizer(min_df=config.min_df, ngram_range=config.ngram_range)
        vec.fit(train_texts)
        Xtr = vec.transform(train_texts)
        Xev = vec.transform(eval_texts)
        custom: BernoulliNB | MultinomialNB
        ref: SklearnBernoulliNB | SklearnMultinomialNB
        if config.model == "bernoulli":
            Xtr = binarize_counts(Xtr)
            Xev = binarize_counts(Xev)
            custom = BernoulliNB(alpha=config.alpha, binarize=None)
            ref = SklearnBernoulliNB(alpha=config.alpha, binarize=None)
        else:
            custom = MultinomialNB(alpha=config.alpha)
            ref = SklearnMultinomialNB(alpha=config.alpha)
        custom_times: list[float] = []
        ref_times: list[float] = []
        for _ in range(RUNTIME_REPEATS):
            t0 = time.perf_counter()
            custom.fit(Xtr, y_train)
            custom.predict(Xev)
            custom_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            ref.fit(Xtr, y_train)
            ref.predict(Xev)
            ref_times.append(time.perf_counter() - t0)
        out[model] = {
            "custom_seconds": float(np.median(custom_times)),
            "sklearn_seconds": float(np.median(ref_times)),
            "repeats": RUNTIME_REPEATS,
            "eval_partition": "validation",
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _analysis_plan(selected_configs: dict[str, dict[str, TextConfig]]) -> dict[str, Any]:
    """Frozen analysis plan: selected configs plus every declared constant."""
    return {
        "seed": SEED,
        "datasets": ["sms_spam", "20newsgroups"],
        "model_kinds": list(MODEL_KINDS),
        "selected_configs": {
            ds: {m: c.as_dict() for m, c in cfgs.items()}
            for ds, cfgs in selected_configs.items()
        },
        "constants": {
            "near_dup_threshold": NEAR_DUP_THRESHOLD,
            "ece_bins": ECE_BINS,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
            "selection_grid": {
                "alphas": list(ALPHAS),
                "min_dfs": list(MIN_DFS),
                "ngram_ranges": [list(n) for n in NGRAM_RANGES],
            },
            "gaussian": {
                "var_smoothing": GAUSSIAN_VAR_SMOOTHING,
                "n_splits": GAUSSIAN_SPLITS,
                "test_size": 0.3,
            },
            "training_size": {
                "fractions": list(TRAINING_FRACTIONS),
                "seeds": list(TRAINING_SEEDS),
            },
            "correlated_feature": {
                "ks": list(CORR_KS),
                "n": CORR_N,
                "seeds": list(CORR_SEEDS),
            },
            "runtime": {"repeats": RUNTIME_REPEATS, "eval_partition": "validation"},
            "sms_split": {"val_fraction": 0.15, "test_fraction": 0.15},
            "news_split": {"val_fraction": 0.15},
        },
    }


def run_benchmark(data_dir: Path, out_dir: Path, git_sha: str = "unknown") -> dict[str, Any]:
    """Run the full benchmark and write metrics.json, manifest.json, CSVs, report."""
    t0 = time.time()
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_dir = out_dir / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)

    prepared = load_prepared(data_dir)

    # ---- SMS split (group-level, near-dup aware) ----
    sms_split, sms_dup = prepare_sms_split(prepared.sms)
    sms_train_texts, y_sms_train = _indices_for(prepared.sms, sms_split.train)
    sms_val_texts, y_sms_val = _indices_for(prepared.sms, sms_split.val)
    sms_test_texts, y_sms_test = _indices_for(prepared.sms, sms_split.test)

    # ---- 20NG split (official + grouped validation + dup audit) ----
    news_split, news_test_nodup, news_dup = prepare_news_split(
        prepared.news_train, prepared.news_test
    )
    news_train_texts, y_news_train = _indices_for(prepared.news_train, news_split.train)
    news_val_texts, y_news_val = _indices_for(prepared.news_train, news_split.val)
    news_test_texts = list(prepared.news_test.texts)
    y_news_test = np.array(prepared.news_test.labels)
    news_test_nodup_texts = list(news_test_nodup.texts)
    y_news_test_nodup = np.array(news_test_nodup.labels)

    text_datasets = [
        TextSplitData(
            name="sms_spam",
            train_texts=sms_train_texts, y_train=y_sms_train,
            val_texts=sms_val_texts, y_val=y_sms_val,
            test_texts=sms_test_texts, y_test=y_sms_test,
            test_texts_nodup=None, y_test_nodup=None,
            split=sms_split, source=prepared.sms,
        ),
        TextSplitData(
            name="20newsgroups",
            train_texts=news_train_texts, y_train=y_news_train,
            val_texts=news_val_texts, y_val=y_news_val,
            test_texts=news_test_texts, y_test=y_news_test,
            test_texts_nodup=news_test_nodup_texts, y_test_nodup=y_news_test_nodup,
            split=news_split, source=prepared.news_train,
        ),
    ]

    selection_tables: dict[str, Any] = {}
    selected_configs: dict[str, dict[str, TextConfig]] = {}
    test_results: dict[str, Any] = {}
    test_preds: dict[str, dict[str, np.ndarray]] = {}

    # Selection on validation only; CSVs written before the first test read.
    for ds in text_datasets:
        for model in MODEL_KINDS:
            best, table = select_text_config(
                ds.train_texts, ds.y_train, ds.val_texts, ds.y_val, model
            )
            selection_tables[f"{ds.name}_{model}"] = table
            selected_configs.setdefault(ds.name, {})[model] = best
            _write_csv(
                sel_dir / f"{ds.name}_{model}_selection.csv",
                table,
                ["model", "alpha", "min_df", "ngram_min", "ngram_max",
                 "macro_f1", "log_loss", "skipped", "reason"],
            )

    # Freeze the analysis plan (selected configs + all constants) and record
    # the epoch at which the plan was written, before any test read.
    plan = _analysis_plan(selected_configs)
    plan_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    (out_dir / "analysis_plan.json").write_bytes(plan_bytes)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    card_write_ns = time.time_ns()
    card_write_epoch = card_write_ns / 1e9

    first_test_read_ns = time.time_ns()
    first_test_read_epoch = first_test_read_ns / 1e9
    ordering_ok = card_write_ns < first_test_read_ns

    # Frozen single test read per (dataset, model).
    for ds in text_datasets:
        all_classes = np.unique(np.concatenate([ds.y_train, ds.y_test]))
        for model in MODEL_KINDS:
            config = selected_configs[ds.name][model]
            primary = _text_test_metrics(ds, model, config, all_classes)
            test_results[f"{ds.name}_{model}"] = primary["metrics"]
            test_preds.setdefault(ds.name, {})[model] = primary["pred"]
            if ds.test_texts_nodup is not None and ds.y_test_nodup is not None:
                nodup = TextSplitData(
                    name=ds.name + "_nodup",
                    train_texts=ds.train_texts, y_train=ds.y_train,
                    val_texts=ds.val_texts, y_val=ds.y_val,
                    test_texts=ds.test_texts_nodup, y_test=ds.y_test_nodup,
                    test_texts_nodup=None, y_test_nodup=None,
                    split=ds.split, source=ds.source,
                )
                secondary = _text_test_metrics(nodup, model, config, all_classes)
                test_results[f"{ds.name}_nodup_{model}"] = secondary["metrics"]

    # Paired bootstrap MNB - BNB on the same resamples (primary test).
    paired: dict[str, Any] = {}
    for ds in text_datasets:
        preds = test_preds[ds.name]
        paired[ds.name] = paired_bootstrap_f1_diff(
            ds.y_test, preds["multinomial"], preds["bernoulli"],
            n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
        )

    gaussian = _gaussian_tabular()
    correlated = _correlated_feature_stress()

    # Controlled experiments (validation / synthetic only, never test).
    training_size: dict[str, Any] = {}
    class_prior: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    for ds in text_datasets:
        cfgs = selected_configs[ds.name]
        training_size[ds.name] = _training_size_study(
            ds.split, ds.source, cfgs, ds.val_texts, ds.y_val
        )
        if ds.name == "sms_spam":
            class_prior[ds.name] = _class_prior_study(
                ds.train_texts, ds.y_train, ds.val_texts, ds.y_val, cfgs
            )
        if ds.name == "20newsgroups":
            runtime[ds.name] = _runtime_study(
                ds.train_texts, ds.y_train, ds.val_texts, cfgs
            )

    manifest = {
        "git_sha": git_sha,
        "protocol": "naive-bayes-benchmark-v2",
        "card_write_epoch": card_write_epoch,
        "first_test_read_epoch": first_test_read_epoch,
        "ordering_ok": bool(ordering_ok),
        "analysis_plan_sha256": plan_sha256,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "ece_bins": ECE_BINS,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "sms_split": {
            "n_groups": sms_split.n_groups,
            **sms_dup,
            "train": int(len(sms_split.train)),
            "val": int(len(sms_split.val)),
            "test": int(len(sms_split.test)),
        },
        "news_split": {
            "n_groups": news_split.n_groups,
            **news_dup,
            "train": int(len(news_split.train)),
            "val": int(len(news_split.val)),
            "test": int(len(prepared.news_test.texts)),
        },
        "selected_configs": {
            ds: {m: c.as_dict() for m, c in cfgs.items()}
            for ds, cfgs in selected_configs.items()
        },
        "data_manifest": prepared.manifest,
        "wall_seconds": time.time() - t0,
        "python_version": sys.version.split()[0],
        "library_versions": _library_versions(),
    }

    metrics_doc = {
        "text_test": test_results,
        "paired_mnb_minus_bnb": paired,
        "gaussian_tabular": gaussian,
        "controlled": {
            "smoothing_grid": "see selection/*.csv",
            "training_size": training_size,
            "correlated_feature": correlated,
            "class_prior": class_prior,
            "runtime": runtime,
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "report.md").write_text(_render_report(metrics_doc, manifest))
    return {"metrics": metrics_doc, "manifest": manifest}


def _library_versions() -> dict[str, str]:
    import contextlib
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn"):
        with contextlib.suppress(md.PackageNotFoundError):
            out[name] = md.version(name)
    return out


def _render_report(metrics: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = ["# Naive Bayes Benchmark Report", ""]
    lines.append("## Frozen test metrics (text)")
    lines.append("")
    lines.append("| dataset | model | accuracy | macro-F1 | log loss | ECE |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    ordered = sorted(metrics["text_test"].values(), key=lambda r: (r["dataset"], r["model"]))
    for m in ordered:
        lines.append(
            f"| {m['dataset']} | {m['model']} | {m['accuracy']:.4f} | "
            f"{m['macro_f1']:.4f} | {m['log_loss']:.4f} | {m['ece']:.4f} |"
        )
    lines.append("")
    lines.append("## Gaussian tabular (10 seeded 70/30 splits)")
    lines.append("")
    lines.append("| dataset | accuracy (mean [CI]) | macro-F1 (mean [CI]) |")
    lines.append("| --- | --- | --- |")
    for name, r in metrics["gaussian_tabular"].items():
        a, f = r["accuracy"], r["macro_f1"]
        lines.append(
            f"| {name} | {a['mean']:.4f} [{a['ci_lower']:.4f}, {a['ci_upper']:.4f}] | "
            f"{f['mean']:.4f} [{f['ci_lower']:.4f}, {f['ci_upper']:.4f}] |"
        )
    lines.append("")
    lines.append("## Failure analysis (frozen test slices)")
    lines.append("")
    lines.append(
        "Slices reuse the single frozen test read (no refits). Length tertiles "
        "use cut points from train token counts; OOV tertiles use cut points "
        "from validation OOV rates against the train unigram vocabulary."
    )
    lines.append("")
    for key in sorted(metrics["text_test"]):
        m = metrics["text_test"][key]
        if "_nodup_" in key:
            continue
        fa = m.get("failure_analysis")
        if not fa:
            continue
        lines.append(f"### {m['dataset']} / {m['model']}")
        lines.append("")
        lines.append("| slice | n | accuracy | macro-F1 |")
        lines.append("| --- | --- | --- | --- |")
        for tname, tkey in (("length", "by_length_tertile"), ("oov", "by_oov_tertile")):
            tertiles = fa[tkey]["tertiles"]
            for ti, t in enumerate(tertiles):
                if t.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {tname} tertile {ti + 1} | {t['n']} | "
                    f"{t['accuracy']:.4f} | {t['macro_f1']:.4f} |"
                )
        lines.append("")
        conf = fa["by_confidence_bin"]
        populated = [b for b in conf if b.get("n", 0) > 0]
        if populated:
            lines.append(
                "Confidence bins (count / accuracy / mean confidence): "
                + "; ".join(
                    f"{b['bin']}:{b['n']}/{b['accuracy']:.3f}/{b['mean_confidence']:.3f}"
                    for b in populated
                )
            )
            lines.append("")
        if "most_confused_pairs" in fa:
            lines.append("Most confused class pairs (20NG):")
            for p in fa["most_confused_pairs"]:
                lines.append(f"- {p['true']} -> {p['predicted']}: {p['count']}")
            lines.append("")
    lines.append(f"git_sha: `{manifest['git_sha']}`")
    lines.append(f"ordering_ok: {manifest['ordering_ok']}")
    lines.append(f"analysis_plan_sha256: `{manifest['analysis_plan_sha256']}`")
    return "\n".join(lines) + "\n"
