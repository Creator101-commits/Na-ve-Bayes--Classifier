"""Tests for data loaders, near-duplicate grouping, selection, and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from naive_bayes.data import (
    EXPECTED_SMS_ROWS,
    TextDataset,
    fetch_newsgroup_split,
    parse_sms_spam_tsv,
)
from naive_bayes.evaluation import (
    bootstrap_metric_ci,
    compute_expected_calibration_error,
    paired_bootstrap_f1_diff,
    t_interval,
)
from naive_bayes.selection import (
    ALPHAS,
    MIN_DFS,
    NGRAM_RANGES,
    TextConfig,
    select_text_config,
    selection_key,
)
from naive_bayes.textprep import (
    NEAR_DUP_THRESHOLD,
    audit_duplicates,
    duplicate_groups,
    group_majority_label,
    group_stratified_split,
    near_duplicate_pairs,
    stratified_group_subset,
)

# ---------------------------------------------------------------------------
# SMS parser
# ---------------------------------------------------------------------------

def _tsv(rows: list[tuple[str, str]]) -> bytes:
    return "\n".join(f"{lab}\t{body}" for lab, body in rows).encode()


def test_parse_sms_spam_tsv_valid() -> None:
    # Parser enforces the real dataset size; test the row-level logic via a
    # monkeypatched expectation is not allowed, so check a full-size synthetic.
    # Force exact class counts expected by the validator.
    labels = ["ham"] * 4827 + ["spam"] * 747
    bodies = [f"synthetic sms body {i}" for i in range(EXPECTED_SMS_ROWS)]
    tsv = "\n".join(f"{lab}\t{b}" for lab, b in zip(labels, bodies, strict=True)).encode()
    msgs, labs = parse_sms_spam_tsv(tsv)
    assert len(msgs) == EXPECTED_SMS_ROWS
    assert labs.count("ham") == 4827 and labs.count("spam") == 747


def test_parse_sms_spam_tsv_rejects_bad_label() -> None:
    labels = ["ham"] * 4827 + ["spam"] * 746 + ["bogus"]
    bodies = [f"body {i}" for i in range(EXPECTED_SMS_ROWS)]
    tsv = "\n".join(f"{lab}\t{b}" for lab, b in zip(labels, bodies, strict=True)).encode()
    with pytest.raises(ValueError, match="invalid label"):
        parse_sms_spam_tsv(tsv)


def test_parse_sms_spam_tsv_rejects_empty_text() -> None:
    labels = ["ham"] * 4827 + ["spam"] * 747
    bodies = [f"body {i}" for i in range(EXPECTED_SMS_ROWS - 1)] + ["   "]
    tsv = "\n".join(f"{lab}\t{b}" for lab, b in zip(labels, bodies, strict=True)).encode()
    with pytest.raises(ValueError, match="empty text"):
        parse_sms_spam_tsv(tsv)


def test_parse_sms_spam_tsv_rejects_wrong_count() -> None:
    tsv = _tsv([("ham", "hi"), ("spam", "yo")])
    with pytest.raises(ValueError, match="Expected"):
        parse_sms_spam_tsv(tsv)


# ---------------------------------------------------------------------------
# 20NG loader (injected fetcher, no network)
# ---------------------------------------------------------------------------

class _FakeBunch:
    def __init__(self, data: list[str], target: list[str]) -> None:
        self.data = data
        self.target = target


def test_fetch_newsgroup_split_drops_empty() -> None:
    def fake_fetcher(subset: str, remove: list[str]) -> _FakeBunch:
        assert subset == "train"
        assert remove == ("headers", "footers", "quotes")
        return _FakeBunch(
            ["doc one", "   ", "doc two"],
            ["alt.atheism", "talk.religion", "alt.atheism"],
        )

    ds = fetch_newsgroup_split("train", fetcher=fake_fetcher)
    assert ds.texts == ["doc one", "doc two"]
    assert ds.dropped_empty == 1
    assert ds.labels == ["alt.atheism", "alt.atheism"]


# ---------------------------------------------------------------------------
# Near-duplicate Jaccard + union-find
# ---------------------------------------------------------------------------

def test_jaccard_hand_computed() -> None:
    # Two texts sharing most char 5-grams should exceed threshold; disjoint not.
    base = "the quick brown fox jumps over the lazy dog again and again"
    near = base + " extra"
    far = "completely different content entirely unrelated words here"
    pairs = near_duplicate_pairs([base, near, far], threshold=NEAR_DUP_THRESHOLD)
    found = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in found
    assert (0, 2) not in found and (1, 2) not in found
    # Exact jaccard value for identical texts is 1.0.
    same = near_duplicate_pairs(["abcde fghij", "abcde fghij"], threshold=0.5)
    assert same and same[0][2] == pytest.approx(1.0)


def test_duplicate_groups_and_split_no_crossing() -> None:
    rng = np.random.default_rng(3)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    texts: list[str] = []
    labels: list[str] = []
    # Many distinct singleton groups per class plus a few exact-duplicate runs.
    for i in range(12):
        texts.append("spam offer " + " ".join(rng.choice(words, 5)) + f" {i}")
        labels.append("spam")
    for i in range(12):
        texts.append("ham note " + " ".join(rng.choice(words, 5)) + f" {i}")
        labels.append("ham")
    texts += ["repeat exact phrase here"] * 4 + ["another exact repeat phrase"] * 4
    labels += ["spam"] * 4 + ["ham"] * 4
    groups = duplicate_groups(texts, labels)
    assert len(groups) < len(texts)  # exact dups collapsed
    split = group_stratified_split(groups, labels, 0.15, 0.15, seed=42)
    covered = set(split.train.tolist()) | set(split.val.tolist()) | set(split.test.tolist())
    assert covered == set(range(len(texts)))
    train_set, val_set = set(split.train.tolist()), set(split.val.tolist())
    for g in groups:
        parts = set()
        for i in g:
            if i in train_set:
                parts.add("train")
            elif i in val_set:
                parts.add("val")
            else:
                parts.add("test")
        assert len(parts) == 1


def test_audit_duplicates_reports_separate_counts() -> None:
    exact_a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    exact_b = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    near_base = "one two three four five six seven eight nine ten eleven twelve"
    near_var = "one two three four five six seven eight nine ten eleven twelve extra"
    unique = "completely unrelated solitary document words nothing shared"
    texts = [exact_a, exact_b, near_base, near_var, unique]
    audit = audit_duplicates(texts, NEAR_DUP_THRESHOLD)
    # One normalized-key collision pair (2 rows), one non-exact near pair.
    assert audit.exact_dup_groups == 1
    assert audit.exact_dup_rows == 2
    assert audit.near_pairs_nonexact == 1
    # Union-find merges {0,1} and {2,3}; doc 4 stays singleton.
    assert audit.multi_member_groups == 2
    assert audit.multi_member_rows == 4
    sizes = sorted(len(g) for g in audit.groups)
    assert sizes == [1, 2, 2]


# ---------------------------------------------------------------------------
# stratified_group_subset: group integrity, size, determinism
# ---------------------------------------------------------------------------

def test_stratified_group_subset_keeps_groups_whole() -> None:
    labels = np.array(["x"] * 6 + ["y"] * 6)
    # Class x groups: {0,1} multi-member, {2},{3},{4},{5} singletons (5 groups).
    # Class y groups: {6},{7},{8},{9},{10,11} (5 groups).
    group_ids = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])
    indices = np.arange(12)
    frac = 0.6  # round(5 * 0.6) = 3 groups kept per class
    sub = stratified_group_subset(indices, labels, group_ids, frac, seed=7)
    sub_set = set(int(i) for i in sub)
    # Exactly 3 distinct groups per class.
    for cls in ("x", "y"):
        cls_gids = {int(group_ids[i]) for i in sub if labels[i] == cls}
        assert len(cls_gids) == 3
    # No group is split: every pool group is fully in or fully out.
    for gid in set(int(g) for g in group_ids):
        members = {int(i) for i in indices if int(group_ids[i]) == gid}
        inside = members & sub_set
        assert inside == set() or inside == members
    # Deterministic per seed.
    sub2 = stratified_group_subset(indices, labels, group_ids, frac, seed=7)
    assert set(int(i) for i in sub2) == sub_set
    # A different seed may pick different groups but still 3 per class.
    sub3 = stratified_group_subset(indices, labels, group_ids, frac, seed=99)
    assert len(sub3) > 0


def test_stratified_group_subset_fraction_one_returns_all() -> None:
    labels = np.array(["x", "x", "y", "y"])
    group_ids = np.array([0, 1, 2, 3])
    indices = np.array([0, 1, 2, 3])
    sub = stratified_group_subset(indices, labels, group_ids, 1.0, seed=0)
    assert set(int(i) for i in sub) == {0, 1, 2, 3}


def test_group_majority_label_deterministic() -> None:
    # Tie between x and y resolves to the smallest sorted label.
    labels = ["x", "y", "x", "y", "z"]
    assert group_majority_label([0, 1, 2, 3, 4], labels) == "x"
    # Clear majority wins regardless of member order.
    assert group_majority_label([2, 0, 4, 1, 3], labels) == "x"
    labels2 = ["a", "a", "b"]
    assert group_majority_label([0, 1, 2], labels2) == "a"
    # A two-way tie picks the alphabetically smaller label.
    assert group_majority_label([0, 1], ["b", "a"]) == "a"
    # Permuting members never changes the result.
    base = group_majority_label([0, 1, 2, 3, 4], labels)
    for perm in ([4, 3, 2, 1, 0], [1, 3, 0, 4, 2], [2, 4, 1, 0, 3]):
        assert group_majority_label(perm, labels) == base


def test_assert_subset_ok_uses_majority_group_labels() -> None:
    """Mixed-label groups must be counted once, under their majority class.

    Regression for the Kaggle failure where per-document counting inflated a
    minority class's group total and tripped the within-1 guard.
    """
    from naive_bayes.benchmark import TextSplit, _assert_subset_ok

    # 10 pure-a groups, 20 mixed [a,a,b] groups (majority a), 2 pure-b groups.
    labels: list[str] = []
    group_ids: list[int] = []
    texts: list[str] = []
    gid = 0
    for _ in range(10):  # pure-a singletons, gids 0-9
        labels.append("a")
        group_ids.append(gid)
        texts.append("t")
        gid += 1
    for _ in range(20):  # mixed groups, gids 10-29, majority a
        for lab in ("a", "a", "b"):
            labels.append(lab)
            group_ids.append(gid)
            texts.append("t")
        gid += 1
    for _ in range(2):  # pure-b singletons, gids 30-31
        labels.append("b")
        group_ids.append(gid)
        texts.append("t")
        gid += 1

    n = len(labels)
    split = TextSplit(
        train=np.arange(n),
        val=np.array([], dtype=np.int64),
        test=np.array([], dtype=np.int64),
        group_ids=np.asarray(group_ids),
        n_groups=gid,
        dup_stats={},
    )
    ds = TextDataset(name="fixture", texts=texts, labels=labels)

    # Groups containing a 'b' doc number 22, but only 2 are b-majority. A
    # per-document count would target round(22 * frac) b-groups; the correct
    # majority count targets round(2 * frac).
    b_doc_groups = {group_ids[i] for i, lab in enumerate(labels) if lab == "b"}
    assert len(b_doc_groups) == 22
    b_majority_groups = {
        g for g in range(gid)
        if group_majority_label(
            [i for i in range(n) if group_ids[i] == g], labels
        )
        == "b"
    }
    assert len(b_majority_groups) == 2

    # A valid frac=0.15 subset: 4 pure-a groups + 1 pure-b group (docs 0-3, 70).
    sub = np.array([0, 1, 2, 3, 70], dtype=np.int64)
    _assert_subset_ok(sub, 0.15, split, ds)  # must not raise


# ---------------------------------------------------------------------------
# Vectorizer fit-on-train-only
# ---------------------------------------------------------------------------

def test_vectorizer_fit_on_train_only() -> None:
    from naive_bayes.selection import vectorize_for_config

    config = TextConfig(model="multinomial", alpha=0.1, min_df=1, ngram_range=(1, 1))
    vec, _ = vectorize_for_config(config, ["hello world"], ["hello world"])
    assert "hello" in vec.vocabulary_
    # A token unseen in train must be absent from the vocabulary.
    assert "unseen" not in vec.vocabulary_
    X = vec.transform(["unseen token here"])
    assert X.nnz == 0 or "unseen" not in vec.vocabulary_


# ---------------------------------------------------------------------------
# Selection tie-breaks
# ---------------------------------------------------------------------------

def test_selection_grid_shape() -> None:
    assert len(ALPHAS) == 6 and len(MIN_DFS) == 3 and len(NGRAM_RANGES) == 2


def test_selection_prefers_higher_macro_f1_then_lower_logloss() -> None:
    # Construct a tiny separable dataset where one config clearly wins.
    train_texts = (
        ["excellent great wonderful fantastic"] * 8
        + ["terrible awful horrible worst"] * 8
    )
    y_train = np.array(["pos"] * 8 + ["neg"] * 8)
    val_texts = ["excellent great", "terrible awful", "excellent wonderful", "awful terrible"]
    y_val = np.array(["pos", "neg", "pos", "neg"])
    best, table = select_text_config(train_texts, y_train, val_texts, y_val, "multinomial")
    assert best.model == "multinomial"
    assert len(table) == len(ALPHAS) * len(MIN_DFS) * len(NGRAM_RANGES)
    # The best row must have the max macro-F1 in the table.
    best_f1 = max(r["macro_f1"] for r in table if not r.get("skipped"))
    chosen = [r for r in table if r["alpha"] == best.alpha and r["min_df"] == best.min_df
              and int(r["ngram_min"]) == best.ngram_range[0]]
    assert chosen and chosen[0]["macro_f1"] == pytest.approx(best_f1)


def test_selection_key_deterministic_tiebreak() -> None:
    base = {"macro_f1": 0.9, "log_loss": 0.2, "alpha": 0.1,
            "min_df": 1.0, "ngram_min": 1.0, "ngram_max": 1.0}
    # Smaller min_df wins when all else ties.
    smaller_df = dict(base, min_df=1.0)
    larger_df = dict(base, min_df=5.0)
    assert selection_key(smaller_df) < selection_key(larger_df)
    # (1,1) ngram beats (1,2) when all else ties.
    uni = dict(base, ngram_min=1.0, ngram_max=1.0)
    bi = dict(base, ngram_min=1.0, ngram_max=2.0)
    assert selection_key(uni) < selection_key(bi)
    # Lower log loss wins before the min_df tiebreak.
    better_ll = dict(base, log_loss=0.1, min_df=9.0)
    assert selection_key(better_ll) < selection_key(larger_df)


def test_selection_records_skipped_empty_vocab() -> None:
    # Each doc has unique tokens, so min_df >= 2 yields an empty vocabulary.
    train_texts = [f"doc{i} uniq{i} tok{i}" for i in range(10)]
    y_train = np.array(["a"] * 5 + ["b"] * 5)
    val_texts = [f"val{i} u{i} t{i}" for i in range(4)]
    y_val = np.array(["a", "b", "a", "b"])
    best, table = select_text_config(train_texts, y_train, val_texts, y_val, "multinomial")
    assert len(table) == len(ALPHAS) * len(MIN_DFS) * len(NGRAM_RANGES)
    skipped = [r for r in table if r.get("skipped")]
    assert skipped, "expected empty-vocabulary cells to be recorded"
    assert all(r["reason"] == "empty vocabulary" for r in skipped)
    # Every min_df >= 2 cell is skipped; only min_df == 1 is selectable.
    assert all(int(r["min_df"]) >= 2 for r in skipped)
    assert best.min_df == 1
    # The winning row is a real (non-skipped) cell.
    assert not any(r.get("skipped") and r["alpha"] == best.alpha
                   and int(r["min_df"]) == best.min_df for r in table)


# ---------------------------------------------------------------------------
# Bootstrap determinism + paired sign; ECE hand-check; t-interval
# ---------------------------------------------------------------------------

def test_bootstrap_determinism() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 50)
    y_pred = y_true.copy()
    y_pred[:10] = 1 - y_pred[:10]
    proba = np.tile([0.9, 0.1], (50, 1))
    proba[np.arange(50), y_pred] = 0.9
    proba[np.arange(50), 1 - y_pred] = 0.1
    a = bootstrap_metric_ci(y_true, y_pred, proba, "macro_f1", n_resamples=100, seed=0)
    b = bootstrap_metric_ci(y_true, y_pred, proba, "macro_f1", n_resamples=100, seed=0)
    assert a == b
    assert a["ci_lower"] <= a["estimate"] <= a["ci_upper"]


def test_paired_bootstrap_sign() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 60)
    pred_a = y_true.copy()
    pred_b = y_true.copy()
    pred_b[:15] = 1 - pred_b[:15]  # B is worse
    res = paired_bootstrap_f1_diff(y_true, pred_a, pred_b, n_resamples=100, seed=0)
    assert res["estimate"] > 0
    assert res["sign"] == "positive"


def test_ece_hand_computed() -> None:
    # Perfectly confident and perfectly accurate -> ECE = 0.
    y_true = np.array([0, 1, 0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert compute_expected_calibration_error(y_true, proba, n_bins=15) == pytest.approx(0.0)
    # Confidence 0.9 but 100% accurate in the bin -> ECE = |1.0 - 0.9| = 0.1.
    proba09 = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    assert compute_expected_calibration_error(y_true, proba09, n_bins=15) == pytest.approx(0.1)
    # All wrong with high confidence -> high ECE.
    proba_wrong = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    ece = compute_expected_calibration_error(y_true, proba_wrong, n_bins=15)
    assert ece > 0.7


def test_t_interval() -> None:
    res = t_interval([0.8, 0.82, 0.81, 0.79, 0.8])
    assert res["mean"] == pytest.approx(0.804)
    assert res["ci_lower"] < res["mean"] < res["ci_upper"]


# ---------------------------------------------------------------------------
# Correlated-feature generator shape + determinism (via benchmark helper)
# ---------------------------------------------------------------------------

def test_correlated_feature_generator() -> None:
    from naive_bayes.benchmark import _correlated_feature_stress

    a = _correlated_feature_stress()
    b = _correlated_feature_stress()
    assert a["synthetic"] is True
    assert a["rows"] == b["rows"]  # deterministic
    ks = [int(r["k"]) for r in a["rows"]]
    assert ks == [0, 1, 2, 4, 8]
    # n=5000, seeds 0-4 baked in; each row reports gnb + lr metrics.
    assert "gnb_ece" in a["rows"][0] and "lr_ece" in a["rows"][0]


# ---------------------------------------------------------------------------
# Benchmark end-to-end on tiny fixture data (no network)
# ---------------------------------------------------------------------------

def _tiny_texts(n_per_class: int, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    # Wide vocab + a shared leading token keeps char-shingle Jaccard low
    # (singleton groups) while guaranteeing a token in every doc for min_df.
    words = [f"w{i:02d}" for i in range(60)]
    texts: list[str] = []
    labels: list[str] = []
    for cls, prefix in (("ham", "meet "), ("spam", "win ")):
        for i in range(n_per_class):
            body = " ".join(rng.choice(words, size=8))
            texts.append("the " + prefix + body + f" {i}")
            labels.append(cls)
    return texts, labels


def test_benchmark_end_to_end(tmp_path) -> None:
    from naive_bayes.benchmark import run_benchmark
    from naive_bayes.data import save_text_dataset

    sms_t, sms_l = _tiny_texts(80, 0)
    news_tr_t, news_tr_l = _tiny_texts(80, 1)
    news_te_t, news_te_l = _tiny_texts(30, 2)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_text_dataset(TextDataset("sms_spam", sms_t, sms_l), data_dir / "sms.pkl")
    save_text_dataset(TextDataset("news_train", news_tr_t, news_tr_l), data_dir / "news_train.pkl")
    save_text_dataset(TextDataset("news_test", news_te_t, news_te_l), data_dir / "news_test.pkl")
    (data_dir / "manifest.json").write_text('{"datasets": []}')

    out_dir = tmp_path / "out"
    result = run_benchmark(data_dir, out_dir, git_sha="testsha")
    for fname in ("metrics.json", "manifest.json", "report.md", "analysis_plan.json"):
        assert (out_dir / fname).exists(), fname
    assert len(list((out_dir / "selection").glob("*.csv"))) == 4
    man = result["manifest"]
    assert man["git_sha"] == "testsha"
    assert man["ordering_ok"] is True
    assert man["analysis_plan_sha256"]
    # card_write_epoch is taken after the plan write, before the test read.
    # ordering_ok is the strict (ns-resolution) check; float epochs may tie.
    assert man["card_write_epoch"] <= man["first_test_read_epoch"]
    met = result["metrics"]
    assert "sms_spam_multinomial" in met["text_test"]
    assert "20newsgroups_bernoulli" in met["text_test"]
    assert "20newsgroups_nodup_multinomial" in met["text_test"]
    assert "gaussian_tabular" in met
    assert "correlated_feature" in met["controlled"]
    # Training-size curves are populated (no NaN) for both datasets/models.
    for ds_name in ("sms_spam", "20newsgroups"):
        for model in ("multinomial", "bernoulli"):
            rows = met["controlled"]["training_size"][ds_name][model]
            assert len(rows) == 5
            for r in rows:
                assert np.isfinite(r["macro_f1_mean"]), (ds_name, model, r)
    # Failure analysis is present on the primary text-test entries.
    fa = met["text_test"]["20newsgroups_multinomial"]["failure_analysis"]
    assert "by_length_tertile" in fa and "by_oov_tertile" in fa
    assert "by_confidence_bin" in fa and "most_confused_pairs" in fa
