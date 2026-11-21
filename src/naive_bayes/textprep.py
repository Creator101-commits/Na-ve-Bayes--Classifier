"""Text normalization, near-duplicate detection, and group-aware splits."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore[import-untyped]

# Fixed near-duplicate threshold declared up front (brief requirement).
NEAR_DUP_THRESHOLD = 0.8
SHINGLE_NGRAM_RANGE = (5, 5)
JACCARD_CHUNK_SIZE = 512


def normalize_text(text: str) -> str:
    """NFKC normalize, casefold, and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def duplicate_key(text: str) -> str:
    """Exact-duplicate key after normalization."""
    return normalize_text(text)


def fit_shingle_vectorizer(texts: list[str]) -> CountVectorizer:
    """Fit char_wb 5-gram binary shingles on the texts being compared."""
    vectorizer = CountVectorizer(
        analyzer="char_wb",
        ngram_range=SHINGLE_NGRAM_RANGE,
        binary=True,
    )
    vectorizer.fit(texts)
    return vectorizer


def _jaccard_candidates(
    X: sparse.spmatrix,
    threshold: float,
    chunk_size: int = JACCARD_CHUNK_SIZE,
) -> list[tuple[int, int, float]]:
    """Find pairs with Jaccard >= threshold via sparse products in row chunks."""
    n = X.shape[0]
    sizes = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = X[start:stop]
        inter = (block @ X.T).toarray()
        for bi in range(stop - start):
            i = start + bi
            row = inter[bi]
            denom = sizes[i] + sizes - row
            with np.errstate(divide="ignore", invalid="ignore"):
                jac = np.where(denom > 0, row / denom, 0.0)
            js = np.nonzero(jac >= threshold)[0]
            for j in js:
                j_int = int(j)
                if j_int > i:
                    pairs.append((i, j_int, float(jac[j_int])))
    return pairs


def near_duplicate_pairs(
    texts_a: list[str],
    texts_b: list[str] | None = None,
    threshold: float = NEAR_DUP_THRESHOLD,
    vectorizer: CountVectorizer | None = None,
) -> list[tuple[int, int, float]]:
    """Return (i, j, jaccard) pairs at or above threshold.

    When ``texts_b`` is None, compares ``texts_a`` against itself (i < j).
    Otherwise compares each ``texts_a[i]`` against ``texts_b[j]``.
    """
    if texts_b is None:
        texts_b = texts_a
        self_compare = True
    else:
        self_compare = False
    if vectorizer is None:
        vectorizer = fit_shingle_vectorizer(list(texts_a) + list(texts_b))
    Xa = vectorizer.transform(texts_a)
    Xb = vectorizer.transform(texts_b)
    sizes_a = np.asarray(Xa.sum(axis=1)).ravel().astype(np.float64)
    sizes_b = np.asarray(Xb.sum(axis=1)).ravel().astype(np.float64)
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, Xa.shape[0], JACCARD_CHUNK_SIZE):
        stop = min(start + JACCARD_CHUNK_SIZE, Xa.shape[0])
        block = Xa[start:stop]
        inter = (block @ Xb.T).toarray()
        for bi in range(stop - start):
            i = start + bi
            denom = sizes_a[i] + sizes_b - inter[bi]
            with np.errstate(divide="ignore", invalid="ignore"):
                jac = np.where(denom > 0, inter[bi] / denom, 0.0)
            for j in np.nonzero(jac >= threshold)[0]:
                j_int = int(j)
                if self_compare and j_int <= i:
                    continue
                pairs.append((i, j_int, float(jac[j_int])))
    return pairs


class UnionFind:
    """Disjoint-set union for grouping duplicate indices."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass(frozen=True)
class DuplicateAudit:
    """Detailed duplicate statistics from one exact+near pass."""

    groups: list[list[int]]
    near_pairs: list[tuple[int, int]]
    near_pairs_nonexact: int
    exact_dup_groups: int
    exact_dup_rows: int
    multi_member_groups: int
    multi_member_rows: int


def audit_duplicates(
    texts: list[str],
    threshold: float = NEAR_DUP_THRESHOLD,
) -> DuplicateAudit:
    """One pass computing exact keys, near-dup pairs, and union-find groups."""
    n = len(texts)
    uf = UnionFind(n)

    key_to_idx: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        key_to_idx.setdefault(duplicate_key(text), []).append(i)
    exact_dup_groups = 0
    exact_dup_rows = 0
    for members in key_to_idx.values():
        if len(members) > 1:
            exact_dup_groups += 1
            exact_dup_rows += len(members)
            for j in members[1:]:
                uf.union(members[0], j)

    vectorizer = fit_shingle_vectorizer(texts)
    X = vectorizer.transform(texts)
    raw_pairs = _jaccard_candidates(X, threshold)
    near_pairs: list[tuple[int, int]] = []
    near_pairs_nonexact = 0
    keys = [duplicate_key(t) for t in texts]
    for i, j, _jac in raw_pairs:
        near_pairs.append((i, j))
        uf.union(i, j)
        if keys[i] != keys[j]:
            near_pairs_nonexact += 1

    groups_map: dict[int, list[int]] = {}
    for i in range(n):
        groups_map.setdefault(uf.find(i), []).append(i)
    groups = [sorted(g) for g in groups_map.values()]
    groups.sort(key=lambda g: g[0])
    multi = [g for g in groups if len(g) > 1]
    return DuplicateAudit(
        groups=groups,
        near_pairs=near_pairs,
        near_pairs_nonexact=near_pairs_nonexact,
        exact_dup_groups=exact_dup_groups,
        exact_dup_rows=exact_dup_rows,
        multi_member_groups=len(multi),
        multi_member_rows=sum(len(g) for g in multi),
    )


def duplicate_groups(
    texts: list[str],
    labels: list[str] | np.ndarray | None = None,
    threshold: float = NEAR_DUP_THRESHOLD,
    progress: bool = False,
) -> list[list[int]]:
    """Union-find over exact + near-duplicate pairs; returns index groups."""
    return audit_duplicates(texts, threshold).groups


@dataclass(frozen=True)
class GroupSplit:
    """Group-level stratified split; no group crosses partitions."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    group_ids: np.ndarray


def group_majority_label(members: Iterable[int], labels: Sequence[str] | np.ndarray) -> str:
    """Majority label among a duplicate group's member indices.

    Ties are broken by the smallest label in sorted order so the result is
    deterministic and independent of member or set iteration order.
    """
    counts: dict[str, int] = {}
    for i in members:
        lab = str(labels[int(i)])
        counts[lab] = counts.get(lab, 0) + 1
    best = max(counts.values())
    return sorted(k for k, v in counts.items() if v == best)[0]


def group_stratified_split(
    groups: list[list[int]],
    labels: list[str] | np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> GroupSplit:
    """Split duplicate groups into stratified train/val/test index arrays."""
    n_groups = len(groups)
    group_labels = np.empty(n_groups, dtype=object)
    for gi, group in enumerate(groups):
        group_labels[gi] = group_majority_label(group, labels)

    total = val_fraction + test_fraction
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=total, random_state=seed
    )
    train_g, temp_g = next(sss1.split(np.zeros((n_groups, 1)), group_labels))
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - val_fraction / total,
        random_state=seed,
    )
    val_g, test_g = next(sss2.split(np.zeros((len(temp_g), 1)), group_labels[temp_g]))
    # val_g/test_g are positions within temp_g; map back to group indices.
    val_groups = temp_g[val_g]
    test_groups = temp_g[test_g]

    def expand(indices: np.ndarray) -> np.ndarray:
        out = [i for gi in indices.ravel() for i in groups[int(gi)]]
        return np.array(sorted(out), dtype=np.int64)

    train_idx = expand(train_g)
    val_idx = expand(val_groups)
    test_idx = expand(test_groups)

    # Assert no group crosses partitions.
    part_of: dict[int, int] = {}
    for i in train_idx:
        part_of[int(i)] = 0
    for i in val_idx:
        part_of[int(i)] = 1
    for i in test_idx:
        part_of[int(i)] = 2
    for group in groups:
        parts = {part_of[i] for i in group if i in part_of}
        assert len(parts) <= 1, "duplicate group crosses split partitions"

    group_ids = np.empty(len(labels), dtype=np.int64)
    for gi, group in enumerate(groups):
        for i in group:
            group_ids[i] = gi

    return GroupSplit(train=train_idx, val=val_idx, test=test_idx, group_ids=group_ids)


def stratified_group_subset(
    indices: np.ndarray,
    labels: list[str] | np.ndarray,
    group_ids: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Stratified subsample of ``indices`` keeping duplicate groups together.

    Groups are shuffled deterministically per class and the first
    ``fraction`` of each class's groups (by group count) is kept, so class
    proportions are preserved and no duplicate group is split.
    """
    if fraction >= 1.0:
        return np.asarray(indices, dtype=np.int64)
    groups_in_subset: dict[int, list[int]] = {}
    for i in indices:
        groups_in_subset.setdefault(int(group_ids[i]), []).append(int(i))
    group_list = list(groups_in_subset.values())
    group_label = np.array(
        [group_majority_label(g, labels) for g in group_list],
        dtype=object,
    )
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for cls in np.unique(group_label):
        cls_groups = [g for lab, g in zip(group_label, group_list, strict=True) if lab == cls]
        perm = rng.permutation(len(cls_groups))
        n_keep = max(1, int(round(len(cls_groups) * fraction)))
        for gi in perm[:n_keep]:
            picked.extend(cls_groups[int(gi)])
    return np.array(sorted(picked), dtype=np.int64)
