"""Data acquisition, validation, and deterministic partitioning."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import (  # type: ignore[import-untyped]
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore[import-untyped]


@dataclass(frozen=True)
class DatasetBundle:
    """Container for dataset arrays and metadata."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_iris_dataset() -> DatasetBundle:
    """Load standard Iris dataset with continuous features."""
    raw: Any = load_iris()
    return DatasetBundle(
        name="iris",
        X=np.asarray(raw.data, dtype=np.float64),
        y=np.asarray(raw.target, dtype=np.int64),
        feature_names=list(raw.feature_names),
        target_names=list(raw.target_names),
    )


def load_wine_dataset() -> DatasetBundle:
    """Load standard Wine dataset with continuous features."""
    raw: Any = load_wine()
    return DatasetBundle(
        name="wine",
        X=np.asarray(raw.data, dtype=np.float64),
        y=np.asarray(raw.target, dtype=np.int64),
        feature_names=list(raw.feature_names),
        target_names=list(raw.target_names),
    )


def load_cancer_dataset() -> DatasetBundle:
    """Load standard Breast Cancer Wisconsin dataset with continuous features."""
    raw: Any = load_breast_cancer()
    return DatasetBundle(
        name="breast_cancer",
        X=np.asarray(raw.data, dtype=np.float64),
        y=np.asarray(raw.target, dtype=np.int64),
        feature_names=list(raw.feature_names),
        target_names=list(raw.target_names),
    )


def load_sms_spam_dataset() -> tuple[list[str], list[str]]:
    """Provide a deterministic text dataset for spam/ham classification.

    Includes representative messages across classes with duplicates to test
    grouping and text vectorization pipelines.
    """
    messages = [
        "Go until jurong point, crazy.. Available only in bugis great world la e buffet...",
        "Ok lar... Joking wif u oni...",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
        "U dun say so early hor... U c already then say...",
        "Nah I don't think he goes to usf, he lives around here though",
        "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun",
        "Even my brother is not like to speak with me. They treat me like aids patent.",
        "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize",
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
        "I've been searching for the right words to thank you for this breather.",
        "I HAVE A DATE ON SUNDAY WITH WILL!!",
        "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message",
        "Oh k...i'm watching here:)",
        "Eh u remember how 2 spell his name... Yes i did. He v naughty make until i v wet.",
        "Fine if that's the way u feel. That's the way its gota b",
        "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077",
        "Is that seriously how you spell his name?",
        "I‘m going to try for 2 months ha ha only joking",
        "So ü pay first lar... Then when is da stock comin...",
        "Aft i finish my lunch then i go str down lor. Ard 3 smth lor.",
        "Ffffffffff. Alright no way I can meet up with you then?",
        "Just forced myself to eat a slice. I'm really not hungry tho. This sucks.",
        "Lol your always so convincing.",
        "Did you catch the bus? Are you frying an egg? Did you make a tea?",
        "I'm back & we're packing the car now, I'll let you know if there's room",
        "Ahhh. Work. I vaguely remember that! What does it feel like?",
        "Wait that's still not all that clear, were you not sure about me being sarcastic?",
        "Yeah he got cancherless and burned his finger with the lighter.",
        "URGENT! We are trying to contact you. Last weekends draw shows that you won a £1000 prize",
        "Your account has been credited with 500 bonus points. Call 0800 now to claim",
        "Congratulations! You won a guaranteed prize. Text CLAIM to 88882",
    ]
    labels = [
        "ham",
        "ham",
        "spam",
        "ham",
        "ham",
        "spam",
        "ham",
        "ham",
        "spam",
        "spam",
        "ham",
        "spam",
        "spam",
        "ham",
        "ham",
        "spam",
        "ham",
        "ham",
        "ham",
        "spam",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "ham",
        "spam",
        "spam",
        "spam",
    ]
    return messages, labels


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data deterministically into stratified train, val, and test partitions."""
    total_test_val = val_size + test_size
    if val_size <= 0 or test_size <= 0 or total_test_val >= 1.0:
        raise ValueError("Sum of val_size and test_size must be less than 1.0.")

    # First split off test+val
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=total_test_val, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    # Then split temp into val and test
    val_relative_size = val_size / total_test_val
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1.0 - val_relative_size,
        random_state=random_state,
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_text_indices(
    texts: list[str], labels: list[str], random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign normalized exact duplicates together in a stratified frozen split."""
    if len(texts) != len(labels):
        raise ValueError("Texts and labels must be aligned.")
    groups: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        key = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        groups.setdefault(key, []).append(i)
    members = list(groups.values())
    for group in members:
        if len({labels[i] for i in group}) != 1:
            raise ValueError("Duplicate texts have conflicting labels.")
    group_ids = np.arange(len(members))[:, None]
    group_labels = np.array([labels[g[0]] for g in members])
    train, val, test, _, _, _ = train_val_test_split(
        group_ids, group_labels, val_size=0.15, test_size=0.15, random_state=random_state
    )

    def expand(indices: np.ndarray) -> np.ndarray:
        return np.array([i for g in indices.ravel() for i in members[int(g)]])

    return expand(train), expand(val), expand(test)


def build_text_pipeline(
    train_texts: list[str],
    binary: bool = False,
    min_df: int = 1,
) -> tuple[CountVectorizer, Any]:
    """Fit a count vectorizer strictly on training text and return vectorizer + matrix."""
    vectorizer = CountVectorizer(binary=binary, min_df=min_df)
    X_train = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train
