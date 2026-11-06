"""Tested from-scratch Naive Bayes estimators (Gaussian, Multinomial, Bernoulli)."""

from naive_bayes.bernoulli import BernoulliNB
from naive_bayes.gaussian import GaussianNB
from naive_bayes.multinomial import MultinomialNB

__version__ = "0.1.0"

__all__ = ["GaussianNB", "MultinomialNB", "BernoulliNB", "__version__"]
