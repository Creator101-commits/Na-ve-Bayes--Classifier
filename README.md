![Banner](./image.png)

# Naive Bayes Classifier

> A tested from-scratch implementation of Gaussian, Multinomial, and Bernoulli Naive Bayes, verified against scikit-learn references and benchmarked on real text and tabular datasets.

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Creator101-commits/Na-ve-Bayes--Classifier?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Creator101-commits/Na-ve-Bayes--Classifier)
![GitHub issues](https://img.shields.io/github/issues-raw/Creator101-commits/Na-ve-Bayes--Classifier)
![GitHub pull requests](https://img.shields.io/github/issues-pr-raw/Creator101-commits/Na-ve-Bayes--Classifier)
![GitHub](https://img.shields.io/github/license/Creator101-commits/Na-ve-Bayes--Classifier)

This repository contains a tested Python implementation of Gaussian, Multinomial, and Bernoulli Naive Bayes built without reusing scikit-learn's model code. The custom estimators are verified against hand-calculated fixtures and scikit-learn reference behavior, then benchmarked end to end on the SMS Spam Collection, 20 Newsgroups, and tabular datasets with leakage-controlled splits, validation-only model selection, frozen test evaluation, bootstrap confidence intervals, and controlled experiments.

Status: **Estimators, benchmark pipeline, and Kaggle reproduction available**. See [PLAN.md](./PLAN.md) for the roadmap and [REVIEW.md](./REVIEW.md) for verification notes.

## Features

- From-scratch `GaussianNB`, `MultinomialNB`, and `BernoulliNB` with log-space probabilities, input validation, and configurable smoothing and priors.
- Hand-calculated correctness checks plus scikit-learn parity comparisons (prediction agreement and probability differences).
- Real-data loaders for the SMS Spam Collection (UCI 228) and 20 Newsgroups (bydate), with sha256-recorded acquisition and a data manifest.
- Near-duplicate control: exact normalization plus char_wb 5-gram Jaccard (threshold 0.8) union-find grouping for duplicate-aware splits.
- Validation-only hyperparameter selection over alpha, min_df, and ngram_range, with deterministic tie-breaks.
- Frozen single test evaluation with bootstrap 95% CIs, per-class metrics, confusion matrices, calibration error, and paired model comparisons.
- Controlled experiments: smoothing grid, training-size curves, correlated-feature stress test, class-prior sensitivity, and runtime.
- A Kaggle kernel that reproduces the full benchmark on CPU.

## Quickstart

```sh
git clone https://github.com/Creator101-commits/Na-ve-Bayes--Classifier.git
cd Na-ve-Bayes--Classifier
uv sync --all-groups
```

Verify the estimators:

```sh
uv run naive-bayes verify-math
uv run naive-bayes compare
```

Run the full benchmark locally (downloads datasets, then evaluates):

```sh
uv run naive-bayes prepare-data --out data
uv run naive-bayes benchmark --data data --out artifacts/benchmark
```

## Installation

Requires Python 3.12 or 3.13 (Kaggle images ship 3.13). Use [uv](https://docs.astral.sh/uv/):

```sh
uv sync --all-groups
```

Runtime dependencies are numpy, scipy, pandas, scikit-learn, and typer. Matplotlib is used only by the figure script in a separate environment.

## CLI Reference

| Command | Description |
| --- | --- |
| `naive-bayes version` | Print the package version. |
| `naive-bayes verify-math` | Run hand-calculated correctness checks. |
| `naive-bayes train --dataset NAME` | Train and evaluate on iris, wine, breast_cancer, or the sms_spam fixture. |
| `naive-bayes compare` | Compare custom estimators against scikit-learn references. |
| `naive-bayes report -o FILE` | Generate the verification report. |
| `naive-bayes prepare-data --out DIR` | Download and validate benchmark datasets; writes a data manifest. |
| `naive-bayes benchmark --data DIR --out DIR` | Run selection, frozen test, metrics, and controlled studies; writes `metrics.json`, `manifest.json`, selection CSVs, and `report.md`. |

## Python API

```python
import numpy as np
from naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Continuous features
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)

# Count features
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_counts, y_train)

# Binary presence features
bnb = BernoulliNB(alpha=1.0, binarize=None)
bnb.fit(X_binary, y_train)
```

`MultinomialNB` and `BernoulliNB` accept an optional `priors` array to override learned class priors. `BernoulliNB` binarizes inputs at `binarize` (default 0.0) or assumes binary input when `binarize=None`.

## Evaluation and Methodology

- **Datasets.** SMS Spam Collection v.1 (UCI 228, CC BY 4.0) and 20 Newsgroups bydate via `sklearn.datasets.fetch_20newsgroups` with headers, footers, and quotes removed; empty documents are dropped and counted. Iris, wine, and breast_cancer are bundled tabular sets.
- **Duplicate control.** Exact duplicates use NFKC + casefold + whitespace normalization. Near-duplicates use char_wb 5-gram binary shingles with Jaccard >= 0.8 (a fixed threshold). SMS is split by union-find group at the group level (stratified 70/15/15, seed 42) so no duplicate group crosses partitions. 20NG keeps the official split, carves a grouped stratified 15% validation from train, and audits test-vs-train duplicates.
- **Selection.** For each (dataset, model) the grid is alpha in {0.01, 0.03, 0.1, 0.3, 1.0, 3.0} x min_df in {1, 2, 5} x ngram_range in {(1,1), (1,2)}. Vectorizers fit on train only. Selection maximizes validation macro-F1, breaking ties by lower validation log loss, smaller alpha, smaller min_df, then the narrower ngram range. Cells whose vectorizer yields an empty vocabulary are recorded in the selection CSV with `skipped` and excluded from selection.
- **Frozen test.** After all selection CSVs are written, `analysis_plan.json` records the selected configs and every declared constant (thresholds, grid, bootstrap, ECE bins, seeds, fractions); the manifest stores its sha256. The selected config is then refit on the training partition only (not train+val) and the test set is read exactly once. The manifest records `card_write_epoch < first_test_read_epoch` (nanosecond ordering).
- **Metrics.** Accuracy, balanced accuracy, macro-F1, log loss, Brier (binary), and top-label ECE (15 equal-width bins). Per-class precision/recall/F1 and confusion matrices are reported. Bootstrap 95% CIs use 2,000 resamples (seed 0); a paired bootstrap reports the MNB-minus-BNB macro-F1 difference on shared resamples.
- **Gaussian tabular.** No tuning (`var_smoothing=1e-9`); 10 seeded stratified 70/30 splits (seeds 0-9) report mean and a 95% t-interval plus sklearn parity.
- **Controlled experiments.** These never touch test: the smoothing grid table, training-size fractions {0.05, 0.1, 0.25, 0.5, 1.0} (stratified, group-aware for SMS; seeds 0-4) on validation macro-F1 and log loss, a synthetic correlated-feature stress test (k exact copies, k in {0,1,2,4,8}, n=5,000, seeds 0-4) with a LogisticRegression reference, SMS class-prior sensitivity, and 20NG fit/predict runtime (median of 5) measured on the validation partition, not the test set.

## Results

Curated benchmark outputs are stored under [`results/`](./results/) and are copied unedited from the Kaggle run. The benchmark writes `metrics.json`, `manifest.json`, per-config selection CSVs under `selection/`, and `report.md`. Every number reported in the README traces to a file in `results/`.

## Findings

Findings are drawn from the frozen test metrics, validation selection tables, and controlled studies in `results/`. They are reported with bootstrap confidence intervals and, where relevant, paired MNB-versus-BNB comparisons and custom-versus-scikit-learn parity. Neutral or negative results are reported as observed.

## Reproduction

Local reproduction:

```sh
uv run naive-bayes prepare-data --out data
uv run naive-bayes benchmark --data data --out artifacts/benchmark
```

The benchmark is also run on Kaggle (CPU, internet enabled) by `scripts/kaggle_run.sh`, which archives HEAD, versions the private source dataset `sreeharshak101/naive-bayes-classifier-src`, pushes kernel `sreeharshak101/naive-bayes-classifier-benchmark`, polls to completion, downloads outputs to `artifacts/kaggle/<timestamp>/`, and verifies the run manifest `git_sha` equals HEAD.

## Development and Testing

```sh
uv run pytest          # tests with coverage
uv run ruff check .    # lint
uv run mypy src        # strict type checking
```

CI runs lint, strict mypy, and pytest on Ubuntu and macOS for Python 3.12 and 3.13. Tests never use the network; dataset loaders are injected or monkeypatched and the real downloads run only in `prepare-data`/Kaggle.

## Provenance

- SMS Spam Collection v.1: UCI Machine Learning Repository, id 228, <https://archive.ics.uci.edu/datasets/228/sms+spam+collection>, CC BY 4.0.
- 20 Newsgroups: fetched via `sklearn.datasets.fetch_20newsgroups` (bydate split).
- Iris, Wine, Breast Cancer: bundled with scikit-learn.
- The 35-message embedded SMS set is a test fixture only (`load_sms_fixture`), not a benchmark.

## Reproducing the figure

`image.png` is rendered from `results/` by `scripts/render_figure.py` using `scripts/figure-data.json`. Run it in a Python environment containing the project dependencies and Matplotlib. Panels show test macro-F1 with confidence intervals per (dataset, model), a reliability/ECE comparison, training-size curves, and correlated-feature ECE versus k.

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## Meta

Sreeharsha Kannegundla

Project link: [https://github.com/Creator101-commits/Na-ve-Bayes--Classifier](https://github.com/Creator101-commits/Na-ve-Bayes--Classifier)
