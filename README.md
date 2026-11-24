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

Curated benchmark outputs are stored under [`results/`](./results/) and are copied unedited from the Kaggle run (`git_sha` `ac5aafdf82d3594c3077674cbbaac57136bdc173`, protocol `naive-bayes-benchmark-v2`). The benchmark writes `metrics.json`, `manifest.json`, `analysis_plan.json`, per-config selection CSVs under `selection/`, and `report.md`. Every number reported in the README traces to a file in `results/`.

![Naive Bayes benchmark results: test macro-F1 and ECE, training-size curves, correlated-feature stress test, 20 Newsgroups confusion matrix and reliability diagram](image.png)

### What the figure shows

- (a) Frozen test macro-F1: both SMS models sit near 0.97 with tight
  intervals, while on 20 Newsgroups multinomial reaches 0.635 and
  bernoulli 0.549; colors are shared with panels (c) and (d).
- (b) Frozen test ECE: SMS is well calibrated at about 0.01 for both
  models, but 20NG ECE is 0.312 (MNB) and 0.390 (BNB), roughly 30-40
  times worse.
- (c) Training-size study: validation macro-F1 rises steadily with
  the training fraction on a log axis for all four dataset-model
  pairs, and the 20NG curves are still climbing at full train size.
- (d) Correlated-feature stress test: GaussianNB ECE grows from 0.023
  at k=0 to 0.183 at k=8 exact feature copies, while the logistic
  reference stays flat at about 0.024.
- (e) Row-normalized 20NG confusion matrix (multinomial): the diagonal
  averages 0.65 of each true class, but class 2 collapses to 0.03
  recall with 155 of its 379 test documents predicted as class 3.
- (f) Reliability diagram: accuracy sits below mean confidence in 14
  of 15 bins, and 90.8% of test predictions (6,646 of 7,317) land in
  the top bin at confidence 0.998 versus accuracy 0.697.

MultinomialNB is clearly better than BernoulliNB on 20 Newsgroups
(+0.087 macro-F1, paired CI excludes zero), the two are
indistinguishable on SMS spam, and 20NG probabilities are badly
overconfident even where accuracy is reasonable.

Frozen test metrics for the validation-selected configs (95% bootstrap CIs, 2,000 resamples, seed 0):

| dataset | model | n_test | accuracy | balanced acc | macro-F1 [CI] | log loss | ECE [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sms_spam | multinomial | 820 | 0.9878 | 0.9664 | 0.9735 [0.9560, 0.9887] | 0.1483 | 0.0097 [0.0047, 0.0185] |
| sms_spam | bernoulli | 820 | 0.9866 | 0.9505 | 0.9701 [0.9509, 0.9872] | 0.1316 | 0.0111 [0.0055, 0.0200] |
| 20newsgroups | multinomial | 7317 | 0.6583 | 0.6472 | 0.6353 [0.6251, 0.6454] | 7.9965 | 0.3124 [0.3017, 0.3231] |
| 20newsgroups | bernoulli | 7317 | 0.5610 | 0.5487 | 0.5487 [0.5372, 0.5594] | 8.6387 | 0.3902 [0.3787, 0.4011] |

Selected configs: SMS MNB alpha 0.3, min_df 1, ngram (1,2); SMS BNB alpha 0.1, min_df 2, ngram (1,1); 20NG MNB alpha 0.01, min_df 1, ngram (1,2); 20NG BNB alpha 0.01, min_df 2, ngram (1,1). The 20NG secondary test (63 near/exact duplicates of train removed, n=7254) gives macro-F1 0.6344 (MNB) and 0.5477 (BNB), essentially unchanged from the primary split.

Gaussian tabular (10 seeded 70/30 splits, mean accuracy [95% t-interval]): iris 0.9489 [0.9179, 0.9798], wine 0.9741 [0.9672, 0.9809], breast_cancer 0.9427 [0.9333, 0.9521]. Custom-versus-sklearn parity is exact on every text and tabular test entry (prediction agreement 1.0, max probability difference at most 1.1e-11).

## Findings

- **SMS Spam is close to saturated for both models.** Macro-F1 is 0.9735 (MNB) and 0.9701 (BNB) with tight intervals, and ECE is about 0.01, so probabilities are well calibrated on this binary task. The paired MNB-minus-BNB macro-F1 difference is +0.0035 with CI [-0.0135, +0.0205], so the two models are statistically indistinguishable here.
- **MultinomialNB is clearly better on 20 Newsgroups.** Macro-F1 is 0.6353 versus 0.5487 for BernoulliNB, and the paired difference +0.0866 has CI [+0.0760, +0.0972], excluding zero. Counts carry more information than binary presence for 20-class text.
- **20NG probabilities are poorly calibrated.** ECE is 0.312 (MNB) and 0.390 (BNB), roughly 30-40x the SMS values. The confidence-bin slices show why: 90.8% of MNB test predictions (6,646 of 7,317) and 82.7% of BNB predictions (6,054 of 7,317) land in the top bin, where accuracy is 0.697 and 0.631 against mean confidence near 1.00. The models are accurate-ish but overconfident on the harder task.
- **Duplicate control did not move 20NG results.** Dropping the 63 test documents that are exact or near duplicates of train (of 7,317) changes macro-F1 by 0.0009 (MNB: 0.6353 to 0.6344) and 0.0010 (BNB: 0.5487 to 0.5477), so near-duplicates are not driving the headline numbers.
- **GaussianNB matches sklearn exactly.** Parity agreement is 1.0 with max probability differences around 1e-15 on iris, wine, and breast_cancer across all 10 splits.
- **More training data helps steadily but with diminishing returns.** On SMS validation, MNB macro-F1 rises from 0.876 at 5% of train to 0.978 at 100%; on 20NG the curves are still rising at full train size. All points are validation-only, seeds 0-4.
- **The naive assumption fails visibly under duplicated features.** In the synthetic correlated-feature stress test, GaussianNB ECE grows from 0.023 at k=0 to 0.183 at k=8 while LogisticRegression stays near 0.024: exact feature copies inflate confidence without adding information. This is a synthetic stress test, not a real-data claim.
- **Learned class priors matter for spam precision.** On SMS validation, switching to uniform priors drops spam precision from 0.990 to 0.886 (MNB) and from 0.980 to 0.962 (BNB) at roughly unchanged recall, so the learned prior is doing real work on the imbalanced task.
- **Custom inference is faster than sklearn.** Median fit+predict on 20NG validation is 0.57s versus 1.10s for MNB and 0.044s versus 0.064s for BNB (median of 5 repeats).
- **Failure modes concentrate in short 20NG documents.** Test accuracy by document-length tertile rises from 0.334 to 0.715 (BNB) and 0.535 to 0.754 (MNB) across tertiles, and OOV-rate tertiles show a milder gradient; details are in `results/report.md`.

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

`image.png` is rendered from `results/` by `scripts/render_figure.py`. Run `uv run --with matplotlib python scripts/render_figure.py` from the repository root in an environment containing the project dependencies. Panels show test macro-F1 and ECE per (dataset, model) with bootstrap intervals, validation training-size curves, correlated-feature ECE versus k, a row-normalized 20 Newsgroups multinomial confusion matrix, and a reliability diagram with marker area by confidence-bin count.

## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

## Meta

Sreeharsha Kannegundla

Project link: [https://github.com/Creator101-commits/Na-ve-Bayes--Classifier](https://github.com/Creator101-commits/Na-ve-Bayes--Classifier)
