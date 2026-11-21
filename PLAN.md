> This document is a roadmap. See README.md for implemented behavior and verification limits; planned capabilities are not measured results.

# Build Plan: Naïve Bayes Classifier from Scratch

## Goal

Build a tested Python implementation of Naïve Bayes classification without using scikit-learn's Naïve Bayes model code for training or prediction. The project will implement Gaussian, Multinomial, and Bernoulli variants, verify their mathematics against small hand-calculated examples, and compare their accuracy, probability calibration, numerical stability, and runtime with trusted scikit-learn reference implementations.

## Scope

### Included in v1

- Create a reusable Python package with a consistent estimator interface.
- Implement Gaussian Naïve Bayes for continuous numerical features.
- Implement Multinomial Naïve Bayes for non-negative count features such as word counts.
- Implement Bernoulli Naïve Bayes for binary features such as word presence.
- Implement class-prior estimation and optional user-supplied priors.
- Implement Laplace or additive smoothing where mathematically appropriate.
- Perform probability calculations in log space to prevent numerical underflow.
- Provide `fit`, `predict`, `predict_log_proba`, and `predict_proba` methods.
- Validate input shapes, target labels, feature domains, smoothing parameters, and fitted state.
- Support binary and multiclass classification.
- Build a small text preprocessing pipeline using scikit-learn vectorizers while keeping the classifier itself independent.
- Evaluate Gaussian Naïve Bayes on public tabular datasets.
- Evaluate Multinomial and Bernoulli Naïve Bayes on a public text-classification dataset.
- Compare predictions and probabilities with equivalent scikit-learn estimators.
- Report classification quality, calibration, runtime, and failure cases.
- Add unit, property-style, integration, and regression tests.
- Provide a command-line experiment workflow and a complete research report.

### Explicitly outside v1

- Reimplementing tokenization, sparse-matrix storage, or every preprocessing algorithm from scratch.
- Reusing scikit-learn Naïve Bayes classes inside the custom models.
- Deep-learning or transformer classifiers.
- A web application or production API.
- Streaming or distributed training.
- GPU acceleration.
- Claims that Naïve Bayes is universally better than more complex classifiers.
- Using private email or message data.
- Fabricated benchmark results or manually selected favorable test examples.
- Categorical Naïve Bayes and Complement Naïve Bayes unless added after the three required variants are complete.

## Research questions

1. Can a from-scratch implementation match scikit-learn's predictions and probabilities within documented numerical tolerances?
2. How do Gaussian, Multinomial, and Bernoulli assumptions affect performance on different data types?
3. How strongly does additive smoothing affect text-classification accuracy and probability calibration?
4. How much does the conditional-independence assumption hurt when features are highly correlated?
5. What runtime and memory costs result from a transparent educational implementation compared with an optimized library implementation?

## Stack

- **Language:** Python 3.12
- **Environment and packaging:** `uv`, `pyproject.toml`, and locked dependencies
- **Core implementation:** NumPy and SciPy sparse matrices
- **Data preparation:** pandas and scikit-learn datasets, splitting, and vectorizers
- **Reference models:** scikit-learn Naïve Bayes estimators used only for comparison tests and benchmarks
- **Visualization:** Matplotlib and seaborn
- **Experiment records:** versioned JSON/CSV artifacts or local MLflow if experiment volume justifies it
- **Command line:** Typer
- **Testing and quality:** pytest, pytest-cov, Hypothesis when justified, Ruff, mypy, and GitHub Actions
- **Documentation:** Markdown report, mathematical derivations, and generated result tables

## Architecture

```text
Public datasets
      |
      v
download/load -> validate -> deterministic splits
      |
      +----------------------+----------------------+
      |                      |                      |
      v                      v                      v
continuous features     word counts           binary features
      |                      |                      |
      v                      v                      v
Gaussian NB            Multinomial NB          Bernoulli NB
      |                      |                      |
      +----------------------+----------------------+
                             |
                             v
              shared prediction/probability API
                             |
             +---------------+----------------+
             |                                |
             v                                v
    correctness comparison              benchmark evaluator
    against scikit-learn                        |
             |                                  v
             +------------------------> tables + figures
                                                |
                                                v
                                         research report
```

### Components

#### `src/naive_bayes/base.py`

Defines shared estimator behavior, fitted-state checks, label encoding, class-prior handling, input validation, stable conversion from joint log likelihoods to normalized probabilities, and serialization conventions. It must not contain assumptions specific to one feature distribution.

#### `src/naive_bayes/gaussian.py`

Implements Gaussian class-conditional likelihoods for continuous features. Training estimates a mean and variance for every class-feature pair. Prediction uses stable joint log likelihoods and a configurable variance floor so constant or nearly constant features do not cause division by zero.

#### `src/naive_bayes/multinomial.py`

Implements count-based likelihoods for non-negative dense or sparse features. Training calculates smoothed feature counts by class, and prediction combines log feature probabilities with log class priors.

#### `src/naive_bayes/bernoulli.py`

Implements binary feature-presence likelihoods, including contributions from features that are absent. It supports a documented binarization threshold but keeps feature transformation separate from probability estimation.

#### `src/naive_bayes/data.py`

Loads public datasets, validates source schemas, creates deterministic stratified partitions, and constructs training-only preprocessing pipelines. Text vectorizers must fit on training text only.

#### `src/naive_bayes/evaluation.py`

Computes classification, probability, calibration, timing, and agreement metrics. It compares custom and reference implementations on identical inputs and preserves per-example predictions for error analysis.

#### `src/naive_bayes/experiments.py`

Loads versioned configurations, sets seeds, trains models, records environment information, writes machine-readable results, and prevents final test data from being used for parameter selection.

#### `src/naive_bayes/reporting.py`

Creates comparison tables, confusion matrices, calibration plots, smoothing curves, class-prior sensitivity plots, correlated-feature experiments, and runtime charts from saved results.

#### `src/naive_bayes/cli.py`

Provides thin commands for data preparation, mathematical verification, model training, comparison, and report generation. Modeling logic remains in tested modules.

## Mathematical design

### Bayes decision rule

For class `c` and feature vector `x`, every implementation predicts the class with the largest posterior probability:

```text
P(c | x) proportional to P(c) P(x | c)
```

Under the Naïve Bayes conditional-independence assumption:

```text
P(x | c) = product over features j of P(x_j | c)
```

The implementation will calculate the equivalent sum in log space:

```text
log P(c, x) = log P(c) + sum over features j of log P(x_j | c)
```

This prevents many small probabilities from underflowing to zero.

### Gaussian likelihood

For every class and continuous feature, estimate the training mean and variance. The likelihood uses the univariate normal density for each feature. A documented variance-smoothing rule prevents zero variance from producing undefined probabilities.

### Multinomial likelihood

For count feature `j` in class `c`, estimate:

```text
P(feature_j | c) =
    (count_jc + alpha) /
    (sum_of_feature_counts_c + alpha * number_of_features)
```

`alpha` controls additive smoothing. The primary experiments will include `alpha = 0`, `0.01`, `0.1`, `0.5`, `1`, and `2`, with safe handling for the unsmoothed zero-probability case.

### Bernoulli likelihood

Estimate the probability that each feature is present in each class. Prediction includes the probability of present and absent features. This difference from Multinomial Naï Bayes must be covered by tests because ignoring absent features changes the model.

### Probability normalization

Convert joint log likelihoods into class probabilities with a stable log-sum-exp calculation. Rows must sum to one within a documented tolerance, contain no unexplained `NaN` or infinite values, and preserve the original class-label order.

## Data

### Tabular datasets

Use small public datasets distributed with scikit-learn so acquisition is deterministic and requires no private data:

- **Iris:** multiclass flower classification with four continuous measurements
- **Wine:** multiclass wine classification with continuous chemical measurements
- **Breast Cancer Wisconsin:** binary classification with continuous measurements

Gaussian Naïve Bayes will be evaluated on these datasets. Any feature scaling experiment must fit transformations on training data only, although Gaussian Naïve Bayes does not inherently require standardized inputs.

### Text dataset

Use the public **SMS Spam Collection** from the UCI Machine Learning Repository, subject to verification of its source, license or usage terms, and schema. It provides short labeled text messages suitable for comparing Multinomial and Bernoulli feature assumptions.

Expected logical schema:

- `message_id`: stable generated identifier
- `label`: `ham` or `spam`
- `text`: original message text
- `split`: train, validation, or test

If dataset usage terms prevent redistribution, only the download script, source metadata, checksums, and small synthetic test fixtures will be committed.

### Acquisition and validation

- Record the official source URL, acquisition date, dataset version when available, file size, checksum, schema, and use terms.
- Preserve downloaded source data unchanged under `data/raw/`.
- Validate labels, empty messages, duplicates, exact class counts, encoding, and malformed rows.
- Assign duplicate or near-duplicate messages to the same partition to reduce leakage.
- Store processed data as Parquet or versioned CSV with a manifest.
- Keep raw and processed benchmark datasets outside Git.

### Splitting protocol

- Create deterministic, stratified train, validation, and test partitions.
- Keep duplicate groups within one partition.
- Fit vocabulary, document-frequency filters, and all transformations on training data only.
- Use validation data for smoothing, vectorizer, and threshold selection.
- Freeze the test partition until model and evaluation decisions are final.
- Use repeated stratified cross-validation on training/development data as a secondary stability analysis, not as a substitute for the final held-out test.

### Text features

- Multinomial model: integer unigram count vectors as the primary representation.
- Bernoulli model: binary word-presence vectors from the same fitted vocabulary.
- Optional sensitivity analysis: unigram-plus-bigram features and minimum document-frequency thresholds.
- TF-IDF is not the primary Multinomial experiment because the model is derived for counts, but it may be included as a clearly labeled practical comparison.

### Synthetic verification data

Tiny hand-authored arrays may be committed only for unit tests where the expected probabilities are calculated explicitly. They are not benchmark data and cannot appear in reported real-world performance results.

## Evaluation design

### Correctness tests

- Compare learned class counts, means, variances, feature counts, and priors with hand-computed examples.
- Compare custom and scikit-learn predictions on fixed datasets.
- Compare log probabilities and normalized probabilities within documented numerical tolerances.
- Test non-contiguous labels, string labels, multiclass inputs, one-feature inputs, sparse matrices, and batch sizes of one.
- Test constant features, unseen vocabulary items, zero counts, highly imbalanced classes, and extremely large feature counts.
- Confirm that predictions before fitting fail with a clear error.
- Confirm that invalid negative Multinomial features and non-binary Bernoulli inputs are rejected or transformed only through documented behavior.

### Primary benchmark metrics

- Accuracy
- Balanced accuracy
- Macro-F1
- Log loss

### Secondary metrics

- Per-class precision, recall, and F1
- Confusion matrix
- Brier score for binary tasks
- Expected calibration error with a predeclared binning method
- Agreement rate with the equivalent scikit-learn estimator
- Maximum and mean absolute probability difference from scikit-learn
- Fit time and prediction time
- Peak memory when practical

### Controlled experiments

#### Smoothing sensitivity

Measure classification and calibration metrics over a fixed `alpha` grid using validation data. Select one primary value before opening the test set.

#### Training-size sensitivity

Train with fixed stratified fractions of the training data to show how quickly each variant reaches stable performance. Repeat sampling with declared seeds.

#### Correlated-feature stress test

Create a clearly labeled simulation that duplicates or correlates informative features while preserving a known data-generating process. Use it only to demonstrate how the conditional-independence assumption can create overconfident probabilities; do not mix it with real dataset benchmarks.

#### Class-prior sensitivity

Compare learned priors with deliberately specified priors on controlled data and one imbalanced real task. Report when priors change recall, precision, and calibration.

### Statistical reporting

- Run final stochastic data-split or training-size experiments with multiple declared seeds.
- Bootstrap held-out examples for paired differences between custom and reference models.
- Report confidence intervals and exact sample counts, not only rounded point estimates.
- Treat near-identical custom/reference results as implementation validation rather than a claim of model superiority.

## Proposed repository structure

```text
.
├── AGENTS.md
├── README.md
├── PLAN.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── gaussian.yaml
│   ├── multinomial.yaml
│   └── bernoulli.yaml
├── src/
│   └── naive_bayes/
│       ├── __init__.py
│       ├── base.py
│       ├── gaussian.py
│       ├── multinomial.py
│       ├── bernoulli.py
│       ├── data.py
│       ├── evaluation.py
│       ├── experiments.py
│       ├── reporting.py
│       └── cli.py
├── tests/
│   ├── fixtures/
│   ├── test_gaussian.py
│   ├── test_multinomial.py
│   ├── test_bernoulli.py
│   ├── test_probabilities.py
│   └── test_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
└── docs/
    └── research-report.md
```

## Planned commands

The implementation should eventually provide commands equivalent to:

```bash
uv sync --all-groups
uv run naive-bayes verify-math
uv run naive-bayes prepare --dataset sms-spam
uv run naive-bayes train --config configs/multinomial.yaml
uv run naive-bayes compare --config configs/multinomial.yaml
uv run naive-bayes report
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Commands must not be advertised in the README as working until their implementation and smoke tests exist.

## Milestones

### 1. Reproducible Python skeleton
**Status:** Complete.

Create `pyproject.toml`, lock dependencies, add the `src/` package, establish the CLI, configure pytest, Ruff, mypy, coverage, and GitHub Actions, and add an appropriate `.gitignore`. Replace the existing README and AGENTS placeholders with accurate Python commands and project rules. Ship a clean-install CI job and one passing import test.

### 2. Shared estimator foundation
**Status:** Complete.

Implement label encoding, class priors, validation, fitted-state checks, stable log-sum-exp normalization, estimator parameters, and the shared API. Add hand-calculated tests for priors, posterior normalization, label order, and invalid inputs.

### 3. Gaussian Naïve Bayes
**Status:** Complete.

Implement class-feature means, variances, variance smoothing, joint log likelihood, prediction, and probabilities. Verify values on tiny arrays and compare against scikit-learn on fixed tabular datasets. Ship the first end-to-end model and benchmark table.

### 4. Multinomial Naïve Bayes
**Status:** Complete.

Implement dense and sparse feature counts, additive smoothing, log feature probabilities, and prediction. Test zero counts, invalid negative values, large counts, sparse inputs, and probability agreement. Ship a count-based text experiment.

### 5. Bernoulli Naïve Bayes
**Status:** Complete.

Implement binary feature likelihoods, absent-feature contributions, smoothing, and configurable binarization behavior. Test cases that distinguish Bernoulli from Multinomial predictions. Ship the shared text comparison.

### 6. Data and experiment pipeline
**Status:** Complete (implementation and tests; end-to-end validation runs on Kaggle).

Public data acquisition is implemented for the SMS Spam Collection (UCI 228, sha256-recorded), the 20 Newsgroups bydate split, and bundled tabular sets. A data manifest records sources, hashes, sizes, per-split and per-class counts, dropped rows, acquisition time, and library versions. Duplicate-aware text splits combine exact normalization with char_wb 5-gram near-duplicate detection (Jaccard >= 0.8) via union-find groups; SMS uses a group-level stratified 70/15/15 split and 20NG keeps its official split with a grouped validation carve and a test-vs-train duplicate audit. Vectorizers fit on the training partition only, and the benchmark writes metrics, manifest, selection tables, and a report.

### 7. Correctness and numerical-stability audit
**Status:** Complete.

Systematic custom-versus-scikit comparisons, hand-calculated checks, probability calibration, sparse/dense handling, and edge cases are covered by the test suite. Intentional differences are documented rather than hidden.

### 8. Controlled research experiments
**Status:** Complete (Kaggle run `ac5aafd`, results in `results/`).

The smoothing grid (alpha x min_df x ngram_range), training-size fractions, correlated-feature stress test, class-prior sensitivity, and runtime studies ran on validation or synthetic data only. Curated outputs are in `results/metrics.json` under `controlled/`.

### 9. Frozen evaluation and failure analysis
**Status:** Complete (Kaggle run `ac5aafd`, results in `results/`).

The benchmark froze datasets, partitions, and the validation-selected configurations, wrote selection tables before the first test read (enforced by the manifest epoch ordering, `ordering_ok: true`), refit on the training partition only, and evaluated the held-out test once with bootstrap confidence intervals, per-class metrics, confusion matrices, calibration error, failure-analysis slices, and custom-versus-scikit parity.

### 10. Final research artifact
**Status:** Complete.

The benchmark, CLI, Kaggle kernel, tests, curated `results/` outputs, the re-rendered `image.png`, and the README findings section are complete; every reported number traces to a file in `results/` from Kaggle run `ac5aafd`.

## Open questions

1. **Primary deliverable:** Decide whether the repository emphasizes educational mathematical transparency, benchmark research, or equal weight for both. The proposed plan supports both but the README should state one primary identity.
2. **Package name:** Choose an import-safe package name such as `naive_bayes`, regardless of the repository's spaces and Unicode characters.
3. **Repository name:** Consider whether the current directory and remote name should eventually be normalized to a URL-safe kebab-case name. Renaming is outside this planning change.
4. **Dependency manager:** Confirm `uv` before creating project files because the current repository contains no implemented Python dependency setup.
5. **Text dataset:** Verify the official SMS Spam Collection source and redistribution terms before adding the downloader.
6. **Sparse support:** Decide whether v1 must support both SciPy CSR and CSC inputs or standardize internally on CSR.
7. **Scikit-learn compatibility:** Decide how closely parameter names and estimator behavior should follow scikit-learn without copying its implementation.
8. **Variance smoothing:** Choose a transparent rule and document any expected probability differences from scikit-learn.
9. **Calibration bins:** Fix the calibration-error method and bin count before final evaluation.
10. **Experiment tracking:** Use versioned JSON/CSV initially; add MLflow only if it materially improves run management rather than adding infrastructure for a small experiment set.
11. **Optional variants:** Add Complement or Categorical Naïve Bayes only after all required models, tests, and research experiments are complete.