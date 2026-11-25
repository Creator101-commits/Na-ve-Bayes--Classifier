# Naive Bayes Classifier

> Naive Bayes from scratch

Can a naive Bayes classifier written from scratch still hold up on real
text? This project implements Gaussian, multinomial and Bernoulli naive
Bayes without scikit-learn's model code, checks them against
hand-calculated cases and sklearn references, and benchmarks them on SMS
spam and 20 newsgroups. Every choice is made on validation data and the
test set is read once.

![Naive Bayes benchmark results: test macro-F1 and ECE, training-size curves, correlated-feature stress test, 20 Newsgroups confusion matrix and reliability diagram](image.png)

The figure follows the evidence: (a) frozen test macro-F1 per dataset
and model, (b) the matching calibration error, (c) validation macro-F1
as the training fraction grows, (d) GaussianNB calibration when features
are duplicated, (e) where 20NG predictions concentrate, and (f)
confidence versus accuracy.

## Results

| dataset | model | test macro-F1 [95% CI] | test ECE |
| :--- | :--- | ---: | ---: |
| sms_spam | multinomial | 0.9735 [0.9560, 0.9887] | 0.0097 |
| sms_spam | bernoulli | 0.9701 [0.9509, 0.9872] | 0.0111 |
| 20newsgroups | multinomial | 0.6353 [0.6251, 0.6454] | 0.3124 |
| 20newsgroups | bernoulli | 0.5487 [0.5372, 0.5594] | 0.3902 |

- **SMS spam is nearly saturated for both models.** Macro-F1 is 0.9735
  (multinomial) and 0.9701 (bernoulli) with ECE near 0.01, and the
  paired difference +0.0035 has CI [-0.0135, +0.0205], so the two are
  statistically indistinguishable here.
- **Multinomial clearly beats Bernoulli on 20NG.** Macro-F1 0.6353 vs
  0.5487, and the paired difference +0.0866 has CI [+0.0760, +0.0972],
  excluding zero. Counts carry more than binary presence for 20-class
  text.
- **20NG is decent but badly overconfident.** ECE is 0.312 and 0.390,
  roughly 30-40x the SMS values; 90.8% of multinomial test predictions
  (6,646 of 7,317) land in the top confidence bin at 0.998 mean
  confidence versus 0.697 accuracy.
- **Near-duplicates do not drive the numbers.** Removing the 63 test
  documents that duplicate train (n=7,254) moves macro-F1 by at most
  0.001 on either model.
- **The custom code matches sklearn exactly.** Prediction agreement is
  1.0 with max probability difference at most 1.1e-11 on every text and
  tabular test; GaussianNB averages 0.9489 accuracy on iris, 0.9741 on
  wine and 0.9427 on breast_cancer.
- **The naive assumption fails visibly under duplicated features.** In
  the synthetic stress test GaussianNB ECE grows from 0.023 at k=0 to
  0.183 at k=8 exact copies while logistic regression stays near 0.024.

## How it works

```text
download + sha256 verify
-> group near-duplicates (char_wb 5-gram Jaccard >= 0.8)
-> grouped stratified split; 20NG keeps its official test split
-> grid alpha x min_df x ngram on validation macro-F1 -> freeze selection
-> write analysis_plan.json + sha256 -> refit on train only
-> read test once -> metrics, bootstrap CIs, confusion, reliability
```

The estimators implement Gaussian, multinomial and Bernoulli log-space
probabilities themselves; scikit-learn is used only as a reference for
agreement checks.

## Run it

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src
```

The full benchmark downloads the datasets, then selects, freezes and
evaluates:

```bash
uv run naive-bayes prepare-data --out data
uv run naive-bayes benchmark --data data --out artifacts/benchmark
```

`scripts/kaggle_run.sh` reproduces the same run on Kaggle CPU.
