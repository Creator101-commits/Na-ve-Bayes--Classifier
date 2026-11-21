# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Fixed zero-smoothing likelihoods without densifying sparse text, finite-domain validation, model-class calibration alignment, and exact-duplicate grouped text splits.

## Benchmark pipeline added

Added real-data loaders for the SMS Spam Collection (UCI 228, CC BY 4.0, sha256 and byte size recorded) and the 20 Newsgroups bydate split, plus bundled iris, wine, and breast_cancer tabular sets. Added char_wb 5-gram near-duplicate detection (Jaccard >= 0.8) with union-find grouping, a group-level stratified SMS split, a grouped 20NG validation carve with a test-vs-train duplicate audit, validation-only hyperparameter selection, bootstrap confidence intervals, paired MNB-minus-BNB differences, Gaussian repeated-split t-intervals, and the controlled studies (smoothing grid, training-size, correlated-feature stress, class-prior sensitivity, runtime). Selection tables are written before the first test read and the manifest records the epoch ordering.

## Verified locally

- 64 tests passed, including SMS TSV parsing and validation errors, near-duplicate Jaccard and union-find grouping, vectorizer fit-on-train-only, selection tie-breaks, bootstrap determinism and paired-difference sign, a hand-computed ECE, the correlated-feature generator, and a benchmark end-to-end run on fixture data that writes every artifact.
- `ruff check .` passed.
- `mypy src` passed (13 source files).
- Real downloads verified locally: SMS Spam Collection parsed to 5,574 rows (4,827 ham, 747 spam) and 20 Newsgroups fetched (11,014 train with 300 empty dropped, 7,317 test with 215 empty dropped).
- The 35-message embedded SMS set is retained only as a test fixture under `load_sms_fixture`.

## Limits

The full-scale benchmark numbers are produced by the Kaggle kernel `sreeharshak101/naive-bayes-classifier-benchmark`; curated `results/` outputs, the re-rendered `image.png`, and the README findings section are pending that run so every reported number traces to a results file. New CI workflows are configuration changes; their hosted execution must be checked after publication.
