# Review and verification

Scope: source modules, CLI behavior, evaluation mathematics, project claims, Kaggle scripts where present, and README figures.

Fixed zero-smoothing likelihoods without densifying sparse text, finite-domain validation, model-class calibration alignment, and exact-duplicate grouped text splits.

## Verified locally

- 48 tests passed with the repository's existing Python environment.
- `ruff check .` passed, including tests, figure source, and Kaggle scripts.
- `mypy src` passed.
- The README image was rendered from code using Matplotlib and visually inspected. Its inputs/results and a source fingerprint are in `scripts/figure-data.json`.

## Limits

Packaged scikit-learn datasets and a small embedded SMS demonstration are supported; large text benchmarks and near-duplicate auditing remain planned.

Kaggle API access failed because the execution environment could not resolve its hostname. The private notebook was also unavailable through the browser. No GPU training or full-scale external benchmark result is claimed by this review. New CI workflows are configuration changes; their hosted execution must be checked after publication.
