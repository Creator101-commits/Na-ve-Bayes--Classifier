"""Report generation and tabular artifact serialization."""

from __future__ import annotations

from naive_bayes.evaluation import ReferenceComparisonResult


def format_continuous_benchmark_table(
    benchmarks: dict[str, ReferenceComparisonResult],
) -> str:
    """Format Gaussian Naive Bayes benchmark results as a Markdown table."""
    header = (
        "| Dataset | Custom Acc | Reference Acc | Agreement Rate | "
        "Max Prob Diff | Mean Prob Diff |"
    )
    divider = "| :--- | :--- | :--- | :--- | :--- | :--- |"
    lines = [header, divider]
    for name, res in benchmarks.items():
        lines.append(
            f"| `{name}` | {res.custom_accuracy:.4f} | {res.ref_accuracy:.4f} | "
            f"{res.agreement_rate * 100:.2f}% | {res.max_prob_diff:.6f} | "
            f"{res.mean_prob_diff:.6f} |"
        )
    return "\n".join(lines)


def format_text_benchmark_table(
    benchmarks: dict[str, dict[str, float]],
) -> str:
    """Format text classification results across models and alpha parameters."""
    lines = [
        "| Model Setting | Test Accuracy | Scikit-Learn Agreement |",
        "| :--- | :--- | :--- |",
    ]
    for setting, res in benchmarks.items():
        lines.append(
            f"| `{setting}` | {res['accuracy']:.4f} | "
            f"{res['agreement_with_sklearn'] * 100:.2f}% |"
        )
    return "\n".join(lines)


def generate_research_report(
    math_results: dict[str, bool],
    continuous_results: dict[str, ReferenceComparisonResult],
    text_results: dict[str, dict[str, float]],
) -> str:
    """Produce the complete Markdown report documenting implementation correctness."""
    math_summary = "\n".join(
        [f"- `{k}`: {'Passed' if v else 'Failed'}" for k, v in math_results.items()]
    )

    report = f"""# Naive Bayes Classifier Research and Verification Report

## Mathematical Verification

All hand-calculated test fixtures were verified against analytical expectations:

{math_summary}

## Gaussian Naive Bayes Benchmark Comparison

{format_continuous_benchmark_table(continuous_results)}

## Text Benchmark Comparison (Multinomial vs Bernoulli)

{format_text_benchmark_table(text_results)}
"""
    return report
