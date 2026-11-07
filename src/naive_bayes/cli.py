"""Command-line interface for Naive Bayes implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from naive_bayes import __version__
from naive_bayes.data import (
    build_text_pipeline,
    load_cancer_dataset,
    load_iris_dataset,
    load_sms_spam_dataset,
    load_wine_dataset,
    split_text_indices,
    train_val_test_split,
)
from naive_bayes.evaluation import compute_classification_metrics
from naive_bayes.experiments import (
    run_continuous_benchmarks,
    run_text_benchmarks,
    verify_mathematics,
)
from naive_bayes.gaussian import GaussianNB
from naive_bayes.multinomial import MultinomialNB
from naive_bayes.reporting import (
    format_continuous_benchmark_table,
    format_text_benchmark_table,
    generate_research_report,
)

app = typer.Typer(help="Naive Bayes from scratch.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command()
def verify_math() -> None:
    """Verify estimator mathematics against hand-calculated cases."""
    results = verify_mathematics()
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        typer.echo(f"  [{status}] {test_name}")
    if all_passed:
        typer.echo("All mathematical verification tests passed.")
    else:
        typer.echo("Some mathematical verification tests failed.")
        raise typer.Exit(code=1)


@app.command()
def train(
    dataset: str = typer.Option(  # noqa: B008
        "iris",
        help="Dataset name: 'iris', 'wine', 'breast_cancer', or 'sms_spam'.",
    ),
    alpha: float = typer.Option(  # noqa: B008
        1.0,
        help="Smoothing alpha parameter for count-based models.",
    ),
) -> None:
    """Train and evaluate custom Naive Bayes on a chosen public dataset."""
    dataset_lower = dataset.lower()
    if dataset_lower == "iris":
        bundle = load_iris_dataset()
    elif dataset_lower == "wine":
        bundle = load_wine_dataset()
    elif dataset_lower == "breast_cancer":
        bundle = load_cancer_dataset()
    elif dataset_lower == "sms_spam":
        texts, labels = load_sms_spam_dataset()
        train_idx, _, test_idx = split_text_indices(texts, labels)
        vec, X_tr_text = build_text_pipeline([texts[i] for i in train_idx])
        X_te_text = vec.transform([texts[i] for i in test_idx])
        y_all = np.asarray(labels)
        y_tr_text = y_all[train_idx]
        y_te_text = y_all[test_idx]
        model = MultinomialNB(alpha=alpha).fit(X_tr_text, y_tr_text)
        acc = model.score(X_te_text, y_te_text)
        typer.echo(f"Trained MultinomialNB on sms_spam: test accuracy = {acc:.4f}")
        return
    else:
        typer.echo(f"Unknown dataset '{dataset}'. Choose iris, wine, breast_cancer, or sms_spam.")
        raise typer.Exit(code=1)

    X_tr_tab, _, X_te_tab, y_tr_tab, _, y_te_tab = train_val_test_split(
        bundle.X, bundle.y, val_size=0.15, test_size=0.15, random_state=42
    )
    gnb = GaussianNB().fit(X_tr_tab, y_tr_tab)
    y_pred = gnb.predict(X_te_tab)
    y_prob = gnb.predict_proba(X_te_tab)
    metrics = compute_classification_metrics(y_te_tab, y_pred, y_prob, classes=gnb.classes_)

    typer.echo(f"Trained GaussianNB on {bundle.name}:")
    typer.echo(f"  Accuracy:         {metrics.accuracy:.4f}")
    typer.echo(f"  Balanced Acc:     {metrics.balanced_accuracy:.4f}")
    typer.echo(f"  Macro-F1:         {metrics.macro_f1:.4f}")
    typer.echo(f"  Calibration ECE:  {metrics.expected_calibration_error:.4f}")


@app.command()
def compare() -> None:
    """Compare custom estimators against scikit-learn references."""
    typer.echo("Running continuous feature comparisons (GaussianNB):")
    cont_results = run_continuous_benchmarks()
    typer.echo(format_continuous_benchmark_table(cont_results))

    typer.echo("\nRunning text feature comparisons (MultinomialNB & BernoulliNB):")
    text_results = run_text_benchmarks()
    typer.echo(format_text_benchmark_table(text_results))


@app.command()
def report(
    output_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Optional path to write markdown report to.",
    ),
) -> None:
    """Generate and display the full research report."""
    math_results = verify_mathematics()
    cont_results = run_continuous_benchmarks()
    text_results = run_text_benchmarks()
    doc = generate_research_report(math_results, cont_results, text_results)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(doc, encoding="utf-8")
        typer.echo(f"Wrote report to {output_file}")
    else:
        typer.echo(doc)


@app.callback()
def _main() -> None:
    """Run Naive Bayes commands. Needed so Typer keeps subcommand mode."""


if __name__ == "__main__":
    app()
