"""Command-line interface for Naive Bayes implementation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import typer

from naive_bayes import __version__
from naive_bayes.data import (
    build_text_pipeline,
    download_sms_spam,
    fetch_newsgroup_split,
    load_cancer_dataset,
    load_iris_dataset,
    load_sms_fixture,
    load_wine_dataset,
    save_text_dataset,
    split_text_indices,
    train_val_test_split,
    write_data_manifest,
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
        texts, labels = load_sms_fixture()
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


@app.command("prepare-data")
def prepare_data(
    out: Path = typer.Option(..., "--out", help="Output directory for prepared data."),  # noqa: B008
) -> None:
    """Download and validate benchmark datasets, writing a data manifest."""
    from naive_bayes.data import TextDataset

    out.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    # SMS Spam Collection (UCI 228).
    zip_bytes, sms_meta = download_sms_spam(out)
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.endswith("SMSSpamCollection")]
        tsv = zf.read(names[0])
    from naive_bayes.data import parse_sms_spam_tsv

    sms_texts, sms_labels = parse_sms_spam_tsv(tsv)
    sms_ds = TextDataset("sms_spam", sms_texts, sms_labels, source=sms_meta["source_url"])
    save_text_dataset(sms_ds, out / "sms.pkl")
    entries.append(
        {
            "name": "sms_spam",
            "source_url": sms_meta["source_url"],
            "sha256": sms_meta["sha256"],
            "byte_size": sms_meta["byte_size"],
            "rows": len(sms_texts),
            "class_counts": {"ham": sms_labels.count("ham"), "spam": sms_labels.count("spam")},
            "license": "CC BY 4.0",
        }
    )

    # 20 Newsgroups bydate.
    for subset in ("train", "test"):
        ds = fetch_newsgroup_split(subset)
        save_text_dataset(ds, out / f"news_{subset}.pkl")
        entries.append(
            {
                "name": f"20newsgroups_{subset}",
                "source": ds.source,
                "rows": len(ds.texts),
                "dropped_empty": ds.dropped_empty,
                "class_counts": ds.class_counts(),
            }
        )

    # Tabular datasets are bundled with scikit-learn (no download).
    for name, loader in (
        ("iris", load_iris_dataset),
        ("wine", load_wine_dataset),
        ("breast_cancer", load_cancer_dataset),
    ):
        bundle = loader()
        entries.append(
            {
                "name": name,
                "source": "sklearn.datasets (bundled)",
                "rows": int(len(bundle.y)),
                "n_features": int(bundle.X.shape[1]),
                "class_counts": {str(c): int((bundle.y == c).sum()) for c in np.unique(bundle.y)},
            }
        )

    write_data_manifest(out / "manifest.json", entries)
    typer.echo(f"Prepared data written to {out}")


@app.command()
def benchmark(
    data: Path = typer.Option(..., "--data", help="Directory produced by prepare-data."),  # noqa: B008
    out: Path = typer.Option(..., "--out", help="Output directory for benchmark artifacts."),  # noqa: B008
) -> None:
    """Run the full benchmark: selection, frozen test, metrics, controlled studies."""
    import subprocess

    from naive_bayes.benchmark import run_benchmark

    git_sha = os.environ.get("GIT_SHA", "")
    if not git_sha:
        try:
            git_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            git_sha = "unknown"
    result = run_benchmark(data, out, git_sha=git_sha)
    typer.echo(f"Benchmark complete. git_sha={result['manifest']['git_sha']}")
    typer.echo(f"Artifacts written to {out}")


@app.callback()
def _main() -> None:
    """Run Naive Bayes commands. Needed so Typer keeps subcommand mode."""


if __name__ == "__main__":
    app()
