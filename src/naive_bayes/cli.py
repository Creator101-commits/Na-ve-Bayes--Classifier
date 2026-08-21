"""Command-line entry point. Commands are added as milestones land."""

import typer

from naive_bayes import __version__

app = typer.Typer(help="Naive Bayes from scratch.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.callback()
def _main() -> None:
    """Run Naive Bayes commands. Needed so Typer keeps subcommand mode."""


if __name__ == "__main__":
    app()
