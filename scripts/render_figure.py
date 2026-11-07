"""Reproduce image.png from actual code outputs; no invented benchmark values.

Requires the project dependencies and an environment with Matplotlib installed.
Run from the repository root with PYTHONPATH=src python scripts/render_figure.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from naive_bayes.experiments import run_continuous_benchmarks
from naive_bayes.multinomial import MultinomialNB

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)
COLORS = ["#4169e1", "#10a68b", "#d95f45", "#8659cf", "#64748b"]
ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")


results = run_continuous_benchmarks()
names = list(results)
x = np.arange(len(names))
axes[0].bar(
    x - 0.18,
    [results[n].custom_accuracy for n in names],
    0.36,
    label="Custom GaussianNB",
    color=COLORS[0],
)
axes[0].bar(
    x + 0.18, [results[n].ref_accuracy for n in names], 0.36, label="scikit-learn", color=COLORS[1]
)
axes[0].set(
    xticks=x,
    xticklabels=["Iris", "Wine", "Breast cancer"],
    ylim=(0, 1.12),
    title="Held-out classification",
    ylabel="Accuracy",
)
axes[0].legend(fontsize=9)
diffs = [results[n].max_prob_diff for n in names]
axes[1].bar(x, diffs, color=COLORS[1])
axes[1].set(
    xticks=x,
    xticklabels=["Iris", "Wine", "Breast cancer"],
    title="Probability agreement",
    ylabel="Maximum absolute probability difference",
)
axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
for i, value in enumerate(diffs):
    axes[1].annotate(
        f"{value:.2e}",
        (i, value),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
axes[1].margins(y=0.3)
alpha = np.logspace(-3, 1, 80)
likelihood = []
for a in alpha:
    model = MultinomialNB(alpha=float(a)).fit(
        np.array([[3.0, 3.0, 0.0], [0.0, 0.0, 6.0]]), np.array([0, 1])
    )
    likelihood.append(float(np.exp(model.feature_log_prob_[0, 2])))
axes[2].plot(alpha, likelihood, color=COLORS[2], lw=2)
axes[2].set(
    xscale="log",
    xlabel="Smoothing alpha",
    ylabel="P(unseen feature | class 0)",
    title="Smoothing: exact model output",
)
fig.suptitle("Naive Bayes | measured reference parity and analytical smoothing", fontsize=16)
fig.supxlabel(
    "Continuous datasets: 70/15/15 stratified split, seed 42. Smoothing panel: two-"
    "row count fixture.",
    fontsize=10,
)
payload = {
    "source": "scikit-learn packaged datasets; seed 42; held-out test partitions",
    "accuracy": {n: results[n].custom_accuracy for n in names},
    "max_probability_difference": dict(zip(names, diffs, strict=True)),
    "alpha": alpha.tolist(),
    "unseen_feature_probability": likelihood,
}

source_hash = hashlib.sha256()
for source_path in sorted((ROOT / "src").rglob("*.py")):
    source_hash.update(str(source_path.relative_to(ROOT)).encode())
    source_hash.update(source_path.read_bytes())
payload["source_sha256"] = source_hash.hexdigest()

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)
fig.savefig(ROOT / "image.png", dpi=180, metadata={"Software": "Matplotlib"})
plt.close(fig)
(ROOT / "scripts" / "figure-data.json").write_text(
    json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
)
