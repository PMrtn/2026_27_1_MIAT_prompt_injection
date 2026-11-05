"""Static report figures (PDF for LaTeX, PNG for slides) from the committed result tables.

Colour follows the entity in every figure: config 'none' blue, 'guardrails' and the regex classifier orange,
'ml' and DistilBERT aqua, 'combined' yellow, TF-IDF violet (validated categorical palette, light mode).
Values are printed on or next to every mark, because two hues are below 3:1 contrast on the surface.

Usage:
    python -m pidefense.reporting.figures
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position
import numpy as np  # noqa: E402  pylint: disable=wrong-import-position
import pandas as pd  # noqa: E402  pylint: disable=wrong-import-position

from pidefense.config import REPO_ROOT  # noqa: E402  pylint: disable=wrong-import-position
from pidefense.logging_utils import setup_logging  # noqa: E402  pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)

FIG_DIR = REPO_ROOT / "docs" / "report" / "figures"
E2E_DIR = REPO_ROOT / "results" / "e2e"
DET_DIR = REPO_ROOT / "results" / "detector"

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CONFIG_COLORS = {"none": "#2a78d6", "guardrails": "#eb6834", "ml": "#1baf7a", "combined": "#eda100"}
CONFIG_LABELS = {"none": "1 none", "guardrails": "2 guardrails", "ml": "3 ML", "combined": "4 combined"}
MODEL_COLORS = {"regex_guardrail": "#eb6834", "tfidf_lr": "#6250d6", "distilbert": "#1baf7a"}
MODEL_LABELS = {"regex_guardrail": "Regex (R1+R2)", "tfidf_lr": "TF-IDF + LR", "distilbert": "DistilBERT"}
SEQUENTIAL = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CONFIG_ORDER = ["none", "guardrails", "ml", "combined"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    logger.info("Wrote figure %s", name)


def _ci_bounds(text: str) -> tuple[float, float]:
    low, high = text.strip("[]").split(",")
    return float(low), float(high)


def _bar_panel(ax: plt.Axes, summary: pd.DataFrame, column: str, title: str, with_ci: bool) -> None:
    configs = [c for c in CONFIG_ORDER if c in set(summary["config"])]
    values = [float(summary.loc[summary["config"] == c, column].iloc[0]) for c in configs]
    x = np.arange(len(configs))
    ax.bar(x, values, width=0.62, color=[CONFIG_COLORS[c] for c in configs], edgecolor=SURFACE, linewidth=1.5)
    if with_ci and f"{column}_ci" in summary:
        bounds = [_ci_bounds(summary.loc[summary["config"] == c, f"{column}_ci"].iloc[0]) for c in configs]
        lower = [v - b[0] for v, b in zip(values, bounds)]
        upper = [b[1] - v for v, b in zip(values, bounds)]
        ax.errorbar(x, values, yerr=[lower, upper], fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        tops = [b[1] for b in bounds]
    else:
        tops = values
    for xi, value, top in zip(x, values, tops):
        ax.text(
            xi,
            top + 0.02,
            f"{value:.0%}" if value >= 0.01 or value == 0 else f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )
    ax.set_xticks(x, [CONFIG_LABELS[c] for c in configs], rotation=0)
    ax.set_ylim(0, max(1.0, max(tops) + 0.12) if column.startswith("utility") else max(0.05, max(tops) * 1.25 + 0.02))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left")


def fig_e2e_overview(summary: pd.DataFrame) -> None:
    """ASR, benign utility and tool-output FPR per configuration (small multiples, one axis each)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    _bar_panel(axes[0], summary, "asr", "Attack success rate (95% CI)", True)
    _bar_panel(axes[1], summary, "utility_benign", "Benign task utility (95% CI)", True)
    _bar_panel(axes[2], summary, "tool_fpr", "False positive rate (benign tool outputs)", False)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout(w_pad=2.0)
    _save(fig, "e2e_overview")


def fig_asr_by_family(table: pd.DataFrame) -> None:
    """Heatmap: ASR per attack family (rows) and configuration (columns), sequential single hue."""
    configs = [c for c in CONFIG_ORDER if c in table.columns]
    data = table[configs].to_numpy(dtype=float)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    vmax = max(0.05, float(np.nanmax(data)))
    ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.grid(False)
    for (row, col), value in np.ndenumerate(data):
        ax.text(
            col,
            row,
            "–" if np.isnan(value) else f"{value:.0%}",
            ha="center",
            va="center",
            fontsize=8,
            color="#ffffff" if value > 0.55 * vmax else INK,
        )
    ax.set_xticks(range(len(configs)), [CONFIG_LABELS[c] for c in configs])
    labels = [f"{family} (n={int(n)})" for family, n in zip(table.index, table["n"])]
    ax.set_yticks(range(len(table.index)), labels)
    ax.set_title("Attack success rate by attack family", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "asr_by_family")


def fig_tradeoff(sweep: pd.DataFrame, summary: pd.DataFrame, deployed: float) -> None:
    """Estimated ASR vs FPR over the detector threshold, with the four measured configurations."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.grid(True, axis="both")
    for config, label in (("ml", "ML only (threshold sweep)"), ("combined", "Guardrails + ML (threshold sweep)")):
        sub = sweep[sweep["config"] == config].sort_values("tool_fpr_est")
        ax.plot(sub["tool_fpr_est"], sub["asr_est"], color=CONFIG_COLORS[config], linewidth=2, label=label)
        at = sub.iloc[(sub["threshold"] - deployed).abs().argsort()[:1]]
        ax.plot(
            at["tool_fpr_est"],
            at["asr_est"],
            marker="o",
            markersize=8,
            color=CONFIG_COLORS[config],
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            linestyle="none",
        )
    for config in CONFIG_ORDER:
        row = summary[summary["config"] == config]
        if row.empty:
            continue
        x, y = float(row["tool_fpr"].iloc[0]), float(row["asr"].iloc[0])
        ax.plot(
            x,
            y,
            marker="s",
            markersize=8,
            color=CONFIG_COLORS[config],
            markeredgecolor=INK,
            markeredgewidth=0.8,
            linestyle="none",
        )
        ax.annotate(CONFIG_LABELS[config], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_xlabel("False positive rate on benign tool outputs")
    ax.set_ylabel("Attack success rate")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlim(left=-0.01)
    ax.set_ylim(bottom=-0.01)
    ax.set_title("Security / usability trade-off (squares = measured)", loc="left")
    ax.legend(loc="upper right")
    _save(fig, "tradeoff")


def fig_latency(latency: pd.DataFrame) -> None:
    """Per-call latency of each layer vs. tool-output length (log-log), direct-labelled."""
    names = {
        "regex_guardrail": ("Regex (R1+R2)", "#eb6834"),
        "tfidf_lr": ("TF-IDF + LR", "#6250d6"),
        "distilbert_gpu": ("DistilBERT, GPU", "#1baf7a"),
        "distilbert_cpu": ("DistilBERT, CPU", "#2a78d6"),
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.grid(True, axis="both")
    for layer, (label, color) in names.items():
        sub = latency[latency["layer"] == layer].sort_values("tokens")
        if sub.empty:
            continue
        ax.plot(sub["tokens"], sub["p50_ms"], color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(
            f"{label}: {sub['p50_ms'].iloc[-1]:.1f} ms",
            (sub["tokens"].iloc[-1], sub["p50_ms"].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_2,
            va="center",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Tool-output length (model tokens)")
    ax.set_ylabel("Median latency per call (ms)")
    ax.set_title("Defense latency per tool output", loc="left")
    ax.legend(loc="upper left")
    _save(fig, "latency")


def fig_classifier(metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    """Recall per attack family (grouped bars) and FPR on benign slices, for the three classifiers."""
    families = ["S1", "S2", "S3", "S4", "S5", "U1", "U2", "U3"]
    models = [m for m in ("regex_guardrail", "tfidf_lr", "distilbert") if m in metrics]
    fig, (ax_r, ax_f) = plt.subplots(1, 2, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [3, 1.3]})
    width = 0.8 / len(models)
    x = np.arange(len(families))
    for index, model in enumerate(models):
        recalls = [metrics[model][f"family_{f}"]["recall"] for f in families]
        ax_r.bar(
            x + (index - (len(models) - 1) / 2) * width,
            recalls,
            width=width * 0.92,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            edgecolor=SURFACE,
            linewidth=1,
        )
    ax_r.set_xticks(x, [f"{f}\n(unseen)" if f.startswith("U") else f for f in families])
    ax_r.axvline(4.5, color=BASELINE, linewidth=1, linestyle="--")
    ax_r.set_ylim(0, 1.05)
    ax_r.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_r.set_title("Recall by attack family (test split)", loc="left")
    ax_r.legend(loc="lower left", ncol=3)
    slices = [("test_all", "all benign"), ("hard_negatives", "hard negatives")]
    if all("external" in metrics[m] for m in models):
        slices.append(("external", "external"))
    xs = np.arange(len(slices))
    for index, model in enumerate(models):
        values = [metrics[model][key]["fpr"] for key, _ in slices]
        positions = xs + (index - (len(models) - 1) / 2) * width
        ax_f.bar(positions, values, width=width * 0.92, color=MODEL_COLORS[model], edgecolor=SURFACE, linewidth=1)
        for pos, value in zip(positions, values):
            ax_f.text(
                pos, value + 0.005, f"{value:.1%}", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90
            )
    ax_f.set_xticks(xs, [label for _, label in slices])
    ax_f.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_f.set_title("False positive rate", loc="left")
    fig.tight_layout(w_pad=2.0)
    _save(fig, "classifier")


def main() -> None:
    """Render every figure whose input table exists."""
    setup_logging()
    _style()
    import json  # pylint: disable=import-outside-toplevel

    if (DET_DIR / "metrics.json").exists():
        fig_classifier(json.loads((DET_DIR / "metrics.json").read_text(encoding="utf-8")))
    if (DET_DIR / "latency.csv").exists():
        fig_latency(pd.read_csv(DET_DIR / "latency.csv"))
    if (E2E_DIR / "summary.csv").exists():
        summary = pd.read_csv(E2E_DIR / "summary.csv")
        fig_e2e_overview(summary)
        if (E2E_DIR / "asr_by_family.csv").exists():
            fig_asr_by_family(pd.read_csv(E2E_DIR / "asr_by_family.csv", index_col=0))
        sweep_path = E2E_DIR / "threshold_sweep.csv"
        threshold_path = REPO_ROOT / "artifacts" / "detector" / "best" / "threshold.json"
        if sweep_path.exists() and threshold_path.exists():
            deployed = json.loads(threshold_path.read_text())["threshold"]
            fig_tradeoff(pd.read_csv(sweep_path), summary, deployed)


if __name__ == "__main__":
    main()
