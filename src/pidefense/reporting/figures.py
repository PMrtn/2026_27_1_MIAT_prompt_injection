"""Static report figures (PNG for the Markdown report and the slides, PDF as vector copy) from the result tables.

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
DET_DIR = REPO_ROOT / "results" / "detector"
LAYERS_DIR = REPO_ROOT / "results" / "layers"

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CONFIG_COLORS = {"none": "#2a78d6", "guardrails": "#eb6834", "ml": "#1baf7a", "combined": "#eda100"}
CONFIG_LABELS = {"none": "1 none", "guardrails": "2 guardrails", "ml": "3 ML", "combined": "4 combined"}
MODEL_COLORS = {
    "regex_guardrail": "#eb6834",
    "tfidf_lr": "#6250d6",
    "distilbert_base": "#2a78d6",
    "distilbert": "#1baf7a",
}
MODEL_LABELS = {
    "regex_guardrail": "Regex (R1+R2)",
    "tfidf_lr": "TF-IDF + LR",
    "distilbert_base": "DistilBERT base",
    "distilbert": "DistilBERT focus",
}
LAYER_COLORS = {"r2": "#eb6834", "ml_base": "#2a78d6", "ml_focus": "#1baf7a"}
LAYER_LABELS = {"r2": "Regex R2", "ml_base": "DistilBERT base", "ml_focus": "DistilBERT focus"}
CONFIG_ORDER = ["none", "guardrails", "ml", "combined"]
DEFENDED = ["guardrails", "ml", "combined"]
WITH_ACTION_POLICY = frozenset({"guardrails", "combined"})
FAMILY_GROUPS = {"S": "Known families\n(S1-S5)", "U": "Unseen families\n(U1-U3)", "P": "Ops-native\n(P1-P8)"}
EVAL_SETS = [
    ("test_seen", "Known families\n(test split)"),
    ("test_unseen", "Unseen families\n(test split)"),
    ("ood_ops_all", "OpsPilot domain set\n(never trained on)"),
    ("focus_all", "Delegated documents\n(focus test split)"),
    ("external", "deepset public set\n(chat prompts)"),
]


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


def _label_bars(ax: plt.Axes, positions: np.ndarray, values: list[float]) -> None:
    for pos, value in zip(positions, values):
        ax.text(pos, value + 0.015, f"{value:.0%}", ha="center", va="bottom", fontsize=7, color=INK)


def _grouped_bars(
    ax: plt.Axes, groups: list[str], series: dict[str, list[float]], colors: dict[str, str], labels: dict[str, str]
) -> None:
    """Vertical grouped bars, one colour per series, value printed above every bar."""
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series.items()):
        positions = x + (index - (len(series) - 1) / 2) * width
        ax.bar(
            positions,
            values,
            width=width * 0.92,
            color=colors[name],
            label=labels[name],
            edgecolor=SURFACE,
            linewidth=1,
        )
        _label_bars(ax, positions, values)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def fig_generalization(metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    """Recall of each classifier on increasingly unfamiliar evaluation sets (thresholds fixed on validation)."""
    models = [m for m in ("regex_guardrail", "tfidf_lr", "distilbert_base", "distilbert") if m in metrics]
    sets = [(key, label) for key, label in EVAL_SETS if all(key in metrics[m] for m in models)]
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    _grouped_bars(
        ax,
        [label for _, label in sets],
        {m: [metrics[m][key]["recall"] for key, _ in sets] for m in models},
        MODEL_COLORS,
        MODEL_LABELS,
    )
    ax.set_title("Share of injected samples detected, from familiar to new attacks", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(models))
    _save(fig, "generalization")


def fig_layers(attacks: pd.DataFrame) -> None:
    """Offline coverage of every attack: carrier withheld (by family group) and attack stopped (by domain)."""
    configs = [c for c in DEFENDED if f"blocked_{c}" in attacks]
    l0 = attacks[attacks["level"] == "L0"].copy()
    l0["group"] = l0["family"].str[0]
    groups = [g for g in FAMILY_GROUPS if g in set(l0["group"])]
    fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [3, 2]})
    _grouped_bars(
        ax_w,
        [f"{FAMILY_GROUPS[g]}\nn={int((l0['group'] == g).sum())}" for g in groups],
        {c: [float(l0.loc[l0["group"] == g, f"blocked_{c}"].astype(bool).mean()) for g in groups] for c in configs},
        CONFIG_COLORS,
        CONFIG_LABELS,
    )
    ax_w.set_title("Injected tool output withheld before the LLM sees it", loc="left")
    domains = [("office", "Office"), ("ops", "OpsPilot")]
    stopped: dict[str, list[float]] = {}
    for config in configs:
        values = []
        for domain, _ in domains:
            sub = l0[l0["domain"] == domain]
            blocked = sub[f"blocked_{config}"].astype(bool)
            if config in WITH_ACTION_POLICY:
                blocked = blocked | sub["r3_blocks_action"].astype(bool)
            values.append(float(blocked.mean()))
        stopped[config] = values
    _grouped_bars(
        ax_s,
        [f"{label}\nn={int((l0['domain'] == domain).sum())}" for domain, label in domains],
        stopped,
        CONFIG_COLORS,
        CONFIG_LABELS,
    )
    ax_s.set_title("Attack stopped (withheld or action vetoed)", loc="left")
    fig.tight_layout(w_pad=2.5)
    handles, labels = ax_w.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(configs))
    _save(fig, "layers")


def fig_runbook(table: pd.DataFrame) -> None:
    """Withheld share of OpsPilot outputs per layer: attacks at L1 (recall) and benign outputs (false positives)."""
    rows = [
        ("OpsPilot undefended successes", "L1", "The attacks\nthat worked"),
        ("OpsPilot runbook attacks (P03/P04)", "L1", "All runbook\nattacks"),
        ("OpsPilot other attacks", "L1", "Attacks in tickets,\nlogs, DB, web"),
        ("benign OpsPilot runbooks (FPR)", "-", "Benign runbooks\n(false alarm)"),
        ("benign OpsPilot outputs (FPR)", "-", "All benign outputs\n(false alarm)"),
    ]
    picked = []
    for slice_name, level, label in rows:
        sub = table[(table["slice"] == slice_name) & (table["level"] == level)]
        if not sub.empty:
            picked.append((sub.iloc[0], label))
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    _grouped_bars(
        ax,
        [f"{label}\nn={int(row['n'])}" for row, label in picked],
        {layer: [float(row[layer]) for row, _ in picked] for layer in LAYER_COLORS},
        LAYER_COLORS,
        LAYER_LABELS,
    )
    ax.axvline(2.5, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_title("OpsPilot tool outputs withheld by each output layer (attacks: after the follow-up)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    _save(fig, "runbook")


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


def _box(ax: plt.Axes, xy: tuple[float, float], size: tuple[float, float], title: str, body: str, color: str) -> None:
    patch = matplotlib.patches.FancyBboxPatch(
        xy, *size, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=SURFACE, edgecolor=color, linewidth=2
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + size[0] / 2,
        xy[1] + size[1] - 0.09,
        title,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        xy[0] + size[0] / 2,
        xy[1] + size[1] - 0.26,
        body,
        ha="center",
        va="top",
        fontsize=7.6,
        color=INK_2,
        linespacing=1.35,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], label: str) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.4})
    ax.text(
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2 + 0.05,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        color=INK_2,
    )


def fig_architecture() -> None:
    """System architecture: agent loop, defense middleware (layers per configuration), MCP servers."""
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    _box(
        ax,
        (0.1, 1.1),
        (2.3, 1.6),
        "Agent loop",
        "gpt-5.6-luna (Azure OpenAI)\nfunction calling, max 8-10 turns\nsame system prompt in\nall configurations",
        CONFIG_COLORS["none"],
    )
    _box(
        ax,
        (3.2, 0.25),
        (3.7, 3.2),
        "Defense middleware",
        "read outputs:\nR1 sanitizer  -> R2 regex signatures  -> ML detector\n(first block withholds the output)\n\n"
        "actions (and SQL):\nR3 action policy / ops permissions\n(veto = action not executed)\n\n"
        "1 none: -   2 guardrails: R1 R2 R3\n3 ML: detector   4 combined: all",
        CONFIG_COLORS["guardrails"],
    )
    _box(
        ax,
        (7.7, 1.9),
        (2.7, 1.5),
        "MCP: office workspace",
        "e-mail, files, web, reviews,\nbank transactions\n12 tasks, goals G1-G5",
        CONFIG_COLORS["ml"],
    )
    _box(
        ax,
        (7.7, 0.25),
        (2.7, 1.5),
        "MCP: OpsPilot workspace",
        "tickets, runbooks, SQLite DB,\nlogs, alerts, configs, notebook\n8 tasks, goals O1-O8",
        CONFIG_COLORS["combined"],
    )
    _arrow(ax, (2.45, 2.15), (3.15, 2.15), "tool call")
    _arrow(ax, (3.15, 1.55), (2.45, 1.55), "filtered result")
    _arrow(ax, (6.95, 2.6), (7.65, 2.6), "MCP")
    _arrow(ax, (6.95, 1.0), (7.65, 1.0), "MCP")
    ax.text(
        0.1,
        0.55,
        "attacker controls tool-output\ncontent only (indirect injection)",
        fontsize=7.5,
        color=INK_2,
        style="italic",
    )
    _save(fig, "architecture")


def main() -> None:
    """Render every figure whose input table exists."""
    setup_logging()
    _style()
    import json  # pylint: disable=import-outside-toplevel

    fig_architecture()
    if (DET_DIR / "metrics.json").exists():
        metrics = json.loads((DET_DIR / "metrics.json").read_text(encoding="utf-8"))
        fig_classifier(metrics)
        if (DET_DIR / "metrics_base.json").exists():
            base = json.loads((DET_DIR / "metrics_base.json").read_text(encoding="utf-8"))
            if "distilbert" in base:
                metrics["distilbert_base"] = base["distilbert"]
        fig_generalization(metrics)
    if (DET_DIR / "latency.csv").exists():
        fig_latency(pd.read_csv(DET_DIR / "latency.csv"))
    if (LAYERS_DIR / "attack_outputs.csv").exists():
        fig_layers(pd.read_csv(LAYERS_DIR / "attack_outputs.csv"))
    if (LAYERS_DIR / "model_comparison.csv").exists():
        fig_runbook(pd.read_csv(LAYERS_DIR / "model_comparison.csv"))


if __name__ == "__main__":
    main()
