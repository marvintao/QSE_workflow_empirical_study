# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_EFFECTIVENESS = os.path.join(
    ROOT,
    "empirical_study",
    "data",
    "processed_data",
    "rq4",
    "rq4_generation_effectivenss.json",
)
DEFAULT_OUT_DIR = os.path.join(ROOT, "empirical_study", "visualization")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq4_ablation.pdf")

VARIANT_LABELS = (
    ("claude", "QPipe"),
    ("mega_prompt_record", "single"),
    ("no_codegen_skill_record", "w/o skill"),
    ("no_knowledge_record", "w/o knowledge"),
    ("no_reviewer_record", "w/o review"),
)
METRIC_LABELS = (
    ("pass@k_compilation", "Cpl."),
    ("pass@k_execution", "Exe."),
    ("pass@k_combination", "Cbi."),
)
GENERATION_METRIC_LABELS = (
    ("QRR_positive", r"$QRR$>0"),
)
PASS_KEY = "pass@1"
COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "Palatino Linotype",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
        "mathtext.bf": "Palatino Linotype:bold",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def pass1_values_by_metric_and_variant(data: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = {
        metric_key: {variant: [] for variant, _ in VARIANT_LABELS}
        for metric_key, _ in METRIC_LABELS
    }

    for row in data["requirements"]:
        for variant, _ in VARIANT_LABELS:
            if variant not in row["variants"]:
                raise ValueError(f"Missing RQ4 variant: {variant}")
            output_metrics = row["variants"][variant]["output_metrics"]
            for metric_key, _ in METRIC_LABELS:
                value = output_metrics[metric_key].get(PASS_KEY)
                if value is None:
                    raise ValueError(
                        f"Unexpected null {PASS_KEY} for {variant} {metric_key} {row['req_id']}"
                    )
                values[metric_key][variant].append(float(value))
    return values


def generation_frequencies_by_variant(data: dict[str, Any]) -> dict[str, dict[str, int]]:
    frequencies: dict[str, dict[str, int]] = {
        variant: {metric_key: 0 for metric_key, _ in GENERATION_METRIC_LABELS}
        for variant, _ in VARIANT_LABELS
    }

    for row in data["requirements"]:
        for variant, _ in VARIANT_LABELS:
            if variant not in row["variants"]:
                raise ValueError(f"Missing RQ4 variant: {variant}")
            generation_metrics = row["variants"][variant]

            qrr = generation_metrics.get("QRR")
            if qrr is not None and float(qrr) > 0:
                frequencies[variant]["QRR_positive"] += 1

    return frequencies


def generate_rq4_boxplot(
    data_path: str = DEFAULT_EFFECTIVENESS,
    out_path: str = DEFAULT_OUT,
    figsize: tuple[float, float] = (4.0, 1.85),
) -> str:
    data = load_json(os.path.abspath(data_path))
    pass1_values = pass1_values_by_metric_and_variant(data)
    generation_frequencies = generation_frequencies_by_variant(data)
    configure_matplotlib_style()

    fig, (ax_pass, ax_generation) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.45, 0.55]},
    )
    group_centers = [2.4, 1.7, 1.0]
    bar_gap = 0.13
    bar_height = 0.088
    offsets = [
        (index - (len(VARIANT_LABELS) - 1) / 2.0) * bar_gap
        for index in range(len(VARIANT_LABELS))
    ]

    for variant_index, ((variant, _), color) in enumerate(zip(VARIANT_LABELS, COLORS)):
        means: list[float] = []
        positions: list[float] = []
        for group_center, (metric_key, _) in zip(group_centers, METRIC_LABELS):
            metric_values = pass1_values[metric_key][variant]
            mean_value = sum(metric_values) / len(metric_values)
            means.append(mean_value)
            positions.append(group_center + offsets[variant_index])

        ax_pass.barh(
            positions,
            means,
            height=bar_height,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.82,
        )
        for position, mean_value in zip(positions, means):
            is_qpipe = variant == "claude"
            ax_pass.text(
                min(mean_value + 0.012, 1.065),
                position,
                f"{mean_value:.2f}",
                ha="left",
                va="center_baseline",
                fontsize=6.4,
                color="#B22222" if is_qpipe else "black",
                fontweight="bold" if is_qpipe else "normal",
                clip_on=True,
            )

    ax_pass.set_yticks(group_centers)
    ax_pass.set_yticklabels([label for _, label in METRIC_LABELS], fontweight="bold")
    ax_pass.set_xlabel("")
    ax_pass.set_xlim(-0.035, 1.10)
    ax_pass.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_pass.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax_pass.set_ylim(0.62, 2.78)
    ax_pass.grid(True, axis="x", color="#E0E0E0", linestyle="--", linewidth=0.75)
    ax_pass.grid(False, axis="y")
    for separator in (1.35, 2.05):
        ax_pass.axhline(
            separator,
            color="#BFBFBF",
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            zorder=0,
        )
    ax_pass.set_axisbelow(True)
    ax_pass.tick_params(axis="x", labelsize=7.5, pad=1.5)
    ax_pass.tick_params(axis="y", length=0, labelsize=8.5)

    generation_centers = list(range(len(GENERATION_METRIC_LABELS)))
    generation_bar_width = 0.088
    generation_gap = 0.19
    generation_offsets = [
        (index - (len(VARIANT_LABELS) - 1) / 2.0) * generation_gap
        for index in range(len(VARIANT_LABELS))
    ]
    max_frequency = 0
    for variant_index, ((variant, _), color) in enumerate(zip(VARIANT_LABELS, COLORS)):
        heights = [
            generation_frequencies[variant][metric_key]
            for metric_key, _ in GENERATION_METRIC_LABELS
        ]
        max_frequency = max(max_frequency, *heights)
        xs = [
            center + generation_offsets[variant_index]
            for center in generation_centers
        ]
        ax_generation.bar(
            xs,
            heights,
            width=generation_bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.82,
        )
        for x, height in zip(xs, heights):
            is_qpipe = variant == "claude"
            ax_generation.text(
                x,
                height + 0.18,
                f"{height}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#B22222" if is_qpipe else "black",
                fontweight="bold" if is_qpipe else "normal",
                clip_on=False,
            )

    ax_generation.set_xticks(generation_centers)
    ax_generation.set_xticklabels(
        [label for _, label in GENERATION_METRIC_LABELS],
        fontweight="bold",
    )
    ax_generation.set_ylabel("Frequency", fontweight="bold", fontsize=8.5, labelpad=1.5)
    ax_generation.set_xlim(
        min(generation_centers) - 0.74,
        max(generation_centers) + 0.74,
    )
    ax_generation.set_ylim(0, max_frequency + 1.35)
    ax_generation.set_yticks(range(0, max_frequency + 2, 2))
    ax_generation.grid(True, axis="y", color="#E0E0E0", linestyle="--", linewidth=0.75)
    ax_generation.grid(False, axis="x")
    if len(generation_centers) > 1:
        for separator in (
            (left + right) / 2.0
            for left, right in zip(generation_centers, generation_centers[1:])
        ):
            ax_generation.axvline(
                separator,
                color="#BFBFBF",
                linestyle=(0, (3, 2)),
                linewidth=0.75,
                zorder=0,
            )
    ax_generation.set_axisbelow(True)
    ax_generation.tick_params(axis="x", length=0, labelsize=8.5, pad=2)
    ax_generation.tick_params(axis="y", labelsize=7.5, pad=1.5)

    for ax in (ax_pass, ax_generation):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    handles = [
        Patch(facecolor=color, edgecolor="#4D4D4D", label=label, alpha=0.82)
        for (_, label), color in zip(VARIANT_LABELS, COLORS)
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(VARIANT_LABELS),
        frameon=True,
        fontsize=7.2,
        columnspacing=0.45,
        handlelength=0.95,
        handletextpad=0.25,
    )
    frame = legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.7)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)

    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.22, wspace=0.30)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(os.path.abspath(out_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


if __name__ == "__main__":
    generate_rq4_boxplot()
