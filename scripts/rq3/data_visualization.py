# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_PATTERN_DATA = os.path.join(
    ROOT,
    "empirical_study",
    "data",
    "processed_data",
    "rq3",
    "rq3_pattern.json",
)
DEFAULT_OUT_DIR = os.path.join(ROOT, "empirical_study", "visualization")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq3_pattern.pdf")

PIE_CHARTS = (
    ("quantum_algorithm", "Quantum\nalgorithm"),
    ("quantum_formulation", "Quantum\nformulation"),
    ("combination_mechanism", "Combination\nmechanism"),
    ("number_of_decompositions", "# of decomposition"),
    ("number_of_workflow_nodes", "# of workflow nodes"),
    ("max_qubits_per_circuit", "Maximum qubits"),
)
COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
)
ZERO_LEGEND_LABELS = {
    "quantum_algorithm": ("others",),
    "quantum_formulation": ("others",),
    "combination_mechanism": ("agent",),
}


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


def distribution_values(data: dict[str, Any], field_key: str) -> tuple[list[str], list[int]]:
    distributions = data.get("distributions")
    if not isinstance(distributions, dict) or field_key not in distributions:
        raise ValueError(f"Missing RQ3 pattern distribution: {field_key}")

    field_distribution = distributions[field_key]
    if not isinstance(field_distribution, dict):
        raise ValueError(f"Unexpected distribution format for: {field_key}")

    labels: list[str] = []
    values: list[int] = []
    for label, item in field_distribution.items():
        if not isinstance(item, dict) or "freq" not in item:
            raise ValueError(f"Unexpected distribution item for {field_key}: {label}")
        labels.append(str(label))
        values.append(int(item["freq"]))

    return labels, values


def format_autopct(percent: float) -> str:
    if percent < 0.5:
        return ""
    return f"{percent:.1f}%"


def center_single_autotext(values: list[int], autotexts: list[plt.Text]) -> None:
    if sum(1 for value in values if value > 0) != 1:
        return
    for autotext in autotexts:
        if autotext.get_text():
            autotext.set_position((0.0, 0.0))
            autotext.set_ha("center")
            autotext.set_va("center")


def offset_autotexts(
    autotexts: list[plt.Text],
    offsets: dict[str, tuple[float, float]],
) -> None:
    for autotext in autotexts:
        text = autotext.get_text()
        if text not in offsets:
            continue
        x, y = autotext.get_position()
        dx, dy = offsets[text]
        autotext.set_position((x + dx, y + dy))


def draw_pie_chart(
    ax: plt.Axes,
    title: str,
    labels: list[str],
    values: list[int],
    zero_legend_labels: tuple[str, ...] = (),
    radius: float = 1.12,
    legend_x: float = -0.02,
    legend_y: float = 0.5,
    title_pad: float = 0.6,
    title_y: float | None = None,
    pctdistance: float = 0.58,
    autotext_offsets: dict[str, tuple[float, float]] | None = None,
    wedge_edgecolor: str = "black",
) -> None:
    positive_items = [
        (label, value)
        for label, value in zip(labels, values)
        if value > 0
    ]
    positive_labels = [label for label, _ in positive_items]
    positive_values = [value for _, value in positive_items]
    pie_colors = COLORS[:len(positive_values)]

    wedges, _, autotexts = ax.pie(
        positive_values,
        colors=pie_colors,
        startangle=90,
        counterclock=False,
        autopct=format_autopct,
        pctdistance=pctdistance,
        radius=radius,
        wedgeprops={
            "edgecolor": wedge_edgecolor,
            "linewidth": 0.55,
        },
        textprops={
            "fontsize": 7.0,
            "fontweight": "bold",
            "color": "white",
        },
    )
    for autotext in autotexts:
        autotext.set_fontsize(7.0)
        autotext.set_fontweight("bold")
    center_single_autotext(positive_values, list(autotexts))
    if autotext_offsets:
        offset_autotexts(list(autotexts), autotext_offsets)

    ax.set_title(title, fontweight="bold", fontsize=9.0, pad=title_pad)
    legend_handles = list(wedges)
    legend_labels = list(positive_labels)
    for offset, label in enumerate(zero_legend_labels):
        if label in legend_labels:
            continue
        legend_handles.append(plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS[(len(positive_values) + offset) % len(COLORS)],
            edgecolor=wedge_edgecolor,
            linewidth=0.55,
        ))
        legend_labels.append(label)
    ax.legend(
        legend_handles,
        legend_labels,
        loc="center right",
        bbox_to_anchor=(legend_x, legend_y),
        ncol=1,
        frameon=False,
        fontsize=6.5,
        handlelength=0.75,
        handletextpad=0.18,
        columnspacing=0.38,
        borderaxespad=0.0,
    )
    if title_y is not None:
        ax.title.set_y(title_y)
    ax.set_aspect("equal")


def generate_rq3_pattern_pie_charts(
    data_path: str = DEFAULT_PATTERN_DATA,
    out_path: str = DEFAULT_OUT,
    figsize: tuple[float, float] = (4.8, 2.05),
) -> str:
    data = load_json(os.path.abspath(data_path))
    configure_matplotlib_style()

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:2]),
        fig.add_subplot(grid[1, 2:4]),
        fig.add_subplot(grid[1, 4:6]),
    ]

    for index, (ax, (field_key, title)) in enumerate(zip(axes, PIE_CHARTS)):
        labels, values = distribution_values(data, field_key)
        draw_pie_chart(
            ax,
            title,
            labels,
            values,
            ZERO_LEGEND_LABELS.get(field_key, ()),
            radius=0.70 if index < 3 else 1.16,
            legend_x=0.16 if index < 3 else -0.07,
            legend_y=0.5,
            title_pad=-7.0 if index < 3 else 8.0,
            title_y=-0.05 if index < 3 else 1.14,
            pctdistance=0.58 if field_key != "number_of_workflow_nodes" else 0.64,
            autotext_offsets={"15.5%": (-0.11, 0.05), "6.9%": (0.13, -0.06)}
            if field_key == "number_of_workflow_nodes"
            else None,
            wedge_edgecolor="black",
        )

    fig.subplots_adjust(
        left=0.015,
        right=0.995,
        top=0.95,
        bottom=0.055,
        wspace=-0.234,
        hspace=0.24,
    )
    for ax, dx in zip(axes[:3], (0.045, 0.0, -0.045)):
        position = ax.get_position()
        ax.set_position([
            position.x0 + dx,
            position.y0,
            position.width,
            position.height,
        ])
    for ax, dx in zip(axes[3:], (0.040, 0.0, -0.040)):
        position = ax.get_position()
        ax.set_position([
            position.x0 + dx,
            position.y0,
            position.width,
            position.height,
        ])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(os.path.abspath(out_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


if __name__ == "__main__":
    generate_rq3_pattern_pie_charts()
