# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_REDUCTION = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq2", "rq2_qubo_reduction.json")
DEFAULT_OUT_DIR = os.path.join(ROOT, "empirical_study", "visualization")
DEFAULT_BOXPLOT = os.path.join(DEFAULT_OUT_DIR, "rq2_boxplot.pdf")

PROBLEM_LABELS = {
    "test case selection": "TCS",
    "test case minimization": "TCM",
}

PROBLEM_ORDER = (
    "test case selection",
    "test case minimization",
)

BOX_COLOR = "#B7B5B7F8"
POINT_COLOR = "grey"
MEDIAN_COLOR = "red"
UPPER_WHISKER_COLOR = "#8E44AD"
LOWER_WHISKER_COLOR = "#16A085"
ZERO_LINE_COLOR = "#B22222"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def requirement_numeric_id(req_id: str) -> int:
    if not req_id.startswith("req_"):
        raise ValueError(f"Unexpected requirement id format: {req_id}")
    return int(req_id.split("_", 1)[1])


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


def format_qrr_percent(value: float) -> str:
    return f"{value:.1f}"


def tukey_box_stats(values: list[float]) -> tuple[float, float, float, float, float]:
    if not values:
        raise ValueError("Cannot compute boxplot statistics for an empty series")
    array = np.asarray(values, dtype=float)
    q1 = float(np.percentile(array, 25))
    median = float(np.percentile(array, 50))
    q3 = float(np.percentile(array, 75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    lower_whisker = float(np.min(array[array >= lower_bound]))
    upper_whisker = float(np.max(array[array <= upper_bound]))
    return lower_whisker, median, upper_whisker, lower_bound, upper_bound


def adjusted_top_label_layout(
    stats: list[tuple[str, float, str]],
    x_lower: float,
    x_upper: float,
) -> list[tuple[str, float, str, float, str, float]]:
    if not stats:
        return []
    sorted_stats = sorted(stats, key=lambda item: item[1])
    x_span = max(x_upper - x_lower, 1.0)
    label_y = 1.235
    x_offsets = [0.0] * len(sorted_stats)
    horizontal_alignments = ["center"] * len(sorted_stats)

    if sorted_stats:
        first_x = sorted_stats[0][1]
        if (first_x - x_lower) / x_span < 0.16:
            x_offsets[0] = -8.0
            horizontal_alignments[0] = "center"

    return [
        (name, x_value, color, x_offset, ha, label_y)
        for (name, x_value, color), x_offset, ha in zip(
            sorted_stats,
            x_offsets,
            horizontal_alignments,
        )
    ]


def hardcoded_top_label_layout(
    problem: str,
    stats: list[tuple[str, float, str]],
    x_lower: float,
    x_upper: float,
) -> list[tuple[str, float, str, float, str, float]] | None:
    by_name = {name: (x_value, color) for name, x_value, color in stats}
    label_y = 1.235
    x_span = x_upper - x_lower

    if problem == "test case selection":
        return [
            (
                "lower",
                by_name["lower"][0],
                by_name["lower"][1],
                x_lower + x_span * 0.12,
                "data",
                label_y,
            ),
            (
                "median",
                by_name["median"][0],
                by_name["median"][1],
                by_name["median"][0],
                "data",
                label_y,
            ),
            (
                "upper",
                by_name["upper"][0],
                by_name["upper"][1],
                by_name["upper"][0],
                "data",
                label_y,
            ),
        ]

    if problem == "test case minimization":
        return [
            (
                "lower",
                by_name["lower"][0],
                by_name["lower"][1],
                x_lower + x_span * 0.08,
                "data",
                label_y,
            ),
            (
                "median",
                by_name["median"][0],
                by_name["median"][1],
                x_lower + x_span * 0.25,
                "data",
                label_y,
            ),
            (
                "upper",
                by_name["upper"][0],
                by_name["upper"][1],
                x_lower + x_span * 0.42,
                "data",
                label_y,
            ),
        ]

    return None


def reduction_values_by_problem(data: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    values: dict[str, list[tuple[str, float]]] = {}
    for problem in PROBLEM_ORDER:
        if problem not in data:
            raise ValueError(f"Missing RQ2 reduction data for problem: {problem}")
        rows: list[tuple[str, float]] = []
        for req_id, item in data[problem].items():
            if not req_id.startswith("req_"):
                continue
            reduction_rate = item.get("reductionRate")
            if is_number(reduction_rate):
                rows.append((req_id, float(reduction_rate)))
        values[problem] = sorted(rows, key=lambda row: requirement_numeric_id(row[0]))
    return values


def generate_rq2_boxplot(
    reduction_path: str = DEFAULT_REDUCTION,
    out_path: str = DEFAULT_BOXPLOT,
    figsize: tuple[float, float] = (4.5, 1.35),
) -> str:
    reduction_data = load_json(os.path.abspath(reduction_path))
    values_by_problem = reduction_values_by_problem(reduction_data)
    configure_matplotlib_style()

    percent_values_by_problem = {
        problem: [(req_id, value * 100.0) for req_id, value in values_by_problem[problem]]
        for problem in PROBLEM_ORDER
    }
    all_values = [
        value
        for problem in PROBLEM_ORDER
        for _, value in percent_values_by_problem[problem]
    ]
    if not all_values:
        raise ValueError("No valid reductionRate values found for RQ2 boxplot")

    fig, axes = plt.subplots(1, len(PROBLEM_ORDER), figsize=figsize, sharex=False)
    if len(PROBLEM_ORDER) == 1:
        axes = [axes]

    for ax, problem in zip(axes, PROBLEM_ORDER):
        label = PROBLEM_LABELS[problem]
        rows = percent_values_by_problem[problem]
        points = [value for _, value in rows]
        x_min = min(points + [0.0])
        x_max = max(points + [0.0])
        x_padding = max((x_max - x_min) * 0.16, 8.0)
        x_lower = x_min - x_padding
        x_upper = x_max + x_padding

        boxes = ax.boxplot(
            [points],
            positions=[1],
            vert=False,
            patch_artist=True,
            widths=0.28,
            showfliers=False,
            medianprops={"color": "#2B2B2B", "linewidth": 1.0},
            whiskerprops={"color": "#4D4D4D", "linewidth": 0.9},
            capprops={"color": "#4D4D4D", "linewidth": 0.9},
            boxprops={"edgecolor": "#4D4D4D", "linewidth": 0.9},
        )

        boxes["boxes"][0].set_facecolor(BOX_COLOR)
        boxes["boxes"][0].set_alpha(0.95)

        lower_whisker, median, upper_whisker, lower_bound, upper_bound = tukey_box_stats(points)
        outliers = [
            (req_id, value)
            for req_id, value in rows
            if value < lower_bound or value > upper_bound
        ]
        outlier_req_ids = {req_id for req_id, _ in outliers}
        regular_points = [
            value
            for req_id, value in rows
            if req_id not in outlier_req_ids
        ]
        offsets = [
            (offset - (len(regular_points) - 1) / 2.0) * 0.025
            for offset in range(len(regular_points))
        ]
        ax.scatter(
            regular_points,
            [1 + offset for offset in offsets],
            s=13,
            color=POINT_COLOR,
            alpha=0.48,
            linewidths=0,
            zorder=3,
        )

        ax.scatter(
            median,
            1,
            color=MEDIAN_COLOR,
            s=30,
            zorder=5,
            label="Median" if problem == PROBLEM_ORDER[0] else "",
        )

        ax.scatter(
            upper_whisker,
            1,
            color=UPPER_WHISKER_COLOR,
            marker=">",
            s=25,
            zorder=5,
            label="Upper whisker" if problem == PROBLEM_ORDER[0] else "",
        )

        ax.scatter(
            lower_whisker,
            1,
            color=LOWER_WHISKER_COLOR,
            marker="<",
            s=25,
            zorder=5,
            label="Lower whisker" if problem == PROBLEM_ORDER[0] else "",
        )

        stats = [
            ("lower", lower_whisker, LOWER_WHISKER_COLOR),
            ("median", median, MEDIAN_COLOR),
            ("upper", upper_whisker, UPPER_WHISKER_COLOR),
        ]
        stat_labels = hardcoded_top_label_layout(problem, stats, x_lower, x_upper)
        if stat_labels is None:
            stat_labels = adjusted_top_label_layout(stats, x_lower, x_upper)

        for _, x_value, color, layout_value, mode, label_y in stat_labels:
            if mode == "data":
                ax.annotate(
                    format_qrr_percent(x_value),
                    (layout_value, label_y),
                    textcoords="data",
                    fontsize=9.5,
                    color=color,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    zorder=6,
                )
            else:
                ax.annotate(
                    format_qrr_percent(x_value),
                    (x_value, label_y),
                    xytext=(layout_value, 0),
                    textcoords="offset points",
                    fontsize=9.5,
                    color=color,
                    ha=mode,
                    va="center",
                    fontweight="bold",
                    zorder=6,
                )

        if outliers:
            ax.scatter(
                [value for _, value in outliers],
                [1.0] * len(outliers),
                s=18,
                facecolors="none",
                edgecolors="black",
                linewidths=0.75,
                zorder=6,
            )

        ax.axvline(
            0.0,
            color=ZERO_LINE_COLOR,
            linewidth=1.35,
            linestyle=(0, (4, 2)),
            zorder=1,
        )
        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(0.70, 1.34)
        ax.set_xlabel(label, fontweight="bold")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.grid(True, axis="x", color="#E0E0E0", linestyle="--", linewidth=0.8)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MEDIAN_COLOR,
            markeredgecolor=MEDIAN_COLOR,
            markersize=6,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            marker=">",
            color="none",
            markerfacecolor=UPPER_WHISKER_COLOR,
            markeredgecolor=UPPER_WHISKER_COLOR,
            markersize=6,
            label="Upper whisker",
        ),
        Line2D(
            [0],
            [0],
            marker="<",
            color="none",
            markerfacecolor=LOWER_WHISKER_COLOR,
            markeredgecolor=LOWER_WHISKER_COLOR,
            markersize=6,
            label="Lower whisker",
        ),
    ]
    plot_area_center = (0.07 + 0.98) / 2.0
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        fontsize=10,
        handlelength=1.0,
        columnspacing=0.85,
        bbox_to_anchor=(plot_area_center, 1.02),
        alignment="center",
    )
    frame = legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.8)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.22, wspace=0.16)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(os.path.abspath(out_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)


if __name__ == "__main__":
    generate_rq2_boxplot()
