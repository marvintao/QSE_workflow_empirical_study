# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_PERFORMANCE = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq1", "rq1_performance.json")
DEFAULT_SOURCE_ANALYSIS = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq1", "rq1_source_analysis_data.json")
DEFAULT_RELATION_TO_REQUIREMENTS = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq1", "rq1_relation_to_requirements.json")
DEFAULT_REQ_OVERVIEW = os.path.join(
    ROOT, "use_cases", "requirements", "req_overview.json")
DEFAULT_OUT_DIR = os.path.join(ROOT, "empirical_study", "visualization")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq1_performance.tex")
DEFAULT_LINE_CHART = os.path.join(DEFAULT_OUT_DIR, "rq1_requirement_correlation.pdf")

PROBLEM_LABELS = {
    "test case selection": "TCS",
    "test case minimization": "TCM",
}

PROBLEM_ORDER = (
    "test case selection",
    "test case minimization",
)

METRIC_LABELS = {
    "time": "Time (s)",
    "tokens": "Token (M)",
    "compilation": "Cpl.",
    "execution": "Exe.",
    "combination": "Cbi.",
}

METRIC_ORDER = (
    "time",
    "tokens",
    "compilation",
    "execution",
    "combination",
)

METRIC_COLORS = {
    "time": "#355C7D",
    "tokens": "#C06C84",
    "compilation": "#6C9A8B",
    "execution": "#F4A261",
    "combination": "#7D5BA6",
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def requirement_numeric_id(req_id: str) -> int:
    if not req_id.startswith("req_"):
        raise ValueError(f"Unexpected requirement id format: {req_id}")
    return int(req_id.split("_", 1)[1])


def load_benchmark_metadata(
    req_overview_path: str,
) -> tuple[dict[str, int], dict[str, str]]:
    overview = load_json(os.path.abspath(req_overview_path))
    benchmark_id_by_name: dict[str, int] = {}
    benchmark_name_by_req_id: dict[str, str] = {}

    for raw_req_id, item in sorted(overview.items(), key=lambda row: int(row[0])):
        specific_problem = item.get("problem")
        if specific_problem not in PROBLEM_ORDER:
            continue

        benchmark_name = item.get("benchmark")
        if not benchmark_name:
            raise ValueError(f"Missing benchmark for requirement {raw_req_id}")

        if benchmark_name not in benchmark_id_by_name:
            benchmark_id_by_name[benchmark_name] = len(benchmark_id_by_name)
        benchmark_name_by_req_id[f"req_{raw_req_id}"] = benchmark_name

    return benchmark_id_by_name, benchmark_name_by_req_id


def format_time(seconds: float | int | None) -> str:
    return "--" if seconds is None else f"{float(seconds):.1f}"


def format_tokens(tokens: float | int | None) -> str:
    return "--" if tokens is None else f"{float(tokens) / 1_000_000:.2f}"


def format_pass(value: float | int | None) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = safe_mean(values)
    assert mean is not None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def average_ranks(values: list[float]) -> list[float]:
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed_values):
        j = i + 1
        while j < len(indexed_values) and indexed_values[j][1] == indexed_values[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed_values[k][0]] = average_rank
        i = j
    return ranks


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    assert mean_x is not None and mean_y is not None
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denominator_x = sum(x * x for x in centered_x) ** 0.5
    denominator_y = sum(y * y for y in centered_y) ** 0.5
    if denominator_x == 0 or denominator_y == 0:
        # Formal correlation is undefined when either variable is constant.
        # For reporting, use 0.0 to indicate no observable variation.
        return 0.0
    return numerator / (denominator_x * denominator_y)


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "font.family": "Palatino Linotype",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Palatino Linotype",
        "mathtext.it": "Palatino Linotype:italic",
        "mathtext.bf": "Palatino Linotype:bold",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.0,
        "lines.markersize": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def compute_all_performance(source_analysis_path: str) -> dict[str, float | None]:
    rows = load_json(os.path.abspath(source_analysis_path))
    return summarize_rows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        "time": safe_mean([
            row["final_metrics"]["wall-clock_time"]["mean"]
            for row in rows
            if row["final_metrics"]["wall-clock_time"]["mean"] is not None
        ]),
        "tokens": safe_mean([
            row["final_metrics"]["tokens"]["mean"]
            for row in rows
            if row["final_metrics"]["tokens"]["mean"] is not None
        ]),
        "compilation": safe_mean([
            row["final_metrics"]["pass@k_compilation"]["pass@1"]
            for row in rows
            if row["final_metrics"]["pass@k_compilation"]["pass@1"] is not None
        ]),
        "execution": safe_mean([
            row["final_metrics"]["pass@k_execution"]["pass@1"]
            for row in rows
            if row["final_metrics"]["pass@k_execution"]["pass@1"] is not None
        ]),
        "combination": safe_mean([
            row["final_metrics"]["pass@k_combination"]["pass@1"]
            for row in rows
            if row["final_metrics"]["pass@k_combination"]["pass@1"] is not None
        ]),
    }


def compute_benchmark_performance(
    source_analysis_path: str,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
) -> dict[int, dict[str, float | None]]:
    benchmark_id_by_name, benchmark_name_by_req_id = load_benchmark_metadata(req_overview_path)
    rows = load_json(os.path.abspath(source_analysis_path))
    rows_by_req_id = {
        row["req_id"]: row
        for row in rows
        if row.get("req_id") in benchmark_name_by_req_id
    }
    missing_req_ids = sorted(
        set(benchmark_name_by_req_id) - set(rows_by_req_id),
        key=requirement_numeric_id,
    )
    if missing_req_ids:
        raise ValueError(
            "RQ1 source analysis data does not contain all benchmark rows. "
            f"Missing requirement rows: {', '.join(missing_req_ids)}"
        )

    rows_by_benchmark: dict[str, list[dict[str, Any]]] = {
        benchmark_name: []
        for benchmark_name in benchmark_id_by_name
    }
    for req_id, benchmark_name in benchmark_name_by_req_id.items():
        rows_by_benchmark[benchmark_name].append(rows_by_req_id[req_id])

    benchmark_performance: dict[int, dict[str, float | None]] = {}
    for benchmark_name, benchmark_id in benchmark_id_by_name.items():
        benchmark_rows = rows_by_benchmark[benchmark_name]
        observed_problems = {
            row.get("se_specific_problem")
            for row in benchmark_rows
        }
        missing_problems = [
            problem
            for problem in PROBLEM_ORDER
            if problem not in observed_problems
        ]
        if missing_problems:
            raise ValueError(
                "RQ1 source analysis data does not contain both TCS and TCM "
                f"rows for benchmark ID {benchmark_id} ({benchmark_name}). "
                f"Missing problems: {', '.join(missing_problems)}"
            )
        benchmark_performance[benchmark_id] = summarize_rows(benchmark_rows)

    return benchmark_performance


def latex_performance_row(label: str, row: dict[str, float | int | None]) -> str:
    return (
        "    "
        f"{label} & "
        f"{format_time(row.get('time'))} & "
        f"{format_tokens(row.get('tokens'))} & "
        f"{format_pass(row.get('compilation'))} & "
        f"{format_pass(row.get('execution'))} & "
        f"{format_pass(row.get('combination'))} "
        r"\\"
    )


def generate_rq1_table(
    performance_path: str = DEFAULT_PERFORMANCE,
    source_analysis_path: str = DEFAULT_SOURCE_ANALYSIS,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
    out_path: str = DEFAULT_OUT,
) -> str:
    performance = load_json(os.path.abspath(performance_path))
    all_performance = compute_all_performance(source_analysis_path)
    benchmark_performance = compute_benchmark_performance(
        source_analysis_path,
        req_overview_path,
    )

    lines = [
        r"\begin{tabular}{c|cc|ccc}",
        r"    \toprule[1pt]",
        (
            r"    \textbf{Case} & \textbf{Time (s)} & \textbf{Token (M)} "
            r"& \textbf{Cpl.} & \textbf{Exe.} & \textbf{Cbi.} \\"
        ),
        (
            r"    \cmidrule(lr){1-1} \cmidrule(lr){2-2} "
            r"\cmidrule(lr){3-3} \cmidrule(lr){4-4} "
            r"\cmidrule(lr){5-5} \cmidrule(lr){6-6}"
        ),
    ]

    for benchmark_id in sorted(benchmark_performance):
        lines.append(latex_performance_row(f"BID-{benchmark_id}", benchmark_performance[benchmark_id]))

    lines.append(
        (
            r"    \cmidrule(lr){1-1} \cmidrule(lr){2-2} "
            r"\cmidrule(lr){3-3} \cmidrule(lr){4-4} "
            r"\cmidrule(lr){5-5} \cmidrule(lr){6-6}"
        )
    )

    for problem in PROBLEM_ORDER:
        if problem not in performance:
            raise ValueError(f"Missing RQ1 performance row for problem: {problem}")
        row = performance[problem]

        # Each cell is the requirement-level macro mean stored in
        # rq1_performance.json: time and tokens average requirement-level means,
        # while Cpl./Exe./Cbi. are macro-averaged pass@1 values for
        # compilation, execution, and final combination, respectively.
        lines.append(latex_performance_row(PROBLEM_LABELS[problem], row))

    lines.extend([
        (
            r"    \cmidrule(lr){1-1} \cmidrule(lr){2-2} "
            r"\cmidrule(lr){3-3} \cmidrule(lr){4-4} "
            r"\cmidrule(lr){5-5} \cmidrule(lr){6-6}"
        ),
        latex_performance_row("All", all_performance),
        r"    \bottomrule[1pt]",
        r"\end{tabular}",
        "",
    ])

    table = "\n".join(lines)
    write_text(os.path.abspath(out_path), table)
    return table


def generate_rq1_line_chart(
    relation_path: str = DEFAULT_RELATION_TO_REQUIREMENTS,
    out_path: str = DEFAULT_LINE_CHART,
    figsize: tuple[float, float] = (4, 4),
) -> str:
    relation = load_json(os.path.abspath(relation_path))
    configure_matplotlib_style()

    xs = sorted(int(size) for size in relation)
    series: dict[str, dict[str, list[float]]] = {
        metric: {"mean": [], "std": []}
        for metric in METRIC_ORDER
    }

    for size in xs:
        items = relation[str(size)]
        for metric in METRIC_ORDER:
            values = [
                float(item[metric]["mean"])
                for item in items
                if item[metric]["mean"] is not None
            ]
            metric_mean = safe_mean(values)

            # If multiple requirements have the same character length, the
            # plotted point is the average of their metric means. The band uses
            # the standard deviation across those same requirement-level means;
            # for a single requirement, it falls back to that requirement's
            # stored repeat-level std from rq1_relation_to_requirements.json.
            if len(values) > 1:
                metric_std = population_std(values)
            elif values:
                stored_std = items[0][metric]["std"]
                metric_std = float(stored_std) if stored_std is not None else 0.0
            else:
                metric_mean = 0.0
                metric_std = 0.0

            value = float(metric_mean) # type: ignore
            if metric == "tokens":
                value /= 1_000_000
                metric_std = (metric_std or 0.0) / 1_000_000

            series[metric]["mean"].append(value)
            series[metric]["std"].append(float(metric_std or 0.0))

    fig, axes = plt.subplots(
        len(METRIC_ORDER),
        1,
        figsize=figsize,
        sharex=True,
    )

    legend_handles = []
    for ax, metric in zip(axes, METRIC_ORDER):
        means = series[metric]["mean"]
        stds = series[metric]["std"]
        lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        color = METRIC_COLORS[metric]

        line, = ax.plot(
            xs,
            means,
            marker="o",
            markersize=1.5,
            color=color,
            label=METRIC_LABELS[metric],
        )
        ax.fill_between(xs, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.set_ylabel(METRIC_LABELS[metric], fontweight="bold")
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.grid(True, axis="x", color="#EAEAEA", linewidth=0.5, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric in {"compilation", "execution", "combination"}:
            ax.set_ylim(0.25, 1.05)
            ax.set_yticks([0.25, 0.50, 0.75, 1.00])
            ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
        rho = spearman_correlation([float(value) for value in xs], means)
        rho_text = r"$\rho$ = n/a" if rho is None else rf"$\rho$ = {rho:.2f}"
        ax.text(
            0.965,
            0.12,
            rho_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#CFCFCF",
                "alpha": 0.86,
            },
        )
        legend_handles.append(line)

    axes[-1].set_xlabel("# of requirement characters", fontweight="bold")
    axes[-1].set_xscale("log")
    axes[-1].set_xticks([3_000, 10_000, 30_000, 100_000, 300_000])
    axes[-1].set_xticklabels(["3k", "10k", "30k", "100k", "300k"])

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.94,
        bottom=0.09,
        hspace=0.18,
    )
    fig.legend(
        handles=legend_handles,
        labels=[METRIC_LABELS[metric] for metric in METRIC_ORDER],
        loc="upper center",
        ncol=len(METRIC_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=0.85,
        labelspacing=0.25,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(os.path.abspath(out_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(out_path)

if __name__ == "__main__":
    generate_rq1_table()
    generate_rq1_line_chart()
