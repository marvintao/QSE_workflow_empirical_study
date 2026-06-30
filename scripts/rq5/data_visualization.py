from __future__ import annotations

import json
import os
from typing import Any


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_GENERATION_EFFECTIVENESS = os.path.join(
    ROOT,
    "empirical_study",
    "data",
    "processed_data",
    "rq5",
    "rq5_generation_effectiveness.json",
)
DEFAULT_OUT_DIR = os.path.join(ROOT, "empirical_study", "visualization")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq5_cross_model.tex")

MODEL_LABELS = (
    ("claude", "Claude"),
    ("deepseek", "DeepSeek"),
    ("llama", "Llama"),
)

PASS_ORDER = ("pass@1", "pass@2", "pass@3")
METRIC_ORDER = (
    ("pass@k_compilation", "Cpl."),
    ("pass@k_execution", "Exe."),
    ("pass@k_combination", "Cbi."),
)
STAGE_ORDER = (
    ("parse", "Par."),
    ("analyze", "Ana."),
    ("blueprint", "Blue."),
    ("encode", "Enc."),
    ("codegen", "Code."),
    ("execute", "Rev."),
    ("combine", "Cbi."),
    ("verify", "Ver."),
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def format_pass(value: float | int | None) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def latex_model_row(label: str, model_summary: dict[str, Any]) -> str:
    values: list[str] = []
    output_metrics = model_summary["output_metrics"]
    stage_metrics = model_summary["stage_metrics"]
    for pass_key in PASS_ORDER:
        for metric_key, _ in METRIC_ORDER:
            values.append(format_pass(output_metrics[metric_key].get(pass_key)))
    for stage_key, _ in STAGE_ORDER:
        values.append(str(stage_metrics[stage_key]["count"]))
    return f"    {label} & " + " & ".join(values) + r" \\"


def generate_rq5_table(
    data_path: str = DEFAULT_GENERATION_EFFECTIVENESS,
    out_path: str = DEFAULT_OUT,
) -> str:
    # Table form:
    # | LLM |     pass@1     |     pass@2     |     pass@3     |  Agent-level success
    #       -------------------------------------------------------------------------
    # |     | Cpl. Exe. Cbi. | Cpl. Exe. Cbi. | Cpl. Exe. Cbi. | Par. Ana. Blue. Enc. Code. Rev. Cbi. Ver.
    # -------------------------------------------------------------------------------
    # Detail data ...
    data = load_json(os.path.abspath(data_path))
    model_summary = data["metadata"]["model_summary"]

    lines = [
        r"\begin{tabular}{c|ccc|ccc|ccc|cccccccc}",
        r"    \toprule[1pt]",
        (
            r"    \textbf{LLM} & \multicolumn{3}{c|}{\textbf{pass@1}} "
            r"& \multicolumn{3}{c|}{\textbf{pass@2}} "
            r"& \multicolumn{3}{c|}{\textbf{pass@3}} "
            r"& \multicolumn{8}{c}{\textbf{Agent-level success}} \\"
        ),
        (
            r"    \cmidrule(lr){2-4} "
            r"\cmidrule(lr){5-7} \cmidrule(lr){8-10} "
            r"\cmidrule(lr){11-18}"
        ),
        (
            r"    & \textbf{Cpl.} & \textbf{Exe.} & \textbf{Cbi.} "
            r"& \textbf{Cpl.} & \textbf{Exe.} & \textbf{Cbi.} "
            r"& \textbf{Cpl.} & \textbf{Exe.} & \textbf{Cbi.} "
            r"& \textbf{Par.} & \textbf{Ana.} & \textbf{Blue.} & \textbf{Enc.} "
            r"& \textbf{Code.} & \textbf{Rev.} & \textbf{Cbi.} & \textbf{Ver.} \\"
        ),
        (
            r"    \cmidrule(lr){1-1} \cmidrule(lr){2-2} "
            r"\cmidrule(lr){3-3} \cmidrule(lr){4-4} "
            r"\cmidrule(lr){5-5} \cmidrule(lr){6-6} "
            r"\cmidrule(lr){7-7} \cmidrule(lr){8-8} "
            r"\cmidrule(lr){9-9} \cmidrule(lr){10-10} "
            r"\cmidrule(lr){11-11} \cmidrule(lr){12-12} "
            r"\cmidrule(lr){13-13} \cmidrule(lr){14-14} "
            r"\cmidrule(lr){15-15} \cmidrule(lr){16-16} "
            r"\cmidrule(lr){17-17} \cmidrule(lr){18-18}"
        ),
    ]

    for model_key, label in MODEL_LABELS:
        if model_key not in model_summary:
            raise ValueError(f"Missing model summary for: {model_key}")
        lines.append(latex_model_row(label, model_summary[model_key]))

    lines.extend([
        r"    \bottomrule[1pt]",
        r"\end{tabular}",
        "",
    ])

    table = "\n".join(lines)
    write_text(os.path.abspath(out_path), table)
    return table


if __name__ == "__main__":
    generate_rq5_table()
