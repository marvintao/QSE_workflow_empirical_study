"""Generate rq5 generation-effectiveness data across model runs.

The script reads structured files under:
  empirical_study/data/unpacked_raw_data/{claude,deepseek,llama}

It keeps only requirements that appear in all three model datasets and computes
output-level pass@k values plus stage-level pass counts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_RAW_ROOT = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data")
DEFAULT_OUT_DIR = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq5")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq5_generation_effectiveness.json")
DEFAULT_REQ_OVERVIEW = os.path.join(
    ROOT, "use_cases", "requirements", "req_overview.json")

MODEL_ORDER = ("claude", "deepseek", "llama")
OUTPUT_METRIC_ORDER = ("compilation", "execution", "combination")
STAGE_ORDER = (
    "parse",
    "analyze",
    "blueprint",
    "encode",
    "codegen",
    "execute",
    "combine",
    "verify",
)
PASS_KS = (1, 2, 3)

SE_GENERAL_BY_SPECIFIC_PROBLEM = {
    "test case selection": "test case optimization",
    "test case minimization": "test case optimization",
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def requirement_numeric_id(req_id: str) -> int:
    if not req_id.startswith("req_"):
        raise ValueError(f"Unexpected requirement id format: {req_id}")
    return int(req_id.split("_", 1)[1])


def load_requirement_metadata(
    req_overview_path: str,
) -> set[str]:
    overview = load_json(req_overview_path)
    req_ids: set[str] = set()

    for raw_req_id, item in overview.items():
        req_id = f"req_{raw_req_id}"
        specific_problem = item.get("problem")
        if specific_problem not in SE_GENERAL_BY_SPECIFIC_PROBLEM:
            continue
        req_ids.add(req_id)

    return req_ids


def pass_at_k(successes: list[bool], k: int) -> float | None:
    n = len(successes)
    c = sum(1 for item in successes if item is True)
    if n < k:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def pass_at_k_metrics(successes: list[bool]) -> dict[str, float | None]:
    return {f"pass@{k}": pass_at_k(successes, k) for k in PASS_KS}


def safe_mean(values: list[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def load_model_runs(raw_root: str) -> dict[str, list[dict[str, Any]]]:
    model_runs: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_ORDER:
        path = os.path.join(raw_root, model, "experiment_runs.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing experiment runs for {model}: {path}")
        model_runs[model] = load_jsonl(path)
    return model_runs


def group_runs_by_req(
    model_runs: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model, rows in model_runs.items():
        by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_req[row["req"]].append(row)
        for runs in by_req.values():
            runs.sort(key=lambda row: (int(row["iter"]), row["rid8"]))
        grouped[model] = dict(by_req)
    return grouped


def common_requirements(
    grouped_runs: dict[str, dict[str, list[dict[str, Any]]]],
    allowed_reqs: set[str],
) -> list[str]:
    req_sets = [set(grouped_runs[model]) & allowed_reqs for model in MODEL_ORDER]
    return sorted(set.intersection(*req_sets), key=requirement_numeric_id)


def build_repeats(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    repeats: dict[str, dict[str, Any]] = {}
    for run in runs:
        repeat_id = str(int(run["iter"]))
        repeats[repeat_id] = {
            "stages": run.get("stages") or {},
            "compilation_success": bool(run.get("compilation_success")),
            "execution_success": bool(run.get("execution_success")),
            "combination_success": bool(run.get("combination_success")),
        }
    return repeats


def build_output_metrics(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        f"pass@k_{metric}": pass_at_k_metrics([
            bool(run.get(f"{metric}_success"))
            for run in runs
        ])
        for metric in OUTPUT_METRIC_ORDER
    }


def count_stage_passes(runs: list[dict[str, Any]], stage: str) -> dict[str, int | str]:
    total = len(runs)
    passed = sum(
        1
        for run in runs
        if bool((run.get("stages") or {}).get(stage))
    )
    return {
        "passed": passed,
        "total": total,
        "count": f"{passed}/{total}",
    }


def build_stage_metrics(runs: list[dict[str, Any]]) -> dict[str, dict[str, int | str]]:
    return {
        stage: count_stage_passes(runs, stage)
        for stage in STAGE_ORDER
    }


def build_model_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        summary[model] = {
            "output_metrics": {},
            "stage_metrics": {},
        }
        for metric in OUTPUT_METRIC_ORDER:
            metric_key = f"pass@k_{metric}"
            summary[model]["output_metrics"][metric_key] = {
                f"pass@{k}": safe_mean([
                    row["models"][model]["output_metrics"][metric_key][f"pass@{k}"]
                    for row in rows
                ])
                for k in PASS_KS
            }
        for stage in STAGE_ORDER:
            passed = sum(
                int(row["models"][model]["stage_metrics"][stage]["passed"])
                for row in rows
            )
            total = sum(
                int(row["models"][model]["stage_metrics"][stage]["total"])
                for row in rows
            )
            summary[model]["stage_metrics"][stage] = {
                "passed": passed,
                "total": total,
                "count": f"{passed}/{total}",
            }
    return summary


def build_generation_effectiveness(
    raw_root: str,
    req_overview_path: str,
) -> dict[str, Any]:
    model_runs = load_model_runs(raw_root)
    grouped_runs = group_runs_by_req(model_runs)
    allowed_req_ids = load_requirement_metadata(req_overview_path)
    req_ids = common_requirements(grouped_runs, allowed_req_ids)

    rows: list[dict[str, Any]] = []
    for req_id in req_ids:
        model_blocks: dict[str, dict[str, Any]] = {}
        for model in MODEL_ORDER:
            runs = grouped_runs[model][req_id]
            model_blocks[model] = {
                "repeats": build_repeats(runs),
                "output_metrics": build_output_metrics(runs),
                "stage_metrics": build_stage_metrics(runs),
            }

        rows.append({
            "req_id": req_id,
            "models": model_blocks,
        })

    return {"metadata": {"model_summary": build_model_summary(rows)}}


def generate_rq5_generation_effectiveness(
    raw_root: str = DEFAULT_RAW_ROOT,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
    out_path: str = DEFAULT_OUT,
    verbose: bool = True,
) -> dict[str, Any]:
    raw_root = os.path.abspath(raw_root)
    req_overview_path = os.path.abspath(req_overview_path)
    out_path = os.path.abspath(out_path)

    result = build_generation_effectiveness(
        raw_root,
        req_overview_path,
    )
    write_json(out_path, result)

    if verbose:
        print(f"Output: {out_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rq5 generation-effectiveness data."
    )
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help=f"Default: {DEFAULT_RAW_ROOT}")
    parser.add_argument("--req-overview", default=DEFAULT_REQ_OVERVIEW, help=f"Default: {DEFAULT_REQ_OVERVIEW}")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Default: {DEFAULT_OUT}")
    args = parser.parse_args()

    generate_rq5_generation_effectiveness(
        raw_root=args.raw_root,
        req_overview_path=args.req_overview,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
