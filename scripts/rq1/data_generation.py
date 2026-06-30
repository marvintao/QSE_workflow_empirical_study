"""Generate RQ1 analysis data from Claude unpacked workflow traces.

The script only reads structured files under:
  empirical_study/data/unpacked_raw_data/claude

It groups available repeated runs by requirement id and writes:
  empirical_study/data/processed_data/rq1/rq1_source_analysis_data.json
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
DEFAULT_RAW_DIR = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data", "claude")
DEFAULT_OUT_DIR = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq1")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq1_source_analysis_data.json")
DEFAULT_PERFORMANCE = os.path.join(DEFAULT_OUT_DIR, "rq1_performance.json")
DEFAULT_RELATION_TO_REQUIREMENTS = os.path.join(
    DEFAULT_OUT_DIR, "rq1_relation_to_requirements.json")
DEFAULT_REQ_OVERVIEW = os.path.join(
    ROOT, "use_cases", "requirements", "req_overview.json")
DEFAULT_REQ_TXT_DIR = os.path.join(
    ROOT, "use_cases", "requirements", "req_txt")

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

SE_SPECIFIC_PROBLEM_ORDER = (
    "test case selection",
    "test case minimization",
)
TESTCASE_PROBLEMS = set(SE_SPECIFIC_PROBLEM_ORDER)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def requirement_numeric_id(req_id: str) -> int:
    if not req_id.startswith("req_"):
        raise ValueError(f"Unexpected requirement id format: {req_id}")
    return int(req_id.split("_", 1)[1])


def requirement_char_count(req_txt_dir: str, req_id: str) -> int:
    path = os.path.join(req_txt_dir, f"{req_id}.txt")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return len(f.read())


def load_requirement_metadata(
    req_overview_path: str,
    req_txt_dir: str,
) -> dict[str, dict[str, Any]]:
    overview = load_json(req_overview_path)
    metadata: dict[str, dict[str, Any]] = {}

    for raw_req_id, item in overview.items():
        req_id = f"req_{raw_req_id}"
        specific_problem = item.get("problem")
        if specific_problem not in TESTCASE_PROBLEMS:
            continue

        metadata[req_id] = {
            "se_specific_problem": specific_problem,
            "req_size": {
                "char_count": requirement_char_count(req_txt_dir, req_id),
            },
        }

    return metadata


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = safe_mean(values)
    assert mean is not None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def metric_summary(values: list[float | None]) -> dict[str, float | None]:
    values = [value for value in values if value is not None]
    return {
        "mean": safe_mean(values),
        "std": population_std(values),
    }


def token_total(token_usage: dict[str, Any] | None) -> int:
    if not isinstance(token_usage, dict):
        return 0
    return sum(int(token_usage.get(field) or 0) for field in TOKEN_FIELDS)


def request_token_total(stages: list[dict[str, Any]]) -> int:
    return sum(token_total(stage.get("token_usage")) for stage in stages)


def request_wall_clock_seconds(stages: list[dict[str, Any]]) -> float:
    return sum((stage.get("duration_ms") or 0) for stage in stages) / 1000.0


def stage_artifacts(stages: list[dict[str, Any]], stage_name: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for stage in stages:
        if stage.get("stage") != stage_name:
            continue
        artifact = stage.get("artifact")
        if isinstance(artifact, dict):
            artifacts.append(artifact)
    return artifacts


def codegen_rounds(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for stage in stages:
        if stage.get("stage") == "codegen":
            rounds.extend(stage.get("rounds") or [])
    return rounds


def compilation_success(stages: list[dict[str, Any]]) -> tuple[bool, str]:
    compile_results: list[dict[str, Any]] = []
    for round_item in codegen_rounds(stages):
        for tool_result in round_item.get("tool_results") or []:
            if not isinstance(tool_result, dict):
                continue
            if tool_result.get("tool_name") == "mcp__codegen__compile_check":
                compile_results.append(tool_result)

    if compile_results:
        return any(
            (not item.get("is_error")) and str(item.get("output", "")).strip().lower() == "ok"
            for item in compile_results
        ), "compile_check_tool"

    codegen_artifacts = stage_artifacts(stages, "codegen")
    return any(bool(artifact.get("code")) for artifact in codegen_artifacts), "codegen_code_fallback"


def execution_success(stages: list[dict[str, Any]]) -> bool:
    execute_artifacts = stage_artifacts(stages, "execute")
    return bool(execute_artifacts) and all(
        artifact.get("status") == "success" for artifact in execute_artifacts
    )


def combination_success(stages: list[dict[str, Any]]) -> bool:
    combine_artifacts = stage_artifacts(stages, "combine")
    return any(
        "value" in artifact and "subresults_summary" in artifact
        for artifact in combine_artifacts
    )


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


def pass_at_k_metrics(successes: list[bool], ks: tuple[int, ...] = (1,)) -> dict[str, float | None]:
    return {f"pass@{k}": pass_at_k(successes, k) for k in ks}


def build_analysis(
    raw_dir: str,
    req_overview_path: str,
    req_txt_dir: str,
) -> list[dict[str, Any]]:
    validate_claude_raw_dir(raw_dir)
    run_rows = load_jsonl(os.path.join(raw_dir, "experiment_runs.jsonl"))
    trace_rows = load_jsonl(os.path.join(raw_dir, "record_traces.jsonl"))
    requirement_metadata = load_requirement_metadata(req_overview_path, req_txt_dir)
    run_rows = [
        row
        for row in run_rows
        if row.get("req") in requirement_metadata
    ]

    run_request_ids = {row["request_id"] for row in run_rows if row.get("request_id")}
    trace_by_request = {
        row["request"]["request_id"]: row
        for row in trace_rows
        if row.get("request", {}).get("request_id") in run_request_ids
    }

    by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run in run_rows:
        request_id = run.get("request_id")
        trace = trace_by_request.get(request_id) if request_id else None
        if trace is None:
            by_req[run["req"]].append({
                "repeat_id": int(run["iter"]),
                "request_id": request_id,
                "rid8": run["rid8"],
                "wall-clock_time": None,
                "tokens": None,
                "compilation_success": run.get("compilation_success"),
                "execution_success": run.get("execution_success"),
                "combination_success": run.get("combination_success"),
            })
            continue

        stages = trace.get("stages") or []
        comp_success, _ = compilation_success(stages)

        by_req[run["req"]].append({
            "repeat_id": int(run["iter"]),
            "request_id": request_id,
            "rid8": run["rid8"],
            "wall-clock_time": request_wall_clock_seconds(stages),
            "tokens": request_token_total(stages),
            "compilation_success": comp_success,
            "execution_success": execution_success(stages),
            "combination_success": combination_success(stages),
        })

    analysis_rows: list[dict[str, Any]] = []

    for req in sorted(by_req, key=requirement_numeric_id):
        repeats = sorted(by_req[req], key=lambda item: item["repeat_id"])
        repeats_obj: dict[str, dict[str, Any]] = {}
        for repeat in repeats:
            repeats_obj[str(repeat["repeat_id"])] = {
                "compilation_success": repeat["compilation_success"],
                "execution_success": repeat["execution_success"],
                "combination_success": repeat["combination_success"],
            }

        wall_clock_values = [
            float(item["wall-clock_time"]) if item["wall-clock_time"] is not None else None
            for item in repeats
        ]
        token_values = [
            float(item["tokens"]) if item["tokens"] is not None else None
            for item in repeats
        ]
        compilation_values = [bool(item["compilation_success"]) for item in repeats]
        execution_values = [bool(item["execution_success"]) for item in repeats]
        combination_values = [bool(item["combination_success"]) for item in repeats]

        final_metrics = {
            "wall-clock_time": metric_summary(wall_clock_values),
            "tokens": metric_summary(token_values),
            "pass@k_compilation": pass_at_k_metrics(compilation_values),
            "pass@k_execution": pass_at_k_metrics(execution_values),
            "pass@k_combination": pass_at_k_metrics(combination_values),
        }

        if req not in requirement_metadata:
            raise ValueError(f"Missing requirement metadata for {req}")

        analysis_rows.append({
            "req_id": req,
            **requirement_metadata[req],
            "repeats": repeats_obj,
            "final_metrics": final_metrics,
        })

    return analysis_rows


def validate_claude_raw_dir(raw_dir: str) -> None:
    if os.path.basename(os.path.normpath(raw_dir)).lower() != "claude":
        raise ValueError(
            "RQ1 is scoped to the Claude dataset; raw_dir must be "
            "empirical_study/data/unpacked_raw_data/claude."
        )


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_rq1_analysis_data(
    raw_dir: str = DEFAULT_RAW_DIR,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
    req_txt_dir: str = DEFAULT_REQ_TXT_DIR,
    out_path: str = DEFAULT_OUT,
    performance_path: str = DEFAULT_PERFORMANCE,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    raw_dir = os.path.abspath(raw_dir)
    req_overview_path = os.path.abspath(req_overview_path)
    req_txt_dir = os.path.abspath(req_txt_dir)
    out_path = os.path.abspath(out_path)
    performance_path = os.path.abspath(performance_path)

    analysis_rows = build_analysis(
        raw_dir,
        req_overview_path,
        req_txt_dir,
    )
    write_json(out_path, analysis_rows)
    generate_rq1_performance_cell(
        analysis_path=out_path,
        out_path=performance_path,
    )

    if verbose:
        print(f"Generated {len(analysis_rows)} RQ1 requirement rows.")
        print(f"Output: {out_path}")
        print(f"Performance: {performance_path}")

    return analysis_rows


def generate_rq1_performance_cell(
    analysis_path: str = DEFAULT_OUT,
    out_path: str = DEFAULT_PERFORMANCE,
) -> dict[str, dict[str, float | None]]:
    analysis_rows = load_json(os.path.abspath(analysis_path))

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in analysis_rows:
        by_problem[row["se_specific_problem"]].append(row)

    performance: dict[str, dict[str, float | None]] = {}
    for se_problem in SE_SPECIFIC_PROBLEM_ORDER:
        rows = by_problem.get(se_problem, [])

        # Problem-level cells use requirement-level macro means:
        # 1. each requirement first contributes its own final_metrics mean;
        # 2. this function averages those requirement-level values within one
        #    se_specific_problem, so requirements with more available traces do
        #    not receive extra weight;
        # 3. time is in seconds, tokens are raw token counts, and success cells
        #    are pass@1 values for compilation/execution/combination.
        time_values = [
            row["final_metrics"]["wall-clock_time"]["mean"]
            for row in rows
            if row["final_metrics"]["wall-clock_time"]["mean"] is not None
        ]
        token_values = [
            row["final_metrics"]["tokens"]["mean"]
            for row in rows
            if row["final_metrics"]["tokens"]["mean"] is not None
        ]
        compilation_values = [
            row["final_metrics"]["pass@k_compilation"]["pass@1"]
            for row in rows
            if row["final_metrics"]["pass@k_compilation"]["pass@1"] is not None
        ]
        execution_values = [
            row["final_metrics"]["pass@k_execution"]["pass@1"]
            for row in rows
            if row["final_metrics"]["pass@k_execution"]["pass@1"] is not None
        ]
        combination_values = [
            row["final_metrics"]["pass@k_combination"]["pass@1"]
            for row in rows
            if row["final_metrics"]["pass@k_combination"]["pass@1"] is not None
        ]

        performance[se_problem] = {
            "time": safe_mean(time_values),
            "tokens": safe_mean(token_values),
            "compilation": safe_mean(compilation_values),
            "execution": safe_mean(execution_values),
            "combination": safe_mean(combination_values),
        }

    write_json(os.path.abspath(out_path), performance)
    return performance


def boolean_metric_summary(values: list[bool]) -> dict[str, float | None]:
    numeric_values = [1.0 if value else 0.0 for value in values]
    return metric_summary(numeric_values)


def generate_rq1_relation_to_requirement_size(
    analysis_path: str = DEFAULT_OUT,
    out_path: str = DEFAULT_RELATION_TO_REQUIREMENTS,
) -> dict[str, list[dict[str, Any]]]:
    analysis_rows = load_json(os.path.abspath(analysis_path))
    relation: dict[str, list[dict[str, Any]]] = {}

    for row in sorted(analysis_rows, key=lambda item: item["req_size"]["char_count"]):
        char_length = int(row["req_size"]["char_count"])
        repeats = row["repeats"].values()

        # This file relates requirement character length to per-requirement
        # performance. time/tokens reuse the mean/std already computed over
        # the requirement's repeated runs. For success metrics, only pass@1 is
        # retained as mean, and std is computed over the three repeated boolean
        # outcomes for that same requirement.
        compilation_values = [
            bool(repeat["compilation_success"])
            for repeat in repeats
        ]
        execution_values = [
            bool(repeat["execution_success"])
            for repeat in repeats
        ]
        combination_values = [
            bool(repeat["combination_success"])
            for repeat in repeats
        ]

        relation.setdefault(str(char_length), []).append({
            "req_id": row["req_id"],
            "time": row["final_metrics"]["wall-clock_time"],
            "tokens": row["final_metrics"]["tokens"],
            "compilation": {
                "mean": row["final_metrics"]["pass@k_compilation"]["pass@1"],
                "std": boolean_metric_summary(compilation_values)["std"],
            },
            "execution": {
                "mean": row["final_metrics"]["pass@k_execution"]["pass@1"],
                "std": boolean_metric_summary(execution_values)["std"],
            },
            "combination": {
                "mean": row["final_metrics"]["pass@k_combination"]["pass@1"],
                "std": boolean_metric_summary(combination_values)["std"],
            },
        })

    write_json(os.path.abspath(out_path), relation)
    return relation


def generate_rq1_relateion_to_requirement_size(
    analysis_path: str = DEFAULT_OUT,
    out_path: str = DEFAULT_RELATION_TO_REQUIREMENTS,
) -> dict[str, list[dict[str, Any]]]:
    return generate_rq1_relation_to_requirement_size(
        analysis_path=analysis_path,
        out_path=out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RQ1 processed analysis data.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help=f"Default: {DEFAULT_RAW_DIR}")
    parser.add_argument("--req-overview", default=DEFAULT_REQ_OVERVIEW, help=f"Default: {DEFAULT_REQ_OVERVIEW}")
    parser.add_argument("--req-txt-dir", default=DEFAULT_REQ_TXT_DIR, help=f"Default: {DEFAULT_REQ_TXT_DIR}")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Default: {DEFAULT_OUT}")
    parser.add_argument("--performance", default=DEFAULT_PERFORMANCE, help=f"Default: {DEFAULT_PERFORMANCE}")
    parser.add_argument(
        "--relation-to-requirements",
        default=DEFAULT_RELATION_TO_REQUIREMENTS,
        help=f"Default: {DEFAULT_RELATION_TO_REQUIREMENTS}",
    )
    args = parser.parse_args()

    generate_rq1_analysis_data(
        raw_dir=args.raw_dir,
        req_overview_path=args.req_overview,
        req_txt_dir=args.req_txt_dir,
        out_path=args.out,
        performance_path=args.performance,
    )
    relation_to_requirements_path = os.path.abspath(args.relation_to_requirements)
    generate_rq1_relation_to_requirement_size(
        analysis_path=os.path.abspath(args.out),
        out_path=relation_to_requirements_path,
    )
    print(f"Relation to requirement size: {relation_to_requirements_path}")


if __name__ == "__main__":
    main()
