"""Generate rq4 ablation generation-effectiveness data.

The script reads:
  empirical_study/data/unpacked_raw_data/claude
  empirical_study/data/unpacked_raw_data/claude_ablation/*

It keeps the 10 requirements used by the ablation study and computes only
output-level effectiveness metrics: compilation, execution, and combination.
Unavailable raw values are preserved as raw_*_success fields, while
effectiveness metrics use conservative boolean pass/fail values.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Any


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_RAW_ROOT = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data")
DEFAULT_REFERENCE_DIR = os.path.join(
    ROOT, "empirical_study", "classical_reference", "reference_data")
DEFAULT_OUT_DIR = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq4")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq4_generation_effectivenss.json")
DEFAULT_REQ_OVERVIEW = os.path.join(
    ROOT, "use_cases", "requirements", "req_overview.json")

VARIANT_ORDER = (
    "claude",
    "mega_prompt_record",
    "no_codegen_skill_record",
    "no_knowledge_record",
    "no_reviewer_record",
)
ABLATION_VARIANTS = VARIANT_ORDER[1:]
OUTPUT_METRIC_ORDER = ("compilation", "execution", "combination")
PASS_KS = (1, 2, 3)
REQ_MARKER_RE = re.compile(r"This is user requirement\s+(\d+)\.")
FITNESS_KEY_RE = re.compile(r"(fitness|score|objective)", re.IGNORECASE)
PREFERRED_QUBO_INDEX = {
    "test case selection": 2,
    "test case minimization": 1,
}

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


def is_valid_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def requirement_numeric_id(req_id: str) -> int:
    if not req_id.startswith("req_"):
        raise ValueError(f"Unexpected requirement id format: {req_id}")
    return int(req_id.split("_", 1)[1])


def derive_req_from_raw_text(raw_text: str) -> str | None:
    match = REQ_MARKER_RE.search(raw_text.lstrip("\ufeff"))
    if match:
        return f"req_{match.group(1)}"
    return None


def reference_file_for_req(reference_dir: str, req_id: str) -> str:
    return os.path.join(reference_dir, "testcase", f"{req_id}_reference.json")


def normalize_numeric_values(value: Any) -> list[float | None]:
    values = value if isinstance(value, list) else [value]
    return [
        float(item) if is_valid_number(item) else None
        for item in values
    ]


def load_reference_qubo_values(
    reference_dir: str,
    req_ids: set[str],
) -> dict[str, list[float | None]]:
    values: dict[str, list[float | None]] = {}
    for req_id in req_ids:
        path = reference_file_for_req(reference_dir, req_id)
        if not os.path.exists(path):
            values[req_id] = []
            continue
        data = load_json(path)
        values[req_id] = normalize_numeric_values(data.get("qubo_val"))
    return values


def load_requirement_metadata(
    req_overview_path: str,
) -> dict[str, dict[str, Any]]:
    overview = load_json(req_overview_path)
    metadata: dict[str, dict[str, Any]] = {}

    for raw_req_id, item in overview.items():
        req_id = f"req_{raw_req_id}"
        specific_problem = item.get("problem")
        if specific_problem not in SE_GENERAL_BY_SPECIFIC_PROBLEM:
            continue

        metadata[req_id] = {"se_specific_problem": specific_problem}

    return metadata


def deep_json(value: Any) -> Any:
    while isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{\"":
            return value
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return value
        if parsed == value:
            return parsed
        value = parsed
    return value


def extract_first_number(value: Any) -> float | None:
    value = deep_json(value)
    if is_valid_number(value):
        return float(value)
    if isinstance(value, list):
        for item in value:
            number = extract_first_number(item)
            if number is not None:
                return number
    if isinstance(value, dict):
        for item in value.values():
            number = extract_first_number(item)
            if number is not None:
                return number
    return None


def compact_sequence(value: Any, max_items: int = 50) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "type": "bitstring" if set(value) <= {"0", "1"} else "string",
            "length": len(value),
            "preview": value[:max_items],
            "truncated": len(value) > max_items,
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "preview": value[:max_items],
            "truncated": len(value) > max_items,
        }
    return {
        "type": type(value).__name__,
        "value": value,
    }


def extract_solution_list(artifact: dict[str, Any]) -> tuple[str | None, Any]:
    solution_keys = (
        "selected_test_cases",
        "selected_ids",
        "selected_indices",
        "solution",
        "solutions",
        "best_solution",
        "bitstring",
        "x",
    )
    for key in solution_keys:
        value = artifact.get(key)
        if isinstance(value, (list, str)) and len(value) > 0:
            return key, compact_sequence(value)
    return None, None


def extract_fitness_values(artifact: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in artifact.items():
        if key in {"value", "qubo_value", "quboVal", "qubo_val"} or FITNESS_KEY_RE.search(key):
            number = extract_first_number(value)
            if number is not None:
                values[key] = number
    return values


def extract_valid_solution(artifact: Any) -> dict[str, Any]:
    artifact = deep_json(artifact)
    if not isinstance(artifact, dict):
        return {
            "is_valid": False,
            "qubo_value": None,
            "fitness_values": {},
            "solution_key": None,
            "solution": None,
        }

    fitness_values = extract_fitness_values(artifact)
    qubo_value = fitness_values.get("value")
    if qubo_value is None:
        for key in ("qubo_value", "quboVal", "qubo_val"):
            if key in fitness_values:
                qubo_value = fitness_values[key]
                break

    solution_key, solution = extract_solution_list(artifact)
    return {
        "is_valid": bool(fitness_values) and solution is not None,
        "qubo_value": qubo_value,
        "fitness_values": fitness_values,
        "solution_key": solution_key,
        "solution": solution,
    }


def calculate_qrr(
    reference_qubo: float | None,
    llm_qubo: float | None,
) -> float | None:
    if not is_valid_number(reference_qubo) or float(reference_qubo) == 0.0:
        return None
    if not is_valid_number(llm_qubo):
        return None
    return (float(reference_qubo) - float(llm_qubo)) / abs(float(reference_qubo))


def reference_value_for_repeat(
    reference_values: list[float | None],
    repeat_index: int | None,
) -> float | None:
    if repeat_index is not None:
        index = repeat_index - 1
        if 0 <= index < len(reference_values) and is_valid_number(reference_values[index]):
            return float(reference_values[index])
    for value in reference_values:
        if is_valid_number(value):
            return float(value)
    return None


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


def apply_effectiveness_policy(row: dict[str, Any]) -> None:
    raw_compilation = row.get("raw_compilation_success")
    raw_execution = row.get("raw_execution_success")
    raw_combination = row.get("raw_combination_success")

    compilation_success = raw_compilation is True
    execution_success = compilation_success and raw_execution is True
    combination_success = execution_success and raw_combination is True

    row["compilation_success"] = compilation_success
    row["execution_success"] = execution_success
    row["combination_success"] = combination_success


def build_output_metrics(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    return {
        f"pass@k_{metric}": pass_at_k_metrics([
            bool(run.get(f"{metric}_success"))
            for run in runs
        ])
        for metric in OUTPUT_METRIC_ORDER
    }


def build_repeats(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    repeats: dict[str, dict[str, Any]] = {}
    for index, run in enumerate(runs, start=1):
        repeat_id = str(int(run.get("iter") or index))
        repeats[repeat_id] = {
            "compilation_success": run.get("compilation_success"),
            "execution_success": run.get("execution_success"),
            "combination_success": run.get("combination_success"),
        }
    return repeats


def requirement_level_qrr(
    runs: list[dict[str, Any]],
    reference_values: list[float | None],
    problem: str,
) -> float | None:
    llm_values = [
        (run.get("valid_solution") or {}).get("qubo_value")
        for run in runs
    ]
    preferred_index = PREFERRED_QUBO_INDEX.get(problem)
    if preferred_index is not None:
        if (
            preferred_index < len(reference_values)
            and preferred_index < len(llm_values)
            and is_valid_number(reference_values[preferred_index])
            and is_valid_number(llm_values[preferred_index])
        ):
            return calculate_qrr(
                float(reference_values[preferred_index]),
                float(llm_values[preferred_index]),
            )

    for reference_value, llm_value in zip(reference_values, llm_values):
        if is_valid_number(reference_value) and is_valid_number(llm_value):
            return calculate_qrr(float(reference_value), float(llm_value))
    return None


def valid_solution_summary(
    runs: list[dict[str, Any]],
    reference_values: list[float | None],
    problem: str,
) -> dict[str, Any]:
    qrr = requirement_level_qrr(runs, reference_values, problem)
    return {
        "# of valid solution": 1 if qrr is not None else 0,
        "QRR": qrr,
    }


def load_claude_runs(
    raw_root: str,
    req_ids: set[str],
    reference_qubo_values: dict[str, list[float | None]],
) -> list[dict[str, Any]]:
    rows = load_jsonl(os.path.join(raw_root, "claude", "experiment_runs.jsonl"))
    solution_by_request = load_claude_solution_by_request(raw_root)
    for row in rows:
        row["raw_compilation_success"] = row.get("compilation_success")
        row["raw_execution_success"] = row.get("execution_success")
        row["raw_combination_success"] = row.get("combination_success")
        apply_effectiveness_policy(row)
        row["valid_solution"] = solution_details_for_run(
            row,
            solution_by_request,
            reference_qubo_values,
        )
    return [
        row
        for row in rows
        if row.get("req") in req_ids
    ]


def truthy_success(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"success", "succeeded", "ok", "passed", "cross_checked", "unverified", "mismatch"}:
            return True
        if lowered in {"failed", "failure", "error", "timeout"}:
            return False
    return None


def infer_stage_success(stage: str, artifact: Any) -> bool | None:
    artifact = deep_json(artifact)
    if not isinstance(artifact, dict):
        return None

    flags = artifact.get("stage_flags")
    if isinstance(flags, dict):
        flag_name = {
            "codegen": "codegen_ok",
            "execute": "execute_ok",
            "combine": "combine_ok",
        }.get(stage)
        if flag_name in flags:
            return truthy_success(flags.get(flag_name))

    if stage == "codegen":
        if artifact.get("failure_reason") or artifact.get("error_type") or artifact.get("error_message"):
            return False
        if artifact.get("code"):
            return True
    if stage == "execute":
        status = truthy_success(artifact.get("status"))
        if status is not None:
            return status
        if artifact.get("failure_reason") or artifact.get("stderr_excerpt"):
            return False
        if "result" in artifact:
            return True
    if stage == "combine":
        if artifact.get("failure_reason") or artifact.get("error"):
            return False
        if "value" in artifact or "instance" in artifact:
            return True
    if stage == "mega_prompt":
        flags = artifact.get("stage_flags")
        if isinstance(flags, dict):
            return truthy_success(flags.get("codegen_ok"))

    return None


def combine_artifact_from_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    for stage in trace.get("stages") or []:
        if stage.get("stage") != "combine":
            continue
        artifact = deep_json(stage.get("artifact"))
        if isinstance(artifact, dict):
            return artifact
    return None


def load_claude_solution_by_request(raw_root: str) -> dict[str, dict[str, Any]]:
    path = os.path.join(raw_root, "claude", "record_traces.jsonl")
    if not os.path.exists(path):
        return {}
    solution_by_request: dict[str, dict[str, Any]] = {}
    for trace in load_jsonl(path):
        request = trace.get("request")
        if not isinstance(request, dict):
            continue
        request_id = request.get("request_id")
        artifact = combine_artifact_from_trace(trace)
        if request_id and artifact is not None:
            solution_by_request[request_id] = extract_valid_solution(artifact)
    return solution_by_request


def solution_details_for_run(
    run: dict[str, Any],
    solution_by_request: dict[str, dict[str, Any]],
    reference_qubo_values: dict[str, list[float | None]],
) -> dict[str, Any]:
    details = solution_by_request.get(str(run.get("request_id") or ""), {
        "is_valid": False,
        "qubo_value": None,
        "fitness_values": {},
        "solution_key": None,
        "solution": None,
    })
    req_id = run.get("req")
    repeat_index = int(run["iter"]) if run.get("iter") is not None else None
    reference_qubo = reference_value_for_repeat(
        reference_qubo_values.get(str(req_id), []),
        repeat_index,
    )
    return {
        **details,
        "reference_qubo_value": reference_qubo,
        "QRR": calculate_qrr(reference_qubo, details.get("qubo_value")),
    }


def load_ablation_runs(
    raw_root: str,
    variant: str,
    reference_qubo_values: dict[str, list[float | None]] | None = None,
) -> list[dict[str, Any]]:
    reference_qubo_values = reference_qubo_values or {}
    raw_dir = os.path.join(raw_root, "claude_ablation", variant)
    requests = load_jsonl(os.path.join(raw_dir, "requests.jsonl"))
    stages = load_jsonl(os.path.join(raw_dir, "stage_traces.jsonl"))

    request_rows: dict[str, dict[str, Any]] = {}
    by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        req = derive_req_from_raw_text(str(request.get("raw_text") or ""))
        row = {
            "req": req,
            "request_id": request["request_id"],
            "rid8": request["request_id"][:8],
            "created_at": request.get("created_at") or "",
            "raw_compilation_success": None,
            "raw_execution_success": None,
            "raw_combination_success": None,
            "compilation_success": False,
            "execution_success": False,
            "combination_success": False,
        }
        request_rows[request["request_id"]] = row
        if req:
            by_req[req].append(row)

    stage_success_by_request: dict[str, dict[str, bool | None]] = defaultdict(dict)
    solution_by_request: dict[str, dict[str, Any]] = {}
    for stage in stages:
        request_id = stage.get("request_id")
        stage_name = stage.get("stage")
        if request_id not in request_rows:
            continue
        if stage_name == "mega_prompt":
            artifact = deep_json(stage.get("artifact"))
            if isinstance(artifact, dict):
                solution_by_request[request_id] = extract_valid_solution(artifact)
            flags = artifact.get("stage_flags") if isinstance(artifact, dict) else {}
            if isinstance(flags, dict):
                stage_success_by_request[request_id]["codegen"] = truthy_success(flags.get("codegen_ok"))
                stage_success_by_request[request_id]["execute"] = truthy_success(flags.get("execute_ok"))
                stage_success_by_request[request_id]["combine"] = truthy_success(flags.get("combine_ok"))
            continue
        if stage_name in {"codegen", "execute", "combine"}:
            artifact = deep_json(stage.get("artifact"))
            if stage_name == "combine" and isinstance(artifact, dict):
                solution_by_request[request_id] = extract_valid_solution(artifact)
            success = infer_stage_success(stage_name, artifact)
            if success is not None:
                stage_success_by_request[request_id][stage_name] = success

    for request_id, row in request_rows.items():
        stage_success = stage_success_by_request.get(request_id, {})
        row["raw_compilation_success"] = stage_success.get("codegen")
        row["raw_execution_success"] = stage_success.get("execute")
        row["raw_combination_success"] = stage_success.get("combine")
        apply_effectiveness_policy(row)

    rows: list[dict[str, Any]] = []
    for runs in by_req.values():
        runs.sort(key=lambda row: (row["created_at"], row["request_id"]))
        for index, run in enumerate(runs, start=1):
            run["iter"] = index
            run["valid_solution"] = solution_details_for_run(
                run,
                solution_by_request,
                reference_qubo_values,
            )
            rows.append(run)
    return rows


def group_runs_by_req(
    variant_runs: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant, rows in variant_runs.items():
        by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("req"):
                by_req[row["req"]].append(row)
        for runs in by_req.values():
            runs.sort(key=lambda row: (int(row.get("iter") or 0), row.get("rid8") or ""))
        grouped[variant] = dict(by_req)
    return grouped


def load_variant_runs(
    raw_root: str,
    reference_qubo_values: dict[str, list[float | None]],
) -> dict[str, list[dict[str, Any]]]:
    ablation_req_ids: set[str] = set()
    ablation_runs: dict[str, list[dict[str, Any]]] = {}
    for variant in ABLATION_VARIANTS:
        runs = load_ablation_runs(raw_root, variant, reference_qubo_values)
        ablation_runs[variant] = runs
        ablation_req_ids.update(row["req"] for row in runs if row.get("req"))

    return {
        "claude": load_claude_runs(raw_root, ablation_req_ids, reference_qubo_values),
        **ablation_runs,
    }


def build_effectiveness_metrics(
    raw_root: str,
    reference_dir: str,
    req_overview_path: str,
) -> dict[str, Any]:
    requirement_metadata = load_requirement_metadata(req_overview_path)
    reference_qubo_values = load_reference_qubo_values(
        reference_dir,
        set(requirement_metadata),
    )
    variant_runs = load_variant_runs(raw_root, reference_qubo_values)
    grouped_runs = group_runs_by_req(variant_runs)
    req_sets = [
        set(grouped_runs[variant]) & set(requirement_metadata)
        for variant in VARIANT_ORDER
    ]
    req_ids = sorted(set.intersection(*req_sets), key=requirement_numeric_id)

    rows: list[dict[str, Any]] = []
    for req_id in req_ids:
        variant_blocks: dict[str, dict[str, Any]] = {}
        for variant in VARIANT_ORDER:
            runs = grouped_runs[variant][req_id]
            variant_blocks[variant] = {
                "repeats": build_repeats(runs),
                "output_metrics": build_output_metrics(runs),
                **valid_solution_summary(
                    runs,
                    reference_qubo_values.get(req_id, []),
                    requirement_metadata[req_id]["se_specific_problem"],
                ),
            }

        rows.append({
            "req_id": req_id,
            "variants": variant_blocks,
        })

    return {"requirements": rows}


def generate_rq4_effectiveness_metrics(
    raw_root: str = DEFAULT_RAW_ROOT,
    reference_dir: str = DEFAULT_REFERENCE_DIR,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
    out_path: str = DEFAULT_OUT,
    verbose: bool = True,
) -> dict[str, Any]:
    raw_root = os.path.abspath(raw_root)
    req_overview_path = os.path.abspath(req_overview_path)
    out_path = os.path.abspath(out_path)

    result = build_effectiveness_metrics(
        raw_root,
        reference_dir,
        req_overview_path,
    )
    write_json(out_path, result)

    if verbose:
        print(f"Generated {len(result['requirements'])} rq4 common requirement rows.")
        print(f"Output: {out_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rq4 ablation generation-effectiveness data."
    )
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT, help=f"Default: {DEFAULT_RAW_ROOT}")
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR, help=f"Default: {DEFAULT_REFERENCE_DIR}")
    parser.add_argument("--req-overview", default=DEFAULT_REQ_OVERVIEW, help=f"Default: {DEFAULT_REQ_OVERVIEW}")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Default: {DEFAULT_OUT}")
    args = parser.parse_args()

    generate_rq4_effectiveness_metrics(
        raw_root=args.raw_root,
        reference_dir=args.reference_dir,
        req_overview_path=args.req_overview,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
