"""Generate RQ2 QUBO-value reduction data for the Claude runs.

The script is scoped to test case optimization requirements only:
test case selection and test case minimization.
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
DEFAULT_REFERENCE_DIR = os.path.join(
    ROOT, "empirical_study", "classical_reference", "reference_data")
DEFAULT_RAW_DIR = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data", "claude")
DEFAULT_OUT_DIR = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq2")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq2_qubo_reduction.json")
DEFAULT_REQ_OVERVIEW = os.path.join(
    ROOT, "use_cases", "requirements", "req_overview.json")

SE_PROBLEM_ORDER = (
    "test case selection",
    "test case minimization",
)
TESTCASE_PROBLEMS = set(SE_PROBLEM_ORDER)
REQ_MARKER_RE = re.compile(r"This is user requirement\s+(\d+)\.")

# The GA reference and Claude repeated-run QUBO lists align with the three
# repeated runs. Existing RQ2 reporting compares one representative repeat per
# problem type: run 3 for selection and run 2 for minimization. req_0 has only
# one non-null reference value, so the helper below falls back to the first
# valid aligned pair when the preferred slot is unavailable.
PREFERRED_QUBO_INDEX = {
    "test case selection": 2,
    "test case minimization": 1,
}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def requirement_numeric_id(req_id: str) -> int:
    if not req_id.startswith("req_"):
        raise ValueError(f"Unexpected requirement id format: {req_id}")
    return int(req_id.split("_", 1)[1])


def is_valid_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_requirement_metadata(req_overview_path: str) -> dict[str, dict[str, Any]]:
    overview = load_json(req_overview_path)
    metadata: dict[str, dict[str, Any]] = {}
    for raw_req_id, item in overview.items():
        req_id = f"req_{raw_req_id}"
        problem = item.get("problem")
        if problem not in TESTCASE_PROBLEMS:
            continue
        metadata[req_id] = {
            "problem": problem,
            "benchmark": item.get("benchmark"),
        }
    return metadata


def reference_file_for_req(reference_dir: str, req_id: str) -> str:
    return os.path.join(reference_dir, "testcase", f"{req_id}_reference.json")


def normalize_qubo_values(value: Any) -> list[float | None]:
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    return [
        float(item) if is_valid_number(item) else None
        for item in values
    ]


def load_ga_qubo_values(
    reference_dir: str,
    req_ids: list[str],
) -> dict[str, list[float | None]]:
    values: dict[str, list[float | None]] = {}
    for req_id in req_ids:
        path = reference_file_for_req(reference_dir, req_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing GA reference data for {req_id}: {path}")
        data = load_json(path)
        values[req_id] = normalize_qubo_values(data.get("qubo_val"))
    return values


def req_id_from_trace(trace: dict[str, Any]) -> str | None:
    raw_text = str(trace.get("request", {}).get("raw_text") or "")
    match = REQ_MARKER_RE.search(raw_text.lstrip("\ufeff"))
    if match:
        return f"req_{match.group(1)}"
    return None


def combine_qubo_value(trace: dict[str, Any]) -> float | None:
    for stage in trace.get("stages") or []:
        if stage.get("stage") != "combine":
            continue
        artifact = stage.get("artifact")
        if not isinstance(artifact, dict):
            continue
        value = artifact.get("value")
        if is_valid_number(value):
            return float(value)
    return None


def load_llm_qubo_values(
    raw_dir: str,
    req_ids: list[str],
) -> dict[str, list[float | None]]:
    req_set = set(req_ids)
    traces = load_jsonl(os.path.join(raw_dir, "record_traces.jsonl"))
    trace_by_request_id = {
        trace["request"]["request_id"]: trace
        for trace in traces
        if isinstance(trace.get("request"), dict)
    }

    runs = load_jsonl(os.path.join(raw_dir, "experiment_runs.jsonl"))
    runs_by_req: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        req_id = run.get("req")
        if req_id in req_set:
            runs_by_req[req_id].append(run)

    values_by_req: dict[str, list[float | None]] = {}
    for req_id in req_ids:
        values: list[float | None] = []
        for run in sorted(runs_by_req.get(req_id, []), key=lambda row: int(row["iter"])):
            request_id = run.get("request_id")
            trace = trace_by_request_id.get(request_id) if request_id else None
            values.append(combine_qubo_value(trace) if trace is not None else None)
        values_by_req[req_id] = values

    # Fallback for ad-hoc raw directories without experiment_runs.jsonl rows.
    for trace in load_jsonl(os.path.join(raw_dir, "record_traces.jsonl")):
        req_id = req_id_from_trace(trace)
        if req_id not in req_set or values_by_req.get(req_id):
            continue
        value = combine_qubo_value(trace)
        values_by_req.setdefault(req_id, []).append(value)
    return {req_id: values_by_req.get(req_id, []) for req_id in req_ids}


def aligned_value_pair(
    ga_values: list[float | None],
    llm_values: list[float | None],
    problem: str,
) -> tuple[float | None, float | None]:
    preferred_index = PREFERRED_QUBO_INDEX[problem]
    if (
        preferred_index < len(ga_values)
        and preferred_index < len(llm_values)
        and is_valid_number(ga_values[preferred_index])
        and is_valid_number(llm_values[preferred_index])
    ):
        return float(ga_values[preferred_index]), float(llm_values[preferred_index])

    for index, (ga_value, llm_value) in enumerate(zip(ga_values, llm_values)):
        if is_valid_number(ga_value) and is_valid_number(llm_value):
            return float(ga_value), float(llm_value)
    return None, None


def calculate_reduction_rate(
    ga_qubo: float | None,
    llm_qubo: float | None,
) -> float | None:
    if not is_valid_number(ga_qubo) or float(ga_qubo) == 0.0:
        return None
    if not is_valid_number(llm_qubo):
        return None

    # reductionRate = (Q_GA - Q_LLM) / |Q_GA|.
    # Because QUBO is minimized, a positive value means the LLM result has a
    # lower QUBO objective than the GA reference; a negative value means worse.
    return (float(ga_qubo) - float(llm_qubo)) / abs(float(ga_qubo))


def build_qubo_reduction_data(
    reference_dir: str,
    raw_dir: str,
    req_overview_path: str,
) -> dict[str, Any]:
    metadata = load_requirement_metadata(req_overview_path)
    req_ids = sorted(metadata, key=requirement_numeric_id)
    ga_values = load_ga_qubo_values(reference_dir, req_ids)
    llm_values = load_llm_qubo_values(raw_dir, req_ids)

    result: dict[str, Any] = {
        problem: {"num_valid_reductionRate": 0}
        for problem in SE_PROBLEM_ORDER
    }

    for req_id in req_ids:
        problem = metadata[req_id]["problem"]
        ga_qubo, llm_qubo = aligned_value_pair(
            ga_values[req_id],
            llm_values[req_id],
            problem,
        )
        reduction_rate = calculate_reduction_rate(ga_qubo, llm_qubo)
        if reduction_rate is not None:
            result[problem]["num_valid_reductionRate"] += 1

        result[problem][req_id] = {
            "gaQuboVal": ga_values[req_id],
            "llmQuboVal": llm_values[req_id],
            "reductionRate": reduction_rate,
        }

    return result


def generate_qubo_reduction(
    reference_dir: str = DEFAULT_REFERENCE_DIR,
    raw_dir: str = DEFAULT_RAW_DIR,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
    out_path: str = DEFAULT_OUT,
    verbose: bool = True,
) -> dict[str, Any]:
    result = build_qubo_reduction_data(
        reference_dir=os.path.abspath(reference_dir),
        raw_dir=os.path.abspath(raw_dir),
        req_overview_path=os.path.abspath(req_overview_path),
    )

    write_json(os.path.abspath(out_path), result)

    if verbose:
        print(f"QUBO reduction output: {os.path.abspath(out_path)}")
        print(json.dumps({
            problem: result[problem]["num_valid_reductionRate"]
            for problem in SE_PROBLEM_ORDER
        }, ensure_ascii=False, indent=2))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RQ2 QUBO-value reduction data."
    )
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--req-overview", default=DEFAULT_REQ_OVERVIEW)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    generate_qubo_reduction(
        reference_dir=args.reference_dir,
        raw_dir=args.raw_dir,
        req_overview_path=args.req_overview,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
