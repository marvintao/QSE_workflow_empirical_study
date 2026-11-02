# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

"""Generate RQ3 workflow-pattern data for Claude successful-combination runs.

The script reads structured files under:
  empirical_study/data/unpacked_raw_data/claude

It keeps only Claude runs whose final combination succeeded and writes:
  empirical_study/data/processed_data/rq3/rq3_pattern.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_RAW_DIR = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data", "claude")
DEFAULT_OUT_DIR = os.path.join(
    ROOT, "empirical_study", "data", "processed_data", "rq3")
DEFAULT_OUT = os.path.join(DEFAULT_OUT_DIR, "rq3_pattern.json")

PATTERN_FIELDS = (
    "quantum_algorithm",
    "quantum_formulation",
    "number_of_decompositions",
    "number_of_workflow_nodes",
    "combination_mechanism",
    "max_qubits_per_circuit",
)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stage_artifact(stage: dict[str, Any]) -> dict[str, Any]:
    artifact = stage.get("artifact")
    if isinstance(artifact, dict):
        return artifact

    llm_output = stage.get("llm_output")
    if isinstance(llm_output, dict):
        artifact = llm_output.get("artifact")
        if isinstance(artifact, dict):
            return artifact

    return {}


def load_successful_combination_runs(raw_dir: str) -> list[dict[str, Any]]:
    runs = load_jsonl(os.path.join(raw_dir, "experiment_runs.jsonl"))
    selected = [
        run for run in runs
        if run.get("combination_success") is True and run.get("request_id")
    ]
    selected.sort(key=lambda row: (int(row.get("req_no", 0)), int(row.get("iter", 0))))
    return selected


def group_stage_traces_by_request(raw_dir: str) -> dict[str, list[dict[str, Any]]]:
    stages = load_jsonl(os.path.join(raw_dir, "stage_traces.jsonl"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stage in stages:
        request_id = stage.get("request_id")
        if request_id:
            grouped[request_id].append(stage)
    return dict(grouped)


def first_stage(stages: list[dict[str, Any]], stage_name: str) -> dict[str, Any] | None:
    for stage in stages:
        if stage.get("stage") == stage_name:
            return stage
    return None


def stages_named(stages: list[dict[str, Any]], stage_name: str) -> list[dict[str, Any]]:
    return [stage for stage in stages if stage.get("stage") == stage_name]


def extract_quantum_algorithm(stages: list[dict[str, Any]]) -> str:
    analyze = first_stage(stages, "analyze")
    artifact = stage_artifact(analyze) if analyze is not None else {}
    quantum_kernel = artifact.get("quantum_kernel")
    if isinstance(quantum_kernel, dict):
        algorithm = quantum_kernel.get("algorithm")
        if algorithm:
            return str(algorithm)
    return "unknown"


def text_mentions(pattern: str, value: Any) -> bool:
    return re.search(pattern, json.dumps(value, ensure_ascii=False), re.IGNORECASE) is not None


def execute_instance_types(stages: list[dict[str, Any]]) -> list[str]:
    types: list[str] = []
    for stage in stages_named(stages, "execute"):
        artifact = stage_artifact(stage)
        candidates = [artifact]
        result = artifact.get("result")
        if isinstance(result, dict):
            candidates.append(result)
        for candidate in candidates:
            instance = candidate.get("instance") if isinstance(candidate, dict) else None
            if isinstance(instance, dict) and instance.get("type"):
                types.append(str(instance["type"]).lower())
    return types


def extract_quantum_formulation(stages: list[dict[str, Any]]) -> str:
    instance_types = execute_instance_types(stages)
    if "qubo" in instance_types:
        return "QUBO"

    analyze = first_stage(stages, "analyze")
    artifact = stage_artifact(analyze) if analyze is not None else {}
    if text_mentions(r"\bQUBO\b", artifact):
        return "QUBO"
    if text_mentions(r"\bIsing\b", artifact):
        return "Ising"
    return "unknown"


def extract_number_of_decompositions(stages: list[dict[str, Any]]) -> str:
    analyze = first_stage(stages, "analyze")
    artifact = stage_artifact(analyze) if analyze is not None else {}
    decomposition = artifact.get("decomposition")
    if isinstance(decomposition, list):
        return str(len(decomposition))
    return "unknown"


def extract_number_of_workflow_nodes(stages: list[dict[str, Any]]) -> str:
    blueprint = first_stage(stages, "blueprint")
    artifact = stage_artifact(blueprint) if blueprint is not None else {}
    nodes = artifact.get("nodes")
    if isinstance(nodes, list):
        return str(len(nodes))
    return "unknown"


def extract_combination_mechanism(stages: list[dict[str, Any]]) -> str:
    combine = first_stage(stages, "combine")
    if combine is None:
        return "unknown"

    artifact = stage_artifact(combine)
    llm_output = combine.get("llm_output")
    via = None
    if isinstance(llm_output, dict):
        via = llm_output.get("via")
    if via is None:
        via = combine.get("via") or artifact.get("via")

    if str(via).lower() == "python_aggregator":
        return "script"
    return "agent"


def numeric_qubit_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def collect_qubits_from_value(value: Any) -> list[int]:
    qubits: list[int] = []
    if isinstance(value, dict):
        for key in ("num_qubits", "num_variables"):
            qubit_value = numeric_qubit_value(value.get(key))
            if qubit_value is not None:
                qubits.append(qubit_value)
        for child in value.values():
            qubits.extend(collect_qubits_from_value(child))
    elif isinstance(value, list):
        for child in value:
            qubits.extend(collect_qubits_from_value(child))
    return qubits


def extract_max_qubits_per_circuit(stages: list[dict[str, Any]]) -> str:
    qubit_values: list[int] = []

    for stage in stages_named(stages, "execute"):
        qubit_values.extend(collect_qubits_from_value(stage_artifact(stage)))

    if not qubit_values:
        encode = first_stage(stages, "encode")
        if encode is not None:
            qubit_values.extend(collect_qubits_from_value(stage_artifact(encode)))

    if qubit_values:
        return str(max(qubit_values))
    return "unknown"


def frequency_table(values: list[str]) -> dict[str, dict[str, Any]]:
    total = len(values)
    counts = Counter(values)
    return {
        str(key): {
            "freq": counts[key],
            "proportion": f"{counts[key]}/{total}",
        }
        for key in sorted(counts, key=lambda item: (item == "unknown", str(item)))
    }


def extract_pattern_record(stages: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "quantum_algorithm": extract_quantum_algorithm(stages),
        "quantum_formulation": extract_quantum_formulation(stages),
        "number_of_decompositions": extract_number_of_decompositions(stages),
        "number_of_workflow_nodes": extract_number_of_workflow_nodes(stages),
        "combination_mechanism": extract_combination_mechanism(stages),
        "max_qubits_per_circuit": extract_max_qubits_per_circuit(stages),
    }


def generate_pattern_data(
    raw_dir: str = DEFAULT_RAW_DIR,
    out_path: str = DEFAULT_OUT,
) -> dict[str, Any]:
    """Generate RQ3 pattern frequency data for Claude combination-passed runs."""
    runs = load_successful_combination_runs(raw_dir)
    stage_traces_by_request = group_stage_traces_by_request(raw_dir)

    records: list[dict[str, Any]] = []
    for run in runs:
        request_id = run["request_id"]
        stages = stage_traces_by_request.get(request_id, [])
        pattern = extract_pattern_record(stages)
        records.append({
            "req": run.get("req"),
            "iter": run.get("iter"),
            "rid8": run.get("rid8"),
            "request_id": request_id,
            **pattern,
        })

    distributions = {
        field: frequency_table([str(record[field]) for record in records])
        for field in PATTERN_FIELDS
    }

    data = {"distributions": distributions}
    write_json(out_path, data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RQ3 pattern data for Claude successful-combination runs.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    data = generate_pattern_data(raw_dir=args.raw_dir, out_path=args.out)
    print(f"Wrote {os.path.abspath(args.out)}.")


if __name__ == "__main__":
    main()
