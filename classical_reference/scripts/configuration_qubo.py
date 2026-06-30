"""QUBO helpers for configuration-reference comparison."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawQuboExpression:
    req_id: str
    request_id: str
    iteration: int
    stage: str
    instance: dict[str, Any]
    observed_value: float | None
    matrix_convention: str


def load_raw_qubo_expressions(repo_root: str) -> dict[str, RawQuboExpression]:
    raw_root = os.path.join(repo_root, "empirical_study", "data", "unpacked_raw_data")
    mapping_path = os.path.join(raw_root, "request_mapping.jsonl")
    traces_path = os.path.join(raw_root, "record_traces.jsonl")
    if not os.path.exists(mapping_path) or not os.path.exists(traces_path):
        return {}

    request_to_run = _load_configuration_request_mapping(mapping_path)
    best_by_req: dict[str, RawQuboExpression] = {}

    with open(traces_path, "r", encoding="utf-8") as handle:
        for line in handle:
            trace = json.loads(line)
            request = trace.get("request")
            if not isinstance(request, dict):
                continue
            request_id = request.get("request_id")
            run = request_to_run.get(request_id)
            if run is None:
                continue
            for stage in trace.get("stages", []):
                expression = _expression_from_stage(stage, run)
                if expression is None:
                    continue
                current = best_by_req.get(expression.req_id)
                if current is None or _expression_rank(expression) < _expression_rank(current):
                    best_by_req[expression.req_id] = expression

    return best_by_req


def evaluate_qubo_for_assignment(
    expression: RawQuboExpression,
    assignment: dict[str, int],
) -> dict[str, Any]:
    instance = expression.instance
    matrix = instance.get("matrix")
    if not isinstance(matrix, list):
        raise ValueError("QUBO instance does not contain a matrix")

    variable_order = _variable_order(instance, len(matrix))
    x = [_assignment_value(assignment, name) for name in variable_order]
    constant = _constant_offset(instance)

    upper_value = _evaluate_upper_triangular(matrix, x) + constant
    full_value = _evaluate_full_matrix(matrix, x) + constant
    selected_value = upper_value
    if expression.matrix_convention == "full":
        selected_value = full_value

    return {
        "upper_triangular_value": upper_value,
        "full_matrix_value": full_value,
        "constant": constant,
        "matrix_convention": expression.matrix_convention,
        "variable_order": variable_order,
        "x": x,
        "source_request_id": expression.request_id,
        "source_iter": expression.iteration,
        "source_stage": expression.stage,
        "source_observed_value": expression.observed_value,
    }


def _load_configuration_request_mapping(path: str) -> dict[str, dict[str, Any]]:
    request_to_run = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            req_no = row.get("req_no")
            if isinstance(req_no, int) and 20 <= req_no <= 79:
                request_to_run[row["request_id"]] = row
    return request_to_run


def _expression_from_stage(
    stage: dict[str, Any],
    run: dict[str, Any],
) -> RawQuboExpression | None:
    stage_name = stage.get("stage")
    if stage_name not in {"execute", "combine"}:
        return None
    artifact = stage.get("artifact")
    if not isinstance(artifact, dict):
        return None

    instance = None
    observed_value = _as_float(artifact.get("value"))
    if isinstance(artifact.get("instance"), dict):
        instance = artifact["instance"]
    elif isinstance(artifact.get("result"), dict):
        result = artifact["result"]
        observed_value = _as_float(result.get("value"))
        if isinstance(result.get("instance"), dict):
            instance = result["instance"]

    if not isinstance(instance, dict):
        return None
    if instance.get("type") != "qubo" or not isinstance(instance.get("matrix"), list):
        return None

    convention = _infer_matrix_convention(instance, artifact, observed_value)
    return RawQuboExpression(
        req_id=str(run["req"]),
        request_id=str(run["request_id"]),
        iteration=int(run["iter"]),
        stage=str(stage_name),
        instance=instance,
        observed_value=observed_value,
        matrix_convention=convention,
    )


def _expression_rank(expression: RawQuboExpression) -> tuple[int, int, int]:
    complete_vars = 0 if _has_variable_order(expression.instance) else 1
    stage_rank = 0 if expression.stage == "execute" else 1
    return (complete_vars, stage_rank, expression.iteration)


def _infer_matrix_convention(
    instance: dict[str, Any],
    artifact: dict[str, Any],
    observed_value: float | None,
) -> str:
    explicit = instance.get("matrix_convention")
    if isinstance(explicit, str):
        lowered = explicit.lower()
        if "upper" in lowered:
            return "upper"
        if "full" in lowered or "symmetric" in lowered:
            return "full"

    assignment = _extract_assignment(artifact)
    if assignment and observed_value is not None:
        try:
            variable_order = _variable_order(instance, len(instance["matrix"]))
            x = [_assignment_value(assignment, name) for name in variable_order]
            constant = _constant_offset(instance)
            upper = _evaluate_upper_triangular(instance["matrix"], x) + constant
            full = _evaluate_full_matrix(instance["matrix"], x) + constant
            if abs(full - observed_value) + 1e-9 < abs(upper - observed_value):
                return "full"
            if abs(upper - observed_value) + 1e-9 < abs(full - observed_value):
                return "upper"
        except (KeyError, TypeError, ValueError):
            pass

    if _is_symmetric_matrix(instance.get("matrix")):
        return "full"
    return "upper"


def _extract_assignment(artifact: dict[str, Any]) -> dict[str, int] | None:
    result = artifact.get("result")
    if not isinstance(result, dict):
        return None
    for key in ("assignment", "selection", "best_assignment"):
        value = result.get(key)
        if isinstance(value, dict):
            return {str(k): int(v) for k, v in value.items()}
    for key in ("top_configurations", "top_configs", "top_k_configurations"):
        value = result.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                assignment = first.get("assignment") or first.get("config")
                if isinstance(assignment, dict):
                    return {str(k): int(v) for k, v in assignment.items()}
                x = first.get("x")
                if isinstance(x, list):
                    variables = _variable_order(result.get("instance", {}), len(x))
                    return dict(zip(variables, [int(v) for v in x]))
    return None


def _variable_order(instance: dict[str, Any], n: int) -> list[str]:
    for key in (
        "variable_order",
        "variables",
        "feature_names",
        "variable_labels",
        "names",
    ):
        value = instance.get(key)
        if isinstance(value, list) and len(value) == n:
            return [_normalize_variable_name(str(item)) for item in value]

    features = instance.get("features")
    if isinstance(features, list) and len(features) == n:
        names = []
        for feature in features:
            if not isinstance(feature, dict) or "name" not in feature:
                break
            names.append(_normalize_variable_name(str(feature["name"])))
        if len(names) == n:
            return names

    return [f"x{i}" for i in range(n)]


def _assignment_value(assignment: dict[str, int], name: str) -> int:
    normalized = _normalize_variable_name(name)
    normalized_assignment = {
        _normalize_variable_name(key): int(value)
        for key, value in assignment.items()
    }
    if normalized not in normalized_assignment:
        raise KeyError(f"assignment does not contain variable {name!r}")
    value = normalized_assignment[normalized]
    if value not in (0, 1):
        raise ValueError(f"assignment value for {name!r} must be 0 or 1")
    return value


def _normalize_variable_name(name: str) -> str:
    if name.startswith("x_"):
        return name[2:]
    return name


def _constant_offset(instance: dict[str, Any]) -> float:
    keys = (
        "constant",
        "constant_offset",
        "offset",
        "qubo_constant",
        "constant_term",
    )
    return sum(_as_float(instance.get(key)) or 0.0 for key in keys)


def _evaluate_upper_triangular(matrix: list[Any], x: list[int]) -> float:
    value = 0.0
    n = len(x)
    for i in range(n):
        row = matrix[i]
        for j in range(i, n):
            value += float(row[j]) * x[i] * x[j]
    return value


def _evaluate_full_matrix(matrix: list[Any], x: list[int]) -> float:
    value = 0.0
    n = len(x)
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            value += float(row[j]) * x[i] * x[j]
    return value


def _is_symmetric_matrix(matrix: Any) -> bool:
    if not isinstance(matrix, list):
        return False
    n = len(matrix)
    for i in range(n):
        if not isinstance(matrix[i], list) or len(matrix[i]) != n:
            return False
        for j in range(i + 1, n):
            if abs(float(matrix[i][j]) - float(matrix[j][i])) > 1e-9:
                return False
    return True


def _has_variable_order(instance: dict[str, Any]) -> bool:
    n = len(instance.get("matrix", []))
    try:
        return _variable_order(instance, n) != [f"x{i}" for i in range(n)]
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
