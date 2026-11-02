# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

"""Raw QUBO helpers for test-case reference comparison."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawQuboBlock:
    matrix: list[Any]
    start: int
    end: int
    constant: float
    matrix_convention: str


@dataclass(frozen=True)
class RawTestcaseQuboExpression:
    req_id: str
    request_id: str
    iteration: int
    blocks: list[RawQuboBlock]


def load_raw_qubo_expressions(repo_root: str) -> dict[str, RawTestcaseQuboExpression]:
    raw_root = os.path.join(repo_root, "empirical_study", "data", "unpacked_raw_data")
    mapping_path = os.path.join(raw_root, "request_mapping.jsonl")
    traces_path = os.path.join(raw_root, "record_traces.jsonl")
    if not os.path.exists(mapping_path) or not os.path.exists(traces_path):
        return {}

    request_to_run = _load_testcase_request_mapping(mapping_path)
    best_by_req: dict[str, RawTestcaseQuboExpression] = {}

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
            blocks = _blocks_from_trace(trace)
            if not blocks:
                continue
            expression = RawTestcaseQuboExpression(
                req_id=str(run["req"]),
                request_id=str(run["request_id"]),
                iteration=int(run["iter"]),
                blocks=blocks,
            )
            current = best_by_req.get(expression.req_id)
            if current is None or _expression_rank(expression) < _expression_rank(current):
                best_by_req[expression.req_id] = expression

    return best_by_req


def evaluate_qubo_for_bits(
    expression: RawTestcaseQuboExpression,
    bits: list[int],
) -> float:
    total = 0.0
    for block in expression.blocks:
        local_bits = bits[block.start:block.end]
        if len(local_bits) != len(block.matrix):
            continue
        if block.matrix_convention == "full":
            total += _evaluate_full_matrix(block.matrix, local_bits)
        else:
            total += _evaluate_upper_triangular(block.matrix, local_bits)
        total += block.constant
    return total


def _load_testcase_request_mapping(path: str) -> dict[str, dict[str, Any]]:
    request_to_run = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            req_no = row.get("req_no")
            if isinstance(req_no, int) and 0 <= req_no <= 19:
                request_to_run[row["request_id"]] = row
    return request_to_run


def _blocks_from_trace(trace: dict[str, Any]) -> list[RawQuboBlock]:
    blocks = []
    sequential_start = 0
    for stage in trace.get("stages", []):
        if stage.get("stage") != "execute":
            continue
        artifact = stage.get("artifact")
        if not isinstance(artifact, dict):
            continue
        result = artifact.get("result")
        if not isinstance(result, dict):
            continue
        instance = result.get("instance")
        if not isinstance(instance, dict):
            continue
        matrix = instance.get("matrix")
        if not isinstance(matrix, list):
            continue
        n = len(matrix)
        start = _as_int(result.get("variable_start"))
        end = _as_int(result.get("variable_end"))
        if start is None or end is None or end - start != n:
            start = sequential_start
            end = start + n
        sequential_start = max(sequential_start, end)
        blocks.append(
            RawQuboBlock(
                matrix=matrix,
                start=start,
                end=end,
                constant=_constant_offset(instance),
                matrix_convention=_matrix_convention(instance),
            )
        )
    return blocks


def _expression_rank(expression: RawTestcaseQuboExpression) -> tuple[int, int]:
    covered = max((block.end for block in expression.blocks), default=0)
    return (-covered, expression.iteration)


def _matrix_convention(instance: dict[str, Any]) -> str:
    explicit = instance.get("matrix_convention")
    if isinstance(explicit, str):
        lowered = explicit.lower()
        if "full" in lowered or "symmetric" in lowered:
            return "full"
        if "upper" in lowered:
            return "upper"
    if _is_symmetric_matrix(instance.get("matrix")):
        return "full"
    return "upper"


def _constant_offset(instance: dict[str, Any]) -> float:
    keys = (
        "constant",
        "constant_offset",
        "offset",
        "qubo_constant",
        "constant_shift",
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


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
