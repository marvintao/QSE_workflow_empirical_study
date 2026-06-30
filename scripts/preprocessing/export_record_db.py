"""Export a record SQLite DB into readable JSON and JSONL artifacts.

The exported record database is a SQLite trace store with three tables:
request, stage_trace, and agent_round. Some JSON fields are stored as TEXT in
the exported DB, so this script parses those fields and writes human-readable
UTF-8 files for empirical analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from typing import Any


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DEFAULT_DB = os.path.join(ROOT, "reports", "all_new_record.db")
DEFAULT_OUT_DIR = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data", "claude")
DEFAULT_REQ_OVERVIEW = os.path.join(
    ROOT, "use_cases", "requirements", "req_overview.json")

STAGE_JSON_FIELDS = ("llm_input", "llm_output", "token_usage", "artifact")
ROUND_JSON_FIELDS = ("tool_calls", "tool_results", "token_usage")
TESTCASE_PROBLEMS = {"test case selection", "test case minimization"}
REQ_MARKER_RE = re.compile(r"This is user requirement\s+(\d+)\.")


def load_allowed_requirements(req_overview_path: str) -> set[str]:
    with open(req_overview_path, "r", encoding="utf-8") as f:
        overview = json.load(f)
    return {
        f"req_{raw_req_id}"
        for raw_req_id, item in overview.items()
        if item.get("problem") in TESTCASE_PROBLEMS
    }


def derive_req_from_raw_text(raw_text: str) -> str | None:
    match = REQ_MARKER_RE.search(raw_text.lstrip("\ufeff"))
    if match:
        return f"req_{match.group(1)}"
    return None


def parse_json_field(value: Any) -> Any:
    """Parse a JSON TEXT field, preserving malformed values for inspection."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return value
    parsed: Any = value
    while isinstance(parsed, str):
        stripped = parsed.strip()
        if not stripped or stripped[0] not in "[{\"":
            return parsed
        try:
            next_parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"_parse_error": True, "raw": parsed}
        if next_parsed == parsed:
            return next_parsed
        parsed = next_parsed
    return parsed


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def load_database(
    db_path: str,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_reqs = load_allowed_requirements(req_overview_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        requests = [
            row_to_dict(row)
            for row in con.execute("select * from request order by request_id")
        ]
        requests = [
            request
            for request in requests
            if derive_req_from_raw_text(str(request.get("raw_text") or "")) in allowed_reqs
        ]
        allowed_request_ids = {request["request_id"] for request in requests}

        stages: list[dict[str, Any]] = []
        for row in con.execute("select * from stage_trace order by id"):
            item = row_to_dict(row)
            if item.get("request_id") not in allowed_request_ids:
                continue
            for field in STAGE_JSON_FIELDS:
                if field in item:
                    item[field] = parse_json_field(item[field])
            stages.append(item)
        allowed_stage_ids = {stage["id"] for stage in stages}

        rounds: list[dict[str, Any]] = []
        for row in con.execute("select * from agent_round order by stage_trace_id, round_number, id"):
            item = row_to_dict(row)
            if item.get("stage_trace_id") not in allowed_stage_ids:
                continue
            for field in ROUND_JSON_FIELDS:
                if field in item:
                    item[field] = parse_json_field(item[field])
            rounds.append(item)
    finally:
        con.close()

    return requests, stages, rounds


def group_by(items: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item[key], []).append(item)
    return grouped


def build_nested_trace(
    requests: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stages_by_request = group_by(stages, "request_id")
    rounds_by_stage = group_by(rounds, "stage_trace_id")

    traces: list[dict[str, Any]] = []
    for request in requests:
        request_stages = []
        for stage in stages_by_request.get(request["request_id"], []):
            stage = dict(stage)
            stage_rounds = rounds_by_stage.get(stage["id"], [])
            if stage_rounds:
                stage["rounds"] = stage_rounds
            request_stages.append(stage)
        traces.append({"request": request, "stages": request_stages})
    return traces


def build_summary(
    requests: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    db_path: str,
) -> dict[str, Any]:
    final_status_counts: dict[str, int] = {}
    verification_status_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}

    for request in requests:
        final = request.get("final_status")
        verification = request.get("verification_status")
        final_status_counts[str(final)] = final_status_counts.get(str(final), 0) + 1
        verification_status_counts[str(verification)] = verification_status_counts.get(str(verification), 0) + 1

    for stage in stages:
        name = str(stage.get("stage"))
        stage_counts[name] = stage_counts.get(name, 0) + 1

    return {
        "source": relative_to_root(db_path),
        "counts": {
            "requests": len(requests),
            "stage_traces": len(stages),
            "agent_rounds": len(rounds),
        },
        "final_status_counts": final_status_counts,
        "verification_status_counts": verification_status_counts,
        "stage_counts": stage_counts,
    }


def relative_to_root(path: str) -> str:
    """Return a stable repo-relative path when possible."""
    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, ROOT)
    except ValueError:
        rel_path = path
    rel_path = rel_path.replace(os.sep, "/")
    if not rel_path.startswith((".", "/")):
        rel_path = f"./{rel_path}"
    return rel_path


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def export(
    db_path: str,
    out_dir: str,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
) -> None:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")

    os.makedirs(out_dir, exist_ok=True)
    requests, stages, rounds = load_database(db_path, req_overview_path)
    traces = build_nested_trace(requests, stages, rounds)
    summary = build_summary(requests, stages, rounds, db_path)

    write_json(os.path.join(out_dir, "record_summary.json"), summary)
    write_json(os.path.join(out_dir, "record_traces.json"), traces)
    write_jsonl(os.path.join(out_dir, "requests.jsonl"), requests)
    write_jsonl(os.path.join(out_dir, "stage_traces.jsonl"), stages)
    write_jsonl(os.path.join(out_dir, "agent_rounds.jsonl"), rounds)
    write_jsonl(os.path.join(out_dir, "record_traces.jsonl"), traces)

    print(f"Exported {len(requests)} requests, {len(stages)} stage traces, {len(rounds)} agent rounds.")
    print(f"Output directory: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a record SQLite DB into readable JSON and JSONL files."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite DB path. Default: {DEFAULT_DB}",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--req-overview",
        default=DEFAULT_REQ_OVERVIEW,
        help=f"Requirement overview used to keep only test case optimization data. Default: {DEFAULT_REQ_OVERVIEW}",
    )
    args = parser.parse_args()

    export(
        os.path.abspath(args.db),
        os.path.abspath(args.out),
        os.path.abspath(args.req_overview),
    )


if __name__ == "__main__":
    main()
