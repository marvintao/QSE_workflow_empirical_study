"""Build structured request/run mappings for empirical-study preprocessing.

Outputs:
  empirical_study/data/unpacked_raw_data/<dataset>/request_mapping.jsonl
  empirical_study/data/unpacked_raw_data/<dataset>/experiment_runs.jsonl

`request_mapping.jsonl` maps request ids present in the source record DB to
requirement files and exported artifact directories.

`experiment_runs.jsonl` is the structured planned-run manifest parsed once from
the experiment record. Downstream RQ scripts should read this JSONL rather
than parse the Markdown report directly.
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
DEFAULT_EXP_RECORD = os.path.join(ROOT, "reports", "record_all_new_exp.md")
DEFAULT_EXP_DATA_DIR = os.path.join(ROOT, "reports", "exp_all_new_data")
DEFAULT_REQ_DIR = os.path.join(ROOT, "use_cases", "requirements", "req_txt")
DEFAULT_REQ_OVERVIEW = os.path.join(ROOT, "use_cases", "requirements", "req_overview.json")
DEFAULT_OUT = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data", "claude", "request_mapping.jsonl")
DEFAULT_RUNS_OUT = os.path.join(
    ROOT, "empirical_study", "data", "unpacked_raw_data", "claude", "experiment_runs.jsonl")

EXP_RECORD_ROW_RE = re.compile(
    r"^\|\s*(?P<req_no>\d+)\s*"
    r"\|\s*(?P<iter>\d+)\s*"
    r"\|\s*(?P<req>req_\d+)\s*"
    r"\|\s*(?P<parse>[^|]+)\s*"
    r"\|\s*(?P<analyze>[^|]+)\s*"
    r"\|\s*(?P<blueprint>[^|]+)\s*"
    r"\|\s*(?P<encode>[^|]+)\s*"
    r"\|\s*(?P<codegen>[^|]+)\s*"
    r"\|\s*(?P<execute>[^|]+)\s*"
    r"\|\s*(?P<combine>[^|]+)\s*"
    r"\|\s*(?P<verify>[^|]+)\s*"
    r"\|\s*`(?P<rid8>[0-9a-f]{8})`\s*"
    r"\|\s*(?P<notes>[^|]*)\|"
)
REQ_MARKER_RE = re.compile(r"This is user requirement\s+(\d+)\.")
STAGE_NAMES = (
    "parse", "analyze", "blueprint", "encode",
    "codegen", "execute", "combine", "verify",
)
TESTCASE_PROBLEMS = {"test case selection", "test case minimization"}


def load_allowed_requirements(req_overview_path: str) -> set[str]:
    with open(req_overview_path, "r", encoding="utf-8") as f:
        overview = json.load(f)
    return {
        f"req_{raw_req_id}"
        for raw_req_id, item in overview.items()
        if item.get("problem") in TESTCASE_PROBLEMS
    }


def relative_to_root(path: str | None) -> str | None:
    if not path:
        return None
    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, ROOT)
    except ValueError:
        rel_path = path
    rel_path = rel_path.replace(os.sep, "/")
    if not rel_path.startswith((".", "/")):
        rel_path = f"./{rel_path}"
    return rel_path


def load_requests(db_path: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        return [
            {"request_id": row["request_id"], "raw_text": row["raw_text"]}
            for row in con.execute("select request_id, raw_text from request order by request_id")
        ]
    finally:
        con.close()


def build_prefix_to_id(request_ids: set[str]) -> dict[str, str]:
    prefix_to_id: dict[str, str] = {}
    for request_id in request_ids:
        rid8 = request_id[:8]
        if rid8 in prefix_to_id:
            raise ValueError(f"request_id prefix collision: {rid8}")
        prefix_to_id[rid8] = request_id
    return prefix_to_id


def status_mark_success(value: str) -> bool | None:
    codes = {ord(ch) for ch in value.strip()}
    if 0x2705 in codes:
        return True
    if 0x274C in codes:
        return False
    return None


def load_experiment_runs(
    exp_record_path: str,
    request_ids: set[str],
    allowed_reqs: set[str] | None = None,
) -> list[dict[str, Any]]:
    prefix_to_id = build_prefix_to_id(request_ids)
    rows: list[dict[str, Any]] = []
    with open(exp_record_path, "r", encoding="utf-8") as f:
        for line in f:
            match = EXP_RECORD_ROW_RE.match(line)
            if not match:
                continue
            req = match.group("req")
            if allowed_reqs is not None and req not in allowed_reqs:
                continue

            stages = {stage: status_mark_success(match.group(stage)) for stage in STAGE_NAMES}
            rid8 = match.group("rid8")
            request_id = prefix_to_id.get(rid8)
            rows.append({
                "req_no": int(match.group("req_no")),
                "req": req,
                "iter": int(match.group("iter")),
                "rid8": rid8,
                "request_id": request_id,
                "stages": stages,
                "compilation_success": stages["codegen"],
                "execution_success": stages["execute"],
                "combination_success": stages["combine"],
                "has_unpacked_trace": request_id is not None,
                "notes": match.group("notes").strip(),
                "sources": {
                    "exp_record": relative_to_root(exp_record_path),
                },
            })

    rows.sort(key=lambda row: (row["req_no"], row["iter"], row["rid8"]))
    return rows


def load_exp_record_mapping(
    exp_record_path: str,
    request_ids: set[str],
    allowed_reqs: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in load_experiment_runs(exp_record_path, request_ids, allowed_reqs):
        request_id = row["request_id"]
        if request_id is None:
            continue
        mapping[request_id] = {
            "rid8": row["rid8"],
            "req_no": row["req_no"],
            "iter": row["iter"],
            "req": row["req"],
        }
    return mapping


def load_exp_data_dirs(exp_data_dir: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not os.path.isdir(exp_data_dir):
        return mapping
    for name in sorted(os.listdir(exp_data_dir)):
        dir_path = os.path.join(exp_data_dir, name)
        request_id_path = os.path.join(dir_path, "request_id")
        if not os.path.isdir(dir_path) or not os.path.exists(request_id_path):
            continue

        with open(request_id_path, "r", encoding="utf-8") as f:
            request_id = f.read().strip()
        if request_id:
            mapping[request_id] = dir_path
    return mapping


def load_req_texts(req_dir: str) -> dict[str, str]:
    texts: dict[str, str] = {}
    for name in sorted(os.listdir(req_dir)):
        if not name.startswith("req_") or not name.endswith(".txt"):
            continue
        req = name[:-4]
        path = os.path.join(req_dir, name)
        with open(path, "r", encoding="utf-8-sig") as f:
            texts[req] = f.read()
    return texts


def derive_req_from_raw_text(raw_text: str) -> str | None:
    match = REQ_MARKER_RE.search(raw_text.lstrip("\ufeff"))
    if match:
        return f"req_{match.group(1)}"
    return None


def build_mapping(
    db_path: str,
    exp_record_path: str,
    exp_data_dir: str,
    req_dir: str,
    req_overview_path: str = DEFAULT_REQ_OVERVIEW,
) -> list[dict[str, Any]]:
    allowed_reqs = load_allowed_requirements(req_overview_path)
    requests = load_requests(db_path)
    request_ids = {row["request_id"] for row in requests}
    exp_record = load_exp_record_mapping(exp_record_path, request_ids, allowed_reqs)
    exp_data_dirs = load_exp_data_dirs(exp_data_dir)
    req_texts = load_req_texts(req_dir)

    rows: list[dict[str, Any]] = []
    for request in requests:
        request_id = request["request_id"]
        raw_text = request["raw_text"].lstrip("\ufeff")
        record_info = exp_record.get(request_id, {})
        req = record_info.get("req") or derive_req_from_raw_text(raw_text)
        if req not in allowed_reqs:
            continue
        req_file = os.path.join(req_dir, f"{req}.txt") if req else None
        exp_dir = exp_data_dirs.get(request_id)

        rows.append({
            "request_id": request_id,
            "rid8": request_id[:8],
            "req": req,
            "req_no": record_info.get("req_no"),
            "iter": record_info.get("iter"),
            "req_file": relative_to_root(req_file),
            "exp_data_dir": relative_to_root(exp_dir),
            "sources": {
                "record_db": relative_to_root(db_path),
                "exp_record": relative_to_root(exp_record_path),
                "exp_data_request_id_file": relative_to_root(
                    os.path.join(exp_dir, "request_id") if exp_dir else None
                ),
            },
            "checks": {
                "found_in_exp_record": request_id in exp_record,
                "found_in_exp_data_all": exp_dir is not None,
                "found_req_file": req in req_texts if req else False,
                "raw_text_matches_req_file": req_texts.get(req) == raw_text if req else False,
            },
        })

    rows.sort(key=lambda row: (
        row["req_no"] if row["req_no"] is not None else 10**9,
        row["iter"] if row["iter"] is not None else 10**9,
        row["request_id"],
    ))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unpacked request/run mapping JSONL files."
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Default: {DEFAULT_DB}")
    parser.add_argument("--exp-record", default=DEFAULT_EXP_RECORD, help=f"Default: {DEFAULT_EXP_RECORD}")
    parser.add_argument("--exp-data-dir", default=DEFAULT_EXP_DATA_DIR, help=f"Default: {DEFAULT_EXP_DATA_DIR}")
    parser.add_argument("--req-dir", default=DEFAULT_REQ_DIR, help=f"Default: {DEFAULT_REQ_DIR}")
    parser.add_argument("--req-overview", default=DEFAULT_REQ_OVERVIEW, help=f"Default: {DEFAULT_REQ_OVERVIEW}")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Default: {DEFAULT_OUT}")
    parser.add_argument("--runs-out", default=DEFAULT_RUNS_OUT, help=f"Default: {DEFAULT_RUNS_OUT}")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    exp_record_path = os.path.abspath(args.exp_record)
    request_ids = {row["request_id"] for row in load_requests(db_path)}

    rows = build_mapping(
        db_path,
        exp_record_path,
        os.path.abspath(args.exp_data_dir),
        os.path.abspath(args.req_dir),
        os.path.abspath(args.req_overview),
    )
    run_rows = load_experiment_runs(
        exp_record_path,
        request_ids,
        load_allowed_requirements(os.path.abspath(args.req_overview)),
    )

    write_jsonl(os.path.abspath(args.out), rows)
    write_jsonl(os.path.abspath(args.runs_out), run_rows)

    checks = {
        "request_mapping_rows": len(rows),
        "experiment_runs": len(run_rows),
        "experiment_runs_with_unpacked_trace": sum(row["has_unpacked_trace"] for row in run_rows),
        "found_in_exp_record": sum(row["checks"]["found_in_exp_record"] for row in rows),
        "found_in_exp_data_all": sum(row["checks"]["found_in_exp_data_all"] for row in rows),
        "found_req_file": sum(row["checks"]["found_req_file"] for row in rows),
        "raw_text_matches_req_file": sum(row["checks"]["raw_text_matches_req_file"] for row in rows),
    }
    print(f"Generated {len(rows)} request mapping rows.")
    print(f"Mapping output: {os.path.abspath(args.out)}")
    print(f"Generated {len(run_rows)} experiment run rows.")
    print(f"Runs output: {os.path.abspath(args.runs_out)}")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
