# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

"""Export all model record databases into unpacked raw-data directories."""
from __future__ import annotations

import os
from dataclasses import dataclass

from build_request_mapping import build_mapping, load_experiment_runs, load_requests, write_jsonl
from export_record_db import export, load_allowed_requirements


ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
RAW_ROOT = os.path.join(ROOT, "empirical_study", "data", "unpacked_raw_data")
REQ_DIR = os.path.join(ROOT, "use_cases", "requirements", "req_txt")
REQ_OVERVIEW = os.path.join(ROOT, "use_cases", "requirements", "req_overview.json")


@dataclass(frozen=True)
class Dataset:
    name: str
    db: str
    exp_record: str
    exp_data_dir: str


DATASETS = (
    Dataset(
        name="claude",
        db=os.path.join(ROOT, "reports", "all_new_record.db"),
        exp_record=os.path.join(ROOT, "reports", "record_all_new_exp.md"),
        exp_data_dir=os.path.join(ROOT, "reports", "exp_all_new_data"),
    ),
    Dataset(
        name="deepseek",
        db=os.path.join(ROOT, "reports", "deepseek_record.db"),
        exp_record=os.path.join(ROOT, "reports", "record_deepseek_exp.md"),
        exp_data_dir=os.path.join(ROOT, "reports", "exp_deepseek_data"),
    ),
    Dataset(
        name="llama",
        db=os.path.join(ROOT, "reports", "llama_record.db"),
        exp_record=os.path.join(ROOT, "reports", "record_llama_exp.md"),
        exp_data_dir=os.path.join(ROOT, "reports", "exp_llama_data"),
    ),
)

ABLATION_DBS = (
    "mega_prompt_record.db",
    "no_codegen_skill_record.db",
    "no_knowledge_record.db",
    "no_reviewer_record.db",
)


def export_dataset(dataset: Dataset) -> None:
    out_dir = os.path.join(RAW_ROOT, dataset.name)
    print(f"\n== Exporting {dataset.name} ==")
    export(dataset.db, out_dir, REQ_OVERVIEW)

    request_ids = {row["request_id"] for row in load_requests(dataset.db)}
    allowed_reqs = load_allowed_requirements(REQ_OVERVIEW)
    mapping_rows = build_mapping(
        dataset.db,
        dataset.exp_record,
        dataset.exp_data_dir,
        REQ_DIR,
        REQ_OVERVIEW,
    )
    run_rows = load_experiment_runs(dataset.exp_record, request_ids, allowed_reqs)

    write_jsonl(os.path.join(out_dir, "request_mapping.jsonl"), mapping_rows)
    write_jsonl(os.path.join(out_dir, "experiment_runs.jsonl"), run_rows)

    print(f"Generated {len(mapping_rows)} request mapping rows.")
    print(f"Generated {len(run_rows)} experiment run rows.")


def export_ablation_db(db_name: str) -> None:
    name = os.path.splitext(db_name)[0]
    out_dir = os.path.join(RAW_ROOT, "claude_ablation", name)
    print(f"\n== Exporting claude_ablation/{name} ==")
    export(os.path.join(ROOT, "reports", db_name), out_dir, REQ_OVERVIEW)


def main() -> None:
    for dataset in DATASETS:
        export_dataset(dataset)
    for db_name in ABLATION_DBS:
        export_ablation_db(db_name)


if __name__ == "__main__":
    main()
