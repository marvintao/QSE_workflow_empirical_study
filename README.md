# Empirical Study

This directory contains the scripts, processed data, and rendered figures used
for the empirical study.

## Directory Layout

```text
empirical_study/
  README.md
  classical_reference/
  data/
    unpacked_raw_data/
      claude/
      deepseek/
      llama/
      claude_ablation/
    processed_data/
      rq1/
      rq2/
      rq3/
      rq4/
      rq5/
  scripts/
    preprocessing/
    rq1/
    rq2/
    rq3/
    rq4/
    rq5/
  visualization/
```

## Inputs

The scripts read only local repository data. Main inputs are:

```text
./empirical_study/data/unpacked_raw_data/
./empirical_study/classical_reference/reference_data/
./use_cases/requirements/req_overview.json
./use_cases/requirements/req_txt/
./reports/
```

The preprocessing scripts export SQLite trace databases and experiment summary
files from `./reports/` into structured JSON/JSONL files under
`./empirical_study/data/unpacked_raw_data/`.

## Preprocessing

Run commands from the repository root.

Export all supported record databases:

```powershell
python empirical_study\scripts\preprocessing\export_all_record_dbs.py
```

Build the Claude request mapping and `experiment_runs.jsonl`:

```powershell
python empirical_study\scripts\preprocessing\build_request_mapping.py
```

The default preprocessing outputs include:

```text
./empirical_study/data/unpacked_raw_data/claude/record_traces.jsonl
./empirical_study/data/unpacked_raw_data/claude/stage_traces.jsonl
./empirical_study/data/unpacked_raw_data/claude/experiment_runs.jsonl
./empirical_study/data/unpacked_raw_data/deepseek/experiment_runs.jsonl
./empirical_study/data/unpacked_raw_data/llama/experiment_runs.jsonl
```

## Reproduction

The study has two steps per RQ:

1. Generate compact processed data under `./empirical_study/data/processed_data/`.
2. Generate the final table or figure under `./empirical_study/visualization/`.

Use the following commands to reproduce every current visualization.

### RQ1

Generate processed data:

```powershell
python empirical_study\scripts\rq1\data_generation.py
```

This writes:

```text
./empirical_study/data/processed_data/rq1/rq1_source_analysis_data.json
./empirical_study/data/processed_data/rq1/rq1_performance.json
./empirical_study/data/processed_data/rq1/rq1_relation_to_requirements.json
```

Generate visualizations:

```powershell
python empirical_study\scripts\rq1\data_visualization.py
```

This writes:

```text
./empirical_study/visualization/rq1_performance.tex
./empirical_study/visualization/rq1_requirement_correlation.pdf
```

Optional explicit command:

```powershell
python empirical_study\scripts\rq1\data_generation.py `
  --raw-dir empirical_study\data\unpacked_raw_data\claude `
  --req-overview use_cases\requirements\req_overview.json `
  --req-txt-dir use_cases\requirements\req_txt `
  --out empirical_study\data\processed_data\rq1\rq1_source_analysis_data.json `
  --performance empirical_study\data\processed_data\rq1\rq1_performance.json `
  --relation-to-requirements empirical_study\data\processed_data\rq1\rq1_relation_to_requirements.json
```

### RQ2

Generate processed data:

```powershell
python empirical_study\scripts\rq2\data_generation.py
```

This writes:

```text
./empirical_study/data/processed_data/rq2/rq2_qubo_reduction.json
```

Generate visualization:

```powershell
python empirical_study\scripts\rq2\data_visualization.py
```

This writes:

```text
./empirical_study/visualization/rq2_boxplot.pdf
```

Optional explicit command:

```powershell
python empirical_study\scripts\rq2\data_generation.py `
  --reference-dir empirical_study\classical_reference\reference_data `
  --raw-dir empirical_study\data\unpacked_raw_data\claude `
  --req-overview use_cases\requirements\req_overview.json `
  --out empirical_study\data\processed_data\rq2\rq2_qubo_reduction.json
```

### RQ3

Generate processed data:

```powershell
python empirical_study\scripts\rq3\data_generation.py
```

This writes:

```text
./empirical_study/data/processed_data/rq3/rq3_pattern.json
```

Generate visualization:

```powershell
python empirical_study\scripts\rq3\data_visualization.py
```

This writes:

```text
./empirical_study/visualization/rq3_pattern.tex
```

Optional explicit command:

```powershell
python empirical_study\scripts\rq3\data_generation.py `
  --raw-dir empirical_study\data\unpacked_raw_data\claude `
  --out empirical_study\data\processed_data\rq3\rq3_pattern.json
```

### RQ4

Generate processed data:

```powershell
python empirical_study\scripts\rq4\data_generation.py
```

This writes:

```text
./empirical_study/data/processed_data/rq4/rq4_generation_effectivenss.json
```

Generate visualization:

```powershell
python empirical_study\scripts\rq4\data_visualization.py
```

This writes:

```text
./empirical_study/visualization/rq4_ablation.pdf
```

Optional explicit command:

```powershell
python empirical_study\scripts\rq4\data_generation.py `
  --raw-root empirical_study\data\unpacked_raw_data `
  --reference-dir empirical_study\classical_reference\reference_data `
  --req-overview use_cases\requirements\req_overview.json `
  --out empirical_study\data\processed_data\rq4\rq4_generation_effectivenss.json
```

### RQ5

Generate processed data:

```powershell
python empirical_study\scripts\rq5\data_generation.py
```

This writes:

```text
./empirical_study/data/processed_data/rq5/rq5_generation_effectiveness.json
```

Generate visualization:

```powershell
python empirical_study\scripts\rq5\data_visualization.py
```

This writes:

```text
./empirical_study/visualization/rq5_cross_model.tex
```

Optional explicit command:

```powershell
python empirical_study\scripts\rq5\data_generation.py `
  --raw-root empirical_study\data\unpacked_raw_data `
  --req-overview use_cases\requirements\req_overview.json `
  --out empirical_study\data\processed_data\rq5\rq5_generation_effectiveness.json
```

## Current Outputs

The current `./empirical_study/visualization/` directory contains:

```text
./empirical_study/visualization/rq1_performance.tex
./empirical_study/visualization/rq1_requirement_correlation.pdf
./empirical_study/visualization/rq2_boxplot.pdf
./empirical_study/visualization/rq3_pattern.tex
./empirical_study/visualization/rq4_ablation.pdf
./empirical_study/visualization/rq5_cross_model.tex
```

The scripts intentionally generate only the processed fields required by these
current visualization files.

## Data Contracts

RQ1 visualization reads:

```text
./empirical_study/data/processed_data/rq1/rq1_source_analysis_data.json
./empirical_study/data/processed_data/rq1/rq1_performance.json
./empirical_study/data/processed_data/rq1/rq1_relation_to_requirements.json
```

RQ2 visualization reads:

```text
./empirical_study/data/processed_data/rq2/rq2_qubo_reduction.json
```

RQ3 visualization reads:

```text
./empirical_study/data/processed_data/rq3/rq3_pattern.json
```

RQ4 visualization reads:

```text
./empirical_study/data/processed_data/rq4/rq4_generation_effectivenss.json
```

RQ5 visualization reads:

```text
./empirical_study/data/processed_data/rq5/rq5_generation_effectiveness.json
```

No additional processed reports, legend-only figures, or unused tables are
required for the current visualization set.
