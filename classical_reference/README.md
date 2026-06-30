# Classical Reference Generation

This directory contains the classical baseline reference generators used for
solution-quality comparison in the empirical study. The current implementation
uses a jMetalPy genetic algorithm shell adapted from the IGDec-QAOA classical
baseline and maps this repository's structured requirements to binary
single-objective optimization problems.

## Directory Layout

- `scripts/ga.py`: shared reusable GA wrapper.
- `scripts/testcase.py`: GA adapter and CLI for test case selection and test
  case minimization requirements.
- `scripts/testcase_qubo.py`: raw QUBO extraction and evaluation helpers for
  test-case requirements.
- `scripts/configuration.py`: GA adapter and CLI for configuration selection
  and configuration prioritization requirements.
- `scripts/configuration_qubo.py`: raw QUBO extraction and evaluation helpers
  for configuration requirements.
- `reference_data/testcase/`: compact JSON reference outputs for `req_0`
  through `req_19`.
- `reference_data/configuration/`: compact JSON reference outputs for `req_20`
  through `req_79`.

The reference output path is one JSON file per requirement:

```text
reference_data/testcase/req_0_reference.json
reference_data/configuration/req_20_reference.json
```

## Environment

Create the conda environment from the local YAML file:

```bash
conda env create -f empirical_study/classical_reference/environment.yml
conda activate qse-classical-reference
```

The scripts require Python 3.11+, NumPy, and jMetalPy. They read benchmark and
requirement inputs from `use_cases/benchmark/json` and
`use_cases/requirements/req_overview.json`, and optionally read raw workflow
QUBO traces from `empirical_study/data/unpacked_raw_data`.

## GA Defaults And IGDec-QAOA Relation

The shared GA implementation is in `scripts/ga.py`. It follows the GA shell
used by the IGDec-QAOA classical baseline artifact as closely as practical:

- jMetalPy single-objective `GeneticAlgorithm`;
- binary solution representation;
- single-point crossover through `SPXCrossover`;
- bit-flip mutation through `BitFlipMutation`;
- equal parent and offspring population sizes;
- fixed stopping budget by number of evaluations.

The default parameters used by both `scripts/testcase.py` and
`scripts/configuration.py` are:

| Parameter | Default | Meaning |
|---|---:|---|
| `--runs` | `20` | Independent GA repetitions per requirement. |
| `--population-size` | `100` | GA population size and offspring population size. |
| `--evaluations` | `400000` | Maximum evaluations per GA repetition. |
| `--crossover-probability` | `1.0` | Probability passed to `SPXCrossover`. |
| `--mutation-probability` | unset | If unset, use `1 / number_of_bits`. |
| `--seed` | unset | If unset, use nondeterministic seeding. |

The batch script exposes the same defaults through environment variables:
`RUNS=20`, `POPULATION_SIZE=100`, `EVALUATIONS=400000`,
`CROSSOVER_PROBABILITY=1.0`, and unset `MUTATION_PROBABILITY`.

This repository does not run the original IGDec-QAOA GA code unchanged over all
80 requirements. Instead, it reuses the same jMetalPy GA design and default
settings, while replacing the fitness wrapper with adapters for this paper's
structured instances:

- `scripts/testcase.py` maps test case selection and minimization requirements
  to binary subsets of test cases.
- `scripts/configuration.py` maps configuration selection and prioritization
  requirements to binary feature selections.

Accordingly, the generated values are references for this repository's
requirement instances and objective definitions, not raw IGDec-QAOA case-study
outputs.

## Output Schema

The compact test-case output schema is:

```json
{
  "best_solution": [0, 1],
  "ga_fitness_val": 0.0,
  "qubo_val": 0.0,
  "bitstring": "10",
  "selected_test_cases": ["t1"],
  "wallclock_time": 1.23
}
```

The compact configuration output schema is:

```json
{
  "best_solution": [0, 1],
  "ga_fitness_val": 0.0,
  "qubo_val": 0.0,
  "bitstring": "10",
  "assignment": {"A": 0, "B": 1},
  "selected_features": ["B"],
  "wallclock_time": 1.23
}
```

For configuration prioritization requirements, the configuration output also
contains `top_configurations`, whose entries keep only solution representation
fields (`bitstring`, `assignment`, and `selected_features`).

`ga_fitness_val` is the fitness used by the structured GA adapter.
`qubo_val` is the value obtained by evaluating the selected GA solution against
the raw QUBO expression extracted from unpacked workflow traces when available.
`wallclock_time` is elapsed wall-clock seconds for the requirement across the
configured GA runs.

## Batch Generation

Run all 80 requirement instances from the repository root:

```bash
bash empirical_study/classical_reference/generate_all_references.sh
```

The default batch budget matches the runner defaults:

- `RUNS=20`
- `POPULATION_SIZE=100`
- `EVALUATIONS=400000`
- `CROSSOVER_PROBABILITY=1.0`
- `MUTATION_PROBABILITY` unset, which makes the scripts use `1 / number_of_bits`

Override the budget with environment variables for smoke tests or server runs:

```bash
RUNS=1 POPULATION_SIZE=10 EVALUATIONS=10 SEED=123 \
  bash empirical_study/classical_reference/generate_all_references.sh
```

Useful batch-script variables:

- `PYTHON`: Python executable, default `python`.
- `RUNS`: GA runs per requirement, default `20`.
- `POPULATION_SIZE`: GA population and offspring size, default `100`.
- `EVALUATIONS`: max evaluations per GA run, default `400000`.
- `SEED`: base random seed. If unset, the runners use nondeterministic seeding.
- `MUTATION_PROBABILITY`: optional explicit mutation probability. If unset, the
  runners use `1 / number_of_bits`.
- `CROSSOVER_PROBABILITY`: crossover probability, default `1.0`.
- `TESTCASE_OUT`: output directory for `req_0` through `req_19`.
- `CONFIGURATION_OUT`: output directory for `req_20` through `req_79`.
- `LOG_DIR`: per-requirement log directory.
- `SKIP_EXISTING=1`: skip a requirement if its target JSON already exists.

The batch script runs one requirement per Python invocation so that server logs
identify the failing instance directly and completed requirement files can be
kept when a later requirement fails.

## Direct Commands

Generate all test-case references in one Python process:

```bash
python empirical_study/classical_reference/scripts/testcase.py \
  --runs 20 \
  --population-size 100 \
  --evaluations 400000 \
  --out empirical_study/classical_reference/reference_data/testcase
```

Generate all configuration references in one Python process:

```bash
python empirical_study/classical_reference/scripts/configuration.py \
  --runs 20 \
  --population-size 100 \
  --evaluations 400000 \
  --out empirical_study/classical_reference/reference_data/configuration
```

Generate a single requirement:

```bash
python empirical_study/classical_reference/scripts/testcase.py --req 0
python empirical_study/classical_reference/scripts/configuration.py --req 20
```

## Validation

Compile the scripts before a server run:

```bash
python -m py_compile \
  empirical_study/classical_reference/scripts/ga.py \
  empirical_study/classical_reference/scripts/testcase.py \
  empirical_study/classical_reference/scripts/testcase_qubo.py \
  empirical_study/classical_reference/scripts/configuration.py \
  empirical_study/classical_reference/scripts/configuration_qubo.py
```

Use a very small budget for smoke validation only:

```bash
RUNS=1 POPULATION_SIZE=10 EVALUATIONS=10 SEED=123 \
  bash empirical_study/classical_reference/generate_all_references.sh
```

Small-budget outputs are useful for checking feasibility and schema consistency,
but should not be reported as final best-known reference values.
