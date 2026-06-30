#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-python}"
RUNS="${RUNS:-20}"
POPULATION_SIZE="${POPULATION_SIZE:-100}"
EVALUATIONS="${EVALUATIONS:-400000}"
CROSSOVER_PROBABILITY="${CROSSOVER_PROBABILITY:-1.0}"
SEED="${SEED:-}"
MUTATION_PROBABILITY="${MUTATION_PROBABILITY:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

TESTCASE_OUT="${TESTCASE_OUT:-${SCRIPT_DIR}/reference_data/testcase}"
CONFIGURATION_OUT="${CONFIGURATION_OUT:-${SCRIPT_DIR}/reference_data/configuration}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/reference_data/logs}"

mkdir -p "${TESTCASE_OUT}" "${CONFIGURATION_OUT}" "${LOG_DIR}"

common_args=(
  --repo-root "${REPO_ROOT}"
  --runs "${RUNS}"
  --population-size "${POPULATION_SIZE}"
  --evaluations "${EVALUATIONS}"
  --crossover-probability "${CROSSOVER_PROBABILITY}"
)

if [[ -n "${SEED}" ]]; then
  common_args+=(--seed "${SEED}")
fi

if [[ -n "${MUTATION_PROBABILITY}" ]]; then
  common_args+=(--mutation-probability "${MUTATION_PROBABILITY}")
fi

run_requirement() {
  local kind="$1"
  local script="$2"
  local req="$3"
  local out_dir="$4"
  local target="${out_dir}/req_${req}_reference.json"
  local log_file="${LOG_DIR}/req_${req}_${kind}.log"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${target}" ]]; then
    echo "[skip] req_${req}: ${target} already exists"
    return
  fi

  echo "[run] req_${req}: ${kind}"
  "${PYTHON}" "${script}" \
    --req "${req}" \
    --out "${out_dir}" \
    "${common_args[@]}" \
    2>&1 | tee "${log_file}"
}

echo "Repository root: ${REPO_ROOT}"
echo "Runs per requirement: ${RUNS}"
echo "Population size: ${POPULATION_SIZE}"
echo "Evaluations per run: ${EVALUATIONS}"
echo "Testcase output: ${TESTCASE_OUT}"
echo "Configuration output: ${CONFIGURATION_OUT}"
echo "Logs: ${LOG_DIR}"

for req in $(seq 0 19); do
  run_requirement \
    "testcase" \
    "${SCRIPT_DIR}/scripts/testcase.py" \
    "${req}" \
    "${TESTCASE_OUT}"
done

for req in $(seq 20 79); do
  run_requirement \
    "configuration" \
    "${SCRIPT_DIR}/scripts/configuration.py" \
    "${req}" \
    "${CONFIGURATION_OUT}"
done

echo "Done. Generated reference files for req_0 through req_79."
