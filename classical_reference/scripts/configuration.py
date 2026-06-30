"""GA reference runner for configuration optimization requirements.

This adapter combines this repository's configuration-optimization inputs with
the IGDec-QAOA GA baseline from https://zenodo.org/records/13911651.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

try:
    from .ga import GeneticAlgorithmConfig, run_binary_ga
    from .configuration_qubo import evaluate_qubo_for_assignment, load_raw_qubo_expressions
except ImportError:
    from ga import GeneticAlgorithmConfig, run_binary_ga
    from configuration_qubo import evaluate_qubo_for_assignment, load_raw_qubo_expressions


CONFIGURATION_PROBLEMS = {
    "configuration selection",
    "configuration prioritization",
}


@dataclass(frozen=True)
class Feature:
    name: str
    cost: float


@dataclass(frozen=True)
class ConfigurationCandidate:
    bits: tuple[int, ...]
    cost: float
    unsatisfied_constraints: int
    fitness: float


class FeatureModelProblem:
    def __init__(
        self,
        features: list[Feature],
        clauses: list[list[tuple[int, bool]]],
        penalty: float,
    ) -> None:
        self.features = features
        self.clauses = clauses
        self.penalty = penalty

    @property
    def n_variables(self) -> int:
        return len(self.features)

    @property
    def total_constraints(self) -> int:
        return len(self.clauses)

    def evaluate(self, bits: tuple[int, ...]) -> ConfigurationCandidate:
        unsatisfied = self.unsatisfied_constraints(bits)
        cost = self.cost(bits)
        return ConfigurationCandidate(
            bits=bits,
            cost=cost,
            unsatisfied_constraints=unsatisfied,
            fitness=cost + self.penalty * unsatisfied,
        )

    def cost(self, bits: tuple[int, ...]) -> float:
        return sum(feature.cost for feature, bit in zip(self.features, bits) if bit)

    def unsatisfied_constraints(self, bits: tuple[int, ...]) -> int:
        unsatisfied = 0
        for clause in self.clauses:
            if not self._clause_satisfied(bits, clause):
                unsatisfied += 1
        return unsatisfied

    def selected_features(self, bits: tuple[int, ...]) -> list[str]:
        return [
            feature.name
            for feature, bit in zip(self.features, bits)
            if bit
        ]

    def _clause_satisfied(
        self,
        bits: tuple[int, ...],
        clause: list[tuple[int, bool]],
    ) -> bool:
        for index, positive in clause:
            value = bool(bits[index])
            if value == positive:
                return True
        return False


class ConfigurationBinaryProblem(BinaryProblem):
    def __init__(self, feature_problem: FeatureModelProblem) -> None:
        super().__init__()
        self.feature_problem = feature_problem
        self.number_of_bits = feature_problem.n_variables
        self.number_of_bits_per_variable = [self.number_of_bits]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["configuration_cost_with_constraint_penalty"]
        self.feasible_seen: dict[tuple[int, ...], ConfigurationCandidate] = {}
        self.candidates_seen: dict[tuple[int, ...], ConfigurationCandidate] = {}

    def number_of_variables(self) -> int:
        return self.number_of_bits

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        bits = tuple(1 if bit else 0 for bit in solution.bits.tolist())
        candidate = self.feature_problem.evaluate(bits)
        solution.objectives[0] = candidate.fitness
        solution.attributes["candidate"] = candidate
        self._record(candidate)
        return solution

    def create_solution(self) -> BinarySolution:
        solution = BinarySolution(
            number_of_variables=self.number_of_bits,
            number_of_objectives=self.number_of_objectives(),
        )
        solution.bits = np.random.choice([True, False], size=self.number_of_bits)
        return solution

    def name(self) -> str:
        return "ConfigurationOptimization"

    def _record(self, candidate: ConfigurationCandidate) -> None:
        existing = self.candidates_seen.get(candidate.bits)
        if existing is None or candidate_sort_key(candidate) < candidate_sort_key(existing):
            self.candidates_seen[candidate.bits] = candidate
        if candidate.unsatisfied_constraints == 0:
            feasible_existing = self.feasible_seen.get(candidate.bits)
            if feasible_existing is None or candidate.cost < feasible_existing.cost:
                self.feasible_seen[candidate.bits] = candidate


def candidate_sort_key(
    candidate: ConfigurationCandidate,
) -> tuple[int, float, float, tuple[int, ...]]:
    return (
        candidate.unsatisfied_constraints,
        candidate.cost,
        candidate.fitness,
        candidate.bits,
    )


def repo_root_from_script() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", "..", ".."))


def load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, data: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def write_reference_outputs(out_path: str, output: dict[str, object]) -> None:
    abs_out_path = os.path.abspath(out_path)
    if abs_out_path.lower().endswith(".json"):
        out_dir = os.path.dirname(abs_out_path)
    else:
        out_dir = abs_out_path
    os.makedirs(out_dir, exist_ok=True)
    for req_id, result in output.items():
        write_json(os.path.join(out_dir, f"{req_id}_reference.json"), result)


def parse_clause(constraint: str, name_to_index: dict[str, int]) -> list[tuple[int, bool]]:
    clause = []
    for raw_literal in constraint.split(" or "):
        literal = raw_literal.strip()
        positive = True
        if literal.startswith("not "):
            positive = False
            literal = literal[4:].strip()
        if literal not in name_to_index:
            raise ValueError(f"Unknown feature in constraint literal: {raw_literal!r}")
        clause.append((name_to_index[literal], positive))
    return clause


def load_feature_problem(repo_root: str, model_index: int) -> FeatureModelProblem:
    json_dir = os.path.join(repo_root, "use_cases", "benchmark", "json")
    cost_path = os.path.join(json_dir, f"feature_model_cost_{model_index}.json")
    constraint_path = os.path.join(json_dir, f"feature_model_constraint_{model_index}.json")

    cost_data = load_json(cost_path)
    constraint_data = load_json(constraint_path)
    if not isinstance(cost_data, list) or not isinstance(constraint_data, list):
        raise ValueError(f"Invalid feature model JSON for feature_model_{model_index}")

    features = [
        Feature(name=str(item["name"]), cost=float(item["cost"]))
        for item in cost_data
    ]
    name_to_index = {feature.name: index for index, feature in enumerate(features)}
    clauses = [
        parse_clause(str(item["constraint"]), name_to_index)
        for item in constraint_data
    ]
    penalty = sum(feature.cost for feature in features) * 1.5
    return FeatureModelProblem(features=features, clauses=clauses, penalty=penalty)


def feature_model_index(benchmark: str) -> int:
    prefix = "feature_model_"
    if not benchmark.startswith(prefix):
        raise ValueError(f"Unexpected configuration benchmark name: {benchmark}")
    return int(benchmark[len(prefix):])


def load_configuration_requirements(repo_root: str) -> dict[int, dict[str, object]]:
    overview_path = os.path.join(repo_root, "use_cases", "requirements", "req_overview.json")
    overview = load_json(overview_path)
    if not isinstance(overview, dict):
        raise ValueError("req_overview.json must contain a JSON object")

    requirements: dict[int, dict[str, object]] = {}
    for req_id_text, metadata in overview.items():
        if not isinstance(metadata, dict):
            continue
        if metadata.get("problem") in CONFIGURATION_PROBLEMS:
            requirements[int(req_id_text)] = metadata
    return dict(sorted(requirements.items()))


def solve_requirement(
    repo_root: str,
    req_no: int,
    metadata: dict[str, object],
    args: argparse.Namespace,
    raw_qubo_expressions: dict[str, object] | None = None,
) -> dict[str, object]:
    problem_name = str(metadata["problem"])
    benchmark = str(metadata["benchmark"])
    model_index = feature_model_index(benchmark)
    feature_problem = load_feature_problem(repo_root, model_index)
    requested_count = 1
    if problem_name == "configuration prioritization":
        requested_count = int(metadata["m_for_prioritization"])

    candidates_by_bits: dict[tuple[int, ...], ConfigurationCandidate] = {}
    run_summaries = []
    started = time.perf_counter()

    for run_index in range(args.runs):
        problem_adapter = ConfigurationBinaryProblem(feature_problem)
        seed = None
        if args.seed is not None:
            seed = args.seed + req_no * 1009 + run_index
        ga_config = GeneticAlgorithmConfig(
            population_size=args.population_size,
            max_evaluations=args.evaluations,
            crossover_probability=args.crossover_probability,
            mutation_probability=args.mutation_probability,
            seed=seed,
        )
        run = run_binary_ga(problem_adapter, ga_config)
        run_summaries.append(
            {
                "evaluations": run.evaluations,
                "population_size": run.population_size,
                "offspring_population_size": run.offspring_population_size,
                "mutation_probability": run.mutation_probability,
                "crossover_probability": run.crossover_probability,
                "elapsed_s": run.elapsed_s,
                "best_fitness": float(run.best_solution.objectives[0]),
                "unique_feasible_candidates": len(problem_adapter.feasible_seen),
            }
        )
        for bits, candidate in problem_adapter.feasible_seen.items():
            existing = candidates_by_bits.get(bits)
            if existing is None or candidate_sort_key(candidate) < candidate_sort_key(existing):
                candidates_by_bits[bits] = candidate

    elapsed = time.perf_counter() - started
    feasible_candidates = sorted(candidates_by_bits.values(), key=candidate_sort_key)
    selected_candidates = feasible_candidates[:requested_count]
    status = "feasible_best_known"
    if len(selected_candidates) < requested_count:
        status = "partial_feasible_best_known"

    configurations = [
        candidate_to_output(feature_problem, candidate, rank=index + 1)
        for index, candidate in enumerate(selected_candidates)
    ]
    raw_qubo_expression = None
    if raw_qubo_expressions:
        raw_qubo_expression = raw_qubo_expressions.get(f"req_{req_no}")
    if raw_qubo_expression is not None:
        attach_qubo_values(configurations, feature_problem, raw_qubo_expression)

    costs = [candidate.cost for candidate in feasible_candidates]
    best_value = selected_candidates[0].cost if selected_candidates else None
    best_solution = list(selected_candidates[0].bits) if selected_candidates else None

    mutation_probability = args.mutation_probability
    if mutation_probability is None:
        mutation_probability = 1.0 / feature_problem.n_variables

    return {
        "req_id": f"req_{req_no}",
        "req_no": req_no,
        "problem": problem_name,
        "benchmark": benchmark,
        "model_index": model_index,
        "objective_direction": "minimize",
        "objective": "total selected feature cost",
        "reference_type": "feasible_best_known",
        "status": status,
        "solver": "jmetalpy_genetic_algorithm",
        "source": "IGDec-QAOA GA baseline, https://zenodo.org/records/13911651",
        "features": [
            {"name": feature.name, "cost": feature.cost}
            for feature in feature_problem.features
        ],
        "constraint_count": feature_problem.total_constraints,
        "penalty": feature_problem.penalty,
        "requested_configurations": requested_count,
        "returned_configurations": len(configurations),
        "best_value": best_value,
        "best_solution": best_solution,
        "configurations": configurations,
        "ga_summary": {
            "runs": args.runs,
            "population_size": args.population_size,
            "offspring_population_size": args.population_size,
            "max_evaluations_per_run": args.evaluations,
            "total_evaluations": args.runs * args.evaluations,
            "mutation_probability": mutation_probability,
            "crossover_probability": args.crossover_probability,
            "unique_feasible_candidates": len(feasible_candidates),
            "elapsed_s": elapsed,
            "best_cost": min(costs) if costs else None,
            "median_feasible_cost": statistics.median(costs) if costs else None,
            "worst_feasible_cost": max(costs) if costs else None,
            "run_summaries": run_summaries,
        },
    }


def compact_result(detailed_result: dict[str, object]) -> dict[str, object]:
    configurations = detailed_result["configurations"]
    ga_summary = detailed_result.get("ga_summary", {})
    wallclock_time = None
    if isinstance(ga_summary, dict):
        wallclock_time = ga_summary.get("elapsed_s")
    if not isinstance(configurations, list) or not configurations:
        return {
            "best_solution": None,
            "ga_fitness_val": None,
            "qubo_val": None,
            "bitstring": None,
            "assignment": {},
            "selected_features": [],
            "status": detailed_result["status"],
            "wallclock_time": wallclock_time,
        }

    first = configurations[0]
    if not isinstance(first, dict):
        raise ValueError("configuration entry must be a JSON object")

    best_solution = first["solution_bits"]
    feature_names = [feature["name"] for feature in detailed_result["features"]]
    assignment = dict(zip(feature_names, best_solution))
    compact = {
        "best_solution": best_solution,
        "ga_fitness_val": first["fitness"],
        "qubo_val": first.get("qubo_value"),
        "bitstring": qiskit_style_bitstring(best_solution),
        "assignment": assignment,
        "selected_features": first["selected_features"],
        "wallclock_time": wallclock_time,
    }

    if detailed_result["problem"] == "configuration prioritization":
        compact["top_configurations"] = [
            {
                "bitstring": qiskit_style_bitstring(configuration["solution_bits"]),
                "assignment": dict(zip(feature_names, configuration["solution_bits"])),
                "selected_features": configuration["selected_features"],
            }
            for configuration in configurations
        ]

    return compact


def qiskit_style_bitstring(bits: Iterable[int]) -> str:
    return "".join(str(int(bit)) for bit in reversed(list(bits)))


def candidate_to_output(
    problem: FeatureModelProblem,
    candidate: ConfigurationCandidate,
    rank: int,
) -> dict[str, object]:
    return {
        "rank": rank,
        "cost": candidate.cost,
        "fitness": candidate.fitness,
        "valid": candidate.unsatisfied_constraints == 0,
        "unsatisfied_constraints": candidate.unsatisfied_constraints,
        "satisfied_constraints": problem.total_constraints - candidate.unsatisfied_constraints,
        "total_constraints": problem.total_constraints,
        "solution_bits": list(candidate.bits),
        "selected_features": problem.selected_features(candidate.bits),
    }


def attach_qubo_values(
    configurations: list[dict[str, object]],
    problem: FeatureModelProblem,
    raw_qubo_expression: object,
) -> None:
    feature_names = [feature.name for feature in problem.features]
    for configuration in configurations:
        bits = configuration["solution_bits"]
        if not isinstance(bits, list):
            continue
        assignment = dict(zip(feature_names, [int(bit) for bit in bits]))
        try:
            qubo = evaluate_qubo_for_assignment(raw_qubo_expression, assignment)
        except (KeyError, TypeError, ValueError):
            continue
        configuration["qubo_value"] = qubo["full_matrix_value"]
        if qubo["matrix_convention"] == "upper":
            configuration["qubo_value"] = qubo["upper_triangular_value"]
        configuration["qubo"] = qubo

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GA reference data for configuration optimization requirements.",
    )
    default_repo_root = repo_root_from_script()
    parser.add_argument(
        "--repo-root",
        default=default_repo_root,
        help="repository root; defaults to three directories above this script",
    )
    parser.add_argument(
        "--req",
        nargs="*",
        type=int,
        help="specific requirement numbers to solve, e.g. --req 20 25",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            default_repo_root,
            "empirical_study",
            "classical_reference",
            "reference_data",
            "configuration",
        ),
        help=(
            "output directory, or a legacy JSON path whose parent directory "
            "will be used; writes [REQ_ID]_reference.json files"
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--evaluations", type=int, default=400000)
    parser.add_argument("--mutation-probability", type=float, default=None)
    parser.add_argument("--crossover-probability", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    logging.getLogger("jmetal").setLevel(logging.WARNING)
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    requirements = load_configuration_requirements(repo_root)
    raw_qubo_expressions = load_raw_qubo_expressions(repo_root)

    selected_req_nos = args.req if args.req else list(requirements)
    unknown = [req_no for req_no in selected_req_nos if req_no not in requirements]
    if unknown:
        raise ValueError(f"Unknown or non-configuration requirement numbers: {unknown}")

    results = []
    for req_no in selected_req_nos:
        print(f"Solving req_{req_no} with GA...")
        results.append(
            solve_requirement(
                repo_root,
                req_no,
                requirements[req_no],
                args,
                raw_qubo_expressions=raw_qubo_expressions,
            )
        )

    output = {
        str(result["req_id"]): compact_result(result)
        for result in results
    }
    write_reference_outputs(args.out, output)
    out_dir = os.path.dirname(os.path.abspath(args.out)) if args.out.lower().endswith(".json") else os.path.abspath(args.out)
    print(f"Wrote {len(output)} reference files to {out_dir}")


if __name__ == "__main__":
    main()
