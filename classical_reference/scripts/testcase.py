"""GA reference runner for test-case optimization requirements."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution

try:
    from .ga import GeneticAlgorithmConfig, run_binary_ga
    from .testcase_qubo import evaluate_qubo_for_bits, load_raw_qubo_expressions
except ImportError:
    from ga import GeneticAlgorithmConfig, run_binary_ga
    from testcase_qubo import evaluate_qubo_for_bits, load_raw_qubo_expressions


TESTCASE_PROBLEMS = {
    "test case selection",
    "test case minimization",
}


@dataclass(frozen=True)
class TestCase:
    id: str
    values: dict[str, float]


@dataclass(frozen=True)
class TestcaseCandidate:
    bits: tuple[int, ...]
    fitness: float
    selected_count: int


class TestcaseOptimizationProblem:
    def __init__(self, test_cases: list[TestCase], problem_name: str) -> None:
        self.test_cases = test_cases
        self.problem_name = problem_name
        self.objective_names = [
            name
            for name in test_cases[0].values
            if name != "id"
        ]
        self.minimize_names, self.maximize_names = objective_groups(self.objective_names)
        self.sums = {
            name: sum(test_case.values.get(name, 0.0) for test_case in test_cases)
            for name in self.objective_names
        }
        self.penalty = max(sum(abs(value) for value in self.sums.values()), 1.0)

    @property
    def n_variables(self) -> int:
        return len(self.test_cases)

    def evaluate(self, bits: tuple[int, ...]) -> TestcaseCandidate:
        if self.problem_name == "test case minimization":
            fitness = self._minimization_fitness(bits)
        else:
            fitness = self._selection_fitness(bits)
        return TestcaseCandidate(
            bits=bits,
            fitness=fitness,
            selected_count=sum(bits),
        )

    def selected_ids(self, bits: tuple[int, ...]) -> list[str]:
        return [
            test_case.id
            for test_case, bit in zip(self.test_cases, bits)
            if bit
        ]

    def _selection_fitness(self, bits: tuple[int, ...]) -> float:
        selected_count = sum(bits)
        terms = []
        for name in self.minimize_names:
            total = self._selected_sum(bits, name)
            terms.append((total / self._denominator(name)) ** 2)
        for name in self.maximize_names:
            selected = self._selected_sum(bits, name)
            target = self._denominator(name)
            terms.append(((selected - target) / target) ** 2)
        if len(self.objective_names) == 2 and "input_div" in self.maximize_names:
            pass
        else:
            terms.append((selected_count / max(self.n_variables, 1)) ** 2)
        return sum(terms) / max(len(terms), 1)

    def _minimization_fitness(self, bits: tuple[int, ...]) -> float:
        selected_count = sum(bits)
        terms = [(selected_count / max(self.n_variables, 1)) ** 2]
        for name in self.minimize_names:
            total = self._selected_sum(bits, name)
            terms.append((total / self._denominator(name)) ** 2)
        penalty = 0.0
        for name in self.maximize_names:
            selected = self._selected_sum(bits, name)
            target = self._denominator(name)
            shortfall = max(0.0, target - selected) / target
            penalty += shortfall * shortfall
        return sum(terms) / max(len(terms), 1) + self.penalty * penalty

    def _selected_sum(self, bits: tuple[int, ...], name: str) -> float:
        return sum(
            test_case.values.get(name, 0.0)
            for test_case, bit in zip(self.test_cases, bits)
            if bit
        )

    def _denominator(self, name: str) -> float:
        return abs(self.sums.get(name, 0.0)) + 1e-20


class TestcaseBinaryProblem(BinaryProblem):
    def __init__(self, testcase_problem: TestcaseOptimizationProblem) -> None:
        super().__init__()
        self.testcase_problem = testcase_problem
        self.number_of_bits = testcase_problem.n_variables
        self.number_of_bits_per_variable = [self.number_of_bits]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["testcase_ga_fitness"]
        self.best_candidate: TestcaseCandidate | None = None

    def number_of_variables(self) -> int:
        return self.number_of_bits

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        bits = tuple(1 if bit else 0 for bit in solution.bits.tolist())
        candidate = self.testcase_problem.evaluate(bits)
        solution.objectives[0] = candidate.fitness
        solution.attributes["candidate"] = candidate
        if self.best_candidate is None or candidate_sort_key(candidate) < candidate_sort_key(self.best_candidate):
            self.best_candidate = candidate
        return solution

    def create_solution(self) -> BinarySolution:
        solution = BinarySolution(
            number_of_variables=self.number_of_bits,
            number_of_objectives=self.number_of_objectives(),
        )
        solution.bits = np.random.choice([True, False], size=self.number_of_bits)
        return solution

    def name(self) -> str:
        return "TestcaseOptimization"


def candidate_sort_key(candidate: TestcaseCandidate) -> tuple[float, int, tuple[int, ...]]:
    return (candidate.fitness, candidate.selected_count, candidate.bits)


def objective_groups(names: list[str]) -> tuple[list[str], list[str]]:
    minimize = [
        name
        for name in names
        if name in {"cost", "time", "execution_cost"}
    ]
    maximize = [name for name in names if name not in minimize]
    if not minimize and names:
        minimize = [names[0]]
        maximize = names[1:]
    return minimize, maximize


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


def load_test_cases(repo_root: str, benchmark: str) -> list[TestCase]:
    path = os.path.join(repo_root, "use_cases", "benchmark", "json", f"{benchmark}.json")
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Invalid benchmark JSON: {path}")
    test_cases = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid test case entry in {path}")
        test_id = str(item.get("id", f"t{index}"))
        values = {
            str(key): float(value)
            for key, value in item.items()
            if key != "id" and isinstance(value, (int, float))
        }
        test_cases.append(TestCase(id=test_id, values=values))
    if not test_cases:
        raise ValueError(f"Benchmark has no test cases: {path}")
    return test_cases


def load_testcase_requirements(repo_root: str) -> dict[int, dict[str, object]]:
    overview_path = os.path.join(repo_root, "use_cases", "requirements", "req_overview.json")
    overview = load_json(overview_path)
    if not isinstance(overview, dict):
        raise ValueError("req_overview.json must contain a JSON object")
    requirements: dict[int, dict[str, object]] = {}
    for req_id_text, metadata in overview.items():
        if isinstance(metadata, dict) and metadata.get("problem") in TESTCASE_PROBLEMS:
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
    test_cases = load_test_cases(repo_root, benchmark)
    testcase_problem = TestcaseOptimizationProblem(test_cases, problem_name)

    best_candidates: list[TestcaseCandidate] = []
    run_summaries = []
    started = time.perf_counter()
    for run_index in range(args.runs):
        problem_adapter = TestcaseBinaryProblem(testcase_problem)
        seed = None
        if args.seed is not None:
            seed = args.seed + req_no * 1009 + run_index
        run = run_binary_ga(
            problem_adapter,
            GeneticAlgorithmConfig(
                population_size=args.population_size,
                max_evaluations=args.evaluations,
                crossover_probability=args.crossover_probability,
                mutation_probability=args.mutation_probability,
                seed=seed,
            ),
        )
        run_summaries.append(
            {
                "evaluations": run.evaluations,
                "elapsed_s": run.elapsed_s,
                "best_fitness": float(run.best_solution.objectives[0]),
            }
        )
        if problem_adapter.best_candidate is not None:
            best_candidates.append(problem_adapter.best_candidate)

    elapsed = time.perf_counter() - started
    if not best_candidates:
        raise RuntimeError(f"GA did not evaluate any candidates for req_{req_no}")
    best = min(best_candidates, key=candidate_sort_key)
    qubo_val = None
    raw_qubo_expression = None
    if raw_qubo_expressions:
        raw_qubo_expression = raw_qubo_expressions.get(f"req_{req_no}")
    if raw_qubo_expression is not None:
        qubo_val = evaluate_qubo_for_bits(raw_qubo_expression, list(best.bits))

    return {
        "req_id": f"req_{req_no}",
        "problem": problem_name,
        "benchmark": benchmark,
        "best_solution": list(best.bits),
        "ga_fitness_val": best.fitness,
        "qubo_val": qubo_val,
        "bitstring": qiskit_style_bitstring(best.bits),
        "selected_test_cases": testcase_problem.selected_ids(best.bits),
        "summary": {
            "runs": args.runs,
            "population_size": args.population_size,
            "offspring_population_size": args.population_size,
            "max_evaluations_per_run": args.evaluations,
            "total_evaluations": args.runs * args.evaluations,
            "crossover_probability": args.crossover_probability,
            "mutation_probability": (
                args.mutation_probability
                if args.mutation_probability is not None
                else 1.0 / testcase_problem.n_variables
            ),
            "elapsed_s": elapsed,
            "run_summaries": run_summaries,
        },
    }


def compact_result(detailed_result: dict[str, object]) -> dict[str, object]:
    summary = detailed_result.get("summary", {})
    wallclock_time = None
    if isinstance(summary, dict):
        wallclock_time = summary.get("elapsed_s")
    return {
        "best_solution": detailed_result["best_solution"],
        "ga_fitness_val": detailed_result["ga_fitness_val"],
        "qubo_val": detailed_result["qubo_val"],
        "bitstring": detailed_result["bitstring"],
        "selected_test_cases": detailed_result["selected_test_cases"],
        "wallclock_time": wallclock_time,
    }


def qiskit_style_bitstring(bits: Iterable[int]) -> str:
    return "".join(str(int(bit)) for bit in reversed(list(bits)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GA reference data for test-case optimization requirements.",
    )
    default_repo_root = repo_root_from_script()
    parser.add_argument("--repo-root", default=default_repo_root)
    parser.add_argument("--req", nargs="*", type=int)
    parser.add_argument(
        "--out",
        default=os.path.join(
            default_repo_root,
            "empirical_study",
            "classical_reference",
            "reference_data",
            "testcase",
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
    requirements = load_testcase_requirements(repo_root)
    raw_qubo_expressions = load_raw_qubo_expressions(repo_root)

    selected_req_nos = args.req if args.req else list(requirements)
    unknown = [req_no for req_no in selected_req_nos if req_no not in requirements]
    if unknown:
        raise ValueError(f"Unknown or non-testcase requirement numbers: {unknown}")

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

    output = {str(result["req_id"]): compact_result(result) for result in results}
    write_reference_outputs(args.out, output)
    out_dir = os.path.dirname(os.path.abspath(args.out)) if args.out.lower().endswith(".json") else os.path.abspath(args.out)
    print(f"Wrote {len(output)} reference files to {out_dir}")


if __name__ == "__main__":
    main()
