# Copyright (c) 2026 QJump.ai
# SPDX-License-Identifier: MIT

"""Reusable GA wrapper following the IGDec-QAOA classical baseline.

Original artifact source: https://zenodo.org/records/13911651

The IGDec-QAOA baseline uses jMetalPy's single-objective GeneticAlgorithm with
binary tournament selection, bit-flip mutation, single-point crossover, equal
population and offspring sizes, and a fixed evaluation budget. This module keeps
that GA shell separate from any repository-specific problem adapter.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.core.problem import BinaryProblem
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 100
    max_evaluations: int = 400000
    crossover_probability: float = 1.0
    mutation_probability: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class GeneticAlgorithmRun:
    best_solution: Any
    elapsed_s: float
    evaluations: int
    population_size: int
    offspring_population_size: int
    crossover_probability: float
    mutation_probability: float


def run_binary_ga(problem: BinaryProblem, config: GeneticAlgorithmConfig) -> GeneticAlgorithmRun:
    if config.population_size < 2:
        raise ValueError("population_size must be at least 2")
    if config.max_evaluations < config.population_size:
        raise ValueError("max_evaluations must be at least population_size")

    number_of_bits = _problem_number_of_bits(problem)
    mutation_probability = config.mutation_probability
    if mutation_probability is None:
        mutation_probability = 1.0 / number_of_bits

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    crossover = SPXCrossover(config.crossover_probability)
    if config.seed is not None and hasattr(crossover, "_rng"):
        crossover._rng = np.random.default_rng(config.seed)  # type: ignore[attr-defined]

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=config.population_size,
        offspring_population_size=config.population_size,
        mutation=BitFlipMutation(mutation_probability),
        crossover=crossover,
        termination_criterion=StoppingByEvaluations(
            max_evaluations=config.max_evaluations,
        ),
    )

    started = time.perf_counter()
    algorithm.run()
    elapsed_s = time.perf_counter() - started

    return GeneticAlgorithmRun(
        best_solution=algorithm.result(),
        elapsed_s=elapsed_s,
        evaluations=config.max_evaluations,
        population_size=config.population_size,
        offspring_population_size=config.population_size,
        crossover_probability=config.crossover_probability,
        mutation_probability=mutation_probability,
    )


def _problem_number_of_bits(problem: BinaryProblem) -> int:
    if hasattr(problem, "number_of_bits"):
        return int(getattr(problem, "number_of_bits"))
    total_number_of_bits = problem.total_number_of_bits()
    if total_number_of_bits <= 0:
        raise ValueError("binary problem must expose at least one bit")
    return int(total_number_of_bits)
