from __future__ import annotations

import math

from model import ACTIONS, CalculationResult, GridConfig, PathResult


def evaluate_policy(config: GridConfig, max_count: int = 100) -> CalculationResult:
    """Evaluate the configured policy for exactly max_count future steps."""
    config.validate()
    if (
        isinstance(max_count, bool)
        or not isinstance(max_count, int)
        or not 1 <= max_count <= 1000
    ):
        raise ValueError("计算步数必须是 1 到 1000 之间的整数。")

    values = [[0.0] * config.columns for _ in range(config.rows)]

    # Each layer uses the previous layer, with zero value after the final step.
    for _ in range(max_count):
        next_values = [[0.0] * config.columns for _ in range(config.rows)]
        for i in range(config.rows):
            for j in range(config.columns):
                value = 0.0
                for k, probability in enumerate(config.cells[i][j].probabilities):
                    if probability == 0:
                        continue
                    (next_i, next_j), reward = config.transition(i, j, ACTIONS[k])
                    value += probability * (
                        reward + config.gamma * values[next_i][next_j]
                    )
                next_values[i][j] = value
        values = next_values

    result = CalculationResult(values=values)
    result.validate(config)
    return result


def solve_optimal_policy(config: GridConfig, max_count: int = 100) -> CalculationResult:
    """Compute finite-step optimal values and paths in the final policy.

    Nonzero input probabilities mark available actions. Displayed paths use
    the final policy, visit each cell once, and stop at their first target visit.
    Their score includes rewards only up to that visit.
    """
    config.validate()
    if (
        isinstance(max_count, bool)
        or not isinstance(max_count, int)
        or not 1 <= max_count <= 1000
    ):
        raise ValueError("计算步数必须是 1 到 1000 之间的整数。")

    values = [[0.0] * config.columns for _ in range(config.rows)]
    optimal_policy = [
        [[0.0] * len(ACTIONS) for _ in range(config.columns)]
        for _ in range(config.rows)
    ]

    # Each layer uses the previous layer, with zero value after the final step.
    for _ in range(max_count):
        next_values = [[0.0] * config.columns for _ in range(config.rows)]
        for i in range(config.rows):
            for j in range(config.columns):
                optimal_policy[i][j] = [0.0] * len(ACTIONS)
        for i in range(config.rows):
            for j in range(config.columns):
                action_values = {}
                for k, probability in enumerate(config.cells[i][j].probabilities):
                    if probability == 0:
                        continue
                    (next_i, next_j), reward = config.transition(i, j, ACTIONS[k])
                    action_values[k] = reward + config.gamma * values[next_i][next_j]
                max_value = max(action_values.values())

                action_indexs = [
                    k
                    for k, value in action_values.items()
                    if math.isclose(value, max_value, rel_tol=0.0, abs_tol=1e-9)
                ]
                for k in action_indexs:
                    optimal_policy[i][j][k] = 1.0 / len(action_indexs)

                next_values[i][j] = max_value
        values = next_values

    PathResults = []
    start_cell = config.start
    target_cell = config.target

    cell_path_list = [[start_cell]]
    target_paths = []
    while cell_path_list:
        new_path = []
        for cell_path in cell_path_list:
            cell = cell_path[-1]
            if cell == target_cell:
                target_paths.append(cell_path)
                continue
            if len(cell_path) - 1 >= max_count:
                continue

            for k, probability in enumerate(optimal_policy[cell[0]][cell[1]]):
                if probability > 0:
                    next_cell, _ = config.transition(cell[0], cell[1], ACTIONS[k])
                    if next_cell not in cell_path:
                        new_path.append(cell_path + [next_cell])
        cell_path_list = new_path

    for path in target_paths:
        score = 0
        for i in range(1, len(path)):
            score += (
                config.gamma ** (i - 1) * config.cells[path[i][0]][path[i][1]].reward
            )
        PathResults.append(PathResult(coordinates=path, score=score))

    if PathResults:
        best_score = max(path.score for path in PathResults)
        PathResults = [
            path
            for path in PathResults
            if math.isclose(
                path.score,
                best_score,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
    result = CalculationResult(
        values=values, optimal_policy=optimal_policy, best_paths=PathResults
    )
    result.validate(config)
    return result
