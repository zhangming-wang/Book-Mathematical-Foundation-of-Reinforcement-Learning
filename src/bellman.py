from __future__ import annotations

from model import ACTIONS, CalculationResult, GridConfig
import math


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

    raise NotImplementedError("最优策略求解功能尚未实现。")
