from __future__ import annotations

import math
from collections.abc import Callable, Iterator

from model import ACTIONS, CalculationResult, GridConfig, PathResult


class CalculationCancelled(Exception):
    """Raised by a checkpoint when the caller stops a calculation."""


def _no_checkpoint() -> None:
    pass


def _update_values(
    config: GridConfig,
    values: list[list[float]],
    optimal: bool,
    checkpoint: Callable[[], None],
) -> tuple[list[list[float]], list[list[list[float]]] | None]:
    next_values = [[0.0] * config.columns for _ in range(config.rows)]
    policy = (
        [[[0.0] * len(ACTIONS) for _ in range(config.columns)] for _ in range(config.rows)]
        if optimal else None
    )
    for i in range(config.rows):
        checkpoint()
        for j in range(config.columns):
            action_values = {}
            value = 0.0
            for k, probability in enumerate(config.cells[i][j].probabilities):
                if probability == 0:
                    continue
                (next_i, next_j), reward = config.transition(i, j, ACTIONS[k])
                action_value = reward + config.gamma * values[next_i][next_j]
                if optimal:
                    action_values[k] = action_value
                else:
                    value += probability * action_value
            if optimal:
                value = max(action_values.values())
                best_actions = [
                    k for k, action_value in action_values.items()
                    if math.isclose(action_value, value, rel_tol=0.0, abs_tol=1e-9)
                ]
                for k in best_actions:
                    policy[i][j][k] = 1.0 / len(best_actions)
            next_values[i][j] = value
    return next_values, policy


def _find_paths(
    config: GridConfig,
    policy: list[list[list[float]]],
    path_max_count: int,
    checkpoint: Callable[[], None],
) -> list[PathResult]:
    """Return all simple goal paths in descending arrival-return order."""
    target = config.target
    pending = [([config.start], 0.0)]
    paths = []
    while pending:
        checkpoint()
        coordinates, score = pending.pop()
        row, column = coordinates[-1]
        if (row, column) == target:
            paths.append(PathResult(coordinates=coordinates, score=score))
            continue
        step = len(coordinates) - 1
        if step >= path_max_count:
            continue
        for k in reversed(range(len(ACTIONS))):
            if policy[row][column][k] == 0:
                continue
            next_cell, reward = config.transition(row, column, ACTIONS[k])
            if next_cell not in coordinates:
                pending.append((coordinates + [next_cell], score + config.gamma ** step * reward))
    checkpoint()
    paths.sort(key=lambda path: path.score, reverse=True)
    checkpoint()
    return paths


def iterate_calculation(
    config: GridConfig,
    max_count: int = 100,
    *,
    optimal: bool = False,
    intermediate: bool = True,
    path_max_count: int | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> Iterator[tuple[int, CalculationResult]]:
    """Yield independent whole-board rounds, with paths for each emitted policy.

    With intermediate=False, only the last round builds paths and is yielded.
    Target cells continue acting. Paths follow the emitted policy throughout;
    path scores stop at first target arrival and exclude later rewards. The
    path length limit is independent of the value horizon and defaults to the
    maximum simple-path length (number of cells minus one).
    """
    check = checkpoint if checkpoint is not None else _no_checkpoint
    check()
    config.validate()
    if isinstance(max_count, bool) or not isinstance(max_count, int) or not 1 <= max_count <= 1000:
        raise ValueError("计算步数必须是 1 到 1000 之间的整数。")
    if path_max_count is None:
        path_max_count = config.rows * config.columns - 1
    if isinstance(path_max_count, bool) or not isinstance(path_max_count, int) or path_max_count < 1:
        raise ValueError("路径最大步数必须是大于 0 的整数。")

    values = [[0.0] * config.columns for _ in range(config.rows)]
    for count in range(1, max_count + 1):
        values, policy = _update_values(config, values, optimal, check)
        if intermediate or count == max_count:
            paths = _find_paths(config, policy, path_max_count, check) if optimal else []
            best_paths = []
            for path in paths:
                check()
                if not math.isclose(path.score, paths[0].score, rel_tol=0.0, abs_tol=1e-9):
                    break
                best_paths.append(path)
            check()
            result = CalculationResult(
                values=values, optimal_policy=policy, best_paths=best_paths, all_paths=paths,
            )
            result.validate(config)
            check()
            yield count, result


def evaluate_policy(
    config: GridConfig,
    max_count: int = 100,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> CalculationResult:
    """Evaluate the configured policy for exactly max_count future steps."""
    return next(iterate_calculation(config, max_count, intermediate=False, checkpoint=checkpoint))[1]


def solve_optimal_policy(
    config: GridConfig,
    max_count: int = 100,
    *,
    path_max_count: int | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> CalculationResult:
    """Compute finite-step optimal values and paths in the final policy."""
    return next(iterate_calculation(
        config, max_count, optimal=True, intermediate=False,
        path_max_count=path_max_count, checkpoint=checkpoint,
    ))[1]
