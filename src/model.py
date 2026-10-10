from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any


ACTIONS = ("up", "down", "left", "right", "stay")
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}
KINDS = ("normal", "start", "target", "forbidden")
DEFAULT_REWARDS = {
    "normal": 0.0,
    "start": 0.0,
    "target": 1.0,
    "forbidden": -1.0,
}
MAX_DIMENSION = 100
PROBABILITY_TOLERANCE = 1e-6


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _validate_probability_values(
    probabilities: object, context: str = "probabilities"
) -> None:
    if not isinstance(probabilities, list):
        raise ValueError(f"{context} must be a list of five numbers.")
    if len(probabilities) != len(ACTIONS):
        raise ValueError(f"{context} must contain exactly five values.")

    total = 0.0
    for index, probability in enumerate(probabilities):
        if not _is_finite_number(probability):
            raise ValueError(f"{context}[{index}] must be a finite number.")
        if probability < 0 or probability > 1:
            raise ValueError(f"{context}[{index}] must be between 0 and 1.")
        total += probability

    if abs(total - 1.0) > PROBABILITY_TOLERANCE:
        raise ValueError(f"{context} must sum to 1 within the allowed tolerance.")


def _validate_coordinate_component(value: object, name: str) -> None:
    if not _is_integer(value):
        raise ValueError(f"{name} must be an integer.")


def _validate_grid_dimension(value: object, name: str) -> None:
    _validate_coordinate_component(value, name)
    if value < 1 or value > MAX_DIMENSION:
        raise ValueError(f"{name} must be between 1 and {MAX_DIMENSION}.")


def _validate_finite_range(
    value: object, name: str, minimum: float, maximum: float
) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


@dataclass
class CellConfig:
    kind: str = "normal"
    reward: float = 0.0
    probabilities: list[float] = field(
        default_factory=lambda: [1.0 / len(ACTIONS)] * len(ACTIONS)
    )

    def validate(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}.")
        if not _is_finite_number(self.reward):
            raise ValueError("reward must be a finite number.")
        _validate_probability_values(self.probabilities)


@dataclass
class GridConfig:
    rows: int
    columns: int
    cell_width_mm: float
    cell_height_mm: float
    gamma: float
    zoom: float
    cells: list[list[CellConfig]]

    @classmethod
    def create(
        cls,
        rows: int = 5,
        columns: int = 5,
        cell_width_mm: float = 10.0,
        cell_height_mm: float = 10.0,
        gamma: float = 0.9,
        zoom: float = 3.0,
    ) -> GridConfig:
        _validate_grid_dimension(rows, "rows")
        _validate_grid_dimension(columns, "columns")
        if rows == 1 and columns == 1:
            raise ValueError("A 1x1 grid is not supported.")
        _validate_finite_range(cell_width_mm, "cell_width_mm", 0.01, 1000.0)
        _validate_finite_range(cell_height_mm, "cell_height_mm", 0.01, 1000.0)
        _validate_finite_range(gamma, "gamma", 0.0, 1.0)
        _validate_finite_range(zoom, "zoom", 0.001, 10.0)

        cells = [[CellConfig() for _ in range(columns)] for _ in range(rows)]
        cells[0][0] = CellConfig(kind="start", reward=DEFAULT_REWARDS["start"])
        cells[rows - 1][columns - 1] = CellConfig(
            kind="target", reward=DEFAULT_REWARDS["target"]
        )
        config = cls(
            rows=rows,
            columns=columns,
            cell_width_mm=cell_width_mm,
            cell_height_mm=cell_height_mm,
            gamma=gamma,
            zoom=zoom,
            cells=cells,
        )
        config.validate()
        return config

    def validate(self) -> None:
        _validate_grid_dimension(self.rows, "rows")
        _validate_grid_dimension(self.columns, "columns")
        if self.rows == 1 and self.columns == 1:
            raise ValueError("A 1x1 grid is not supported.")
        _validate_finite_range(self.cell_width_mm, "cell_width_mm", 0.01, 1000.0)
        _validate_finite_range(self.cell_height_mm, "cell_height_mm", 0.01, 1000.0)
        _validate_finite_range(self.gamma, "gamma", 0.0, 1.0)
        _validate_finite_range(self.zoom, "zoom", 0.001, 10.0)

        if not isinstance(self.cells, list):
            raise ValueError("cells must be a list of rows.")
        if len(self.cells) != self.rows:
            raise ValueError("cells does not match rows.")

        start_count = 0
        target_count = 0
        for row_index, row in enumerate(self.cells):
            if not isinstance(row, list):
                raise ValueError(f"cells[{row_index}] must be a list.")
            if len(row) != self.columns:
                raise ValueError(f"cells[{row_index}] does not match columns.")
            for column_index, cell in enumerate(row):
                if not isinstance(cell, CellConfig):
                    raise ValueError(
                        f"cells[{row_index}][{column_index}] must be a CellConfig."
                    )
                cell.validate()
                if cell.kind == "start":
                    start_count += 1
                elif cell.kind == "target":
                    target_count += 1

        if start_count != 1:
            raise ValueError("The grid must contain exactly one start cell.")
        if target_count != 1:
            raise ValueError("The grid must contain exactly one target cell.")

    def _role_position(self, kind: str) -> tuple[int, int]:
        for row_index, row in enumerate(self.cells):
            for column_index, cell in enumerate(row):
                if cell.kind == kind:
                    return row_index, column_index
        raise ValueError(f"The grid has no {kind} cell.")

    @property
    def start(self) -> tuple[int, int]:
        return self._role_position("start")

    @property
    def target(self) -> tuple[int, int]:
        return self._role_position("target")

    def _validate_coordinate(self, row: object, column: object) -> None:
        _validate_coordinate_component(row, "row")
        _validate_coordinate_component(column, "column")
        if row < 0 or row >= self.rows:
            raise ValueError("row is outside the grid.")
        if column < 0 or column >= self.columns:
            raise ValueError("column is outside the grid.")

    def set_cell(
        self,
        row: int,
        column: int,
        kind: str,
        reward: float,
        probabilities: list[float],
    ) -> None:
        self.validate()
        self._validate_coordinate(row, column)

        replacement = CellConfig(
            kind=kind,
            reward=reward,
            probabilities=probabilities,
        )
        replacement.validate()

        current = self.cells[row][column]
        if current.kind in ("start", "target") and kind != current.kind:
            raise ValueError(
                f"Cannot change the existing {current.kind} cell; "
                f"move the {current.kind} role first."
            )

        replacement.probabilities = list(probabilities)
        if kind in ("start", "target") and kind != current.kind:
            former_row, former_column = self._role_position(kind)
            former = self.cells[former_row][former_column]
            former_replacement = CellConfig(
                kind="normal",
                reward=DEFAULT_REWARDS["normal"],
                probabilities=list(former.probabilities),
            )
            former_replacement.validate()
            # The complete edit is valid and preserves both unique roles.
            self.cells[former_row][former_column] = former_replacement
        self.cells[row][column] = replacement

    def apply_policy_to_all(self, probabilities: list[float]) -> None:
        self.validate()
        _validate_probability_values(probabilities)
        copied = list(probabilities)
        for row in self.cells:
            for cell in row:
                cell.probabilities = list(copied)

    def transition(
        self, row: int, column: int, action: str
    ) -> tuple[tuple[int, int], float]:
        """Return the next coordinate and destination reward.

        Call validate() once before repeated transitions on this config.
        """
        self._validate_coordinate(row, column)
        if not isinstance(action, str) or action not in ACTION_DELTAS:
            raise ValueError(f"action must be one of {ACTIONS}.")

        delta_row, delta_column = ACTION_DELTAS[action]
        next_row = min(max(row + delta_row, 0), self.rows - 1)
        next_column = min(max(column + delta_column, 0), self.columns - 1)
        return (next_row, next_column), self.cells[next_row][next_column].reward


def save_config(config: GridConfig, path: os.PathLike[str] | str) -> None:
    if not isinstance(config, GridConfig):
        raise TypeError("config must be a GridConfig.")
    config.validate()
    data = {
        "format_version": 1,
        "rows": config.rows,
        "columns": config.columns,
        "cell_width_mm": config.cell_width_mm,
        "cell_height_mm": config.cell_height_mm,
        "gamma": config.gamma,
        "zoom": config.zoom,
        "cells": [
            [
                {
                    "kind": cell.kind,
                    "reward": cell.reward,
                    "probabilities": {
                        action: cell.probabilities[index]
                        for index, action in enumerate(ACTIONS)
                    },
                }
                for cell in row
            ]
            for row in config.cells
        ],
    }
    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination) or os.curdir
    prefix = f".{os.path.basename(destination)}."
    temporary_path: str | None = None

    try:
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=prefix,
            suffix=".tmp",
            delete=False,
        )
        temporary_path = temporary_file.name
        with temporary_file:
            json.dump(
                data, temporary_file, indent=2, ensure_ascii=True, allow_nan=False
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value {value} is not allowed.")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has non-canonical keys.")


def load_config(path: os.PathLike[str] | str) -> GridConfig:
    with open(os.fspath(path), "r", encoding="utf-8") as input_file:
        data = json.load(
            input_file,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    top_level = _require_mapping(data, "top-level JSON value")
    _require_exact_keys(
        top_level,
        {
            "format_version",
            "rows",
            "columns",
            "cell_width_mm",
            "cell_height_mm",
            "gamma",
            "zoom",
            "cells",
        },
        "top-level object",
    )

    format_version = top_level["format_version"]
    if not _is_integer(format_version) or format_version != 1:
        raise ValueError("format_version must be the integer 1.")

    rows = top_level["rows"]
    columns = top_level["columns"]
    _validate_grid_dimension(rows, "rows")
    _validate_grid_dimension(columns, "columns")
    if rows == 1 and columns == 1:
        raise ValueError("A 1x1 grid is not supported.")
    cell_width_mm = top_level["cell_width_mm"]
    cell_height_mm = top_level["cell_height_mm"]
    gamma = top_level["gamma"]
    zoom = top_level["zoom"]
    _validate_finite_range(cell_width_mm, "cell_width_mm", 0.01, 1000.0)
    _validate_finite_range(cell_height_mm, "cell_height_mm", 0.01, 1000.0)
    _validate_finite_range(gamma, "gamma", 0.0, 1.0)
    _validate_finite_range(zoom, "zoom", 0.001, 10.0)

    raw_cells = top_level["cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) != rows:
        raise ValueError("cells does not match rows.")

    cells: list[list[CellConfig]] = []
    expected_cell_keys = {"kind", "reward", "probabilities"}
    expected_probability_keys = set(ACTIONS)
    for row_index, raw_row in enumerate(raw_cells):
        if not isinstance(raw_row, list) or len(raw_row) != columns:
            raise ValueError(f"cells[{row_index}] does not match columns.")
        loaded_row: list[CellConfig] = []
        for column_index, raw_cell in enumerate(raw_row):
            cell_data = _require_mapping(
                raw_cell, f"cells[{row_index}][{column_index}]"
            )
            _require_exact_keys(
                cell_data,
                expected_cell_keys,
                f"cells[{row_index}][{column_index}]",
            )
            probability_data = _require_mapping(
                cell_data["probabilities"],
                f"cells[{row_index}][{column_index}].probabilities",
            )
            _require_exact_keys(
                probability_data,
                expected_probability_keys,
                f"cells[{row_index}][{column_index}].probabilities",
            )
            cell = CellConfig(
                kind=cell_data["kind"],
                reward=cell_data["reward"],
                probabilities=[probability_data[action] for action in ACTIONS],
            )
            try:
                cell.validate()
            except ValueError as error:
                raise ValueError(
                    f"Invalid cell at ({row_index}, {column_index}): {error}"
                ) from error
            loaded_row.append(cell)
        cells.append(loaded_row)

    config = GridConfig(
        rows=rows,
        columns=columns,
        cell_width_mm=cell_width_mm,
        cell_height_mm=cell_height_mm,
        gamma=gamma,
        zoom=zoom,
        cells=cells,
    )
    config.validate()
    return config


@dataclass
class PathResult:
    coordinates: list[tuple[int, int]]
    score: float


@dataclass
class CalculationResult:
    values: list[list[float]]
    optimal_policy: list[list[list[float]]] | None = None
    best_paths: list[PathResult] = field(default_factory=list)

    def validate(self, config: GridConfig) -> None:
        if not isinstance(config, GridConfig):
            raise TypeError("config must be a GridConfig.")
        config.validate()

        if not isinstance(self.values, list) or len(self.values) != config.rows:
            raise ValueError("values does not match the grid rows.")
        for row_index, row in enumerate(self.values):
            if not isinstance(row, list) or len(row) != config.columns:
                raise ValueError(
                    f"values[{row_index}] does not match the grid columns."
                )
            for column_index, value in enumerate(row):
                if not _is_finite_number(value):
                    raise ValueError(
                        f"values[{row_index}][{column_index}] must be finite."
                    )

        if self.optimal_policy is not None:
            if not isinstance(self.optimal_policy, list) or len(
                self.optimal_policy
            ) != config.rows:
                raise ValueError("optimal_policy does not match the grid rows.")
            for row_index, row in enumerate(self.optimal_policy):
                if not isinstance(row, list) or len(row) != config.columns:
                    raise ValueError(
                        f"optimal_policy[{row_index}] does not match the grid columns."
                    )
                for column_index, probabilities in enumerate(row):
                    _validate_probability_values(
                        probabilities,
                        f"optimal_policy[{row_index}][{column_index}]",
                    )

        if not isinstance(self.best_paths, list):
            raise ValueError("best_paths must be a list.")
        start, target = config.start, config.target
        for path_index, path in enumerate(self.best_paths):
            if not isinstance(path, PathResult):
                raise ValueError(f"best_paths[{path_index}] must be a PathResult.")
            if not _is_finite_number(path.score):
                raise ValueError(f"best_paths[{path_index}].score must be finite.")
            if not isinstance(path.coordinates, list) or not path.coordinates:
                raise ValueError(
                    f"best_paths[{path_index}].coordinates must be non-empty."
                )

            for coordinate_index, coordinate in enumerate(path.coordinates):
                if not isinstance(coordinate, tuple) or len(coordinate) != 2:
                    raise ValueError(
                        f"best_paths[{path_index}].coordinates"
                        f"[{coordinate_index}] must be a coordinate tuple."
                    )
                row, column = coordinate
                if not _is_integer(row) or not _is_integer(column):
                    raise ValueError(
                        f"best_paths[{path_index}].coordinates"
                        f"[{coordinate_index}] must contain integers."
                    )
                if not (0 <= row < config.rows and 0 <= column < config.columns):
                    raise ValueError(
                        f"best_paths[{path_index}].coordinates"
                        f"[{coordinate_index}] is outside the grid."
                    )
            if path.coordinates[0] != start:
                raise ValueError(
                    f"best_paths[{path_index}] must start at the start cell."
                )
            if path.coordinates[-1] != target:
                raise ValueError(
                    f"best_paths[{path_index}] must end at the target cell."
                )
            for previous, current in zip(
                path.coordinates, path.coordinates[1:]
            ):
                delta = (current[0] - previous[0], current[1] - previous[1])
                if delta not in ACTION_DELTAS.values():
                    raise ValueError(
                        f"best_paths[{path_index}] contains a non-adjacent move."
                    )
