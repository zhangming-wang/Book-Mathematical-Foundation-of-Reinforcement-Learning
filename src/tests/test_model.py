import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model import (
    ACTIONS,
    CalculationResult,
    CellConfig,
    GridConfig,
    PathResult,
    load_config,
    save_config,
)


class ModelTest(unittest.TestCase):
    def test_defaults_and_independent_cell_lists(self) -> None:
        config = GridConfig.create()
        self.assertEqual((config.rows, config.columns), (5, 5))
        self.assertEqual((config.cell_width_mm, config.cell_height_mm), (10.0, 10.0))
        self.assertEqual((config.gamma, config.zoom), (0.9, 3.0))
        self.assertEqual((config.start, config.target), ((0, 0), (4, 4)))
        self.assertEqual(config.cells[0][0].probabilities, [0.2] * 5)
        self.assertIsNot(config.cells[0], config.cells[1])
        self.assertIsNot(config.cells[0][0], config.cells[0][1])
        self.assertIsNot(
            config.cells[0][0].probabilities, config.cells[0][1].probabilities
        )
        first = CellConfig()
        second = CellConfig()
        first.probabilities[0] = 0.1
        self.assertEqual(second.probabilities, [0.2] * 5)

    def test_metadata_bounds_and_strict_numeric_types(self) -> None:
        GridConfig.create(
            rows=100, columns=100, cell_width_mm=0.01, cell_height_mm=1000,
            gamma=0, zoom=0.001,
        ).validate()
        GridConfig.create(
            rows=1, columns=2, cell_width_mm=1000, cell_height_mm=0.01,
            gamma=1, zoom=10,
        ).validate()
        with self.assertRaises(ValueError):
            GridConfig.create(rows=1, columns=1)

        invalid_fields = {
            "rows": [0, -1, 101, 2.0, "2", True, None],
            "columns": [0, -1, 101, 2.0, "2", False, None],
            "cell_width_mm": [0, 0.009, 1000.01, True, "10", math.nan, math.inf],
            "cell_height_mm": [0, 0.009, 1000.01, False, "10", math.nan, -math.inf],
            "gamma": [-0.001, 1.001, True, "0.9", None, math.nan, math.inf],
            "zoom": [0, 0.0009, 10.01, True, "3", None, math.nan, math.inf],
        }
        for name, bad_values in invalid_fields.items():
            for value in bad_values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        GridConfig.create(**{name: value})
                    config = GridConfig.create(rows=2, columns=2)
                    setattr(config, name, value)
                    with self.assertRaises(ValueError):
                        config.validate()

    def test_cell_kinds_and_rewards_are_validated(self) -> None:
        for kind in ("normal", "start", "target", "forbidden"):
            CellConfig(kind=kind, reward=-3.5).validate()
        for kind in ("unknown", "START", True, None, [], 1):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                CellConfig(kind=kind).validate()
        for reward in (True, False, "1", None, math.nan, math.inf, -math.inf):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                CellConfig(reward=reward).validate()

    def test_grid_structure_and_unique_roles_are_validated(self) -> None:
        valid = GridConfig.create(rows=2, columns=2)
        malformed_cells = [
            None,
            tuple(valid.cells),
            valid.cells[:1],
            [tuple(valid.cells[0]), valid.cells[1]],
            [valid.cells[0][:1], valid.cells[1]],
            [[None, valid.cells[0][1]], valid.cells[1]],
        ]
        for cells in malformed_cells:
            config = copy.deepcopy(valid)
            config.cells = cells
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                config.validate()
        for position, kind in (
            ((0, 0), "normal"), ((1, 1), "normal"),
            ((0, 1), "start"), ((0, 1), "target"),
        ):
            config = copy.deepcopy(valid)
            config.cells[position[0]][position[1]].kind = kind
            with self.subTest(position=position, kind=kind), self.assertRaises(ValueError):
                config.validate()

    def test_transition_boundaries_corners_and_single_axis_grids(self) -> None:
        config = GridConfig.create(rows=3, columns=3)
        self.assertEqual(config.transition(0, 0, "up"), ((0, 0), 0.0))
        self.assertEqual(config.transition(0, 0, "left"), ((0, 0), 0.0))
        self.assertEqual(config.transition(0, 0, "down"), ((1, 0), 0.0))
        self.assertEqual(config.transition(2, 2, "down"), ((2, 2), 1.0))
        self.assertEqual(config.transition(2, 2, "right"), ((2, 2), 1.0))
        self.assertEqual(config.transition(0, 2, "up"), ((0, 2), 0.0))
        self.assertEqual(config.transition(0, 2, "right"), ((0, 2), 0.0))
        self.assertEqual(config.transition(2, 0, "down"), ((2, 0), 0.0))
        self.assertEqual(config.transition(2, 0, "left"), ((2, 0), 0.0))
        for action, expected in zip(
            ACTIONS, ((0, 1), (2, 1), (1, 0), (1, 2), (1, 1))
        ):
            with self.subTest(action=action):
                self.assertEqual(config.transition(1, 1, action), (expected, 0.0))

        one_row = GridConfig.create(rows=1, columns=4)
        self.assertEqual(one_row.start, (0, 0))
        self.assertEqual(one_row.target, (0, 3))
        self.assertEqual(one_row.transition(0, 1, "up"), ((0, 1), 0.0))
        self.assertEqual(one_row.transition(0, 1, "down"), ((0, 1), 0.0))
        self.assertEqual(one_row.transition(0, 2, "right"), ((0, 3), 1.0))

        one_column = GridConfig.create(rows=4, columns=1)
        self.assertEqual(one_column.transition(1, 0, "left"), ((1, 0), 0.0))
        self.assertEqual(one_column.transition(1, 0, "right"), ((1, 0), 0.0))
        self.assertEqual(one_column.transition(2, 0, "down"), ((3, 0), 1.0))

    def test_invalid_transition_coordinates_and_actions(self) -> None:
        config = GridConfig.create(rows=2, columns=2)
        for row, column in (
            (-1, 0), (2, 0), (0, -1), (0, 2), (True, 0), (0, False),
            (0.0, 0), (0, 0.0), ("0", 0), (0, None),
        ):
            with self.subTest(row=row, column=column), self.assertRaises(ValueError):
                config.transition(row, column, "stay")
        for action in ("diagonal", "UP", "", 0, None, True, [], (0, 1)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                config.transition(0, 0, action)

    def test_custom_rewards_and_enterable_forbidden_cells(self) -> None:
        config = GridConfig.create(rows=2, columns=3)
        probabilities = [1.0, 0.0, 0.0, 0.0, 0.0]
        config.set_cell(0, 1, "forbidden", -7.5, probabilities)
        config.set_cell(0, 0, "start", 2.5, probabilities)
        config.set_cell(1, 2, "target", 8.0, probabilities)

        self.assertEqual(config.transition(0, 0, "right"), ((0, 1), -7.5))
        self.assertEqual(config.transition(0, 0, "up"), ((0, 0), 2.5))
        self.assertEqual(config.transition(0, 0, "stay"), ((0, 0), 2.5))
        self.assertEqual(config.transition(0, 1, "up"), ((0, 1), -7.5))
        self.assertEqual(config.transition(0, 1, "stay"), ((0, 1), -7.5))
        self.assertEqual(config.transition(0, 1, "down"), ((1, 1), 0.0))
        self.assertEqual(config.transition(1, 1, "right"), ((1, 2), 8.0))
        self.assertEqual(config.transition(1, 2, "left"), ((1, 1), 0.0))

    def test_role_edits_move_roles_and_reject_direct_role_changes_atomically(self) -> None:
        config = GridConfig.create(rows=2, columns=3)
        original_start_policy = [0.0, 1.0, 0.0, 0.0, 0.0]
        config.set_cell(0, 0, "start", 3.0, original_start_policy)
        config.set_cell(0, 1, "start", 4.0, [0.2] * 5)
        self.assertEqual(config.start, (0, 1))
        self.assertEqual(config.cells[0][0].kind, "normal")
        self.assertEqual(config.cells[0][0].reward, 0.0)
        self.assertEqual(config.cells[0][0].probabilities, original_start_policy)

        before = copy.deepcopy(config)
        with self.assertRaisesRegex(ValueError, "move the start role first"):
            config.set_cell(0, 1, "normal", 9.0, [0.2] * 5)
        self.assertEqual(config, before)

        with self.assertRaisesRegex(ValueError, "move the target role first"):
            config.set_cell(1, 2, "forbidden", -3.0, [0.2] * 5)
        self.assertEqual(config, before)

        original_target_policy = [0.0, 0.0, 0.0, 1.0, 0.0]
        config.set_cell(1, 2, "target", 11.0, original_target_policy)
        config.set_cell(1, 1, "forbidden", -9.0, [0.2] * 5)
        config.set_cell(1, 1, "target", 6.0, [0.2] * 5)
        self.assertEqual(config.target, (1, 1))
        self.assertEqual(config.cells[1][2].kind, "normal")
        self.assertEqual(config.cells[1][2].reward, 0.0)
        self.assertEqual(config.cells[1][2].probabilities, original_target_policy)
        config.validate()

    def test_invalid_cell_edits_and_overlap_leave_grid_unchanged(self) -> None:
        config = GridConfig.create(rows=2, columns=2)
        before = copy.deepcopy(config)
        invalid_edits = [
            (0, 0, "target", 5.0, [0.2] * 5),
            (1, 1, "start", 5.0, [0.2] * 5),
            (0, 1, "start", math.nan, [0.2] * 5),
            (1, 0, "target", 5.0, [0.0] * 5),
            (0, 1, "unknown", 0.0, [0.2] * 5),
            (-1, 0, "start", 0.0, [0.2] * 5),
            (0, 2, "target", 0.0, [0.2] * 5),
            (True, 0, "normal", 0.0, [0.2] * 5),
            (0, 1.0, "normal", 0.0, [0.2] * 5),
        ]
        for edit in invalid_edits:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                config.set_cell(*edit)
            self.assertEqual(config, before)

    def test_probability_validation_and_copy_isolation(self) -> None:
        invalid_values = (
            [math.nan, 0.0, 0.0, 0.0, 1.0],
            [math.inf, 0.0, 0.0, 0.0, 0.0],
            [True, 0.0, 0.0, 0.0, 0.0],
            [False, 0.0, 0.0, 0.0, 1.0],
            ["1", 0.0, 0.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0, 0.0, 1.1],
            [0.5, 0.5],
            [0.2] * 6,
            (0.2,) * 5,
            None,
            [0.2, 0.2, 0.2, 0.2, 0.3],
        )
        for probabilities in invalid_values:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    CellConfig(probabilities=probabilities).validate()
                config = GridConfig.create(rows=2, columns=2)
                before = copy.deepcopy(config)
                with self.assertRaises(ValueError):
                    config.apply_policy_to_all(probabilities)
                self.assertEqual(config, before)

        CellConfig(probabilities=[0.2, 0.2, 0.2, 0.2, 0.2000005]).validate()
        with self.assertRaises(ValueError):
            CellConfig(probabilities=[0.2, 0.2, 0.2, 0.2, 0.200002]).validate()

        config = GridConfig.create(rows=2, columns=2)
        config.set_cell(0, 1, "forbidden", -8.0, [0.2] * 5)
        roles_and_rewards = [
            [(cell.kind, cell.reward) for cell in row] for row in config.cells
        ]
        probabilities = [1.0, 0.0, 0.0, 0.0, 0.0]
        config.apply_policy_to_all(probabilities)
        probabilities[0] = 0.0
        probabilities[1] = 1.0
        self.assertEqual(config.cells[0][0].probabilities, [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(config.cells[0][0].probabilities, config.cells[1][1].probabilities)
        self.assertIsNot(
            config.cells[0][0].probabilities,
            config.cells[1][1].probabilities,
        )
        self.assertEqual(
            [[(cell.kind, cell.reward) for cell in row] for row in config.cells],
            roles_and_rewards,
        )
        config.cells[0][0].probabilities[0] = 0.0
        self.assertEqual(config.cells[1][1].probabilities, [1.0, 0.0, 0.0, 0.0, 0.0])
        config.cells[0][0].probabilities[0] = 1.0

        cell_policy = [0.0, 0.0, 1.0, 0.0, 0.0]
        config.set_cell(0, 1, "normal", 0.0, cell_policy)
        cell_policy[2] = 0.0
        cell_policy[3] = 1.0
        self.assertEqual(config.cells[0][1].probabilities, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_random_forbidden_replaces_old_cells_and_preserves_roles_and_policies(self) -> None:
        config = GridConfig.create(rows=2, columns=4)
        config.set_cell(1, 1, "start", 3.0, [0.2] * 5)
        config.set_cell(0, 1, "target", 4.0, [0.2] * 5)
        config.set_cell(0, 0, "forbidden", -9.0, [0.2] * 5)
        config.set_cell(0, 2, "normal", 7.0, [0.2] * 5)
        for i, row in enumerate(config.cells):
            for j, cell in enumerate(row):
                cell.probabilities = [0.0] * 5
                cell.probabilities[(i * config.columns + j) % 5] = 1.0
        before = copy.deepcopy(config)

        positions = [(1, 0), (1, 3)]
        with patch("model.random.sample", return_value=positions) as sample:
            self.assertEqual(config.randomize_forbidden(2), positions)
        candidates, count = sample.call_args.args
        self.assertEqual(count, 2)
        self.assertNotIn(config.start, candidates)
        self.assertNotIn(config.target, candidates)
        self.assertIn((0, 0), candidates)
        self.assertEqual(config.cells[0][0].kind, "normal")
        self.assertEqual(config.cells[0][0].reward, 0.0)
        self.assertEqual(config.cells[0][2].reward, 7.0)
        for i, row in enumerate(config.cells):
            for j, cell in enumerate(row):
                self.assertEqual(cell.probabilities, before.cells[i][j].probabilities)
                self.assertEqual(cell.kind == "forbidden", (i, j) in positions)
                if cell.kind == "forbidden":
                    self.assertEqual(cell.reward, -1.0)
        for i, j in (config.start, config.target):
            self.assertEqual(config.cells[i][j], before.cells[i][j])
        config.validate()

    def test_random_forbidden_count_limits_and_clear(self) -> None:
        for rows, columns in ((2, 3), (1, 5), (5, 1), (1, 2)):
            config = GridConfig.create(rows=rows, columns=columns)
            for count in (rows * columns - 2, 0, 1 if rows * columns > 2 else 0):
                with self.subTest(rows=rows, columns=columns, count=count):
                    positions = config.randomize_forbidden(count)
                    self.assertEqual(len(set(positions)), count)
                    self.assertEqual(sum(cell.kind == "forbidden" for row in config.cells for cell in row), count)
                    self.assertNotIn(config.start, positions)
                    self.assertNotIn(config.target, positions)
                    for row in config.cells:
                        for cell in row:
                            self.assertEqual(cell.reward, {"normal": 0.0, "start": 0.0, "target": 1.0, "forbidden": -1.0}[cell.kind])
                    config.validate()

    def test_invalid_random_forbidden_requests_do_not_change_cells(self) -> None:
        config = GridConfig.create(rows=2, columns=3)
        config.set_cell(0, 1, "forbidden", -3.0, [0.2] * 5)
        before = copy.deepcopy(config)
        for count in (-1, 5, True, False, 2.0, "2", None):
            with self.subTest(count=count):
                with patch("model.random.sample") as sample:
                    with self.assertRaises(ValueError):
                        config.randomize_forbidden(count)
                    sample.assert_not_called()
                self.assertEqual(config, before)
        with patch("model.random.sample", side_effect=ValueError("sampling failed")):
            with self.assertRaises(ValueError):
                config.randomize_forbidden(2)
        self.assertEqual(config, before)

    def test_json_round_trip_and_action_keyed_probabilities(self) -> None:
        config = GridConfig.create(
            rows=2,
            columns=3,
            cell_width_mm=12.5,
            cell_height_mm=8.25,
            gamma=0.75,
            zoom=2.0,
        )
        config.set_cell(0, 1, "forbidden", -4.5, [0.0, 0.0, 0.0, 1.0, 0.0])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            save_config(config, path)
            raw_data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw_data["format_version"], 1)
            self.assertEqual(
                list(raw_data["cells"][0][1]["probabilities"]), list(ACTIONS)
            )
            self.assertNotIn("values", raw_data)
            loaded = load_config(path)
            path.write_text(json.dumps(raw_data, sort_keys=True), encoding="utf-8")
            self.assertEqual(load_config(path), config)

        self.assertEqual(loaded, config)
        loaded.cells[0][1].probabilities[3] = 0.5
        self.assertEqual(config.cells[0][1].probabilities[3], 1.0)

    def test_malformed_noncanonical_and_invalid_role_files_are_rejected(self) -> None:
        config = GridConfig.create(rows=2, columns=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            save_config(config, path)
            original_text = path.read_text(encoding="utf-8")
            original_data = json.loads(original_text)
            bad_changes = [
                (("format_version",), 2),
                (("format_version",), True),
                (("format_version",), 1.0),
                (("format_version",), "1"),
                (("rows",), True),
                (("columns",), 2.0),
                (("rows",), 101),
                (("columns",), 0),
                (("cell_width_mm",), "10"),
                (("cell_height_mm",), False),
                (("cell_width_mm",), 0),
                (("cell_height_mm",), 1001),
                (("gamma",), "0.9"),
                (("gamma",), True),
                (("gamma",), 1.01),
                (("zoom",), False),
                (("zoom",), 0.0001),
                (("zoom",), 10.01),
                (("values",), [[0.0] * 2 for _ in range(2)]),
                (("cells",), None),
                (("cells",), original_data["cells"][:1]),
                (("cells", 1), original_data["cells"][1][:1]),
                (("cells", 0, 0), None),
                (("cells", 0, 0, "kind"), "normal"),
                (("cells", 1, 1, "kind"), "normal"),
                (("cells", 0, 1, "kind"), "start"),
                (("cells", 0, 1, "kind"), "target"),
                (("cells", 0, 0, "kind"), "unknown"),
                (("cells", 0, 0, "kind"), True),
                (("cells", 0, 0, "reward"), "0"),
                (("cells", 0, 0, "reward"), False),
                (("cells", 0, 0, "extra"), 0),
                (("cells", 0, 0, "probabilities"), [0.2] * 5),
                (("cells", 0, 0, "probabilities"), {"up": 1.0}),
                (("cells", 0, 0, "probabilities", "other"), 0),
                (("cells", 0, 0, "probabilities", "up"), True),
                (("cells", 0, 0, "probabilities", "up"), "0.2"),
                (("cells", 0, 0, "probabilities", "up"), 0.4),
                (("cells", 0, 0, "probabilities", "up"), -0.1),
            ]
            for keys, value in bad_changes:
                data = copy.deepcopy(original_data)
                container = data
                for key in keys[:-1]:
                    container = container[key]
                container[keys[-1]] = value
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(keys=keys, value=value), self.assertRaises(ValueError):
                    load_config(path)

            bad_texts = [
                "{not json", "[]", "null", '{"format_version": 1}',
                original_text.replace('"rows": 2', '"rows": 2, "rows": 2'),
                original_text.replace('"up": 0.2', '"up": 0.2, "up": 0.2'),
            ]
            for constant in ("NaN", "Infinity", "-Infinity", "1e999"):
                for field, value in (("gamma", "0.9"), ("reward", "0.0"), ("up", "0.2")):
                    bad_texts.append(original_text.replace(
                        f'"{field}": {value}', f'"{field}": {constant}', 1
                    ))
            for text in bad_texts:
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text[:80]), self.assertRaises(ValueError):
                    load_config(path)

    def test_atomic_save_preserves_destination_and_cleans_up_on_error(self) -> None:
        config = GridConfig.create(rows=2, columns=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            save_config(config, path)
            before = path.read_bytes()
            config.set_cell(0, 1, "forbidden", -5.0, [0.2] * 5)

            for operation in ("model.os.replace", "model.json.dump"):
                with self.subTest(operation=operation):
                    with patch(operation, side_effect=OSError("save failed")) as failing:
                        with self.assertRaisesRegex(OSError, "save failed"):
                            save_config(config, path)
                        if operation == "model.os.replace":
                            self.assertEqual(Path(failing.call_args.args[0]).parent, path.parent)
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(list(Path(directory).iterdir()), [path])

            config.gamma = True
            with patch("model.tempfile.NamedTemporaryFile") as create_temp:
                with self.assertRaises(ValueError):
                    save_config(config, path)
                create_temp.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_file_os_errors_propagate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing" / "grid.json"
            with self.assertRaises(OSError):
                load_config(missing)
            with self.assertRaises(OSError):
                save_config(GridConfig.create(), missing)

    def test_failed_load_leaves_existing_config_unchanged(self) -> None:
        original = GridConfig.create(rows=2, columns=2)
        before = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text('{"format_version": 1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
        self.assertEqual(original, before)

    def test_calculation_result_contract(self) -> None:
        config = GridConfig.create(rows=2, columns=3)
        config.set_cell(0, 1, "forbidden", -100.0, [0.2] * 5)
        CalculationResult(values=[[0.0] * 3 for _ in range(2)]).validate(config)
        result = CalculationResult(
            values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            optimal_policy=[[[0.2] * 5 for _ in range(3)] for _ in range(2)],
            best_paths=[
                PathResult(
                    coordinates=[
                        (0, 0), (0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 2)
                    ],
                    score=4.5,
                )
            ],
        )
        result.validate(config)
        first = CalculationResult(values=result.values)
        second = CalculationResult(values=result.values)
        first.best_paths.append(result.best_paths[0])
        self.assertEqual(second.best_paths, [])

    def test_result_values_and_policy_validate_types_finiteness_and_shape(self) -> None:
        config = GridConfig.create(rows=2, columns=3)
        valid_values = [[0.0] * 3 for _ in range(2)]
        bad_matrices = [
            None, (), [], [[0.0] * 3], [[0.0] * 2 for _ in range(2)],
            [(0.0, 0.0, 0.0), [0.0] * 3],
        ]
        for bad_value in (True, "0", None, math.nan, math.inf, -math.inf):
            matrix = copy.deepcopy(valid_values)
            matrix[0][1] = bad_value
            bad_matrices.append(matrix)
        for values in bad_matrices:
            with self.subTest(values=values), self.assertRaises(ValueError):
                CalculationResult(values=values).validate(config)

        valid_policy = [[[0.2] * 5 for _ in range(3)] for _ in range(2)]
        bad_policies = [
            (), [], valid_policy[:1], [valid_policy[0][:2], valid_policy[1]],
            [tuple(valid_policy[0]), valid_policy[1]],
        ]
        for probabilities in (
            [], [0.2] * 4, (0.2,) * 5, [0.1] * 5,
            [True, 0.0, 0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0, 1.0],
            [math.inf, 0.0, 0.0, 0.0, 0.0],
        ):
            policy = copy.deepcopy(valid_policy)
            policy[0][1] = probabilities
            bad_policies.append(policy)
        for policy in bad_policies:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                CalculationResult(valid_values, optimal_policy=policy).validate(config)

    def test_result_paths_validate_endpoints_coordinates_and_moves(self) -> None:
        config = GridConfig.create(rows=2, columns=3)
        values = [[0.0] * 3 for _ in range(2)]
        bad_coordinates = [
            [],
            ((0, 0), (0, 1), (0, 2), (1, 2)),
            [(0, 1), (0, 2), (1, 2)],
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 1), (1, 2)],
            [(0, 0), (0, 2), (1, 2)],
            [(0, 0), (0, -1), (0, 1), (0, 2), (1, 2)],
            [(0, 0), (2, 0), (1, 0), (1, 1), (1, 2)],
            [(False, 0), (0, 1), (0, 2), (1, 2)],
            [(0.0, 0), (0, 1), (0, 2), (1, 2)],
            [(0, "0"), (0, 1), (0, 2), (1, 2)],
            [[0, 0], (0, 1), (0, 2), (1, 2)],
            [(0, 0, 0), (0, 1), (0, 2), (1, 2)],
        ]
        for coordinates in bad_coordinates:
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                CalculationResult(
                    values, best_paths=[PathResult(coordinates, 0.0)]
                ).validate(config)

        valid_coordinates = [(0, 0), (0, 1), (0, 2), (1, 2)]
        for score in (True, "0", None, math.nan, math.inf, -math.inf):
            with self.subTest(score=score), self.assertRaises(ValueError):
                CalculationResult(
                    values, best_paths=[PathResult(valid_coordinates, score)]
                ).validate(config)
        for paths in (None, (), [{}]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                CalculationResult(values, best_paths=paths).validate(config)

if __name__ == "__main__":
    unittest.main()
