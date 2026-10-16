import copy
import unittest
from unittest.mock import patch

import bellman
from bellman import CalculationCancelled, evaluate_policy, iterate_calculation, solve_optimal_policy
from fixtures import path_selection_config
from model import GridConfig


class PolicyEvaluationTest(unittest.TestCase):
    def test_exact_horizon_and_continuing_target(self) -> None:
        for gamma in (0.0, 0.9, 1.0):
            config = GridConfig.create(rows=1, columns=2, gamma=gamma)
            config.cells[0][0].probabilities = [0.0, 0.0, 0.0, 1.0, 0.0]
            config.cells[0][1].probabilities = [0.0, 0.0, 0.0, 0.0, 1.0]
            before = copy.deepcopy(config)
            for steps in (1, 3, 1000):
                with self.subTest(gamma=gamma, steps=steps):
                    expected = float(steps) if gamma == 1 else (1 - gamma ** steps) / (1 - gamma)
                    result = evaluate_policy(config, steps)
                    self.assertEqual(len(result.values), 1)
                    for value in result.values[0]:
                        self.assertAlmostEqual(value, expected)
                    self.assertIsNone(result.optimal_policy)
                    self.assertEqual(result.best_paths, [])
                    self.assertEqual(result.all_paths, [])
                    self.assertEqual(config, before)

    def test_alternating_states_use_the_previous_step_values(self) -> None:
        config = GridConfig.create(rows=1, columns=2, gamma=0.5)
        config.set_cell(0, 0, "start", 2.0, [0.0, 0.0, 0.0, 1.0, 0.0])
        config.set_cell(0, 1, "target", -1.0, [0.0, 0.0, 1.0, 0.0, 0.0])
        for steps, expected in ((1, [-1.0, 2.0]), (2, [0.0, 1.5]), (3, [-0.25, 2.0])):
            with self.subTest(steps=steps):
                self.assertEqual(evaluate_policy(config, steps).values, [expected])

    def test_weighted_actions_include_wall_and_forbidden_rewards(self) -> None:
        config = GridConfig.create(rows=1, columns=3, gamma=0.5)
        config.set_cell(0, 0, "start", 2.0, [0.25, 0.0, 0.25, 0.5, 0.0])
        config.set_cell(0, 1, "forbidden", -4.0, [0.0, 0.25, 0.0, 0.0, 0.75])
        config.set_cell(0, 2, "target", 5.0, [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(evaluate_policy(config, 1).values, [[-1.0, -4.0, 5.0]])
        self.assertEqual(evaluate_policy(config, 2).values, [[-2.25, -6.0, 7.5]])

    def test_small_nonzero_probabilities_are_not_filtered(self) -> None:
        config = GridConfig.create(rows=1, columns=2)
        probability = 1e-8
        config.cells[0][0].probabilities = [0.0, 0.0, 0.0, probability, 1 - probability]
        self.assertEqual(evaluate_policy(config, 1).values[0][0], probability)

    def test_default_and_invalid_horizons(self) -> None:
        config = GridConfig.create(rows=1, columns=2, gamma=1.0)
        config.apply_policy_to_all([0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(evaluate_policy(config).values, [[100.0, 100.0]])
        for steps in (0, -1, 1001, True, False, 3.0, "3", None):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                evaluate_policy(config, steps)
        config.cells[0][0].probabilities = [0.0] * 5
        with self.assertRaises(ValueError):
            evaluate_policy(config, 3)


class OptimalPolicyTest(unittest.TestCase):
    def test_round_29_retains_straight_paths_and_all_tied_best_paths(self) -> None:
        config = path_selection_config()
        before = copy.deepcopy(config)
        result = solve_optimal_policy(config, 29)
        self.assertEqual(len(result.all_paths), 8)
        self.assertEqual(len(result.best_paths), 2)
        self.assertEqual([len(path.coordinates) - 1 for path in result.all_paths], [22] * 2 + [20] * 6)
        self.assertEqual([f"{path.score:.3f}" for path in result.all_paths], ["-0.821"] * 2 + ["-1.013"] * 6)
        for path in result.best_paths:
            self.assertIn(path, result.all_paths)
            self.assertAlmostEqual(path.score, -.8208655246871895)
        best_edges = {tuple(sorted((a, b))) for path in result.best_paths for a, b in zip(path.coordinates, path.coordinates[1:])}
        all_edges = {tuple(sorted((a, b))) for path in result.all_paths for a, b in zip(path.coordinates, path.coordinates[1:])}
        self.assertEqual(all_edges - best_edges, {((0, 4), (0, 5)), ((1, 3), (1, 4))})
        self.assertEqual(config, before)
        result.validate(config)

    def test_tied_actions_and_paths_show_distinct_values_and_path_returns(self) -> None:
        config = GridConfig.create(rows=2, columns=2, gamma=0.9)
        before = copy.deepcopy(config)
        result = solve_optimal_policy(config, 3)
        result.validate(config)
        self.assertAlmostEqual(result.values[0][0], 1.71)
        self.assertAlmostEqual(result.values[1][1], 2.71)
        self.assertEqual(result.optimal_policy[0][0], [0.0, 0.5, 0.0, 0.5, 0.0])
        self.assertEqual(result.optimal_policy[1][1], [0.0, 1 / 3, 0.0, 1 / 3, 1 / 3])
        self.assertEqual(
            {tuple(path.coordinates) for path in result.best_paths},
            {((0, 0), (0, 1), (1, 1)), ((0, 0), (1, 0), (1, 1))},
        )
        for path in result.best_paths:
            self.assertAlmostEqual(path.score, 0.9)
        self.assertEqual(config, before)

    def test_missing_goal_paths_still_return_values_and_policy(self) -> None:
        config = GridConfig.create(rows=1, columns=3)
        result = solve_optimal_policy(config, 1, path_max_count=1)
        result.validate(config)
        self.assertEqual(result.best_paths, [])
        self.assertEqual(result.values, [[0.0, 1.0, 1.0]])

        config.apply_policy_to_all([0.0, 0.0, 0.0, 0.0, 1.0])
        config.set_cell(0, 0, "start", 2.0, [0.0, 0.0, 0.0, 0.0, 1.0])
        result = solve_optimal_policy(config, 3)
        self.assertAlmostEqual(result.values[0][0], 5.42)
        self.assertEqual(result.optimal_policy[0][0], [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(result.best_paths, [])

    def test_one_value_update_can_show_a_complete_multi_step_goal_path(self) -> None:
        config = GridConfig.create(rows=1, columns=3)
        result = solve_optimal_policy(config, 1)
        self.assertEqual(result.values, [[0.0, 1.0, 1.0]])
        self.assertEqual(len(result.best_paths), 1)
        self.assertEqual(result.best_paths[0].coordinates, [(0, 0), (0, 1), (0, 2)])
        self.assertAlmostEqual(result.best_paths[0].score, .9)
        limited = solve_optimal_policy(config, 1, path_max_count=1)
        self.assertEqual(limited.values, result.values)
        self.assertEqual(limited.optimal_policy, result.optimal_policy)
        self.assertEqual(limited.best_paths, [])
        for count in (2, 5):
            self.assertEqual(solve_optimal_policy(config, 1, path_max_count=count).best_paths, result.best_paths)

    def test_invalid_path_limits_are_rejected(self) -> None:
        config = GridConfig.create(rows=1, columns=3)
        for count in (0, -1, True, False, 1.0, "2"):
            with self.subTest(path_max_count=count), self.assertRaises(ValueError):
                solve_optimal_policy(config, 1, path_max_count=count)

    def test_only_nonzero_input_actions_are_available(self) -> None:
        config = GridConfig.create(rows=1, columns=2)
        config.cells[0][0].probabilities = [0.0, 0.0, 0.0, 1e-8, 1 - 1e-8]
        result = solve_optimal_policy(config, 1)
        self.assertEqual(result.optimal_policy[0][0], [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(result.values, [[1.0, 1.0]])
        self.assertEqual(result.best_paths[0].coordinates, [(0, 0), (0, 1)])
        self.assertEqual(result.best_paths[0].score, 1.0)

    def test_negative_rewards_choose_the_greatest_value(self) -> None:
        config = GridConfig.create(rows=1, columns=2, gamma=1.0)
        config.set_cell(0, 0, "start", -2.0, [0.0, 0.0, 0.0, 0.5, 0.5])
        config.set_cell(0, 1, "target", -1.0, [0.0, 0.0, 0.0, 0.0, 1.0])
        result = solve_optimal_policy(config, 1000)
        self.assertEqual(result.values, [[-1000.0, -1000.0]])
        self.assertEqual(result.optimal_policy[0][0], [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(result.best_paths[0].score, -1.0)

    def test_invalid_horizons_are_rejected(self) -> None:
        config = GridConfig.create(rows=1, columns=2)
        for steps in (0, -1, 1001, True, False, 3.0, "3", None):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                solve_optimal_policy(config, steps)


class CalculationRoundsTest(unittest.TestCase):
    def test_each_round_is_an_independent_result_matching_its_horizon(self) -> None:
        config = GridConfig.create(rows=2, columns=3, gamma=.7)
        config.set_cell(0, 1, "forbidden", -1.0, [.2] * 5)
        before = copy.deepcopy(config)
        for optimal, function in ((False, evaluate_policy), (True, solve_optimal_policy)):
            with self.subTest(optimal=optimal):
                rounds = list(iterate_calculation(config, 5, optimal=optimal))
                self.assertEqual([count for count, _ in rounds], [1, 2, 3, 4, 5])
                for count, result in rounds:
                    self.assertEqual(result, function(config, count))
                    result.validate(config)
                self.assertIsNot(rounds[0][1].values, rounds[-1][1].values)
                if optimal:
                    self.assertIsNot(rounds[0][1].optimal_policy, rounds[-1][1].optimal_policy)
                    self.assertIsNot(rounds[0][1].all_paths, rounds[-1][1].all_paths)
                    self.assertTrue(rounds[-1][1].best_paths)
                self.assertEqual(config, before)

    def test_batch_builds_only_final_paths_while_playback_builds_every_round(self) -> None:
        config = GridConfig.create(rows=2, columns=2)
        with patch("bellman._find_paths", wraps=bellman._find_paths) as search:
            batch_result = solve_optimal_policy(config, 4)
            self.assertEqual(search.call_count, 1)
            self.assertEqual(search.call_args.args[2], 3)
        with patch("bellman._find_paths", wraps=bellman._find_paths) as search:
            rounds = list(iterate_calculation(config, 4, optimal=True))
            self.assertEqual([call.args[2] for call in search.call_args_list], [3, 3, 3, 3])
        self.assertEqual(rounds[-1][1], batch_result)

    def test_playback_uses_the_same_path_budget_in_every_round(self) -> None:
        config = GridConfig.create(rows=1, columns=4)
        for path_max_count in (2, 3):
            with self.subTest(path_max_count=path_max_count):
                with patch("bellman._find_paths", wraps=bellman._find_paths) as search:
                    rounds = list(iterate_calculation(config, 2, optimal=True, path_max_count=path_max_count))
                self.assertEqual([call.args[2] for call in search.call_args_list], [path_max_count] * 2)
                self.assertEqual(rounds[-1][1], solve_optimal_policy(config, 2, path_max_count=path_max_count))
                self.assertEqual(bool(rounds[0][1].best_paths), path_max_count == 3)

    def test_checkpoints_can_cancel_value_updates_and_path_enumeration(self) -> None:
        config = GridConfig.create(rows=3, columns=3)
        before = copy.deepcopy(config)
        for function in (evaluate_policy, solve_optimal_policy):
            calls = []

            def cancel_during_update():
                calls.append(True)
                if len(calls) == 4:
                    raise CalculationCancelled()

            with self.subTest(function=function.__name__), self.assertRaises(CalculationCancelled):
                function(config, 100, checkpoint=cancel_during_update)
            self.assertEqual(config, before)

        searching = False
        search_checks = []
        original_search = bellman._find_paths

        def cancel_during_search():
            if searching:
                search_checks.append(True)
                if len(search_checks) == 3:
                    raise CalculationCancelled()

        def search(*args):
            nonlocal searching
            searching = True
            return original_search(*args)

        with patch("bellman._find_paths", side_effect=search), self.assertRaises(CalculationCancelled):
            solve_optimal_policy(config, 5, checkpoint=cancel_during_search)
        self.assertEqual(len(search_checks), 3)
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
