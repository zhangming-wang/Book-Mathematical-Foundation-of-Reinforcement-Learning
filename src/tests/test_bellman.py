import copy
import unittest

from bellman import evaluate_policy, solve_optimal_policy
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
        result = solve_optimal_policy(config, 1)
        result.validate(config)
        self.assertEqual(result.best_paths, [])
        self.assertEqual(result.values, [[0.0, 1.0, 1.0]])

        config.apply_policy_to_all([0.0, 0.0, 0.0, 0.0, 1.0])
        config.set_cell(0, 0, "start", 2.0, [0.0, 0.0, 0.0, 0.0, 1.0])
        result = solve_optimal_policy(config, 3)
        self.assertAlmostEqual(result.values[0][0], 5.42)
        self.assertEqual(result.optimal_policy[0][0], [0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(result.best_paths, [])

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


if __name__ == "__main__":
    unittest.main()
