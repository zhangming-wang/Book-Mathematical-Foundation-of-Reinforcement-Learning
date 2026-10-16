import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from threading import Event
from time import monotonic
from unittest.mock import patch

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from bellman import evaluate_policy, solve_optimal_policy
from calculation_worker import CalculationWorker
from fixtures import path_selection_config
from model import CalculationResult, GridConfig, PathResult, load_config, save_config
from window import MainWindow


class WindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.shutdown_calculation()
        self.window.hide()
        self.window.deleteLater()
        self.app.processEvents()

    def _wait_until(self, predicate, timeout: int = 3000) -> None:
        deadline = monotonic() + timeout / 1000
        while not predicate():
            if monotonic() >= deadline:
                self.fail("等待界面计算状态超时")
            QTest.qWait(5)

    def _wait_for_calculation(self) -> None:
        self._wait_until(lambda: self.window._calculation_worker is None)

    def _assert_reset_display(self) -> None:
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.board_view.board_item.result)
        self.assertEqual(self.window.board_view.board_item.path_edges, set())
        self.assertEqual(self.window.policy_mode.currentIndex(), 0)
        self.assertFalse(self.window.policy_mode.model().item(1).isEnabled())
        self.assertFalse(self.window.show_paths_box.isEnabled())
        self.assertFalse(self.window.path_selection_box.isEnabled())
        self.assertEqual(self.window.path_selection_box.count(), 0)
        self.assertEqual(self.window._displayed_round, 0)
        self.assertEqual(self.window.progress_label.text(), "尚未开始")
        self.assertEqual(self.window.result_label.text(), "尚未计算")
        self.assertEqual(self.window.value_label.text(), "选中格状态价值 V：尚未计算")
        self.assertEqual(self.window.paths_label.text(), "最高得分路径：尚未计算")
        self.assertFalse(self.window.pause_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())
        self.assertFalse(self.window._playback_timer.isActive())
        self.assertTrue(self.window.reset_board_button.isEnabled())

    def test_select_apply_and_move_start_role(self) -> None:
        self.window._select_cell(0, 1)
        self.window.kind_box.setCurrentIndex(1)
        self.window.reward_edit.setText("2.5")

        self.assertTrue(self.window._apply_cell())
        self.assertEqual(self.window.config.start, (0, 1))
        self.assertEqual(self.window.config.cells[0][0].kind, "normal")
        self.assertEqual(self.window.config.cells[0][0].reward, 0.0)
        self.assertEqual(self.window.config.cells[0][1].reward, 2.5)

    def test_resize_preserves_cells_and_rebuild_resets_them(self) -> None:
        self.window._select_cell(1, 1)
        self.window.reward_edit.setText("4")
        self.assertTrue(self.window._apply_cell())

        self.window.width_box.setValue(12)
        self.assertTrue(self.window._apply_grid())
        self.assertEqual(self.window.config.cells[1][1].reward, 4.0)
        self.assertEqual(self.window.config.cell_width_mm, 12.0)

        self.window.rows_box.setValue(2)
        self.window.columns_box.setValue(3)
        with patch.object(self.window, "_confirm", return_value=True):
            self.assertTrue(self.window._apply_grid())
        self.assertEqual((self.window.config.rows, self.window.config.columns), (2, 3))
        self.assertEqual(self.window.config.start, (0, 0))
        self.assertEqual(self.window.config.target, (1, 2))
        self.assertEqual(self.window.config.cells[1][1].reward, 0.0)

    def test_random_forbidden_button_refreshes_selected_cell_and_clears_results(self) -> None:
        self.window.config.set_cell(1, 1, "forbidden", -8.0, [0.2] * 5)
        self.window.config.set_cell(1, 2, "normal", 7.0, [0.0, 0.0, 0.0, 1.0, 0.0])
        self.window._select_cell(1, 1)
        self.window.optimize_button.click()
        self._wait_for_calculation()
        before = deepcopy(self.window.config)
        self.assertEqual(self.window.forbidden_count_box.maximum(), 23)
        self.window.forbidden_count_box.setValue(2)
        self.assertFalse(self.window._grid_pending())
        with patch("model.random.sample", return_value=[(1, 2), (2, 2)]):
            self.window.randomize_forbidden_button.click()
        self.assertEqual(self.window.selected, (1, 1))
        self.assertEqual(self.window.kind_box.currentData(), "normal")
        self.assertEqual(float(self.window.reward_edit.text()), 0.0)
        self.assertFalse(self.window._cell_pending())
        self.assertEqual(self.window.config.cells[1][2].reward, -1.0)
        self.assertEqual(sum(cell.kind == "forbidden" for row in self.window.config.cells for cell in row), 2)
        for i, row in enumerate(self.window.config.cells):
            for j, cell in enumerate(row):
                self.assertEqual(cell.probabilities, before.cells[i][j].probabilities)
        self.assertEqual(self.window.config.start, before.start)
        self.assertEqual(self.window.config.target, before.target)
        self.assertIsNone(self.window.result)
        self.assertEqual(self.window.board_view.board_item.path_edges, set())
        self.assertEqual(self.window.policy_mode.currentIndex(), 0)
        self.assertFalse(self.window.show_paths_box.isEnabled())
        self.assertTrue(self.window.dirty)
        self.window.forbidden_count_box.setValue(0)
        self.window.randomize_forbidden_button.click()
        self.assertFalse(any(cell.kind == "forbidden" for row in self.window.config.cells for cell in row))
        self.assertEqual(self.window.config.cells[1][2].reward, 0.0)

    def test_random_forbidden_handles_pending_edit_choices(self) -> None:
        self.window.forbidden_count_box.setValue(1)
        for choice in ("取消", "放弃修改", "应用"):
            with self.subTest(choice=choice):
                self.window.evaluate_button.click()
                self._wait_for_calculation()
                before = deepcopy(self.window.config)
                previous_result = self.window.result
                previous_dirty = self.window.dirty
                draft_reward = before.cells[0][0].reward + 2
                self.window.reward_edit.setText(str(draft_reward))
                choices = []

                def choose_edit_action():
                    dialog = self.app.activeModalWidget()
                    for button in dialog.buttons():
                        if button.text() == choice:
                            choices.append(choice)
                            button.click()
                            return
                    dialog.reject()

                QTimer.singleShot(0, choose_edit_action)
                with patch("model.random.sample", return_value=[(1, 1)]) as sample:
                    self.window.randomize_forbidden_button.click()
                self.assertEqual(choices, [choice])
                if choice == "取消":
                    sample.assert_not_called()
                    self.assertEqual(self.window.config, before)
                    self.assertIs(self.window.result, previous_result)
                    self.assertEqual(self.window.dirty, previous_dirty)
                    self.assertTrue(self.window._cell_pending())
                    self.assertEqual(float(self.window.reward_edit.text()), draft_reward)
                    self.window._load_cell_editor()
                else:
                    sample.assert_called_once()
                    expected_reward = draft_reward if choice == "应用" else before.cells[0][0].reward
                    self.assertEqual(self.window.config.cells[0][0].reward, expected_reward)
                    self.assertEqual(self.window.config.cells[1][1].kind, "forbidden")
                    self.assertFalse(self.window._cell_pending())
                    self.assertIsNone(self.window.result)

    def test_random_forbidden_waits_for_global_settings(self) -> None:
        before = deepcopy(self.window.config)
        self.window.columns_box.setValue(3)
        self.window.forbidden_count_box.setValue(2)
        with patch.object(self.window, "_message") as message, patch("model.random.sample") as sample:
            self.window.randomize_forbidden_button.click()
        message.assert_called_once()
        sample.assert_not_called()
        self.assertEqual(self.window.config, before)
        self.assertEqual(self.window.columns_box.value(), 3)

    def test_random_forbidden_layout_round_trips_and_range_updates(self) -> None:
        self.window.forbidden_count_box.setValue(5)
        self.window.randomize_forbidden_button.click()
        expected = deepcopy(self.window.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "random-grid.json"
            with patch("window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._save_dialog())
            self.window.forbidden_count_box.setValue(0)
            self.window.randomize_forbidden_button.click()
            with patch("window.QFileDialog.getOpenFileName", return_value=(str(path), "")), patch.object(self.window, "_confirm_unsaved", return_value=True):
                self.assertTrue(self.window._load_dialog())
            self.assertEqual(self.window.config, expected)
            self.assertEqual(self.window.forbidden_count_box.value(), 5)
            self.assertFalse(self.window.dirty)
        self.window.rows_box.setValue(1)
        self.window.columns_box.setValue(2)
        with patch.object(self.window, "_confirm", return_value=True):
            self.assertTrue(self.window._apply_grid())
        self.assertEqual(self.window.forbidden_count_box.maximum(), 0)
        self.assertEqual(self.window.forbidden_count_box.value(), 0)
        self.assertTrue(self.window._randomize_forbidden())
        self.window.config.validate()

    def test_result_enables_optimal_policy_and_draws_all_unique_edges(self) -> None:
        values = [[0.0] * 5 for _ in range(5)]
        policy = [[[0.2] * 5 for _ in range(5)] for _ in range(5)]
        paths = [
            PathResult([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)], 1.0),
            PathResult([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)], 1.0),
        ]
        self.window.show_result(CalculationResult(values, policy, paths), optimal=True)
        self.app.processEvents()

        self.assertTrue(self.window.policy_mode.model().item(1).isEnabled())
        self.assertEqual(len(self.window.board_view.board_item.path_edges), 14)
        self.assertEqual(self.window.paths_label.text().splitlines()[0], "最高得分路径：2 条  可选目标路径：2 条")

    def test_path_selector_lists_round_29_routes_and_reveals_straight_segments(self) -> None:
        self.window.config = path_selection_config()
        self.window.selected = self.window.config.start
        self.window.board_view.set_config(self.window.config)
        self.window._load_grid_editor()
        self.window._load_cell_editor()
        self.window.max_count_box.setValue(29)
        with patch.object(self.window, "_message") as message:
            self.window.optimize_button.click()
            self._wait_for_calculation()
        message.assert_not_called()
        selector = self.window.path_selection_box
        self.assertTrue(selector.isEnabled())
        self.assertEqual(selector.count(), 9)
        self.assertEqual(selector.currentIndex(), 0)
        self.assertEqual(selector.itemText(0), "最优路径")
        for index in range(1, 9):
            score_text, steps_text = selector.itemText(index).split(" - ")
            path = selector.itemData(index)
            self.assertEqual(score_text, str(path.score))
            self.assertEqual(float(score_text), path.score)
            self.assertEqual(steps_text, f"{len(path.coordinates) - 1} 步")
        self.assertEqual(selector.itemText(1), selector.itemText(2))
        self.assertEqual(len(self.window.board_view.board_item.path_edges), 24)
        self.assertIn("最高得分路径：2 条", self.window.paths_label.text())
        self.assertIn("可选目标路径：8 条", self.window.paths_label.text())
        result = self.window.result
        policy_before = deepcopy(result.optimal_policy)

        for straight in (((0, 4), (0, 5)), ((1, 3), (1, 4))):
            with self.subTest(straight=straight):
                self.assertNotIn(straight, self.window.board_view.board_item.path_edges)
                index = next(
                    index for index in range(1, selector.count())
                    if straight in {tuple(sorted((a, b))) for a, b in zip(selector.itemData(index).coordinates, selector.itemData(index).coordinates[1:])}
                )
                selector.setCurrentIndex(index)
                selected = selector.currentData()
                self.assertIn(straight, self.window.board_view.board_item.path_edges)
                self.assertEqual(len(self.window.board_view.board_item.path_edges), 20)
                self.assertEqual(selector.currentText(), f"{selected.score} - 20 步")
                self.assertEqual(self.window.paths_label.text(), "最高得分路径：2 条  可选目标路径：8 条")
                self.assertIn(f"路径 {index}", selector.itemData(index, Qt.ItemDataRole.ToolTipRole))
                self.assertIn("(1, 1)", selector.itemData(index, Qt.ItemDataRole.ToolTipRole))
                self.assertIs(self.window.result, result)
                self.assertEqual(result.optimal_policy, policy_before)
                self.window.show_paths_box.setChecked(False)
                self.assertFalse(self.window.board_view.board_item.paths_visible)
                self.window.show_paths_box.setChecked(True)
                selector.setCurrentIndex(0)
                self.assertEqual(len(self.window.board_view.board_item.path_edges), 24)

        self.window.reset_board_button.click()
        self._assert_reset_display()

    def test_path_selector_sorts_full_precision_and_preserves_duplicate_score_items(self) -> None:
        values = [[0.0] * 5 for _ in range(5)]
        first = PathResult([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)], .12341)
        second = PathResult([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)], .12349)
        self.window.show_result(CalculationResult(values=values, best_paths=[second], all_paths=[first, second]))
        selector = self.window.path_selection_box
        self.assertEqual(selector.count(), 3)
        self.assertEqual(selector.itemText(1), "0.12349 - 8 步")
        self.assertEqual(selector.itemText(2), "0.12341 - 8 步")
        self.assertEqual(selector.itemData(1).coordinates, second.coordinates)
        self.assertEqual(selector.itemData(2).coordinates, first.coordinates)
        self.assertEqual(selector.itemData(1).score, .12349)
        self.assertEqual(selector.itemData(2).score, .12341)
        selector.setCurrentIndex(2)
        first_edges = set(self.window.board_view.board_item.path_edges)
        selector.setCurrentIndex(1)
        self.assertNotEqual(self.window.board_view.board_item.path_edges, first_edges)

    def test_playback_preserves_selected_route_by_coordinates_and_falls_back_when_it_disappears(self) -> None:
        config = path_selection_config()
        self.window.config = config
        self.window.selected = config.start
        self.window.board_view.set_config(config)
        self.window._load_grid_editor()
        self.window._load_cell_editor()
        self.window.max_count_box.setValue(2)
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(10)
        round_29 = solve_optimal_policy(config, 29)
        reordered = deepcopy(round_29)
        reordered.all_paths.reverse()
        selector = self.window.path_selection_box
        # Reorder tied routes in the next snapshot so selection cannot rely on its index.
        pending_frames = iter([(1, round_29), (2, reordered)])
        with patch("calculation_worker.bellman.iterate_calculation", return_value=pending_frames):
            self.window.optimize_button.click()
            self._wait_until(lambda: self.window._displayed_round == 1)
            selector.setCurrentIndex(3)
            coordinates = deepcopy(selector.currentData().coordinates)
            self.window.interval_box.setValue(0)
            self._wait_for_calculation()
        self.assertEqual(selector.currentData().coordinates, coordinates)
        self.assertNotEqual(selector.currentIndex(), 3)
        self.assertEqual(self.window.board_view.board_item.path_edges,
                         {tuple(sorted((a, b))) for a, b in zip(coordinates, coordinates[1:])})
        final = solve_optimal_policy(config, 100)
        self.window.show_result(final, optimal=True)
        self.assertEqual(selector.currentIndex(), 0)
        self.assertEqual(self.window.board_view.board_item.path_edges,
                         {tuple(sorted((a, b))) for path in final.best_paths for a, b in zip(path.coordinates, path.coordinates[1:])})

    def test_compute_button_shows_real_values_for_the_selected_horizon(self) -> None:
        for box in self.window.probability_boxes:
            box.setValue(0)
        self.window.probability_boxes[3].setValue(1)
        with patch.object(self.window, "_confirm", return_value=True):
            self.window.bulk_policy_button.click()
        self.window.max_count_box.setValue(3)
        with patch.object(self.window, "_message") as message:
            self.window.evaluate_button.click()
            self._wait_for_calculation()
        message.assert_not_called()
        self.assertAlmostEqual(self.window.result.values[4][4], 2.71)
        self.assertEqual(self.window.board_view.board_item.result.values, self.window.result.values)
        self.assertIn("3 步", self.window.result_label.text())
        self.window._select_cell(4, 4)
        self.assertIn("2.71", self.window.value_label.text())

        self.window.max_count_box.setValue(1000)
        self.assertIsNone(self.window.result)
        self.assertIsNone(self.window.board_view.board_item.result)
        with patch.object(self.window, "_message") as message:
            self.window.evaluate_button.click()
            self._wait_for_calculation()
        message.assert_not_called()
        self.assertAlmostEqual(self.window.result.values[4][4], 10.0)
        self.assertIn("1000 步", self.window.result_label.text())

    def test_optimal_button_displays_real_policy_values_and_all_tied_paths(self) -> None:
        self.window.max_count_box.setValue(100)
        with patch.object(self.window, "_message") as message:
            self.window.optimize_button.click()
            self._wait_for_calculation()
        self.app.processEvents()
        message.assert_not_called()
        self.assertEqual(self.window.policy_mode.currentIndex(), 1)
        self.assertTrue(self.window.policy_mode.model().item(1).isEnabled())
        self.assertIn("100 步", self.window.result_label.text())
        self.assertIn("选中格状态价值 V：4.782", self.window.value_label.text())
        self.assertEqual(self.window.result.optimal_policy[0][0], [0.0, 0.5, 0.0, 0.5, 0.0])
        self.assertEqual(len(self.window.result.best_paths), 70)
        self.assertEqual(len(self.window.board_view.board_item.path_edges), 40)
        self.assertIn("70 条", self.window.paths_label.text())
        self.assertEqual(self.window.paths_label.text(), "最高得分路径：70 条  可选目标路径：70 条")
        self.assertTrue(self.window.show_paths_box.isEnabled())
        self.assertTrue(self.window.board_view.board_item.paths_visible)
        self.assertEqual(self.window.board_view.board_item.probabilities(0, 0), [0.0, 0.5, 0.0, 0.5, 0.0])

        self.window.policy_mode.setCurrentIndex(0)
        self.assertEqual(self.window.board_view.board_item.probabilities(0, 0), [0.2] * 5)
        self.window.policy_mode.setCurrentIndex(1)
        self.window.show_paths_box.setChecked(False)
        self.assertFalse(self.window.board_view.board_item.paths_visible)
        self.window.show_paths_box.setChecked(True)
        self.window._select_cell(4, 4)
        self.assertIn("9.999", self.window.value_label.text())

        self.window.max_count_box.setValue(3)
        self.assertIsNone(self.window.result)
        self.assertEqual(self.window.value_label.text(), "选中格状态价值 V：尚未计算")

    def test_optimal_button_without_goal_path_keeps_the_policy_visible(self) -> None:
        self.window.max_count_box.setValue(3)
        self.window.path_max_count_box.setValue(7)
        with patch.object(self.window, "_message") as message:
            self.window.optimize_button.click()
            self._wait_for_calculation()
        message.assert_not_called()
        self.assertIsNotNone(self.window.result)
        self.assertEqual(self.window.result.best_paths, [])
        self.assertEqual(self.window.path_selection_box.count(), 0)
        self.assertFalse(self.window.path_selection_box.isEnabled())
        self.assertEqual(self.window.policy_mode.currentIndex(), 1)
        self.assertIn("没有可展示的目标路径", self.window.paths_label.text())
        self.assertFalse(self.window.show_paths_box.isEnabled())
        self.assertEqual(self.window.board_view.board_item.path_edges, set())

    def test_both_calculation_buttons_pass_the_selected_horizon(self) -> None:
        values = [[0.0] * 5 for _ in range(5)]
        policy = [[[0.2] * 5 for _ in range(5)] for _ in range(5)]
        for button, function in (
            (self.window.evaluate_button, "calculation_worker.bellman.evaluate_policy"),
            (self.window.optimize_button, "calculation_worker.bellman.solve_optimal_policy"),
        ):
            for steps in (1, 37, 1000):
                with self.subTest(function=function, steps=steps):
                    self.window.max_count_box.setValue(steps)
                    with patch(function, return_value=CalculationResult(values, policy)) as calculate:
                        button.click()
                        self._wait_for_calculation()
                    calculate.assert_called_once()
                    passed_config, passed_steps = calculate.call_args.args
                    self.assertEqual(passed_steps, steps)
                    self.assertEqual(passed_config, self.window.config)
                    self.assertIsNot(passed_config, self.window.config)
                    if button is self.window.optimize_button:
                        self.assertEqual(calculate.call_args.kwargs["path_max_count"], 24)

    def test_playback_shows_each_round_and_matches_batch_results(self) -> None:
        self.window.rows_box.setValue(2)
        self.window.columns_box.setValue(2)
        with patch.object(self.window, "_confirm", return_value=True):
            self.assertTrue(self.window._apply_grid())
        self.window.dirty = False
        self.window.max_count_box.setValue(3)
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(.01)
        for button, function in ((self.window.evaluate_button, evaluate_policy),
                                 (self.window.optimize_button, solve_optimal_policy)):
            with self.subTest(function=function.__name__):
                frames = []
                display = self.window.show_result

                def collect(result, optimal=False):
                    frames.append(deepcopy(result))
                    display(result, optimal)

                with patch.object(self.window, "show_result", side_effect=collect), patch.object(self.window, "_message") as message:
                    button.click()
                    self._wait_for_calculation()
                message.assert_not_called()
                self.assertEqual(len(frames), 3)
                for count, frame in enumerate(frames, 1):
                    self.assertEqual(frame, function(self.window.config, count))
                self.assertEqual(self.window.result, function(self.window.config, 3))
                self.assertIn("已完成，第 3／3 轮", self.window.progress_label.text())
                self.assertFalse(self.window.dirty)
                if function is solve_optimal_policy:
                    self.assertEqual([len(frame.best_paths) for frame in frames], [2, 2, 2])
                    self.assertEqual(len(self.window.board_view.board_item.path_edges), 4)

    def test_reset_clears_completed_results_and_preserves_configuration(self) -> None:
        self.window.config.set_cell(1, 1, "forbidden", -3.0, [.2] * 5)
        self.window._select_cell(1, 1)
        self.window.zoom_box.setValue(250)
        self.window.max_count_box.setValue(8)
        self.window.path_max_count_box.setValue(12)
        self.window.interval_box.setValue(.123)
        before = deepcopy(self.window.config)
        transform = self.window.board_view.viewportTransform()
        parameters = (8, 12, 0, .123)
        for button, dirty in ((self.window.evaluate_button, False), (self.window.optimize_button, True)):
            with self.subTest(button=button.text(), dirty=dirty):
                self.window.dirty = dirty
                button.click()
                self._wait_for_calculation()
                self.assertIsNotNone(self.window.result)
                if button is self.window.optimize_button:
                    self.assertTrue(self.window.board_view.board_item.path_edges)
                self.window.reset_board_button.click()
                self._assert_reset_display()
                self.assertEqual(self.window.config, before)
                self.assertEqual(self.window.selected, (1, 1))
                self.assertEqual(self.window.dirty, dirty)
                self.assertEqual(self.window.board_view.viewportTransform(), transform)
                self.assertEqual(self.window.board_view.board_item.probabilities(1, 1), [.2] * 5)
                self.assertEqual(
                    (self.window.max_count_box.value(), self.window.path_max_count_box.value(),
                     self.window.calculation_mode_box.currentIndex(), self.window.interval_box.value()),
                    parameters,
                )

    def test_reset_during_playback_and_pause_clears_after_cancellation(self) -> None:
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.max_count_box.setValue(4)
        self.window.interval_box.setValue(10)
        before = deepcopy(self.window.config)
        for button, paused in ((self.window.evaluate_button, False), (self.window.optimize_button, True)):
            with self.subTest(button=button.text(), paused=paused):
                button.click()
                self._wait_until(lambda: self.window._displayed_round == 1)
                if paused:
                    self.window.pause_button.click()
                self.assertTrue(self.window.reset_board_button.isEnabled())
                self.window.reset_board_button.click()
                self.assertTrue(self.window._reset_after_calculation)
                self.assertFalse(self.window.reset_board_button.isEnabled())
                self._wait_for_calculation()
                QTest.qWait(30)
                self._assert_reset_display()
                self.assertFalse(self.window._reset_after_calculation)
                self.assertIsNone(self.window._pending_frame)
                self.assertEqual(self.window.config, before)
                self.assertFalse(self.window.dirty)
                self.assertEqual(self.window.interval_box.value(), 10)
                self.assertTrue(self.window.evaluate_button.isEnabled())
                self.assertTrue(self.window.load_button.isEnabled())
        self.window.calculation_mode_box.setCurrentIndex(0)
        self.window.evaluate_button.click()
        self._wait_for_calculation()
        self.assertEqual(self.window.result, evaluate_policy(self.window.config, 4))

    def test_reset_interrupts_busy_path_search_before_any_frame(self) -> None:
        entered_search = Event()
        delay = Event()
        before = deepcopy(self.window.config)

        def slow_search(config, policy, count, checkpoint):
            entered_search.set()
            while True:
                checkpoint()
                delay.wait(.001)

        with patch("bellman._find_paths", side_effect=slow_search), patch.object(self.window, "_message") as message:
            self.window.optimize_button.click()
            self._wait_until(entered_search.is_set)
            self.window.reset_board_button.click()
            self._wait_for_calculation()
        message.assert_not_called()
        self._assert_reset_display()
        self.assertEqual(self.window.config, before)

    def test_reset_discards_a_final_frame_already_queued_for_display(self) -> None:
        emitted = Event()
        publish = CalculationWorker._publish

        def publish_and_notify(worker, count, result):
            worker.frame_ready.connect(lambda count, frame: emitted.set(), Qt.ConnectionType.DirectConnection)
            publish(worker, count, result)

        with patch.object(CalculationWorker, "_publish", new=publish_and_notify), patch.object(self.window, "show_result", wraps=self.window.show_result) as display:
            self.window.evaluate_button.click()
            self.assertTrue(emitted.wait(2))
            self.window.reset_board_button.click()
            self._wait_for_calculation()
            QTest.qWait(30)
            display.assert_not_called()
        self._assert_reset_display()
        self.window.evaluate_button.click()
        self._wait_for_calculation()
        self.assertEqual(self.window.result, evaluate_policy(self.window.config, 100))

    def test_reset_preserves_unapplied_cell_and_grid_edits(self) -> None:
        self.window.evaluate_button.click()
        self._wait_for_calculation()
        before = deepcopy(self.window.config)
        self.window.reward_edit.setText("3.5")
        self.window.probability_boxes[0].setValue(.3)
        self.window.rows_box.setValue(7)
        cell_draft = self.window._cell_snapshot()
        grid_draft = self.window._grid_snapshot()
        self.window.reset_board_button.click()
        self._assert_reset_display()
        self.assertEqual(self.window.config, before)
        self.assertEqual(self.window._cell_snapshot(), cell_draft)
        self.assertEqual(self.window._grid_snapshot(), grid_draft)
        self.assertTrue(self.window._cell_pending())
        self.assertTrue(self.window._grid_pending())
        self.assertFalse(self.window.dirty)

    def test_first_round_shows_complete_goal_paths_with_an_independent_path_limit(self) -> None:
        self.assertEqual(self.window.path_max_count_box.value(), 24)
        self.window.max_count_box.setValue(1)
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(0)
        with patch.object(self.window, "_message") as message:
            self.window.optimize_button.click()
            self._wait_for_calculation()
        message.assert_not_called()
        result = deepcopy(self.window.result)
        self.assertEqual(result.values[0][0], 0.0)
        self.assertEqual(len(result.best_paths), 70)
        self.assertTrue(all(len(path.coordinates) - 1 == 8 for path in result.best_paths))
        self.assertEqual(len(self.window.board_view.board_item.path_edges), 40)
        self.assertTrue(self.window.show_paths_box.isEnabled())
        self.assertIn("第 1／1 轮", self.window.progress_label.text())
        self.assertFalse(self.window.dirty)

        self.window.path_max_count_box.setValue(7)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.dirty)
        self.window.calculation_mode_box.setCurrentIndex(0)
        self.window.optimize_button.click()
        self._wait_for_calculation()
        self.assertEqual(self.window.result.values, result.values)
        self.assertEqual(self.window.result.optimal_policy, result.optimal_policy)
        self.assertEqual(self.window.result.best_paths, [])

        self.window.path_max_count_box.setValue(8)
        self.window.optimize_button.click()
        self._wait_for_calculation()
        self.assertEqual(self.window.result.values, result.values)
        self.assertEqual(self.window.result.optimal_policy, result.optimal_policy)
        self.assertEqual(self.window.result.best_paths, result.best_paths)
        self.assertLess(len(self.window.result.all_paths), len(result.all_paths))

    def test_path_limit_range_tracks_applied_board_dimensions(self) -> None:
        self.window.rows_box.setValue(2)
        self.window.columns_box.setValue(3)
        with patch.object(self.window, "_confirm", return_value=True):
            self.assertTrue(self.window._apply_grid())
        self.assertEqual(self.window.path_max_count_box.maximum(), 5)
        self.assertEqual(self.window.path_max_count_box.value(), 5)
        self.window.path_max_count_box.setValue(3)
        self.window.rows_box.setValue(3)
        with patch.object(self.window, "_confirm", return_value=True):
            self.assertTrue(self.window._apply_grid())
        self.assertEqual(self.window.path_max_count_box.maximum(), 8)
        self.assertEqual(self.window.path_max_count_box.value(), 3)

    def test_pause_live_interval_change_and_resume(self) -> None:
        self.window.max_count_box.setValue(3)
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(.2)
        self.window.evaluate_button.click()
        self._wait_until(lambda: self.window._displayed_round == 1)
        self.window.pause_button.click()
        paused_result = self.window.result
        QTest.qWait(250)
        self.assertIs(self.window.result, paused_result)
        self.assertEqual(self.window._displayed_round, 1)
        self.assertIn("已暂停", self.window.progress_label.text())
        self.assertEqual(self.window.pause_button.text(), "继续")
        self.assertTrue(self.window.interval_box.isEnabled())
        self.window.interval_box.setValue(.01)
        self.window.pause_button.click()
        self._wait_for_calculation()
        self.assertEqual(self.window.result, evaluate_policy(self.window.config, 3))
        self.assertFalse(self.window.dirty)

    def test_changing_interval_reschedules_the_next_round(self) -> None:
        self.window.max_count_box.setValue(2)
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(10)
        self.window.evaluate_button.click()
        self._wait_until(lambda: self.window._displayed_round == 1)
        QTest.qWait(20)
        self.assertEqual(self.window._displayed_round, 1)
        self.window.interval_box.setValue(0)
        self._wait_for_calculation()
        self.assertEqual(self.window._displayed_round, 2)

    def test_stop_keeps_displayed_result_and_allows_a_new_run(self) -> None:
        self.window.max_count_box.setValue(20)
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(.2)
        self.window.optimize_button.click()
        self._wait_until(lambda: self.window._displayed_round == 1)
        displayed_result = self.window.result
        self.assertFalse(self.window.grid_settings_group.isEnabled())
        self.assertFalse(self.window.reward_edit.isEnabled())
        self.assertFalse(self.window.load_button.isEnabled())
        self.assertFalse(self.window.max_count_box.isEnabled())
        self.assertFalse(self.window.path_max_count_box.isEnabled())
        self.assertFalse(self.window.apply_cell_button.isEnabled())
        self.assertFalse(self.window.randomize_forbidden_button.isEnabled())
        self.window._select_cell(0, 1)
        self.assertEqual(self.window.selected, (0, 1))
        self.assertFalse(self.window.bulk_policy_button.isEnabled())
        self.window.stop_button.click()
        self._wait_for_calculation()
        QTest.qWait(250)
        self.assertIs(self.window.result, displayed_result)
        self.assertEqual(self.window._displayed_round, 1)
        self.assertIn("已停止", self.window.progress_label.text())
        self.assertTrue(self.window.grid_settings_group.isEnabled())
        self.assertTrue(self.window.load_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())
        self.window.calculation_mode_box.setCurrentIndex(0)
        self.window.max_count_box.setValue(2)
        self.window.evaluate_button.click()
        self._wait_for_calculation()
        self.assertEqual(self.window.result, evaluate_policy(self.window.config, 2))

    def test_pause_and_stop_interrupt_busy_path_search_without_freezing_ui(self) -> None:
        entered_search = Event()
        delay = Event()
        ticks = []

        def slow_search(config, policy, count, checkpoint):
            entered_search.set()
            while True:
                checkpoint()
                delay.wait(.001)

        timer = QTimer(self.window)
        timer.timeout.connect(lambda: ticks.append(True))
        timer.start(5)
        with patch("bellman._find_paths", side_effect=slow_search), patch.object(self.window, "_message") as message:
            self.window.optimize_button.click()
            self._wait_until(entered_search.is_set)
            self.window.pause_button.click()
            QTest.qWait(30)
            self.assertTrue(ticks)
            self.assertIsNone(self.window.result)
            self.window.stop_button.click()
            self._wait_for_calculation()
        timer.stop()
        message.assert_not_called()
        self.assertIn("已停止", self.window.progress_label.text())
        self.assertTrue(self.window.optimize_button.isEnabled())

    def test_failed_calculation_releases_worker_and_restores_controls(self) -> None:
        with patch("calculation_worker.bellman.evaluate_policy", side_effect=ValueError("测试计算失败")), patch.object(self.window, "_message") as message:
            self.window.evaluate_button.click()
            self._wait_for_calculation()
        message.assert_called_once()
        self.assertIn("测试计算失败", message.call_args.args[1])
        self.assertIsNone(self.window.result)
        self.assertTrue(self.window.evaluate_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

    def test_close_and_shutdown_finish_a_paused_worker(self) -> None:
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(10)
        self.window.evaluate_button.click()
        self._wait_until(lambda: self.window._displayed_round == 1)
        self.window.pause_button.click()
        self.window.close()
        self._wait_until(lambda: self.window._calculation_worker is None and not self.window.isVisible())

    def test_shutdown_joins_worker_even_before_first_frame_is_displayed(self) -> None:
        self.window.evaluate_button.click()
        worker = self.window._calculation_worker
        self.window.shutdown_calculation()
        self.assertFalse(worker.isRunning())
        self.app.processEvents()
        self.assertIsNone(self.window._calculation_worker)
        self.assertIsNone(self.window.result)

    def test_pause_holds_an_already_queued_final_frame_until_resume(self) -> None:
        emitted = Event()
        publish = CalculationWorker._publish

        def publish_and_notify(worker, count, result):
            worker.frame_ready.connect(lambda count, frame: emitted.set(), Qt.ConnectionType.DirectConnection)
            publish(worker, count, result)

        with patch.object(CalculationWorker, "_publish", new=publish_and_notify):
            self.window.evaluate_button.click()
            # Keep the GUI queue pending until the worker has sent its frame.
            self.assertTrue(emitted.wait(2))
            self.window.pause_button.click()
            self._wait_until(lambda: self.window._pending_frame is not None)
            self.assertIsNone(self.window.result)
            self.assertTrue(self.window._calculation_worker.isRunning())
            self.window.pause_button.click()
            self._wait_for_calculation()
        self.assertEqual(self.window.result, evaluate_policy(self.window.config, 100))

    def test_calculation_horizon_is_not_saved_as_grid_configuration(self) -> None:
        self.assertEqual(self.window.max_count_box.value(), 100)
        self.window.max_count_box.setValue(7)
        self.window.path_max_count_box.setValue(12)
        self.window.calculation_mode_box.setCurrentIndex(1)
        self.window.interval_box.setValue(.123)
        self.assertFalse(self.window.dirty)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            with patch("window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._save_dialog())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("max_count", data)
            self.assertNotIn("path_max_count", data)
            self.assertNotIn("calculation_mode", data)
            self.assertNotIn("interval", data)
            self.assertEqual(load_config(path), self.window.config)
            with patch("window.QFileDialog.getOpenFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._load_dialog())
            self.assertEqual(self.window.max_count_box.value(), 7)
            self.assertEqual(self.window.path_max_count_box.value(), 12)

    def test_rounded_probability_fields_preserve_loaded_precision(self) -> None:
        config = GridConfig.create(cell_width_mm=10.123456, gamma=.912345)
        probabilities = [1 / 3, 1 / 3, 1 / 3, 0.0, 0.0]
        config.cells[0][0].probabilities = probabilities
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            save_config(config, path)
            with patch("window.QFileDialog.getOpenFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._load_dialog())

            self.assertEqual(self.window.probability_boxes[0].value(), .333)
            self.assertEqual(self.window.probability_total_label.text(), "概率合计：1.000  /  1.000")
            self.assertTrue(self.window.bulk_policy_button.isEnabled())
            self.assertFalse(self.window._cell_pending())
            self.window.reward_edit.setText("2.5")
            self.assertTrue(self.window.apply_cell_button.isEnabled())
            self.assertTrue(self.window._apply_cell())

            self.window.height_box.setValue(12)
            self.assertTrue(self.window._apply_grid())
            with patch("window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._save_dialog())
            saved = load_config(path)
            self.assertEqual(saved.cells[0][0].probabilities, probabilities)
            self.assertEqual(saved.cell_width_mm, config.cell_width_mm)
            self.assertEqual(saved.gamma, config.gamma)

        self.window.probability_boxes[0].setValue(.334)
        self.assertFalse(self.window.apply_cell_button.isEnabled())
        self.assertFalse(self.window.bulk_policy_button.isEnabled())
        self.window.probability_boxes[0].setValue(.333)
        with patch.object(self.window, "_confirm", return_value=True):
            self.window._apply_policy_to_all()
        for row in self.window.config.cells:
            for cell in row:
                self.assertEqual(cell.probabilities, probabilities)
        self.assertFalse(self.window._cell_pending())

    def test_scrollbar_updates_headers_without_signal_argument_error(self) -> None:
        changes = []
        self.window.board_view.view_changed.connect(lambda: changes.append(True))
        self.window.board_view._scrollbar_changed(1)
        self.app.processEvents()
        self.assertTrue(changes)

    def test_board_starts_centered_when_it_fits(self) -> None:
        view = self.window.board_view
        self.app.processEvents()
        board_center = view.mapFromScene(view.board_item.boundingRect().center())
        viewport_center = view.viewport().rect().center()
        self.assertLessEqual(abs(board_center.x() - viewport_center.x()), 2)
        self.assertLessEqual(abs(board_center.y() - viewport_center.y()), 2)

    def test_left_click_selects_but_left_drag_pans(self) -> None:
        view = self.window.board_view
        target = view.mapFromScene(QPointF(view.board_item.cell_width * 1.5, view.board_item.cell_height * .5))
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, target)
        self.assertEqual(self.window.selected, (0, 1))

        board_center = view.board_item.boundingRect().center()
        for zoom in (300, 50, 1000):
            with self.subTest(zoom_percent=zoom):
                self.window.zoom_box.setValue(zoom)
                self.app.processEvents()
                before = view.mapFromScene(board_center)
                start = view.viewport().rect().center()
                delta = QPoint(-120, -80)
                QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
                QTest.mouseMove(view.viewport(), start + delta)
                QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + delta)
                after = view.mapFromScene(board_center)
                self.assertLessEqual(abs((after - before).x() - delta.x()), 2)
                self.assertLessEqual(abs((after - before).y() - delta.y()), 2)
                self.assertEqual(self.window.selected, (0, 1))

    def test_continuous_drag_preserves_each_axis(self) -> None:
        view = self.window.board_view
        board_center = view.board_item.boundingRect().center()
        selected = self.window.selected
        for zoom in (300, 50, 1000):
            self.window.zoom_box.setValue(zoom)
            self.app.processEvents()
            for step in (QPoint(-1, 0), QPoint(1, 0), QPoint(0, -1), QPoint(0, 1)):
                with self.subTest(zoom_percent=zoom, step=step):
                    before = view.mapFromScene(board_center)
                    start = view.viewport().rect().center()
                    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
                    for index in range(1, 81):
                        QTest.mouseMove(view.viewport(), start + step * index)
                    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + step * 80)
                    displacement = view.mapFromScene(board_center) - before
                    expected = step * 80
                    self.assertLessEqual(abs(displacement.x() - expected.x()), 1)
                    self.assertLessEqual(abs(displacement.y() - expected.y()), 1)
                    self.assertEqual(self.window.selected, selected)

    def _send_wheel(self, anchor: QPoint, delta: int, pixel: bool = False) -> None:
        viewport = self.window.board_view.viewport()
        event = QWheelEvent(
            QPointF(anchor), QPointF(viewport.mapToGlobal(anchor)),
            QPoint(0, delta) if pixel else QPoint(),
            QPoint() if pixel else QPoint(0, delta),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        self.app.sendEvent(viewport, event)
        self.app.processEvents()
        self.assertTrue(event.isAccepted())

    def test_wheel_zoom_keeps_off_center_anchor_and_syncs_saved_zoom(self) -> None:
        view = self.window.board_view
        anchor = QPoint(view.viewport().width() // 3, view.viewport().height() // 3)
        for delta in (120, -120):
            with self.subTest(wheel_delta=delta):
                scene_before = view.mapToScene(anchor)
                zoom_before = self.window.zoom_box.value()
                self._send_wheel(anchor, delta)
                self.assertGreater((self.window.zoom_box.value() - zoom_before) * delta, 0)
                self.assertEqual(self.window.config.zoom, self.window.zoom_box.value() / 100)
                anchor_after = view.mapFromScene(scene_before)
                self.assertLessEqual(abs(anchor_after.x() - anchor.x()), 2)
                self.assertLessEqual(abs(anchor_after.y() - anchor.y()), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zoom.json"
            with patch("window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._save_dialog())
            self.assertEqual(load_config(path).zoom, self.window.config.zoom)

    def test_wheel_zoom_limits_and_pixel_scrolling(self) -> None:
        view = self.window.board_view
        anchor = view.viewport().rect().center()
        self._send_wheel(anchor, 60, pixel=True)
        self.assertGreater(self.window.zoom_box.value(), 300)
        for limit, delta in ((1000, 120), (.1, -120)):
            with self.subTest(zoom_percent=limit):
                self.window.zoom_box.setValue(limit)
                self._send_wheel(anchor, delta)
                self.assertEqual(self.window.zoom_box.value(), limit)
                self.assertEqual(self.window.config.zoom, limit / 100)

    def test_middle_button_drag_does_not_zoom(self) -> None:
        view = self.window.board_view
        anchor = view.viewport().rect().center()
        before = self.window.config.zoom
        QTest.mousePress(view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, anchor)
        QTest.mouseMove(view.viewport(), anchor + QPoint(0, -40))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, anchor + QPoint(0, -40))
        self.assertEqual(self.window.config.zoom, before)

    def test_fit_and_configuration_changes_recenter_the_board(self) -> None:
        view = self.window.board_view
        start = view.viewport().rect().center()
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(view.viewport(), start + QPoint(-120, -80))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(-120, -80))
        self.window.fit_button.click()
        self.app.processEvents()
        center = view.mapFromScene(view.board_item.boundingRect().center())
        self.assertLessEqual((center - view.viewport().rect().center()).manhattanLength(), 2)
        visible = view.mapToScene(view.viewport().rect()).boundingRect()
        self.assertTrue(visible.contains(view.board_item.boundingRect()))

        self.window.rows_box.setValue(1)
        self.window.columns_box.setValue(2)
        with patch.object(self.window, "_confirm", return_value=True):
            self.assertTrue(self.window._apply_grid())
        self.app.processEvents()
        center = view.mapFromScene(view.board_item.boundingRect().center())
        self.assertLessEqual((center - view.viewport().rect().center()).manhattanLength(), 2)


if __name__ == "__main__":
    unittest.main()
