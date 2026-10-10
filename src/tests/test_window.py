import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

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
        self.window.hide()
        self.window.deleteLater()
        self.app.processEvents()

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
        self.assertEqual(self.window.paths_label.text().splitlines()[0], "最高得分路径：2 条")

    def test_compute_button_shows_real_values_for_the_selected_horizon(self) -> None:
        for box in self.window.probability_boxes:
            box.setValue(0)
        self.window.probability_boxes[3].setValue(1)
        with patch.object(self.window, "_confirm", return_value=True):
            self.window.bulk_policy_button.click()
        self.window.max_count_box.setValue(3)
        with patch.object(self.window, "_message") as message:
            self.window.evaluate_button.click()
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
        message.assert_not_called()
        self.assertAlmostEqual(self.window.result.values[4][4], 10.0)
        self.assertIn("1000 步", self.window.result_label.text())

    def test_both_calculation_buttons_pass_the_selected_horizon(self) -> None:
        values = [[0.0] * 5 for _ in range(5)]
        policy = [[[0.2] * 5 for _ in range(5)] for _ in range(5)]
        for button, function in (
            (self.window.evaluate_button, "window.bellman.evaluate_policy"),
            (self.window.optimize_button, "window.bellman.solve_optimal_policy"),
        ):
            for steps in (1, 37, 1000):
                with self.subTest(function=function, steps=steps):
                    self.window.max_count_box.setValue(steps)
                    with patch(function, return_value=CalculationResult(values, policy)) as calculate:
                        button.click()
                    calculate.assert_called_once()
                    passed_config, passed_steps = calculate.call_args.args
                    self.assertEqual(passed_steps, steps)
                    self.assertEqual(passed_config, self.window.config)
                    self.assertIsNot(passed_config, self.window.config)

    def test_calculation_horizon_is_not_saved_as_grid_configuration(self) -> None:
        self.assertEqual(self.window.max_count_box.value(), 100)
        self.window.max_count_box.setValue(7)
        self.assertFalse(self.window.dirty)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.json"
            with patch("window.QFileDialog.getSaveFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._save_dialog())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("max_count", data)
            self.assertEqual(load_config(path), self.window.config)
            with patch("window.QFileDialog.getOpenFileName", return_value=(str(path), "")):
                self.assertTrue(self.window._load_dialog())
            self.assertEqual(self.window.max_count_box.value(), 7)

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
