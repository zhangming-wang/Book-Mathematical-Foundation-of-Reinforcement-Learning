from __future__ import annotations

from threading import Condition

from PySide6.QtCore import QThread, Signal

import bellman
from model import CalculationResult, GridConfig


class CalculationWorker(QThread):
    """Calculate on a frozen config; the UI requests and acknowledges each frame."""

    frame_ready = Signal(int, object)

    def __init__(self, config: GridConfig, max_count: int, optimal: bool, animated: bool, parent=None,
                 *, path_max_count: int | None = None):
        super().__init__(parent)
        self.config = config
        self.max_count = max_count
        self.optimal = optimal
        self.animated = animated
        self.path_max_count = path_max_count
        self.outcome = "running"
        self.error = ""
        self._condition = Condition()
        self._cancelled = False
        self._paused = False
        self._requested = True
        self._awaiting_display = False

    def request_next(self) -> None:
        with self._condition:
            self._requested = True
            self._condition.notify_all()

    def acknowledge_frame(self) -> None:
        with self._condition:
            self._awaiting_display = False
            self._condition.notify_all()

    def set_paused(self, paused: bool) -> None:
        with self._condition:
            self._paused = paused
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def _checkpoint(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._cancelled or not self._paused)
            if self._cancelled:
                raise bellman.CalculationCancelled()

    def _wait_for_request(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._cancelled or (self._requested and not self._paused))
            if self._cancelled:
                raise bellman.CalculationCancelled()
            self._requested = False

    def _publish(self, count: int, result: CalculationResult) -> None:
        self._checkpoint()
        with self._condition:
            self._awaiting_display = True
        self.frame_ready.emit(count, result)
        # Even the final frame must reach the UI before the run can complete.
        with self._condition:
            self._condition.wait_for(lambda: self._cancelled or not self._awaiting_display)
            if self._cancelled:
                raise bellman.CalculationCancelled()

    def run(self) -> None:
        try:
            if self.animated:
                rounds = bellman.iterate_calculation(
                    self.config, self.max_count, optimal=self.optimal,
                    path_max_count=self.path_max_count, checkpoint=self._checkpoint,
                )
                for _ in range(self.max_count):
                    self._wait_for_request()
                    count, result = next(rounds)
                    self._publish(count, result)
            else:
                if self.optimal:
                    result = bellman.solve_optimal_policy(
                        self.config, self.max_count, path_max_count=self.path_max_count,
                        checkpoint=self._checkpoint,
                    )
                else:
                    result = bellman.evaluate_policy(self.config, self.max_count, checkpoint=self._checkpoint)
                self._publish(self.max_count, result)
            self.outcome = "completed"
        except bellman.CalculationCancelled:
            self.outcome = "stopped"
        except Exception as error:
            self.outcome = "failed"
            self.error = str(error) or type(error).__name__
