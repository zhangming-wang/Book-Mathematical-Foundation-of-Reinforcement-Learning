import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from window import MainWindow


def _install_sigint_handler(app: QApplication) -> tuple[QTimer, object]:
    """Let terminal Ctrl+C quit the Qt event loop promptly."""
    timer = QTimer(app)
    timer.setInterval(100)
    timer.timeout.connect(lambda: None)
    timer.start()

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: app.quit())
    return timer, previous_handler


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Bellman Grid Lab")
    app.setStyle("Fusion")
    app.setFont(QFont("Sans Serif", 10))
    window = MainWindow()
    window.show()
    interrupt_timer, previous_handler = _install_sigint_handler(app)
    try:
        return app.exec()
    finally:
        interrupt_timer.stop()
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    sys.exit(main())
