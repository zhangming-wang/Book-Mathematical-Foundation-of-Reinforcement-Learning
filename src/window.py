from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QLocale, QPoint, QPointF, QRectF, Qt, QTimer, QSignalBlocker, Signal, Slot
from PySide6.QtGui import QColor, QDoubleValidator, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsItem, QGraphicsScene, QGraphicsView, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from calculation_worker import CalculationWorker
from model import (
    ACTIONS, DEFAULT_REWARDS, KINDS, MAX_DIMENSION, PROBABILITY_TOLERANCE,
    CalculationResult, CellConfig, GridConfig, PathResult, load_config, save_config,
)


KIND_LABELS = {"normal": "普通", "start": "开始", "target": "目标", "forbidden": "禁止"}
ACTION_LABELS = {"up": "上", "down": "下", "left": "左", "right": "右", "stay": "原地"}
CELL_COLORS = {"normal": "#ffffff", "start": "#ffffff", "target": "#91c6ff", "forbidden": "#ffdf6f"}
SCENE_UNITS_PER_MM = 10.0


def number_text(value: float) -> str:
    return format(value, ".4g")


class BoardItem(QGraphicsItem):
    def __init__(self, config: GridConfig):
        super().__init__()
        self.config = config
        self.selected = config.start
        self.result: CalculationResult | None = None
        self.use_optimal = False
        self.paths_visible = True
        self.path_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption)

    @property
    def cell_width(self) -> float:
        return self.config.cell_width_mm * SCENE_UNITS_PER_MM

    @property
    def cell_height(self) -> float:
        return self.config.cell_height_mm * SCENE_UNITS_PER_MM

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.config.columns * self.cell_width, self.config.rows * self.cell_height)

    def set_config(self, config: GridConfig) -> None:
        self.prepareGeometryChange()
        self.config = config
        self.update()

    def set_result(self, result: CalculationResult | None) -> None:
        self.result = result
        self.set_paths(result.best_paths if result is not None else [])

    def set_paths(self, paths: list[PathResult]) -> None:
        self.path_edges.clear()
        for path in paths:
            for first, second in zip(path.coordinates, path.coordinates[1:]):
                if first != second:
                    self.path_edges.add(tuple(sorted((first, second))))
        self.update()

    def probabilities(self, row: int, column: int) -> list[float]:
        if self.use_optimal and self.result is not None and self.result.optimal_policy is not None:
            return self.result.optimal_policy[row][column]
        return self.config.cells[row][column].probabilities

    @staticmethod
    def _text(painter: QPainter, rect: QRectF, text: str, size: float, color: str = "#203043",
              alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter) -> None:
        font = QFont("DejaVu Sans Mono")
        font.setPixelSize(max(1, round(size)))
        metrics = QFontMetricsF(font)
        width = metrics.horizontalAdvance(text)
        if width > rect.width():
            font.setPixelSize(max(1, int(size * rect.width() / width)))
        painter.setFont(font)
        painter.setPen(QColor(color))
        painter.drawText(rect, alignment, text)

    @staticmethod
    def _arrow(painter: QPainter, start: QPointF, end: QPointF, size: float) -> None:
        painter.drawLine(start, end)
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        head = QPolygonF([
            end,
            end - QPointF(size * math.cos(angle - 0.5), size * math.sin(angle - 0.5)),
            end - QPointF(size * math.cos(angle + 0.5), size * math.sin(angle + 0.5)),
        ])
        painter.drawPolygon(head)

    def _cell_content(self, painter: QPainter, row: int, column: int) -> None:
        w, h = self.cell_width, self.cell_height
        unit = min(w, h)
        cell = self.config.cells[row][column]
        painter.save()
        painter.translate(column * w, row * h)
        painter.setClipRect(QRectF(1, 1, w - 2, h - 2))
        reward_left = max(w * .04, unit * .18) if cell.kind == "start" else w * .04
        self._text(painter, QRectF(reward_left, h * .015, w * .96 - reward_left, h * .13), f"r={number_text(cell.reward)}", unit * .09)
        if self.result is not None:
            self._text(painter, QRectF(w * .04, h * .86, w * .92, h * .13), f"V={number_text(self.result.values[row][column])}", unit * .09, "#3d5372")

        arrows = (
            (QPointF(w * .5, h * .36), QPointF(w * .5, h * .18), QRectF(w * .5 + unit * .035, h * .20, w * .38, h * .14)),
            (QPointF(w * .5, h * .64), QPointF(w * .5, h * .82), QRectF(w * .5 + unit * .035, h * .66, w * .38, h * .14)),
            (QPointF(w * .36, h * .5), QPointF(w * .14, h * .5), QRectF(w * .03, h * .32, w * .32, h * .14)),
            (QPointF(w * .64, h * .5), QPointF(w * .86, h * .5), QRectF(w * .65, h * .32, w * .32, h * .14)),
        )
        probabilities = self.probabilities(row, column)
        for index, probability in enumerate(probabilities):
            if probability == 0:
                continue
            painter.setPen(QPen(QColor("#233243"), unit * .016))
            painter.setBrush(QColor("#233243"))
            if index < 4:
                start, end, text_rect = arrows[index]
                self._arrow(painter, start, end, unit * .055)
            else:
                center, radius = QPointF(w * .5, h * .5), unit * .11
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawArc(QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2), 45 * 16, 285 * 16)
                angle = math.radians(330)
                end = center + QPointF(radius * math.cos(angle), -radius * math.sin(angle))
                tangent = QPointF(-math.sin(angle), -math.cos(angle))
                painter.setBrush(QColor("#233243"))
                self._arrow(painter, end - tangent * radius * .35, end, unit * .035)
                text_rect = QRectF(center.x() - unit * .09, center.y() - unit * .06, unit * .18, unit * .12)
            alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter if index < 2 else Qt.AlignmentFlag.AlignCenter
            self._text(painter, text_rect, format(probability, ".3g"), unit * .09, alignment=alignment)

        if cell.kind == "start":
            center = QPointF(unit * .09, unit * .09)
            points = []
            for index in range(10):
                angle = -math.pi / 2 + index * math.pi / 5
                radius = unit * (.066 if index % 2 == 0 else .029)
                points.append(center + QPointF(math.cos(angle) * radius, math.sin(angle) * radius))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#b55612"))
            painter.drawPolygon(QPolygonF(points))
        painter.restore()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        exposed = option.exposedRect.intersected(self.boundingRect())
        w, h = self.cell_width, self.cell_height
        first_row = max(0, int(exposed.top() // h))
        last_row = min(self.config.rows - 1, int(exposed.bottom() // h))
        first_column = max(0, int(exposed.left() // w))
        last_column = min(self.config.columns - 1, int(exposed.right() // w))
        border = QPen(QColor("#b6c2cf"), 1)
        border.setCosmetic(True)
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                rect = QRectF(column * w, row * h, w, h)
                painter.fillRect(rect, QColor(CELL_COLORS[self.config.cells[row][column].kind]))
                painter.setPen(border)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)

        if self.paths_visible:
            pen = QPen(QColor("#17884b"), 4)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            for first, second in self.path_edges:
                painter.drawLine(QPointF((first[1] + .5) * w, (first[0] + .5) * h), QPointF((second[1] + .5) * w, (second[0] + .5) * h))

        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                self._cell_content(painter, row, column)
        row, column = self.selected
        pen = QPen(QColor("#c45b12"), 2.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(column * w + 1, row * h + 1, w - 2, h - 2))


class BoardView(QGraphicsView):
    cell_clicked = Signal(int, int)
    view_changed = Signal()
    zoom_changed = Signal(float)

    def __init__(self, config: GridConfig):
        super().__init__()
        self._initial_centering = True
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.board_item = BoardItem(config)
        scene.addItem(self.board_item)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#e9eef3"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self._zoom = config.zoom
        self._left_press_position: QPoint | None = None
        self._left_scene_anchor: QPointF | None = None
        self._left_dragging = False
        self.horizontalScrollBar().valueChanged.connect(self._scrollbar_changed)
        self.verticalScrollBar().valueChanged.connect(self._scrollbar_changed)
        self.set_zoom(config.zoom)

    @Slot(int)
    def _scrollbar_changed(self, value: int) -> None:
        self.view_changed.emit()

    def _center_view(self, center: QPointF) -> None:
        width = self.viewport().width() / self.transform().m11()
        height = self.viewport().height() / self.transform().m22()
        visible = QRectF(center.x() - width / 2, center.y() - height / 2, width, height)
        # Leave room to pan even when the whole board fits in the viewport.
        bounds = self.board_item.boundingRect().united(visible)
        self.setSceneRect(bounds.adjusted(-width, -height, width, height))
        self.centerOn(center)
        self.view_changed.emit()

    def set_zoom(self, zoom: float) -> None:
        center = self.board_item.boundingRect().center() if self._initial_centering else self.mapToScene(self.viewport().rect().center())
        self._zoom = zoom
        self.resetTransform()
        self.scale(self.logicalDpiX() / 25.4 / SCENE_UNITS_PER_MM * zoom,
                   self.logicalDpiY() / 25.4 / SCENE_UNITS_PER_MM * zoom)
        self._center_view(center)

    def _zoom_at(self, zoom: float, anchor: QPoint) -> None:
        zoom = round(max(.001, min(10.0, zoom)) * 100, 3) / 100
        if math.isclose(zoom, self._zoom, rel_tol=0.0, abs_tol=1e-9):
            return
        scene_anchor = self.mapToScene(anchor)
        self.set_zoom(zoom)
        viewport_center = self.viewport().rect().center()
        offset = QPointF((anchor.x() - viewport_center.x()) / self.transform().m11(),
                         (anchor.y() - viewport_center.y()) / self.transform().m22())
        self._center_view(scene_anchor - offset)

    def set_config(self, config: GridConfig) -> None:
        self.board_item.set_config(config)
        self.set_zoom(config.zoom)
        self._center_view(self.board_item.boundingRect().center())

    def coordinate_at(self, point) -> tuple[int, int] | None:
        position = self.mapToScene(point)
        if not self.board_item.boundingRect().contains(position):
            return None
        row = int(position.y() // self.board_item.cell_height)
        column = int(position.x() // self.board_item.cell_width)
        config = self.board_item.config
        if 0 <= row < config.rows and 0 <= column < config.columns:
            return row, column
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position().toPoint()
            self._left_press_position = position
            self._left_scene_anchor = self.viewportTransform().inverted()[0].map(event.position())
            self._left_dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        position = event.position().toPoint()
        if self._left_press_position is not None and self._left_scene_anchor is not None:
            distance = position - self._left_press_position
            if not self._left_dragging and distance.manhattanLength() >= QApplication.startDragDistance():
                self._left_dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._left_dragging:
                # Anchor to the press point so center rounding cannot accumulate.
                offset = QPointF((event.position().x() - self.viewport().width() / 2) / self.transform().m11(),
                                 (event.position().y() - self.viewport().height() / 2) / self.transform().m22())
                self._center_view(self._left_scene_anchor - offset)
                event.accept()
                return

        coordinate = self.coordinate_at(event.position().toPoint())
        if coordinate is None:
            self.setToolTip("")
        else:
            row, column = coordinate
            cell = self.board_item.config.cells[row][column]
            probabilities = self.board_item.probabilities(row, column)
            policy = "，".join(f"{ACTION_LABELS[action]}={probability:.9g}" for action, probability in zip(ACTIONS, probabilities))
            self.setToolTip(f"第 {row + 1} 行，第 {column + 1} 列 · {KIND_LABELS[cell.kind]}\n得分 r={cell.reward:.12g}\n{policy}")
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._left_press_position is not None:
            position = event.position().toPoint()
            was_dragging = self._left_dragging
            self._left_press_position = None
            self._left_scene_anchor = None
            self._left_dragging = False
            self.unsetCursor()
            if not was_dragging:
                coordinate = self.coordinate_at(position)
                if coordinate is not None:
                    self.cell_clicked.emit(*coordinate)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if delta:
            previous_zoom = self._zoom
            self._zoom_at(self._zoom * math.pow(1.2, delta / 120), event.position().toPoint())
            if self._zoom != previous_zoom:
                self.zoom_changed.emit(self._zoom)
        event.accept()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._initial_centering:
            self._center_view(self.board_item.boundingRect().center())
            self._initial_centering = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        center = self.board_item.boundingRect().center() if self._initial_centering else self.mapToScene(self.viewport().rect().center())
        self._center_view(center)


class AxisHeader(QWidget):
    def __init__(self, view: BoardView, horizontal: bool):
        super().__init__()
        self.view, self.horizontal = view, horizontal
        if horizontal:
            self.setFixedHeight(30)
        else:
            self.setFixedWidth(40)
        view.view_changed.connect(self._view_changed)

    @Slot()
    def _view_changed(self) -> None:
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f2f5f8"))
        painter.setPen(QColor("#53677d"))
        item = self.view.board_item
        count = item.config.columns if self.horizontal else item.config.rows
        size = item.cell_width if self.horizontal else item.cell_height
        scale = self.view.transform().m11() if self.horizontal else self.view.transform().m22()
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        first = max(0, int((visible.left() if self.horizontal else visible.top()) // size))
        last = min(count - 1, int((visible.right() if self.horizontal else visible.bottom()) // size))
        font = QFont(self.font())
        font.setPixelSize(max(5, min(12, int(size * scale * .45))))
        painter.setFont(font)
        for index in range(first, last + 1):
            center = (index + .5) * size
            position = self.view.mapFromScene(QPointF(center, 0) if self.horizontal else QPointF(0, center))
            if self.horizontal:
                rect = QRectF(position.x() - size * scale / 2, 0, size * scale, self.height())
            else:
                rect = QRectF(0, position.y() - size * scale / 2, self.width(), size * scale)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(index + 1))


def double_box(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setLocale(QLocale.c())
    box.setDecimals(3)
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


class MainWindow(QMainWindow):
    def __init__(self, config: GridConfig | None = None):
        super().__init__()
        self.config = deepcopy(config) if config is not None else GridConfig.create()
        self.config.validate()
        self.selected = self.config.start
        self.file_path: Path | None = None
        self.dirty = False
        self.result: CalculationResult | None = None
        self._calculation_worker: CalculationWorker | None = None
        self._displayed_round = 0
        self._calculation_total = 0
        self._calculation_optimal = False
        self._animated = False
        self._paused = False
        self._stopping = False
        self._waiting_for_next = False
        self._pending_frame = None
        self._calculation_error = ""
        self._reset_after_calculation = False
        self._close_after_calculation = False
        self._playback_timer = QTimer(self)
        self._playback_timer.setSingleShot(True)
        self._playback_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._playback_timer.timeout.connect(self._request_next_round)
        self._loading = True
        self._cell_baseline = None
        self._grid_baseline = None
        self.resize(1260, 860)
        self.setMinimumSize(820, 600)
        self.setStyleSheet("""
            QMainWindow, QWidget#sidePanel { background: #f4f6f8; }
            QGroupBox { font-weight: 600; border: 1px solid #d3dce5;
                        border-radius: 3px; margin-top: 12px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { padding: 5px 10px; }
            QPushButton#primary { background: #245d86; color: white; border: 1px solid #245d86; border-radius: 3px; }
            QPushButton#primary:disabled { background: #b4c2ce; border-color: #b4c2ce; }
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox { min-height: 24px; }
        """)
        self._build_ui()
        self._load_grid_editor()
        self._load_cell_editor()
        self._loading = False
        self._editor_changed()
        self._update_document_status()
        QApplication.instance().aboutToQuit.connect(self.shutdown_calculation)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 12, 12, 12)
        title_row = QHBoxLayout()
        title = QLabel("策略棋盘")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #203c53;")
        title_row.addWidget(title)
        title_row.addStretch()
        self.board_summary = QLabel()
        title_row.addWidget(self.board_summary)
        left_layout.addLayout(title_row)
        view_tools = QHBoxLayout()
        self.policy_mode = QComboBox()
        self.policy_mode.addItems(["当前策略", "最优策略"])
        self.policy_mode.model().item(1).setEnabled(False)
        self.policy_mode.currentIndexChanged.connect(self._change_policy_mode)
        view_tools.addWidget(self.policy_mode)
        view_tools.addStretch()
        self.zoom_box = double_box(.1, 1000, self.config.zoom * 100)
        self.zoom_box.setSuffix(" %")
        self.zoom_box.setMaximumWidth(105)
        self.zoom_box.setSingleStep(25)
        self.zoom_box.valueChanged.connect(self._zoom_changed)
        view_tools.addWidget(self.zoom_box)
        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("＋")
        self.zoom_out_button.setToolTip("缩小棋盘")
        self.zoom_in_button.setToolTip("放大棋盘")
        for button in (self.zoom_out_button, self.zoom_in_button):
            button.setFixedWidth(32)
            view_tools.addWidget(button)
        self.zoom_out_button.clicked.connect(lambda: self.zoom_box.setValue(self.zoom_box.value() / 1.25))
        self.zoom_in_button.clicked.connect(lambda: self.zoom_box.setValue(self.zoom_box.value() * 1.25))
        self.fit_button = QPushButton("适应窗口")
        self.fit_button.clicked.connect(self._fit_board)
        view_tools.addWidget(self.fit_button)
        left_layout.addLayout(view_tools)
        axes = QGridLayout()
        axes.setSpacing(0)
        self.board_view = BoardView(self.config)
        self.board_view.cell_clicked.connect(self._select_cell)
        self.board_view.zoom_changed.connect(self._board_zoom_changed)
        self.column_header = AxisHeader(self.board_view, True)
        self.row_header = AxisHeader(self.board_view, False)
        axes.addWidget(self.column_header, 0, 1)
        axes.addWidget(self.row_header, 1, 0)
        axes.addWidget(self.board_view, 1, 1)
        axes.setRowStretch(1, 1)
        axes.setColumnStretch(1, 1)
        left_layout.addLayout(axes, 1)
        legend = QLabel('<span style="color:#b55612">★</span> 开始　 '
                        '<span style="background:#91c6ff">　</span> 目标　 '
                        '<span style="background:#ffdf6f">　</span> 禁止　 '
                        '<span style="color:#17884b">━</span> 最高得分路径')
        left_layout.addWidget(legend)

        side = QWidget()
        side.setObjectName("sidePanel")
        side.setMinimumWidth(330)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 12, 12, 12)
        file_group = QGroupBox("配置文件")
        file_layout = QVBoxLayout(file_group)
        file_buttons = QHBoxLayout()
        self.save_button = QPushButton("保存配置")
        self.save_button.setObjectName("primary")
        self.load_button = QPushButton("加载配置")
        self.save_button.clicked.connect(self._save_dialog)
        self.load_button.clicked.connect(self._load_dialog)
        file_buttons.addWidget(self.save_button)
        file_buttons.addWidget(self.load_button)
        file_layout.addLayout(file_buttons)
        self.file_label = QLabel()
        self.file_label.setTextFormat(Qt.TextFormat.PlainText)
        file_layout.addWidget(self.file_label)
        side_layout.addWidget(file_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        grid_group = QGroupBox("棋盘设置")
        self.grid_settings_group = grid_group
        grid_layout = QVBoxLayout(grid_group)
        form = QFormLayout()
        self.rows_box, self.columns_box = QSpinBox(), QSpinBox()
        for box in (self.rows_box, self.columns_box):
            box.setRange(1, MAX_DIMENSION)
            box.setKeyboardTracking(False)
        self.width_box = double_box(.01, 1000, self.config.cell_width_mm)
        self.height_box = double_box(.01, 1000, self.config.cell_height_mm)
        self.width_box.setSuffix(" mm")
        self.height_box.setSuffix(" mm")
        self.gamma_box = double_box(0, 1, self.config.gamma)
        self.gamma_box.setSingleStep(.05)
        for label, box in (("行数", self.rows_box), ("列数", self.columns_box),
                           ("方块宽度", self.width_box), ("方块高度", self.height_box),
                           ("折扣系数 γ", self.gamma_box)):
            form.addRow(label, box)
            box.valueChanged.connect(self._editor_changed)
        grid_layout.addLayout(form)
        self.grid_hint = QLabel()
        self.grid_hint.setWordWrap(True)
        grid_layout.addWidget(self.grid_hint)
        self.apply_grid_button = QPushButton("应用棋盘设置")
        self.apply_grid_button.clicked.connect(self._apply_grid)
        grid_layout.addWidget(self.apply_grid_button)
        forbidden_form = QFormLayout()
        self.forbidden_count_box = QSpinBox()
        self.forbidden_count_box.setKeyboardTracking(False)
        forbidden_form.addRow("禁止格数量", self.forbidden_count_box)
        grid_layout.addLayout(forbidden_form)
        forbidden_hint = QLabel("重新生成共 n 个：避开开始和目标；0 可清除。")
        forbidden_hint.setWordWrap(True)
        grid_layout.addWidget(forbidden_hint)
        self.randomize_forbidden_button = QPushButton("随机生成禁止格")
        self.randomize_forbidden_button.setToolTip("原有禁止格恢复普通格、得分 0；随机选中格设为禁止格、得分 -1；保留各格动作概率。")
        self.randomize_forbidden_button.clicked.connect(self._randomize_forbidden)
        grid_layout.addWidget(self.randomize_forbidden_button)
        content_layout.addWidget(grid_group)

        cell_group = QGroupBox("选中方块")
        cell_layout = QVBoxLayout(cell_group)
        cell_form = QFormLayout()
        self.kind_box = QComboBox()
        for kind in KINDS:
            self.kind_box.addItem(KIND_LABELS[kind], kind)
        self.kind_box.currentIndexChanged.connect(self._kind_changed)
        self.reward_edit = QLineEdit()
        self.reward_edit.setPlaceholderText("例如 0、-1、1.5")
        validator = QDoubleValidator(self.reward_edit)
        validator.setLocale(QLocale.c())
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.reward_edit.setValidator(validator)
        self.reward_edit.textChanged.connect(self._editor_changed)
        cell_form.addRow("方块类型", self.kind_box)
        cell_form.addRow("得分 r", self.reward_edit)
        self.probability_boxes = []
        for action in ACTIONS:
            box = double_box(0, 1, .2)
            box.setSingleStep(.05)
            box.valueChanged.connect(self._editor_changed)
            self.probability_boxes.append(box)
            cell_form.addRow(f"{ACTION_LABELS[action]}的概率", box)
        cell_layout.addLayout(cell_form)
        self.probability_total_label = QLabel()
        cell_layout.addWidget(self.probability_total_label)
        self.cell_hint = QLabel()
        self.cell_hint.setWordWrap(True)
        cell_layout.addWidget(self.cell_hint)
        cell_buttons = QHBoxLayout()
        self.apply_cell_button = QPushButton("应用当前方块")
        self.apply_cell_button.setObjectName("primary")
        self.apply_cell_button.clicked.connect(self._apply_cell)
        self.reset_cell_button = QPushButton("恢复默认输入")
        self.reset_cell_button.setToolTip("恢复当前类型的默认得分及五个 0.2 概率，点击应用后生效")
        self.reset_cell_button.clicked.connect(self._reset_cell_inputs)
        cell_buttons.addWidget(self.apply_cell_button)
        cell_buttons.addWidget(self.reset_cell_button)
        cell_layout.addLayout(cell_buttons)
        self.bulk_policy_button = QPushButton("将这组概率应用到全部方块")
        self.bulk_policy_button.setToolTip("仅复制五项概率，保留每格已有的类型和得分")
        self.bulk_policy_button.clicked.connect(self._apply_policy_to_all)
        cell_layout.addWidget(self.bulk_policy_button)
        content_layout.addWidget(cell_group)

        result_group = QGroupBox("计算与结果")
        result_layout = QVBoxLayout(result_group)
        count_form = QFormLayout()
        self.max_count_box = QSpinBox()
        self.max_count_box.setRange(1, 1000)
        self.max_count_box.setValue(100)
        self.max_count_box.setKeyboardTracking(False)
        self.max_count_box.setToolTip("每轮更新整张棋盘；第 n 轮计算未来 n 步的价值，与路径搜索长度分开。")
        count_form.addRow("计算轮数", self.max_count_box)
        self.path_max_count_box = QSpinBox()
        self.path_max_count_box.setRange(1, self.config.rows * self.config.columns - 1)
        self.path_max_count_box.setValue(self.config.rows * self.config.columns - 1)
        self.path_max_count_box.setKeyboardTracking(False)
        self.path_max_count_box.setToolTip("每轮按该轮策略搜索最多这些步的目标路径，独立于计算轮数；不重复经过方块。")
        count_form.addRow("路径最大步数", self.path_max_count_box)
        self.calculation_mode_box = QComboBox()
        self.calculation_mode_box.addItem("一次算完", False)
        self.calculation_mode_box.addItem("逐轮播放", True)
        self.calculation_mode_box.currentIndexChanged.connect(self._update_playback_controls)
        count_form.addRow("计算方式", self.calculation_mode_box)
        self.interval_box = double_box(0, 60, .5)
        self.interval_box.setSingleStep(.1)
        self.interval_box.setSuffix(" 秒")
        self.interval_box.setToolTip("每轮结果显示后等待的时间，播放中可调整。")
        self.interval_box.setEnabled(False)
        self.interval_box.valueChanged.connect(self._interval_changed)
        count_form.addRow("播放间隔", self.interval_box)
        result_layout.addLayout(count_form)
        calculation_buttons = QHBoxLayout()
        self.evaluate_button = QPushButton("贝尔曼方程")
        self.optimize_button = QPushButton("贝尔曼最优方程")
        self.evaluate_button.clicked.connect(lambda: self._calculate(False))
        self.optimize_button.clicked.connect(lambda: self._calculate(True))
        calculation_buttons.addWidget(self.evaluate_button, 1)
        calculation_buttons.addWidget(self.optimize_button, 1)
        result_layout.addLayout(calculation_buttons)
        self.reset_board_button = QPushButton("重置棋盘")
        self.reset_board_button.setToolTip("清除价值、最优策略和绿色路径，显示配置中的动作概率；计算中会先停止任务。")
        self.reset_board_button.clicked.connect(self._reset_board)
        playback_buttons = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_calculation)
        playback_buttons.addWidget(self.pause_button, 1)
        playback_buttons.addWidget(self.stop_button, 1)
        playback_buttons.addWidget(self.reset_board_button, 1)
        result_layout.addLayout(playback_buttons)
        self.progress_label = QLabel("尚未开始")
        result_layout.addWidget(self.progress_label)
        self.result_label = QLabel("尚未计算")
        self.value_label = QLabel("选中格状态价值 V：尚未计算")
        self.paths_label = QLabel("最高得分路径：尚未计算")
        self.paths_label.setToolTip("在显示策略允许的、不重复经过方块的目标路径中比较得分；回报只累计到首次到达目标。")
        for label in (self.result_label, self.value_label):
            label.setWordWrap(True)
            result_layout.addWidget(label)
        path_form = QFormLayout()
        self.path_selection_box = QComboBox()
        self.path_selection_box.setEnabled(False)
        self.path_selection_box.setMaxVisibleItems(12)
        self.path_selection_box.setToolTip("默认显示全部并列最优路径；按得分从大到小列出每条路线，选择后单独显示。")
        self.path_selection_box.currentIndexChanged.connect(self._change_path_selection)
        path_form.addRow("路径选择", self.path_selection_box)
        result_layout.addLayout(path_form)
        self.paths_label.setWordWrap(True)
        result_layout.addWidget(self.paths_label)
        self.show_paths_box = QCheckBox("显示绿色路径")
        self.show_paths_box.setChecked(True)
        self.show_paths_box.setEnabled(False)
        self.show_paths_box.toggled.connect(self._toggle_paths)
        result_layout.addWidget(self.show_paths_box)
        self.max_count_box.valueChanged.connect(lambda value: self._clear_result())
        self.path_max_count_box.valueChanged.connect(self._path_limit_changed)
        content_layout.addWidget(result_group)
        content_layout.addStretch()
        scroll.setWidget(content)
        side_layout.addWidget(scroll, 1)
        splitter.addWidget(left)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([850, 380])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("左键单击选格，按住拖动平移；滚动鼠标滚轮围绕鼠标位置缩放。")

    def _cell_snapshot(self):
        return (self.kind_box.currentData(), self.reward_edit.text().strip(),
                tuple(box.value() for box in self.probability_boxes))

    def _grid_snapshot(self):
        return (self.rows_box.value(), self.columns_box.value(), self.width_box.value(),
                self.height_box.value(), self.gamma_box.value())

    def _cell_pending(self) -> bool:
        return self._cell_baseline is not None and self._cell_snapshot() != self._cell_baseline

    def _grid_pending(self) -> bool:
        return self._grid_baseline is not None and self._grid_snapshot() != self._grid_baseline

    def _load_grid_editor(self) -> None:
        previous, self._loading = self._loading, True
        for box, value in zip((self.rows_box, self.columns_box, self.width_box, self.height_box, self.gamma_box),
                              (self.config.rows, self.config.columns, self.config.cell_width_mm, self.config.cell_height_mm, self.config.gamma)):
            box.setValue(value)
        self.zoom_box.setValue(self.config.zoom * 100)
        self.forbidden_count_box.setRange(0, self.config.rows * self.config.columns - 2)
        self.forbidden_count_box.setValue(sum(cell.kind == "forbidden" for row in self.config.cells for cell in row))
        was_maximum = self.path_max_count_box.value() == self.path_max_count_box.maximum()
        self.path_max_count_box.setMaximum(self.config.rows * self.config.columns - 1)
        if was_maximum:
            self.path_max_count_box.setValue(self.path_max_count_box.maximum())
        self._grid_baseline = self._grid_snapshot()
        self._loading = previous

    def _load_cell_editor(self) -> None:
        previous, self._loading = self._loading, True
        row, column = self.selected
        cell = self.config.cells[row][column]
        for index, kind in enumerate(KINDS):
            self.kind_box.model().item(index).setEnabled(cell.kind not in ("start", "target") or kind == cell.kind)
        self.kind_box.setCurrentIndex(KINDS.index(cell.kind))
        self.reward_edit.setText(format(cell.reward, ".12g"))
        for box, probability in zip(self.probability_boxes, cell.probabilities):
            box.setValue(probability)
        self._cell_baseline = self._cell_snapshot()
        self._loading = previous
        self._update_result_labels()

    def _probability_values(self) -> list[float]:
        current = self.config.cells[self.selected[0]][self.selected[1]]
        return [original if box.value() == baseline else box.value() for original, box, baseline
                in zip(current.probabilities, self.probability_boxes, self._cell_baseline[2])]

    def _cell_values(self) -> tuple[str, float, list[float]]:
        current = self.config.cells[self.selected[0]][self.selected[1]]
        kind, reward_text, _ = self._cell_snapshot()
        # Preserve loaded precision when a displayed, rounded field was not edited.
        reward = current.reward if reward_text == self._cell_baseline[1] else float(reward_text)
        values = self._probability_values()
        CellConfig(kind, reward, values).validate()
        return kind, reward, values

    def _editor_changed(self, *args) -> None:
        if self._loading:
            return
        total = math.fsum(self._probability_values())
        valid_probabilities = abs(total - 1) <= PROBABILITY_TOLERANCE
        self.probability_total_label.setText(f"概率合计：{total:.3f}  /  1.000")
        self.probability_total_label.setToolTip(f"实际概率合计：{total:.12g}")
        self.probability_total_label.setStyleSheet("color: #226e49;" if valid_probabilities else "color: #b33125;")
        try:
            self._cell_values()
            valid_cell = True
        except ValueError:
            valid_cell = False
        current = self.config.cells[self.selected[0]][self.selected[1]]
        if not valid_probabilities:
            hint = "五个动作的概率合计必须为 1，请调整后应用。"
        elif not valid_cell:
            hint = "请输入有效的有限数值得分。"
        elif self._cell_pending():
            hint = "当前修改尚未应用到棋盘。"
        elif current.kind in ("start", "target"):
            hint = f"要移动{KIND_LABELS[current.kind]}位置，请在其他方块设置新的{KIND_LABELS[current.kind]}格。"
        else:
            hint = "修改后点击“应用当前方块”。"
        self.cell_hint.setText(hint)
        idle = self._calculation_worker is None
        self.apply_cell_button.setEnabled(idle and valid_cell and self._cell_pending())
        self.bulk_policy_button.setEnabled(idle and valid_probabilities)
        self.grid_hint.setText("参数尚未应用；行列改变时将重建棋盘。" if self._grid_pending() else "改变宽高或 γ 会保留各格配置；改变行列会重建棋盘。")
        self.apply_grid_button.setEnabled(idle and self._grid_pending())
        self._update_document_status()

    def _kind_changed(self, *args) -> None:
        if not self._loading:
            self.reward_edit.setText(format(DEFAULT_REWARDS[self.kind_box.currentData()], "g"))
            self._editor_changed()

    def _update_document_status(self) -> None:
        changed = self.dirty or self._cell_pending() or self._grid_pending()
        name = self.file_path.name if self.file_path is not None else "新棋盘"
        state = "已修改，尚未保存" if changed else ("已保存" if self.file_path is not None else "尚未保存")
        shown_name = name if len(name) <= 26 else name[:23] + "…"
        self.file_label.setText(f"{shown_name} · {state}")
        self.file_label.setToolTip(str(self.file_path) if self.file_path else "点击保存配置，选择 JSON 文件。")
        self.setWindowTitle(f"{'* ' if changed else ''}{name} · 贝尔曼棋盘")
        self.board_summary.setText(f"{self.config.rows} × {self.config.columns}  ·  {self.config.cell_width_mm:g} × {self.config.cell_height_mm:g} mm")

    def _message(self, title: str, text: str, error: bool = False) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setIcon(QMessageBox.Icon.Critical if error else QMessageBox.Icon.Information)
        dialog.setText(text)
        dialog.addButton("确定", QMessageBox.ButtonRole.AcceptRole)
        dialog.exec()

    def _confirm(self, title: str, text: str, action: str = "继续") -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(text)
        accept = dialog.addButton(action, QMessageBox.ButtonRole.AcceptRole)
        cancel = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        return dialog.clickedButton() == accept

    def _apply_cell(self) -> bool:
        if not self._cell_pending():
            return True
        if self._calculation_worker is not None:
            return False
        try:
            kind, reward, probabilities = self._cell_values()
            self.config.set_cell(*self.selected, kind, reward, probabilities)
        except ValueError as error:
            self._message("方块配置未应用", f"请检查得分、概率和开始／目标位置。\n{error}", True)
            return False
        self.dirty = True
        self._clear_result()
        self._load_cell_editor()
        self.board_view.board_item.update()
        self._editor_changed()
        self.statusBar().showMessage("当前方块已应用。", 5000)
        return True

    def _resolve_cell_edits(self) -> bool:
        if not self._cell_pending():
            return True
        dialog = QMessageBox(self)
        dialog.setWindowTitle("当前方块尚未应用")
        dialog.setText("切换前如何处理当前方块的修改？")
        apply = dialog.addButton("应用", QMessageBox.ButtonRole.AcceptRole)
        discard = dialog.addButton("放弃修改", QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        if dialog.clickedButton() == apply:
            return self._apply_cell()
        if dialog.clickedButton() == discard:
            self._load_cell_editor()
            self._editor_changed()
            return True
        return False

    def _select_cell(self, row: int, column: int) -> None:
        if (row, column) == self.selected or not self._resolve_cell_edits():
            return
        self.selected = (row, column)
        self.board_view.board_item.selected = self.selected
        self.board_view.board_item.update()
        self._load_cell_editor()
        self._editor_changed()

    def _reset_cell_inputs(self) -> None:
        if self._calculation_worker is not None:
            return
        self.reward_edit.setText(format(DEFAULT_REWARDS[self.kind_box.currentData()], "g"))
        for box in self.probability_boxes:
            box.setValue(.2)
        self._editor_changed()

    def _apply_policy_to_all(self) -> None:
        if self._calculation_worker is not None:
            return
        probabilities = self._probability_values()
        if not self._confirm("统一动作概率", "将这五个概率应用到全部方块？每格已有的类型和得分将保留。", "应用到全部方块"):
            return
        try:
            self.config.apply_policy_to_all(probabilities)
        except ValueError as error:
            self._message("概率未应用", str(error), True)
            return
        kind, reward_text, _ = self._cell_baseline
        self._cell_baseline = (kind, reward_text, tuple(box.value() for box in self.probability_boxes))
        self.dirty = True
        self._clear_result()
        self.board_view.board_item.update()
        self._editor_changed()
        self.statusBar().showMessage("五项概率已应用到全部方块。", 5000)

    def _randomize_forbidden(self) -> bool:
        if self._calculation_worker is not None:
            return False
        if self._grid_pending():
            self._message("棋盘参数尚未应用", "请先点击“应用棋盘设置”，再随机生成禁止格。")
            return False
        if not self._resolve_cell_edits():
            return False

        count = self.forbidden_count_box.value()
        try:
            self.config.randomize_forbidden(count)
        except ValueError as error:
            self._message("禁止格未生成", str(error), True)
            return False
        self.dirty = True
        self._clear_result()
        self._load_cell_editor()
        self.board_view.board_item.update()
        self._editor_changed()
        message = f"已随机生成共 {count} 个禁止格。" if count else "已清除全部禁止格。"
        self.statusBar().showMessage(message, 5000)
        return True

    def _apply_grid(self) -> bool:
        if self._calculation_worker is not None:
            return False
        values = self._grid_snapshot()
        old = (self.config.rows, self.config.columns, self.config.cell_width_mm, self.config.cell_height_mm, self.config.gamma)
        values = tuple(original if shown == baseline else shown for original, shown, baseline in zip(old, values, self._grid_baseline))
        rows, columns, width, height, gamma = values
        rebuilding = (rows, columns) != (self.config.rows, self.config.columns)
        try:
            if rebuilding:
                candidate = GridConfig.create(rows, columns, width, height, gamma, self.config.zoom)
            else:
                candidate = deepcopy(self.config)
                candidate.cell_width_mm, candidate.cell_height_mm, candidate.gamma = width, height, gamma
                candidate.validate()
        except ValueError as error:
            self._message("棋盘设置未应用", f"行列范围为 1–{MAX_DIMENSION}，至少需要两个方块。\n{error}", True)
            return False
        if rebuilding:
            if not self._confirm("重新生成棋盘", "行列数已改变，将重置所有方块的类型、得分和概率，以及当前未应用的编辑。", "重新生成"):
                return False
        elif not self._resolve_cell_edits():
            return False
        else:
            candidate.cells = deepcopy(self.config.cells)
        self.config = candidate
        self.dirty = True
        if rebuilding:
            self.selected = self.config.start
        self._clear_result()
        self.board_view.set_config(self.config)
        self.board_view.board_item.selected = self.selected
        self._load_grid_editor()
        self._load_cell_editor()
        self._editor_changed()
        self.statusBar().showMessage("棋盘设置已应用。", 5000)
        return True

    def _zoom_changed(self, value: float) -> None:
        if self._loading:
            return
        self.config.zoom = value / 100
        self.board_view.set_zoom(self.config.zoom)
        self.dirty = True
        self._update_document_status()

    def _board_zoom_changed(self, zoom: float) -> None:
        if self._loading:
            return
        self.config.zoom = zoom
        self.zoom_box.blockSignals(True)
        try:
            self.zoom_box.setValue(zoom * 100)
        finally:
            self.zoom_box.blockSignals(False)
        self.dirty = True
        self._update_document_status()

    def _fit_board(self) -> None:
        rect = self.board_view.board_item.boundingRect()
        viewport = self.board_view.viewport()
        zoom = min((viewport.width() - 12) / rect.width() / (self.board_view.logicalDpiX() / 25.4 / SCENE_UNITS_PER_MM),
                   (viewport.height() - 12) / rect.height() / (self.board_view.logicalDpiY() / 25.4 / SCENE_UNITS_PER_MM))
        self.zoom_box.setValue(max(.1, min(1000, zoom * 100)))
        self.board_view._center_view(rect.center())

    def _save_dialog(self) -> bool:
        if self._grid_pending():
            self._message("棋盘参数尚未应用", "请先点击“应用棋盘设置”，再保存配置。")
            return False
        if not self._apply_cell():
            return False
        suggested = str(self.file_path) if self.file_path else "棋盘配置.json"
        path, _ = QFileDialog.getSaveFileName(self, "保存棋盘配置", suggested, "棋盘配置 (*.json)")
        if not path:
            return False
        if not Path(path).suffix:
            path += ".json"
        try:
            save_config(self.config, path)
        except (OSError, ValueError) as error:
            self._message("保存失败", f"配置没有保存，请检查目标文件位置。\n{error}", True)
            return False
        self.file_path = Path(path)
        self.dirty = False
        self._update_document_status()
        self.statusBar().showMessage(f"配置已保存：{self.file_path.name}", 7000)
        return True

    def _confirm_unsaved(self, action: str) -> bool:
        if not (self.dirty or self._cell_pending() or self._grid_pending()):
            return True
        dialog = QMessageBox(self)
        dialog.setWindowTitle("存在未保存修改")
        dialog.setText(f"{action}前是否保存当前棋盘？")
        save = dialog.addButton("保存", QMessageBox.ButtonRole.AcceptRole)
        discard = dialog.addButton("放弃", QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        if dialog.clickedButton() == save:
            return self._save_dialog()
        return dialog.clickedButton() == discard

    def _load_dialog(self) -> bool:
        if self._calculation_worker is not None:
            return False
        path, _ = QFileDialog.getOpenFileName(self, "加载棋盘配置", str(self.file_path.parent) if self.file_path else "", "棋盘配置 (*.json)")
        if not path:
            return False
        try:
            candidate = load_config(path)
        except (OSError, ValueError, RecursionError) as error:
            self._message("加载失败", f"当前棋盘已保留。请检查文件格式、唯一开始／目标格和概率合计。\n{error}", True)
            return False
        if not self._confirm_unsaved("加载其他配置"):
            return False
        try:
            # Saving in the confirmation dialog may have updated this same file.
            candidate = load_config(path)
        except (OSError, ValueError, RecursionError) as error:
            self._message("加载失败", f"当前棋盘已保留。\n{error}", True)
            return False
        self.config = candidate
        self.selected = candidate.start
        self.file_path = Path(path)
        self.dirty = False
        self._clear_result()
        self.board_view.set_config(candidate)
        self.board_view.board_item.selected = self.selected
        self._load_grid_editor()
        self._load_cell_editor()
        self._editor_changed()
        self.statusBar().showMessage(f"配置已加载：{self.file_path.name}", 7000)
        return True

    def _reset_board(self) -> None:
        if self._calculation_worker is not None:
            self._reset_after_calculation = True
            self._stop_calculation()
            self.progress_label.setText("正在重置棋盘…")
            return

        self._reset_after_calculation = False
        self._playback_timer.stop()
        self._waiting_for_next = False
        self._pending_frame = None
        self._paused = False
        self._stopping = False
        self._calculation_error = ""
        self._calculation_total = 0
        self._clear_result()
        self.result_label.setText("尚未计算")
        self._set_calculation_controls()
        self.statusBar().showMessage("已重置棋盘显示，可重新计算。", 5000)

    def _clear_result(self) -> None:
        if self._calculation_worker is not None:
            self._stop_calculation()
        self._displayed_round = 0
        self.result = None
        self.board_view.board_item.set_result(None)
        self.policy_mode.setCurrentIndex(0)
        self.policy_mode.model().item(1).setEnabled(False)
        with QSignalBlocker(self.path_selection_box):
            self.path_selection_box.clear()
        self.path_selection_box.setEnabled(False)
        self.show_paths_box.setEnabled(False)
        self.progress_label.setText("尚未开始")
        self._update_result_labels()

    def _path_limit_changed(self, *args) -> None:
        if not self._loading:
            self._clear_result()

    def _update_result_labels(self) -> None:
        if self.result is None:
            self.result_label.setText("公式尚未接入" if self.result_label.text() == "公式尚未接入" else "尚未计算")
            self.value_label.setText("选中格状态价值 V：尚未计算")
            self.paths_label.setText("最高得分路径：尚未计算")
            return
        row, column = self.selected
        self.value_label.setText(f"选中格状态价值 V：{self.result.values[row][column]:.12g}")
        paths = self.result.best_paths
        all_paths = self.result.all_paths or paths
        if not all_paths:
            self.paths_label.setText("最高得分路径：当前策略下没有可展示的目标路径")
        else:
            self.paths_label.setText(
                f"最高得分路径：{len(paths)} 条  可选目标路径：{len(all_paths)} 条"
            )

    def show_result(self, result: CalculationResult, optimal: bool = False) -> None:
        if not isinstance(result, CalculationResult):
            raise ValueError("计算接口必须返回 CalculationResult。")
        result.validate(self.config)
        if optimal and result.optimal_policy is None:
            raise ValueError("最优方程接口需要返回 optimal_policy。")
        previous_path = self.path_selection_box.currentData()
        previous_coordinates = previous_path.coordinates if previous_path is not None else None
        self.result = deepcopy(result)
        self.board_view.board_item.set_result(self.result)
        available = result.optimal_policy is not None
        self.policy_mode.model().item(1).setEnabled(available)
        self.policy_mode.setCurrentIndex(1 if optimal else 0)
        paths = sorted(self.result.all_paths or self.result.best_paths, key=lambda path: path.score, reverse=True)
        with QSignalBlocker(self.path_selection_box):
            self.path_selection_box.clear()
            selected_index = 0
            if paths:
                self.path_selection_box.addItem("最优路径")
                self.path_selection_box.setItemData(
                    0, f"同时显示 {len(self.result.best_paths)} 条并列最高分路径", Qt.ItemDataRole.ToolTipRole,
                )
                for index, path in enumerate(paths, 1):
                    self.path_selection_box.addItem(f"{path.score} - {len(path.coordinates) - 1} 步", path)
                    coordinates = " → ".join(f"({i + 1}, {j + 1})" for i, j in path.coordinates)
                    self.path_selection_box.setItemData(
                        index, f"路径 {index} · {len(path.coordinates) - 1} 步\n{coordinates}",
                        Qt.ItemDataRole.ToolTipRole,
                    )
                    if path.coordinates == previous_coordinates:
                        selected_index = index
                self.path_selection_box.setCurrentIndex(selected_index)
        self.path_selection_box.setEnabled(bool(paths))
        self.show_paths_box.setEnabled(bool(paths))
        self.result_label.setText("已返回最优策略与状态价值" if optimal else "已返回当前策略的状态价值")
        self._change_path_selection(self.path_selection_box.currentIndex())

    def _calculate(self, optimal: bool) -> None:
        if self._calculation_worker is not None:
            return
        if self._grid_pending():
            self._message("棋盘参数尚未应用", "请先点击“应用棋盘设置”。")
            return
        if not self._apply_cell():
            return
        self._clear_result()
        self._calculation_total = self.max_count_box.value()
        self._calculation_optimal = optimal
        self._animated = bool(self.calculation_mode_box.currentData())
        self._paused = False
        self._stopping = False
        self._waiting_for_next = False
        self._pending_frame = None
        self._calculation_error = ""
        self._reset_after_calculation = False
        worker = CalculationWorker(
            deepcopy(self.config), self._calculation_total, optimal, self._animated, self,
            path_max_count=self.path_max_count_box.value(),
        )
        self._calculation_worker = worker
        worker.frame_ready.connect(self._receive_calculation_frame)
        worker.finished.connect(self._calculation_finished)
        self.result_label.setText("正在计算最优价值与路径…" if optimal else "正在计算状态价值…")
        self.progress_label.setText(f"第 0／{self._calculation_total} 轮")
        self._set_calculation_controls()
        worker.start()

    def _set_calculation_controls(self) -> None:
        running = self._calculation_worker is not None
        self.grid_settings_group.setEnabled(not running)
        for widget in (self.kind_box, self.reward_edit, self.reset_cell_button, *self.probability_boxes):
            widget.setEnabled(not running)
        for widget in (self.load_button, self.max_count_box, self.path_max_count_box, self.calculation_mode_box,
                       self.evaluate_button, self.optimize_button):
            widget.setEnabled(not running)
        self.pause_button.setEnabled(running and not self._stopping)
        self.pause_button.setText("继续" if self._paused else "暂停")
        self.stop_button.setEnabled(running and not self._stopping)
        self.reset_board_button.setEnabled(not self._stopping)
        self._update_playback_controls()
        self._editor_changed()

    def _update_playback_controls(self, *args) -> None:
        self.interval_box.setEnabled(bool(self.calculation_mode_box.currentData()) and not self._stopping)

    def _interval_changed(self, *args) -> None:
        if self._waiting_for_next and not self._paused and not self._stopping:
            self._playback_timer.start(round(self.interval_box.value() * 1000))

    @Slot(int, object)
    def _receive_calculation_frame(self, count: int, result: CalculationResult) -> None:
        if self.sender() is not self._calculation_worker or self._stopping:
            return
        if self._paused:
            self._pending_frame = (count, result)
            return
        self._display_calculation_frame(count, result)

    def _display_calculation_frame(self, count: int, result: CalculationResult) -> None:
        worker = self._calculation_worker
        if worker is None or self._stopping:
            return
        try:
            previous_mode = self.policy_mode.currentIndex()
            had_frame = self._displayed_round > 0
            self.show_result(result, self._calculation_optimal)
            if had_frame:
                self.policy_mode.setCurrentIndex(previous_mode)
        except Exception as error:
            self._calculation_error = str(error)
            self._stop_calculation()
            return
        self._displayed_round = count
        self.progress_label.setText(f"第 {count}／{self._calculation_total} 轮")
        if self._calculation_optimal:
            self.result_label.setText(f"已计算未来 {count} 步的最优价值与策略")
        else:
            self.result_label.setText(f"已计算当前策略未来 {count} 步的状态价值")
        worker.acknowledge_frame()
        self._waiting_for_next = self._animated and count < self._calculation_total
        if self._waiting_for_next:
            self._playback_timer.start(round(self.interval_box.value() * 1000))

    @Slot()
    def _request_next_round(self) -> None:
        if self._calculation_worker is None or self._paused or self._stopping or not self._waiting_for_next:
            return
        self._waiting_for_next = False
        self._calculation_worker.request_next()

    def _toggle_pause(self) -> None:
        worker = self._calculation_worker
        if worker is None or self._stopping:
            return
        self._paused = not self._paused
        worker.set_paused(self._paused)
        if self._paused:
            self._playback_timer.stop()
            self.progress_label.setText(f"已暂停，第 {self._displayed_round}／{self._calculation_total} 轮")
        else:
            self.progress_label.setText(f"第 {self._displayed_round}／{self._calculation_total} 轮")
            if self._pending_frame is not None:
                count, result = self._pending_frame
                self._pending_frame = None
                self._display_calculation_frame(count, result)
            elif self._waiting_for_next:
                self._playback_timer.start(round(self.interval_box.value() * 1000))
        self._set_calculation_controls()

    def _stop_calculation(self) -> None:
        if self._calculation_worker is None:
            return
        self._stopping = True
        self._playback_timer.stop()
        self._waiting_for_next = False
        self._pending_frame = None
        self._calculation_worker.cancel()
        self.progress_label.setText(f"正在停止，第 {self._displayed_round}／{self._calculation_total} 轮")
        self._set_calculation_controls()

    @Slot()
    def _calculation_finished(self) -> None:
        worker = self._calculation_worker
        if worker is None or self.sender() is not worker:
            return
        worker.wait()
        error = self._calculation_error or worker.error
        stopped = self._stopping or worker.outcome == "stopped"
        self._calculation_worker = None
        self._playback_timer.stop()
        self._waiting_for_next = False
        self._pending_frame = None
        self._paused = False
        self._stopping = False
        worker.deleteLater()
        self._set_calculation_controls()
        if self._reset_after_calculation:
            self._reset_board()
        elif error:
            self.progress_label.setText(f"计算失败，第 {self._displayed_round}／{self._calculation_total} 轮")
            self.result_label.setText(f"计算失败，保留第 {self._displayed_round} 轮结果" if self.result is not None else "计算失败，尚未生成结果")
            if not self._close_after_calculation:
                self._message("计算失败", f"{error}\n已保留最后显示的结果。", True)
        elif stopped:
            self.progress_label.setText(f"已停止，第 {self._displayed_round}／{self._calculation_total} 轮")
            self.result_label.setText(f"已停止，保留第 {self._displayed_round} 轮结果" if self.result is not None else "已停止，尚未生成结果")
        else:
            self.progress_label.setText(f"已完成，第 {self._displayed_round}／{self._calculation_total} 轮")
        if self._close_after_calculation:
            QTimer.singleShot(0, self.close)

    @Slot()
    def shutdown_calculation(self) -> None:
        """Join the worker before Qt tears down its objects, also on Ctrl+C."""
        self._playback_timer.stop()
        worker = self._calculation_worker
        if worker is not None:
            self._stopping = True
            worker.cancel()
            worker.wait()
            self._calculation_worker = None
            self._pending_frame = None
            worker.deleteLater()

    def _change_policy_mode(self, index: int) -> None:
        self.board_view.board_item.use_optimal = index == 1
        self.board_view.board_item.update()

    def _change_path_selection(self, index: int) -> None:
        paths = []
        if self.result is not None:
            selected_path = self.path_selection_box.itemData(index)
            paths = [selected_path] if selected_path is not None else self.result.best_paths
        self.board_view.board_item.set_paths(paths)
        self._update_result_labels()

    def _toggle_paths(self, visible: bool) -> None:
        self.board_view.board_item.paths_visible = visible
        self.board_view.board_item.update()

    def closeEvent(self, event) -> None:
        if not self._close_after_calculation and not self._confirm_unsaved("关闭窗口"):
            event.ignore()
            return
        if self._calculation_worker is not None:
            self._close_after_calculation = True
            self._stop_calculation()
            event.ignore()
            return
        event.accept()
