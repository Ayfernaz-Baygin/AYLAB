"""Parametre tanımlarından (processing.registry) üretilen giriş kontrolleri.

Her kontrol aynı arayüzü sunar: ``spec``, ``value()``, ``set_value()``,
``reset()`` ve ``value_changed(key, value)`` sinyali. ``create_control``
fabrikası, tanımın türüne göre doğru widget'ı seçer; böylece parametre paneli
algoritmaları tanımak zorunda kalmaz.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QLocale,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app import theme
from processing.registry import BoolParam, ChoiceParam, FloatParam, IntParam, MaskParam, ParamSpec


class ParameterControl(QWidget):
    value_changed = Signal(str, object)

    def __init__(self, spec: ParamSpec, parent=None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setToolTip(spec.help or "")

    def value(self) -> Any:  # pragma: no cover - alt sınıflar uygular
        raise NotImplementedError

    def set_value(self, value: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        self.set_value(self.spec.default)

    def _header(self, layout: QVBoxLayout, trailing: QWidget | None = None) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        # Yalnızca Latin harfleri büyüt: σ, γ gibi matematik sembolleri korunur
        label = QLabel("".join(ch.upper() if ch.isascii() else ch for ch in self.spec.label))
        label.setObjectName("ParamLabel")
        row.addWidget(label, 1)
        if trailing is not None:
            row.addWidget(trailing)
        layout.addLayout(row)


class SliderControl(ParameterControl):
    """Tam sayı veya ondalık parametre: ince ray + eş aralıklı sayısal okuma."""

    def __init__(self, spec: IntParam | FloatParam, parent=None) -> None:
        super().__init__(spec, parent)
        self._is_float = isinstance(spec, FloatParam)
        self._step = spec.step if self._is_float else spec.effective_step
        self._syncing = False

        if self._is_float:
            self.field = QDoubleSpinBox()
            self.field.setDecimals(spec.decimals)
            self.field.setSingleStep(spec.step)
        else:
            self.field = QSpinBox()
            self.field.setSingleStep(self._step)
        # Teknik okumalar her yerde aynı ondalık ayırıcıyı ("."): sistem diline bağlı değil
        self.field.setLocale(QLocale.c())
        self.field.setRange(spec.minimum, spec.maximum)
        self.field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.field.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.field.setFixedWidth(78)
        self.field.setKeyboardTracking(False)
        if spec.suffix:
            self.field.setSuffix(f" {spec.suffix}")

        steps = round((spec.maximum - spec.minimum) / self._step)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(1, steps))
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)

        bounds = QHBoxLayout()
        bounds.setContentsMargins(0, 0, 0, 0)
        lo, hi = QLabel(self._fmt(spec.minimum)), QLabel(self._fmt(spec.maximum))
        lo.setObjectName("Faint")
        hi.setObjectName("Faint")
        bounds.addWidget(lo)
        bounds.addStretch(1)
        bounds.addWidget(hi)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._header(layout, self.field)
        layout.addWidget(self.slider)
        layout.addLayout(bounds)

        self.slider.valueChanged.connect(self._from_slider)
        self.field.valueChanged.connect(self._from_field)
        self.set_value(spec.default)

    def _fmt(self, v: float) -> str:
        return f"{v:.{self.spec.decimals}f}" if self._is_float else str(int(v))

    def _to_pos(self, value: float) -> int:
        return round((value - self.spec.minimum) / self._step)

    def _from_pos(self, pos: int) -> float:
        return self.spec.minimum + pos * self._step

    def value(self) -> int | float:
        return self.spec.validate(self.field.value())

    def set_value(self, value: Any) -> None:
        v = self.spec.validate(value)
        self._syncing = True
        self.field.setValue(v)
        self.slider.setValue(self._to_pos(v))
        self._syncing = False
        self.value_changed.emit(self.spec.key, v)

    def _from_slider(self, pos: int) -> None:
        if not self._syncing:
            self.set_value(self._from_pos(pos))

    def _from_field(self, raw: float) -> None:
        if self._syncing:
            return
        v = self.spec.validate(raw)
        if v != raw:  # ör. tek sayıya yuvarlama
            self.set_value(v)
        else:
            self._syncing = True
            self.slider.setValue(self._to_pos(v))
            self._syncing = False
            self.value_changed.emit(self.spec.key, v)


class ChoiceControl(ParameterControl):
    """Az seçenek → segmentli düğmeler; çok seçenek → açılır liste."""

    SEGMENT_LIMIT = 3

    def __init__(self, spec: ChoiceParam, parent=None) -> None:
        super().__init__(spec, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self._header(layout)

        self.combo: QComboBox | None = None
        self.buttons: dict[str, QPushButton] = {}
        if len(spec.options) <= self.SEGMENT_LIMIT:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            self._group = QButtonGroup(self)
            self._group.setExclusive(True)
            for value, label in spec.options:
                b = QPushButton(label.upper())
                b.setObjectName("Segment")
                b.setCheckable(True)
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                b.clicked.connect(lambda _c=False, v=value: self.set_value(v))
                self._group.addButton(b)
                row.addWidget(b, 1)
                self.buttons[value] = b
            layout.addLayout(row)
        else:
            self.combo = QComboBox()
            for value, label in spec.options:
                self.combo.addItem(label, value)
            self.combo.currentIndexChanged.connect(lambda i: self.set_value(self.combo.itemData(i)))
            layout.addWidget(self.combo)
        self._value = spec.default
        self.set_value(spec.default)

    def value(self) -> str:
        return self._value

    def set_value(self, value: Any) -> None:
        v = self.spec.validate(value)
        self._value = v
        if self.combo is not None:
            idx = self.combo.findData(v)
            if idx != self.combo.currentIndex():
                self.combo.blockSignals(True)
                self.combo.setCurrentIndex(idx)
                self.combo.blockSignals(False)
        else:
            self.buttons[v].setChecked(True)
        self.value_changed.emit(self.spec.key, v)


class ToggleSwitch(QAbstractButton):
    """Kayan düğmeli açma/kapama anahtarı."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(34, 16)

    def _get_knob(self) -> float:
        return self._pos

    def _set_knob(self, v: float) -> None:
        self._pos = v
        self.update()

    knob = Property(float, _get_knob, _set_knob)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        accent = self.palette().highlight().color()
        track = QRectF(0.5, 3.5, self.width() - 1, self.height() - 7)
        p.setPen(QColor(theme.LINE) if self._pos < 0.5 else accent)
        p.setBrush(theme.with_alpha(accent, int(70 * self._pos)))
        p.drawRect(track)
        kx = 2 + self._pos * (self.width() - 14)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent if self._pos > 0.5 else QColor(theme.TEXT_DIM))
        p.drawRect(QRectF(kx, 1, 10, self.height() - 2))
        p.end()


class ToggleControl(ParameterControl):
    def __init__(self, spec: BoolParam, parent=None) -> None:
        super().__init__(spec, parent)
        self.switch = ToggleSwitch()
        self.switch.setFixedSize(34, 16)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self._header(layout, self.switch)
        self.switch.toggled.connect(lambda on: self.value_changed.emit(self.spec.key, bool(on)))
        self.set_value(spec.default)

    def value(self) -> bool:
        return self.switch.isChecked()

    def set_value(self, value: Any) -> None:
        v = self.spec.validate(value)
        if self.switch.isChecked() != v:
            self.switch.setChecked(v)
        else:
            self.value_changed.emit(self.spec.key, v)

    def set_accent(self, color: QColor) -> None:
        pal = self.switch.palette()
        pal.setColor(pal.ColorRole.Highlight, color)
        self.switch.setPalette(pal)


class MaskGrid(QWidget):
    """Tıklanabilir ikili ızgara: sol tık hücreyi aç/kapat, sağ tık orijini taşı."""

    changed = Signal()
    CELL = 22

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.grid: list[list[int]] = [[1]]
        self.origin = (0, 0)
        self.accent = QColor(theme.NEUTRAL_ACCENT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Left click: toggle a cell · Right click: move the origin")

    def set_grid(self, grid, origin) -> None:
        self.grid = [list(r) for r in grid]
        self.origin = tuple(origin)
        self.setFixedSize(self.sizeHint())
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(len(self.grid[0]) * self.CELL + 1, len(self.grid) * self.CELL + 1)

    def _cell_at(self, event: QMouseEvent) -> tuple[int, int] | None:
        pos = event.position()
        r, c = int(pos.y() // self.CELL), int(pos.x() // self.CELL)
        return (r, c) if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]) else None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        cell = self._cell_at(event)
        if cell is None:
            return
        r, c = cell
        if event.button() == Qt.MouseButton.RightButton:
            self.origin = (r, c)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.grid[r][c] ^= 1
            if not any(any(row) for row in self.grid):      # en az bir hücre kalmalı
                self.grid[r][c] = 1
                return
        else:
            return
        self.update()
        self.changed.emit()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        n = self.CELL
        for r, row in enumerate(self.grid):
            for c, v in enumerate(row):
                p.fillRect(c * n, r * n, n, n, theme.with_alpha(self.accent, 170) if v else QColor(theme.PANEL_RAISED))
        p.setPen(QPen(QColor(theme.LINE), 1))
        for r in range(len(self.grid) + 1):
            p.drawLine(0, r * n, len(self.grid[0]) * n, r * n)
        for c in range(len(self.grid[0]) + 1):
            p.drawLine(c * n, 0, c * n, len(self.grid) * n)
        r, c = self.origin                                       # orijin: halka + nokta
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(theme.TEXT), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(c * n + 5, r * n + 5, n - 10, n - 10))
        p.setBrush(QColor(theme.TEXT))
        p.drawEllipse(QRectF(c * n + n / 2 - 2, r * n + n / 2 - 2, 4, 4))
        p.end()


class MaskControl(ParameterControl):
    """Özel yapı elemanı: ızgara boyutu (3 / 5 / 7) + hücre düzenleyici + orijin."""

    SIZES = (3, 5, 7)

    def __init__(self, spec: MaskParam, parent=None) -> None:
        super().__init__(spec, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.origin_label = QLabel()
        self.origin_label.setObjectName("Faint")
        self._header(layout, self.origin_label)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.size_buttons: dict[int, QPushButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for n in self.SIZES:
            if n > spec.max_size:
                continue
            b = QPushButton(f"{n} × {n}")
            b.setObjectName("Segment")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, k=n: self._resize(k))
            group.addButton(b)
            row.addWidget(b, 1)
            self.size_buttons[n] = b
        layout.addLayout(row)
        self.grid = MaskGrid()
        self.grid.changed.connect(self._emit)
        hint = QLabel("left click: cell · right click: origin")
        hint.setObjectName("Faint")
        grid_row = QHBoxLayout()
        grid_row.setContentsMargins(0, 2, 0, 0)
        grid_row.addWidget(self.grid)
        grid_row.addSpacing(10)
        grid_row.addWidget(hint, 1, Qt.AlignmentFlag.AlignTop)
        hint.setWordWrap(True)
        layout.addLayout(grid_row)
        self.set_value(spec.default)

    def value(self) -> str:
        return MaskParam.encode(self.grid.grid, self.grid.origin)

    def set_value(self, value: Any) -> None:
        v = self.spec.validate(value)
        grid, origin = MaskParam.decode(v, self.spec.max_size)
        self.grid.set_grid(grid, origin)
        self._sync()
        self.value_changed.emit(self.spec.key, v)

    def _resize(self, n: int) -> None:
        """Yeni n × n ızgara; eski hücreler merkezlenerek kopyalanır, orijin merkeze alınır."""
        old = self.grid.grid
        h, w = len(old), len(old[0])
        new = [[0] * n for _ in range(n)]
        dy, dx = (n - h) // 2, (n - w) // 2
        for r in range(h):
            for c in range(w):
                if 0 <= r + dy < n and 0 <= c + dx < n:
                    new[r + dy][c + dx] = old[r][c]
        if not any(any(row) for row in new):
            new[n // 2][n // 2] = 1
        self.set_value(MaskParam.encode(new, (n // 2, n // 2)))

    def _emit(self) -> None:
        self._sync()
        self.value_changed.emit(self.spec.key, self.value())

    def _sync(self) -> None:
        g = self.grid.grid
        n = len(g)
        for k, b in self.size_buttons.items():
            b.setChecked(k == n and len(g[0]) == n)
        r, c = self.grid.origin
        member = "" if g[r][c] else " (not in B)"
        self.origin_label.setText(f"origin {r},{c}{member}")

    def set_accent(self, color: QColor) -> None:
        self.grid.accent = QColor(color)
        self.grid.update()


def create_control(spec: ParamSpec) -> ParameterControl:
    if isinstance(spec, (IntParam, FloatParam)):
        return SliderControl(spec)
    if isinstance(spec, ChoiceParam):
        return ChoiceControl(spec)
    if isinstance(spec, BoolParam):
        return ToggleControl(spec)
    if isinstance(spec, MaskParam):
        return MaskControl(spec)
    raise TypeError(f"Desteklenmeyen parametre türü: {type(spec).__name__}")
