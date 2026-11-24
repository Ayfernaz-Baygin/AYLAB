"""Alt durum çubuğu için "telemetri" alanları.

Standart QStatusBar korunur (geçici mesajlar solda gösterilir); sağ tarafa
kalıcı, eş aralıklı ölçüm alanları eklenir.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget

from app import theme
from processing.analysis import ImageInfo


class TelemetryField(QWidget):
    """'ANAHTAR değer' biçiminde tek ölçüm alanı."""

    def __init__(self, key: str, value: str = "—", min_width: int = 0, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(6)
        self.key_label = QLabel(key)
        self.key_label.setFont(theme.display_font(7.5, QFont.Weight.DemiBold, 1.2))
        self.key_label.setStyleSheet(f"color: {theme.TEXT_FAINT};")
        self.value_label = QLabel(value)
        self.value_label.setFont(theme.mono_font(8.5))
        self.value_label.setStyleSheet(f"color: {theme.TEXT};")
        if min_width:
            self.value_label.setMinimumWidth(min_width)
        layout.addWidget(self.key_label)
        layout.addWidget(self.value_label)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text or "—")

    def value(self) -> str:
        return self.value_label.text()


class StateIndicator(QWidget):
    """İşlem durumu: renkli sinyal ışığı + durum adı. PROCESSING'de nabız atar."""

    COLORS = {
        "NO IMAGE": theme.TEXT_FAINT,
        "READY": theme.OK,
        "PENDING": theme.WARN,
        "PROCESSING": theme.NEUTRAL_ACCENT,
        "DONE": theme.OK,
        "ERROR": theme.ERROR,
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state = "NO IMAGE"
        self._glow = 0.0
        self._anim = QPropertyAnimation(self, b"glow", self)
        self._anim.setDuration(900)
        self._anim.setStartValue(0.0)
        self._anim.setKeyValueAt(0.5, 1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.setFixedHeight(22)

    def _get_glow(self) -> float:
        return self._glow

    def _set_glow(self, v: float) -> None:
        self._glow = v
        self.update()

    glow = Property(float, _get_glow, _set_glow)

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in self.COLORS:
            raise ValueError(state)
        self._state = state
        self._anim.stop()
        self._anim.setLoopCount(-1 if state == "PROCESSING" else 1)
        self._anim.start()  # durum değişiminde kısa bir parlama
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(120, 22)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(self.COLORS[self._state])
        h = self.height()
        if self._glow > 0:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.with_alpha(color, int(90 * self._glow)))
            p.drawRect(QRectF(6, h / 2 - 7, 14, 14))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRect(QRectF(10, h / 2 - 3, 6, 6))
        p.setPen(color)
        p.setFont(theme.display_font(8, QFont.Weight.DemiBold, 1.5))
        p.drawText(QRectF(26, 0, self.width() - 26, h), Qt.AlignmentFlag.AlignVCenter, self._state)
        p.end()


class Telemetry:
    """Durum çubuğuna kalıcı ölçüm alanlarını ekler ve günceller."""

    def __init__(self, status_bar: QStatusBar) -> None:
        self.cursor = TelemetryField("PX", "", 230)
        self.size = TelemetryField("SIZE", min_width=74)
        self.channels = TelemetryField("CH", min_width=56)
        self.depth = TelemetryField("DEPTH")
        self.memory = TelemetryField("MEM", min_width=56)
        self.zoom = TelemetryField("ZOOM", min_width=40)
        self.algorithm = TelemetryField("ALGO", min_width=90)
        self.history = TelemetryField("UNDO", "0")
        self.state = StateIndicator()
        self.state.setMinimumWidth(118)
        for w in (self.cursor, self.size, self.channels, self.depth, self.memory, self.zoom,
                  self.algorithm, self.history, self.state):
            status_bar.addPermanentWidget(w)

    def set_image_info(self, info: ImageInfo | None) -> None:
        if info is None:
            for f in (self.size, self.channels, self.depth, self.memory, self.zoom):
                f.set_value("—")
            self.set_cursor("")
            return
        self.size.set_value(f"{info.width}×{info.height}")
        self.channels.set_value(f"{info.channels} · {info.color_model}")
        self.depth.set_value(info.dtype)
        mb = info.nbytes / (1024 * 1024)
        self.memory.set_value(f"{mb:.1f} MB" if mb >= 0.1 else f"{info.nbytes / 1024:.0f} KB")

    def set_zoom(self, scale: float) -> None:
        self.zoom.set_value(f"{scale * 100:.0f}%")

    def set_cursor(self, text: str) -> None:
        # İmleç görüntü dışındayken alan gizlenir; geçici mesajlara yer kalır.
        self.cursor.set_value(text)
        self.cursor.setVisible(bool(text))

    def set_algorithm(self, name: str | None) -> None:
        self.algorithm.set_value(name or "—")

    def set_history(self, depth: int, limit: int) -> None:
        self.history.set_value(f"{depth}/{limit}")

    def set_state(self, state: str) -> None:
        self.state.set_state(state)
