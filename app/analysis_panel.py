"""Histogram osiloskobu ve kanal okumaları.

Sayılar ``processing.analysis`` tarafından hesaplanır; bu modül yalnızca çizer.
İşlenmiş görüntünün histogramı dolu eğriler olarak, orijinalinki ise arkada
ince bir "hayalet" çizgi olarak gösterilir — işlemin tona etkisi tek bakışta
okunur.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from app import theme
from processing.analysis import channel_stats, histograms

# Kanal renkleri — görüntü verisine ait oldukları için tema vurgusundan bağımsız.
CHANNEL_COLORS = {"R": "#FF5E57", "G": "#6BD98B", "B": "#5B8DFF", "L": "#DCE3EF"}


def _smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return values
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    return np.convolve(values, kernel, mode="same")


class HistogramScope(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current: dict[str, np.ndarray] = {}
        self._reference: dict[str, np.ndarray] = {}
        self._log = False
        self.setMinimumHeight(118)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(260, 130)

    def set_log_scale(self, enabled: bool) -> None:
        self._log = enabled
        self.update()

    def set_images(self, current: np.ndarray | None, reference: np.ndarray | None = None) -> None:
        self._current = histograms(current) if current is not None else {}
        self._reference = histograms(reference) if reference is not None else {}
        self.update()

    def has_data(self) -> bool:
        return bool(self._current)

    def _normalise(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for k, v in data.items():
            v = v.astype(float)
            out[k] = _smooth(np.log1p(v) if self._log else v, 1)
        peak = max((float(v.max()) for v in out.values()), default=0.0)
        return {k: v / peak for k, v in out.items()} if peak > 0 else out

    def _path(self, values: np.ndarray, rect: QRectF, closed: bool) -> QPainterPath:
        n = len(values)
        path = QPainterPath(QPointF(rect.left(), rect.bottom()))
        for i, v in enumerate(values):
            path.lineTo(rect.left() + rect.width() * i / (n - 1), rect.bottom() - v * rect.height())
        if closed:
            path.lineTo(rect.right(), rect.bottom())
            path.closeSubpath()
        return path

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillPath(theme.chamfer_path(0, 0, w, h, 8, "tr"), QColor(theme.BG_DEEP))
        plot = QRectF(8, 8, w - 16, h - 26)

        # Izgara: çeyrek çizgileri ve ölçek etiketleri
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        for i in range(1, 4):
            x = plot.left() + plot.width() * i / 4
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        p.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        p.setFont(theme.mono_font(7))
        p.setPen(QColor(theme.TEXT_FAINT))
        for i, label in enumerate(("0", "64", "128", "192", "255")):
            x = plot.left() + plot.width() * i / 4
            align = Qt.AlignmentFlag.AlignLeft if i == 0 else (
                Qt.AlignmentFlag.AlignRight if i == 4 else Qt.AlignmentFlag.AlignHCenter)
            rx = x if i == 0 else (x - 40 if i == 4 else x - 20)
            p.drawText(QRectF(rx, plot.bottom() + 3, 40, 12), align, label)

        if not self._current:
            p.setFont(theme.mono_font(8))
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, "no signal")
            p.end()
            return

        both = self._normalise({**{f"ref_{k}": v for k, v in self._reference.items()}, **self._current})
        reference = {k[4:]: v for k, v in both.items() if k.startswith("ref_")}
        current = {k: v for k, v in both.items() if not k.startswith("ref_")}

        # Orijinal: kesikli hayalet çizgi
        for name, values in reference.items():
            p.setPen(QPen(theme.with_alpha(CHANNEL_COLORS.get(name, theme.TEXT), 90), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self._path(values, plot, closed=False))

        # İşlenmiş: toplamsal (additive) karışımla dolu eğriler — RGB örtüşmesi beyaza yaklaşır
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for name, values in current.items():
            color = QColor(CHANNEL_COLORS.get(name, theme.TEXT))
            grad = QLinearGradient(0, plot.top(), 0, plot.bottom())
            grad.setColorAt(0, theme.with_alpha(color, 120))
            grad.setColorAt(1, theme.with_alpha(color, 25))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(grad)
            p.drawPath(self._path(values, plot, closed=True))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        for name, values in current.items():
            p.setPen(QPen(QColor(CHANNEL_COLORS.get(name, theme.TEXT)), 1.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(self._path(values, plot, closed=False))
        p.end()


class ChannelMeter(QWidget):
    """Tek kanal: renk işareti, ortalama ± std ve min–maks aralık çubuğu."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.name = ""
        self.mean = self.std = 0.0
        self.lo, self.hi = 0, 255
        self.setFixedHeight(22)

    def set_stats(self, name: str, mean: float, std: float, lo: int, hi: int) -> None:
        self.name, self.mean, self.std, self.lo, self.hi = name, mean, std, lo, hi
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        color = QColor(CHANNEL_COLORS.get(self.name, theme.TEXT))
        p.fillRect(QRectF(0, 6, 3, h - 12), color)
        p.setFont(theme.mono_font(8, QFont.Weight.Bold))
        p.setPen(color)
        p.drawText(QRectF(10, 0, 14, h), Qt.AlignmentFlag.AlignVCenter, self.name)

        bar = QRectF(28, h / 2 - 1, max(10, w - 150), 2)
        p.fillRect(bar, QColor(theme.LINE))
        x0 = bar.left() + bar.width() * self.lo / 255
        x1 = bar.left() + bar.width() * self.hi / 255
        p.fillRect(QRectF(x0, bar.top(), max(1.0, x1 - x0), bar.height()), theme.with_alpha(color, 120))
        xm = bar.left() + bar.width() * self.mean / 255
        p.fillRect(QRectF(xm - 1, bar.top() - 4, 2, 10), color)

        p.setFont(theme.mono_font(8))
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(bar.right() + 8, 0, w - bar.right() - 8, h),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   f"μ {self.mean:6.1f}  σ {self.std:5.1f}")
        p.end()


class AnalysisPanel(QWidget):
    """Histogram + kanal okumalarını bir arada tutan bölüm."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        head = QHBoxLayout()
        title = QLabel("HISTOGRAM")
        title.setObjectName("SectionTitle")
        head.addWidget(title)
        head.addStretch(1)
        legend = QLabel("━ B   ┅ A")
        legend.setObjectName("Faint")
        legend.setToolTip("Filled: output (B)  ·  dashed: source (A)")
        head.addWidget(legend)
        self.log_button = QToolButton()
        self.log_button.setObjectName("StageTool")
        self.log_button.setText("LOG")
        self.log_button.setCheckable(True)
        self.log_button.setToolTip("Logarithmic count axis")
        head.addWidget(self.log_button)
        layout.addLayout(head)

        self.scope = HistogramScope()
        layout.addWidget(self.scope)
        self.log_button.toggled.connect(self.scope.set_log_scale)

        ch_title = QLabel("CHANNELS")
        ch_title.setObjectName("SectionTitle")
        layout.addSpacing(6)
        layout.addWidget(ch_title)
        self._meters_box = QVBoxLayout()
        self._meters_box.setSpacing(0)
        layout.addLayout(self._meters_box)
        self.meters: list[ChannelMeter] = []
        self._empty = QLabel("Channel statistics appear after loading an image.")
        self._empty.setObjectName("Faint")
        self._empty.setWordWrap(True)
        self._meters_box.addWidget(self._empty)

    def update_images(self, current: np.ndarray | None, reference: np.ndarray | None) -> None:
        self.scope.set_images(current, reference)
        stats = channel_stats(current) if current is not None else []
        stats = list(reversed(stats)) if [s.name for s in stats] == ["B", "G", "R"] else stats  # R, G, B sırası
        while len(self.meters) < len(stats):
            m = ChannelMeter()
            self._meters_box.addWidget(m)
            self.meters.append(m)
        for i, m in enumerate(self.meters):
            if i < len(stats):
                s = stats[i]
                m.set_stats(s.name, s.mean, s.std, s.minimum, s.maximum)
                m.show()
            else:
                m.hide()
        self._empty.setVisible(not stats)
