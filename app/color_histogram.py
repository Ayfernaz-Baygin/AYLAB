"""Renk histogramı raporu: her kanal kendi gerçek aralığında (H 0–360°, L* 0–100, a*/b* −128–128 …).

Sağ panelde, yalnızca Color Histogram işlemi uygulandığında görünür. Sayılar
``processing.color_processing.ColorHistogramReport`` içinden gelir; bu modül yalnızca çizer.
Üstteki genel HISTOGRAM paneli (B görüntüsünün B/G/R kanalları, 0–255) değişmez.
"""

from __future__ import annotations

import colorsys

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from app import theme
from app.analysis_panel import CHANNEL_COLORS
from processing.color_processing import ColorHistogramReport, MODEL_NAMES

ROW_HEIGHT = 62
CHANNEL_TINTS = dict(CHANNEL_COLORS)                     # R, G, B, L; diğer kanallar nötr
INK_TINTS = {"C": "#3FD0E0", "M": "#E05FD0", "Y": "#E8D44D", "K": "#9AA3B2"}


def _fmt(v: float) -> str:
    return f"{v:g}" if float(v).is_integer() else f"{v:.2f}"


class ColorHistogramPlot(QWidget):
    """Kanal başına bir satır: çubuklar (en yüksek kutuya göre ölçekli), eksen uçları ve birim."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.report: ColorHistogramReport | None = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def set_report(self, report: ColorHistogramReport | None) -> None:
        self.report = report
        self.setFixedHeight(ROW_HEIGHT * (len(report.channels) if report else 1))
        self.update()

    def _bar_color(self, name: str, model: str, k: int, n: int) -> QColor:
        if name == "H":
            r, g, b = colorsys.hsv_to_rgb((k + 0.5) / n, 0.85, 0.95)
            return QColor(int(r * 255), int(g * 255), int(b * 255))
        if model in ("cmy", "cmyk"):
            return QColor(INK_TINTS[name])
        return QColor(CHANNEL_TINTS.get(name, theme.TEXT))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        w = self.width()
        rep = self.report
        if rep is None:
            p.end()
            return
        for row, ch in enumerate(rep.channels):
            top = row * ROW_HEIGHT
            plot = QRectF(34, top + 4, max(20.0, w - 40), ROW_HEIGHT - 20)
            p.fillRect(QRectF(0, top, w, ROW_HEIGHT - 2), QColor(theme.BG_DEEP))
            p.setFont(theme.mono_font(8))
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(4, top + 4, 30, 14), Qt.AlignmentFlag.AlignLeft, ch.name)
            counts = ch.counts.astype(float)
            peak = counts.max() if counts.size else 0
            n = counts.size
            bw = plot.width() / n
            for k, c in enumerate(counts):
                if c <= 0:
                    continue
                hgt = plot.height() * c / peak
                p.fillRect(QRectF(plot.left() + k * bw, plot.bottom() - hgt, max(1.0, bw), hgt),
                           self._bar_color(ch.name, rep.model, k, n))
            p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
            p.drawLine(plot.bottomLeft(), plot.bottomRight())
            if ch.total:                                     # ortalama: kesikli çizgi
                xm = plot.left() + plot.width() * (ch.mean - ch.lo) / (ch.hi - ch.lo)
                p.setPen(QPen(QColor(theme.TEXT_DIM), 1, Qt.PenStyle.DashLine))
                p.drawLine(int(xm), int(plot.top()), int(xm), int(plot.bottom()))
            p.setFont(theme.mono_font(7))
            p.setPen(QColor(theme.TEXT_FAINT))
            y = plot.bottom() + 2
            mid = (ch.lo + ch.hi) / 2
            p.drawText(QRectF(plot.left(), y, 60, 12), Qt.AlignmentFlag.AlignLeft, _fmt(ch.lo) + ch.unit)
            p.drawText(QRectF(plot.center().x() - 30, y, 60, 12), Qt.AlignmentFlag.AlignHCenter, _fmt(mid) + ch.unit)
            p.drawText(QRectF(plot.right() - 60, y, 60, 12), Qt.AlignmentFlag.AlignRight, _fmt(ch.hi) + ch.unit)
        p.end()


class ColorHistogramView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.title = QLabel("COLOR HISTOGRAM")
        self.title.setObjectName("SectionTitle")
        layout.addWidget(self.title)
        self.info = QLabel("")
        self.info.setFont(theme.mono_font(8.5))
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.plot = ColorHistogramPlot()
        layout.addWidget(self.plot)
        self.stats = QLabel("")
        self.stats.setObjectName("Faint")
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)
        self.report: ColorHistogramReport | None = None
        self.set_report(None)

    def set_report(self, report) -> None:
        if not isinstance(report, ColorHistogramReport):     # başka raporlar burada gösterilmez
            self.report = None
            self.plot.set_report(None)
            self.hide()
            return
        self.report = report
        self.title.setText(f"COLOR HISTOGRAM · {MODEL_NAMES[report.model]}")
        self.info.setText(f"{report.bins} bins per channel · {report.pixels} px"
                          + (f" · {report.note}" if report.note else ""))
        self.plot.set_report(report)
        self.stats.setText("  ".join(f"{c.name}: μ {c.mean:.3g}{c.unit}, {_fmt(round(c.minimum, 3))}…"
                                     f"{_fmt(round(c.maximum, 3))}{c.unit}, Σ {c.total}" for c in report.channels))
        self.show()
