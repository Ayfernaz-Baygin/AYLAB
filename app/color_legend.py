"""Pseudo-color lejantı: gri seviye → renk çubuğu (LUT) ve dilim aralıkları.

Sağ panelde yalnızca Intensity Slicing / Color Coding uygulandığında görünür. Renkler
``processing.color_processing.PseudoColorReport`` içindeki 256 girdili tablodan gelir; bu modül
yalnızca çizer ("renklerin gerçek bir anlamı yoktur — renk çubuğu şarttır", laboratuvar).
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from app import theme
from processing.color_processing import PseudoColorReport

TICKS = (0, 64, 128, 192, 255)


class ColorBar(QWidget):
    """256 sütunluk renk çubuğu; altta gri seviye ölçeği, dilim düzlemleri kesikli çizgi."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.report: PseudoColorReport | None = None
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def set_report(self, report: PseudoColorReport | None) -> None:
        self.report = report
        self.update()

    def x_of(self, level: float, bar: QRectF) -> float:
        return bar.left() + bar.width() * level / 256

    def paintEvent(self, event) -> None:  # noqa: N802
        if self.report is None:
            return
        p = QPainter(self)
        bar = QRectF(4, 2, max(20.0, self.width() - 8), 20)
        lut = self.report.lut
        for v in range(256):
            r, g, b = (int(c) for c in lut[v])
            x0, x1 = self.x_of(v, bar), self.x_of(v + 1, bar)
            p.fillRect(QRectF(x0, bar.top(), x1 - x0 + 0.6, bar.height()), QColor(r, g, b))
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawRect(bar)
        for lo, _, _ in self.report.slices[1:]:                  # düzlemler l_i
            x = self.x_of(lo, bar)
            p.setPen(QPen(QColor(theme.TEXT), 1, Qt.PenStyle.DashLine))
            p.drawLine(int(x), int(bar.top()) - 1, int(x), int(bar.bottom()) + 3)
        p.setFont(theme.mono_font(7))
        p.setPen(QColor(theme.TEXT_FAINT))
        for t in TICKS:
            x = self.x_of(t + 0.5, bar)
            p.drawLine(int(x), int(bar.bottom()), int(x), int(bar.bottom()) + 3)
            align = Qt.AlignmentFlag.AlignLeft if t == 0 else (Qt.AlignmentFlag.AlignRight if t == 255 else Qt.AlignmentFlag.AlignHCenter)
            rx = x if t == 0 else (x - 40 if t == 255 else x - 20)
            p.drawText(QRectF(rx, bar.bottom() + 5, 40, 12), align, str(t))
        p.end()


class ColorLegendView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.title = QLabel("COLOR LEGEND")
        self.title.setObjectName("SectionTitle")
        layout.addWidget(self.title)
        self.bar = ColorBar()
        layout.addWidget(self.bar)
        self.slices = QVBoxLayout()
        self.slices.setSpacing(2)
        layout.addLayout(self.slices)
        self.note = QLabel("")
        self.note.setObjectName("Faint")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.report: PseudoColorReport | None = None
        self.set_report(None)

    def _clear_slices(self) -> None:
        while self.slices.count():
            item = self.slices.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def rows(self) -> list[str]:
        """Dilim satırlarının metni (test ve doğrulama için)."""
        return [self.slices.itemAt(i).widget().text() for i in range(self.slices.count())]

    def set_report(self, report) -> None:
        self._clear_slices()
        if not isinstance(report, PseudoColorReport):        # başka raporlar burada gösterilmez
            self.report = None
            self.bar.set_report(None)
            self.hide()
            return
        self.report = report
        self.title.setText(f"COLOR LEGEND · {report.title.upper()}")
        self.bar.set_report(report)
        for k, (lo, hi, rgb) in enumerate(report.slices):
            pm = QPixmap(10, 10)
            pm.fill(QColor(*rgb))
            self.slices.addWidget(_SwatchRow(pm, f"c{k + 1}  [{lo}, {hi})  RGB {rgb[0]}, {rgb[1]}, {rgb[2]}",
                                             f"Intensities {lo} … {hi - 1} → c{k + 1}"))
        self.note.setText(("Gray level f (0–255) → color.  " if not report.slices else
                           "Slices [lo, hi): f = l_i belongs to the upper slice.  ")
                          + ("Color input converted to Y first." if report.gray_from_color else ""))
        self.show()


class _SwatchRow(QLabel):
    """Renk örneği + aralık metni (tek satır)."""

    def __init__(self, pixmap: QPixmap, text: str, tip: str) -> None:
        super().__init__(text)
        self._pm = pixmap
        self.setFont(theme.mono_font(8))
        self.setToolTip(tip)
        self.setContentsMargins(16, 0, 0, 0)
        self.setFixedHeight(16)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        p = QPainter(self)
        p.drawPixmap(1, (self.height() - 10) // 2, self._pm)
        p.end()
