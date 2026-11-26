"""Tam renkli işlemler için aktarım eğrileri: s = T(r), r, s ∈ [0, 1] (slayt 52–56'daki küçük grafikler).

Sağ panelde yalnızca ``processing.color_processing.FullColorReport`` gelince görünür. Eğriler (kanal başına
ya da R, G, B için ortak) rapordan gelir; bu modül yalnızca çizer. Eğrisi olmayan raporlar (Color Slicing)
yalnızca satırları ve varsa örnek rengi (prototip / nötr piksel) gösterir.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from app import theme
from processing.color_processing import FullColorReport

TICKS = (0.0, 0.5, 1.0)


class CurvePlot(QWidget):
    """Kare çizim alanı: kesikli birim doğru, eğriler, 0 / 0.5 / 1 ölçekleri ve sağ üstte eğri adları."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.curves = ()
        self.setFixedHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def set_curves(self, curves) -> None:
        self.curves = tuple(curves)
        self.update()

    def frame(self) -> QRectF:
        side = min(self.height() - 22.0, max(40.0, self.width() - 30.0))
        return QRectF(22, 4, side, side)

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self.curves:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = self.frame()
        at = lambda x, y: QPointF(box.left() + box.width() * x, box.bottom() - box.height() * y)
        p.setPen(QPen(QColor(theme.LINE), 1))
        for t in TICKS[1:-1]:
            p.drawLine(at(t, 0), at(t, 1))
            p.drawLine(at(0, t), at(1, t))
        p.drawRect(box)
        p.setPen(QPen(QColor(theme.TEXT_FAINT), 1, Qt.PenStyle.DashLine))
        p.drawLine(at(0, 0), at(1, 1))
        accent = self.palette().highlight().color()
        for k, c in enumerate(self.curves):
            path = QPainterPath()
            vals = c.values
            for v in range(256):
                y = min(1.0, max(0.0, float(vals[v])))
                pt = at(v / 255, y)
                path.moveTo(pt) if v == 0 else path.lineTo(pt)
            color = QColor(*c.rgb) if c.rgb is not None else accent
            p.setPen(QPen(color, 2.2))
            p.drawPath(path)
            p.setFont(theme.mono_font(7))
            p.drawText(QRectF(box.left() + 4, box.top() + 3 + 11 * k, box.width() - 8, 11),
                       Qt.AlignmentFlag.AlignLeft, c.name)
        p.setPen(QColor(theme.TEXT_FAINT))
        p.setFont(theme.mono_font(7))
        for t in TICKS:
            x = at(t, 0).x()
            p.drawText(QRectF(x - 14, box.bottom() + 3, 28, 12), Qt.AlignmentFlag.AlignHCenter, f"{t:g}")
            y = at(0, t).y()
            p.drawText(QRectF(0, y - 6, 18, 12), Qt.AlignmentFlag.AlignRight, f"{t:g}")
        p.drawText(QRectF(box.right() + 3, box.bottom() - 12, 10, 12), Qt.AlignmentFlag.AlignLeft, "r")
        p.end()


class TransferCurveView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.title = QLabel("TRANSFER CURVES")
        self.title.setObjectName("SectionTitle")
        self.title.setWordWrap(True)                          # uzun başlıklar panel genişliğini aşmasın
        layout.addWidget(self.title)
        self.swatch = QLabel("")
        self.swatch.setFont(theme.mono_font(8))
        layout.addWidget(self.swatch)
        self.plot = CurvePlot()
        layout.addWidget(self.plot)
        self.lines = QLabel("")
        self.lines.setObjectName("Faint")
        self.lines.setWordWrap(True)
        self.lines.setFont(theme.mono_font(8))
        layout.addWidget(self.lines)
        self.report: FullColorReport | None = None
        self.set_report(None)

    def rows(self) -> list[str]:
        """Gösterilen satırlar (test ve doğrulama için)."""
        return self.lines.text().split("\n") if self.lines.text() else []

    def set_report(self, report) -> None:
        if not isinstance(report, FullColorReport):          # başka raporlar burada gösterilmez
            self.report = None
            self.plot.set_curves(())
            self.hide()
            return
        self.report = report
        self.title.setText(("TRANSFER CURVES · " if report.curves else "COLOR SLICE · ") + report.title.upper())
        self.plot.setVisible(bool(report.curves))
        self.plot.set_curves(report.curves)
        if report.swatch is not None:
            r, g, b = report.swatch
            self.swatch.setText(f'<span style="color: rgb({r}, {g}, {b}); font-size: 14px;">■■</span>'
                                f"&nbsp;&nbsp;RGB {r}, {g}, {b}")
            self.swatch.setToolTip("Prototype / neutral color picked from the image")
            self.swatch.show()
        else:
            self.swatch.hide()
        self.lines.setText("\n".join(report.rows))
        self.show()
