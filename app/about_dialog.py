"""AYLAB hakkında penceresi."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout

from app import theme
from app.command_bar import paint_logo

APP_NAME = theme.BRAND_NAME
APP_SUBTITLE = theme.BRAND_DESCRIPTOR
APP_TAGLINE = theme.BRAND_TAGLINE
APP_VERSION = "0.2.0 · interface stage"


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setFixedSize(460, 330)
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("CLOSE")
        close.setObjectName("GhostAction")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(theme.BG_DEEP))
        p.fillPath(theme.chamfer_path(14, 14, w - 28, h - 70, 16, "tl,br"), QColor(theme.PANEL))
        p.fillRect(QRectF(14, h - 58, w - 28, 2), theme.spectrum_gradient(14, 0, w - 14, 0, alpha=160))

        paint_logo(p, 36, 40, cell=8, gap=2)
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.display_font(22, QFont.Weight.Bold, 7))
        p.drawText(QRectF(80, 32, 300, 34), Qt.AlignmentFlag.AlignVCenter, APP_NAME)
        p.setPen(QColor(theme.TEXT_DIM))
        p.setFont(theme.display_font(10, QFont.Weight.Normal, 2))
        p.drawText(QRectF(82, 64, 340, 18), Qt.AlignmentFlag.AlignVCenter, APP_SUBTITLE.upper())

        p.setFont(theme.body_font(9.5))
        p.setPen(QColor(theme.TEXT))
        text = ("A desktop laboratory for exploring digital image processing: geometric and "
                "intensity transformations, spatial filtering, morphology, segmentation, color "
                "processing and feature extraction — side by side with the source.")
        p.drawText(QRectF(36, 100, w - 72, 90), Qt.TextFlag.TextWordWrap, text)

        p.setFont(theme.mono_font(8))
        p.setPen(QColor(theme.TEXT_FAINT))
        p.drawText(QRectF(36, 194, w - 72, 16), Qt.AlignmentFlag.AlignLeft, f"v{APP_VERSION}")
        p.drawText(QRectF(36, 210, w - 72, 16), Qt.AlignmentFlag.AlignLeft,
                   "Python · PySide6 · OpenCV · NumPy · Matplotlib")

        # Slogan — panelin alt kenarında, tayf çizgisinin hemen üstünde
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.display_font(10.5, QFont.Weight.DemiBold, 3.5))
        p.drawText(QRectF(36, 236, w - 72, 22), Qt.AlignmentFlag.AlignVCenter, APP_TAGLINE)
        p.end()
