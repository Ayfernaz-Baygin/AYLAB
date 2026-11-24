"""Üst komut şeridi: AYLAB kimliği, komutlar ve tayf çizgisi.

Klasik menü çubuğu yerine tek satırlık bir komut şeridi kullanılır. Her
komut bir ``QAction``'a bağlıdır; kısayollar ve etkinlik durumu tek yerden
yönetilir. Şeridin alt kenarı, seçili kategorinin dalga boyunda parlayan
bir "emisyon çizgisi" taşıyan ince bir tayftır.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QSizePolicy, QWidget

from app import theme

WORDMARK_WIDTH = 286


def paint_logo(p: QPainter, x: float, y: float, cell: float = 5.0, gap: float = 1.5) -> None:
    """3×3 piksel matrisinden oluşan AYLAB amblemi: köşegen bir tayf."""
    stops = [QColor(c) for _, c in reversed(theme.SPECTRAL_STOPS)]
    p.save()
    p.setPen(Qt.PenStyle.NoPen)
    for i in range(3):
        for j in range(3):
            d = i + j  # 0..4 köşegen indeksi
            if (i, j) in ((2, 0), (0, 2)):
                color = QColor(theme.LINE)
            else:
                color = stops[min(len(stops) - 1, round(d * (len(stops) - 1) / 4))]
            p.setBrush(color)
            p.drawRect(QRectF(x + i * (cell + gap), y + j * (cell + gap), cell, cell))
    p.restore()


class CommandButton(QAbstractButton):
    """Komut: büyük harfli etiket + soluk kısayol ipucu; altında hover çizgisi."""

    def __init__(self, action: QAction, label: str, hint: str = "", parent=None) -> None:
        super().__init__(parent)
        self._action = action
        self._label = label
        self._hint = hint
        self._hover = 0.0
        self._anim = QPropertyAnimation(self, b"hover", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.accent = QColor(theme.NEUTRAL_ACCENT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(action.toolTip())
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.clicked.connect(action.trigger)
        action.changed.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        self.setEnabled(self._action.isEnabled())
        self.update()

    def _get_hover(self) -> float:
        return self._hover

    def _set_hover(self, v: float) -> None:
        self._hover = v
        self.update()

    hover = Property(float, _get_hover, _set_hover)

    def _animate(self, target: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._hover)
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, e) -> None:  # noqa: N802
        if self.isEnabled():
            self._animate(1.0)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._animate(0.0)

    def sizeHint(self) -> QSize:  # noqa: N802
        f1 = theme.display_font(9.5, QFont.Weight.DemiBold, 1.5)
        f2 = theme.mono_font(7.5)
        w = QFontMetrics(f1).horizontalAdvance(self._label)
        if self._hint:
            w += 8 + QFontMetrics(f2).horizontalAdvance(self._hint)
        return QSize(int(w) + 24, 40)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        enabled = self.isEnabled()
        k = self._hover if enabled else 0.0

        if k > 0:
            p.fillRect(QRectF(0, 6, w, h - 12), theme.with_alpha(theme.PANEL_RAISED, int(255 * k)))
            p.fillRect(QRectF(12, h - 8, (w - 24) * k, 2), self.accent)

        p.setFont(theme.display_font(9.5, QFont.Weight.DemiBold, 1.5))
        color = QColor(theme.TEXT) if enabled else QColor(theme.TEXT_FAINT)
        if enabled and k == 0:
            color = QColor("#C3CCDC")
        p.setPen(color)
        fm = p.fontMetrics()
        label_w = fm.horizontalAdvance(self._label)
        p.drawText(QRectF(12, 0, label_w + 2, h), Qt.AlignmentFlag.AlignVCenter, self._label)
        if self._hint:
            p.setFont(theme.mono_font(7.5))
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(QRectF(12 + label_w + 8, 1, w, h), Qt.AlignmentFlag.AlignVCenter, self._hint)
        p.end()


class CommandBar(QWidget):
    """AYLAB kimliği + komut düğmeleri + dosya okuması + tayf kenarı."""

    HEIGHT = 56

    def __init__(self, actions: dict[str, QAction], parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self._accent = QColor(theme.NEUTRAL_ACCENT)
        self._peak = -1.0
        self._peak_anim = QPropertyAnimation(self, b"peak", self)
        self._peak_anim.setDuration(520)
        self._peak_anim.setEasingCurve(QEasingCurve.Type.OutQuint)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(WORDMARK_WIDTH + 18, 6, 18, 8)
        layout.setSpacing(0)

        specs = (
            ("open", "OPEN", "Ctrl+O"),
            ("save", "SAVE", "Ctrl+S"),
            ("reset", "RESET", ""),
            ("undo", "UNDO", "Ctrl+Z"),
        )
        self.buttons: dict[str, CommandButton] = {}
        for key, label, hint in specs:
            b = CommandButton(actions[key], label, hint)
            layout.addWidget(b)
            self.buttons[key] = b
        layout.addStretch(1)

        self.file_label = QLabel("no file loaded")
        self.file_label.setFont(theme.mono_font(8.5))
        self.file_label.setObjectName("Faint")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.file_label.setMinimumWidth(80)
        layout.addWidget(self.file_label)
        layout.addSpacing(14)

        about = CommandButton(actions["about"], "ABOUT")
        layout.addWidget(about)
        self.buttons["about"] = about

        # Eski kodla / testlerle uyumluluk için kısa adlar
        self.open_button = self.buttons["open"]
        self.save_button = self.buttons["save"]
        self.reset_button = self.buttons["reset"]
        self.undo_button = self.buttons["undo"]
        self.about_button = about

    # --- tayf tepe noktası ------------------------------------------------
    def _get_peak(self) -> float:
        return self._peak

    def _set_peak(self, v: float) -> None:
        self._peak = v
        self.update()

    peak = Property(float, _get_peak, _set_peak)

    def set_accent(self, color: QColor, wavelength: int | None) -> None:
        self._accent = QColor(color)
        for b in self.buttons.values():
            b.accent = self._accent
            b.update()
        target = -1.0 if wavelength is None else theme.wavelength_to_unit(wavelength)
        if self._peak < 0 or target < 0:
            self._set_peak(target)
        else:
            self._peak_anim.stop()
            self._peak_anim.setStartValue(self._peak)
            self._peak_anim.setEndValue(target)
            self._peak_anim.start()

    def set_file_text(self, text: str) -> None:
        self.file_label.setText(text)
        self.file_label.setToolTip(text)

    # --- çizim ----------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(theme.PANEL))

        # Sol kimlik bloğu: sağ kenarı eğik (asimetrik) daha koyu bir plaka
        slab = QPainterPath()
        slab.moveTo(0, 0)
        slab.lineTo(WORDMARK_WIDTH, 0)
        slab.lineTo(WORDMARK_WIDTH - 22, h)
        slab.lineTo(0, h)
        slab.closeSubpath()
        p.fillPath(slab, QColor(theme.BG_DEEP))
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawLine(QPointF(WORDMARK_WIDTH, 0), QPointF(WORDMARK_WIDTH - 22, h))

        paint_logo(p, 18, h / 2 - 9.5)
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.display_font(15, QFont.Weight.Bold, 5.0))
        p.drawText(QRectF(48, 9, 200, 24), Qt.AlignmentFlag.AlignVCenter, theme.BRAND_NAME)
        p.setPen(QColor(theme.TEXT_FAINT))
        p.setFont(theme.mono_font(7.2))
        p.drawText(QRectF(49, 31, 230, 14), Qt.AlignmentFlag.AlignVCenter, theme.BRAND_DESCRIPTOR.upper())

        # Alt kenar: soluk tayf + seçili dalga boyunda parlayan emisyon çizgisi
        y = h - 1.5
        p.fillRect(QRectF(0, y, w, 1.5), theme.spectrum_gradient(0, 0, w, 0, alpha=70))
        if self._peak >= 0:
            x = self._peak * w
            glow = QRadialGradient(QPointF(x, y), 90)
            glow.setColorAt(0, theme.with_alpha(self._accent, 120))
            glow.setColorAt(1, theme.with_alpha(self._accent, 0))
            p.fillRect(QRectF(x - 90, y - 10, 180, 11.5), glow)
            p.fillRect(QRectF(x - 40, y - 0.5, 80, 2), self._accent)
        p.end()
