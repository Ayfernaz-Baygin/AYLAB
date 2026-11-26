"""Görüntü tuvali: sığdırma, yakınlaştırma, kaydırma, piksel ızgarası ve
"split" (önce/sonra perdesi) karşılaştırması.

Tuval tamamen QPainter ile çizilir; böylece görüntü her zaman sahnenin
yıldızı olur ve çevresindeki her şey (kayıt işaretleri, etiketler,
ölçüm okumaları) görüntünün üzerine ince bir alet katmanı gibi biner.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from app import theme
from processing.analysis import pixel_value
from processing.image_io import SUPPORTED_EXTENSIONS

MIN_ZOOM = 0.02
MAX_ZOOM = 40.0
ZOOM_STEP = 1.25
PIXEL_GRID_ZOOM = 12.0  # Bu ölçeğin üstünde piksel sınırları çizilir
CLICK_SLOP = 4.0  # px: basma ile bırakma arası bundan az hareket "tıklama" sayılır (sürükleme değil)


def ndarray_to_qimage(image: np.ndarray) -> QImage:
    """OpenCV görüntüsünü (gri, BGR veya BGRA) QImage'e dönüştürür."""
    if image.dtype != np.uint8:
        raise ValueError(f"Yalnızca uint8 görüntüler desteklenir (gelen: {image.dtype})")

    image = np.ascontiguousarray(image)
    if image.ndim == 2:
        h, w = image.shape
        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_Grayscale8)
    elif image.ndim == 3 and image.shape[2] == 3:
        h, w, _ = image.shape
        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_BGR888)
    elif image.ndim == 3 and image.shape[2] == 4:
        h, w, _ = image.shape
        # OpenCV BGRA, little-endian makinede Qt ARGB32 ile aynı bayt düzenidir.
        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_ARGB32)
    else:
        raise ValueError(f"Desteklenmeyen görüntü şekli: {image.shape}")

    # QImage NumPy belleğini paylaşır; kopyalayarak bağımsız hale getiriyoruz.
    return qimg.copy()


def _dot_grid_tile(step: int = 16) -> QPixmap:
    """Sahne zemini için tekrar eden nokta ızgarası (kesim matı / lam hissi)."""
    tile = QPixmap(step, step)
    tile.fill(QColor(theme.BG_STAGE))
    p = QPainter(tile)
    p.setPen(QColor(theme.LINE_SOFT))
    p.drawPoint(0, 0)
    p.end()
    return tile


class ImageViewer(QWidget):
    """Tek bir görüntüyü gösteren etkileşimli tuval.

    * Varsayılan olarak *fit* modundadır: görüntü alana orantılı sığar.
    * Tekerlek → imleç konumuna göre yakınlaştırma, sürükleme → kaydırma,
      çift tıklama → fit / 1:1 arasında geçiş.
    * ``set_compare_image`` ile ikinci bir görüntü verilirse ve split modu
      açıksa, dikey bir perde ile önce/sonra karşılaştırması çizilir.
    """

    view_changed = Signal(float, bool, QPointF)  # zoom, fit, normalize merkez
    zoom_changed = Signal(float)
    cursor_info = Signal(str)
    file_dropped = Signal(str)
    pixel_clicked = Signal(int, int)  # x, y — sürüklemeden sol tıklama (ör. Region Growing tohumu)

    def __init__(self, title: str = "", parent=None, *, tag: str = "", caption: str = "",
                 empty_hint: str = "", tagline: str = "") -> None:
        super().__init__(parent)
        self._tagline = tagline  # Karşılama ekranında gösterilen marka sloganı (isteğe bağlı)
        self._title = (title or "Image").upper()
        self._tag = tag or self._title[:1]
        self._caption = caption or "NO SIGNAL"
        self._empty_hint = empty_hint or "Drop an image here  ·  Ctrl+O"
        self._accent = QColor(theme.NEUTRAL_ACCENT)

        self._array: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._compare_pixmap: QPixmap | None = None
        self._split_enabled = False
        self._split_pos = 0.5
        self._dragging_split = False

        self._fit = True
        self._zoom = 1.0
        self._center = QPointF(0.5, 0.5)
        self._drag_origin: QPointF | None = None
        self._drag_center = QPointF(0.5, 0.5)
        self._press_pos: QPointF | None = None
        self._hover_px: tuple[int, int] | None = None

        self._grid_brush = QBrush(_dot_grid_tile())
        self._phase_clock = QElapsedTimer()
        self._phase_clock.start()
        self._anim = QTimer(self)
        self._anim.setInterval(50)
        self._anim.timeout.connect(self.update)

        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setMinimumSize(220, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.clear_image()

    # ------------------------------------------------------------------
    # Genel API
    # ------------------------------------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        old_size = self._pixmap.size() if self._pixmap is not None else None
        self._array = image
        self._pixmap = QPixmap.fromImage(ndarray_to_qimage(image))
        if old_size is None or old_size != self._pixmap.size():
            self._fit = True
            self._center = QPointF(0.5, 0.5)
        self._anim.stop()
        self._emit_zoom()
        self.update()

    def clear_image(self) -> None:
        self._array = None
        self._pixmap = None
        self._fit = True
        self._center = QPointF(0.5, 0.5)
        self._hover_px = None
        if self.isVisible():
            self._anim.start()
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None

    def image_array(self) -> np.ndarray | None:
        return self._array

    def set_accent(self, color: QColor) -> None:
        self._accent = QColor(color)
        self.update()

    def set_compare_image(self, image: np.ndarray | None) -> None:
        self._compare_pixmap = None if image is None else QPixmap.fromImage(ndarray_to_qimage(image))
        self.update()

    def set_split_enabled(self, enabled: bool) -> None:
        self._split_enabled = enabled
        self.update()

    def split_enabled(self) -> bool:
        return self._split_enabled

    def split_position(self) -> float:
        return self._split_pos

    def set_split_position(self, pos: float) -> None:
        self._split_pos = max(0.0, min(1.0, pos))
        self.update()

    # --- yakınlaştırma ---------------------------------------------------
    def is_fit(self) -> bool:
        return self._fit

    def effective_scale(self) -> float:
        if self._pixmap is None:
            return 1.0
        return self._fit_scale() if self._fit else self._zoom

    def zoom_in(self) -> None:
        self.set_zoom(self.effective_scale() * ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self.effective_scale() / ZOOM_STEP)

    def actual_size(self) -> None:
        self.set_zoom(1.0)

    def fit_to_view(self) -> None:
        self._fit = True
        self._center = QPointF(0.5, 0.5)
        self._after_view_change()

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        if self._pixmap is None:
            return
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if anchor is not None:
            # İmlecin altındaki görüntü noktası yakınlaştırmadan sonra da aynı yerde kalsın.
            before = self._widget_to_image(anchor)
            self._fit, self._zoom = False, zoom
            pw, ph = self._pixmap.width(), self._pixmap.height()
            cx = before.x() - (anchor.x() - self.width() / 2) / zoom
            cy = before.y() - (anchor.y() - self.height() / 2) / zoom
            self._center = QPointF(cx / pw, cy / ph)
        else:
            self._fit, self._zoom = False, zoom
        self._clamp_center()
        self._after_view_change()

    def apply_view(self, zoom: float, fit: bool, center: QPointF) -> None:
        """Bağlı görüntüleyiciden gelen görünümü sinyal yaymadan uygular."""
        self._fit, self._zoom, self._center = fit, zoom, QPointF(center)
        self._clamp_center()
        self._emit_zoom()
        self.update()

    # ------------------------------------------------------------------
    # Geometri
    # ------------------------------------------------------------------
    def _fit_scale(self) -> float:
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0
        margin = 28
        w = max(1, self.width() - 2 * margin)
        h = max(1, self.height() - 2 * margin)
        return min(w / self._pixmap.width(), h / self._pixmap.height())

    def _image_rect(self) -> QRectF:
        assert self._pixmap is not None
        s = self.effective_scale()
        w, h = self._pixmap.width() * s, self._pixmap.height() * s
        cx = self._center.x() * w
        cy = self._center.y() * h
        return QRectF(self.width() / 2 - cx, self.height() / 2 - cy, w, h)

    def _widget_to_image(self, pos: QPointF) -> QPointF:
        rect = self._image_rect()
        s = self.effective_scale()
        return QPointF((pos.x() - rect.x()) / s, (pos.y() - rect.y()) / s)

    def _clamp_center(self) -> None:
        if self._pixmap is None:
            return
        s = self.effective_scale()
        cx, cy = self._center.x(), self._center.y()
        for axis, view, size in (("x", self.width(), self._pixmap.width() * s),
                                 ("y", self.height(), self._pixmap.height() * s)):
            if size <= view:
                value = 0.5
            else:
                half = (view / 2) / size
                value = max(half, min(1 - half, cx if axis == "x" else cy))
            if axis == "x":
                cx = value
            else:
                cy = value
        self._center = QPointF(cx, cy)

    def _after_view_change(self) -> None:
        self._emit_zoom()
        self.view_changed.emit(self._zoom, self._fit, QPointF(self._center))
        self.update()

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(self.effective_scale())

    def _split_x(self) -> float:
        rect = self._image_rect()
        left = max(0.0, rect.left())
        right = min(float(self.width()), rect.right())
        return left + (right - left) * self._split_pos

    def _split_active(self) -> bool:
        return self._split_enabled and self._compare_pixmap is not None and self._pixmap is not None

    # ------------------------------------------------------------------
    # Çizim
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self._grid_brush)

        if self._pixmap is None:
            self._paint_placeholder(p)
        else:
            self._paint_image(p)
        self._paint_frame(p)
        self._paint_labels(p)
        p.end()

    def _paint_image(self, p: QPainter) -> None:
        assert self._pixmap is not None
        rect = self._image_rect()
        scale = self.effective_scale()
        smooth = scale < 2.0  # Büyütmede piksel doğruluğu: en yakın komşu
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)

        # Görüntünün altına yumuşak bir gölge — sahneden "kalkmış" levha hissi.
        shadow = QColor(0, 0, 0, 90)
        p.fillRect(rect.translated(6, 8), shadow)

        if self._split_active():
            sx = self._split_x()
            p.save()
            p.setClipRect(QRectF(0, 0, sx, self.height()))
            p.drawPixmap(rect, self._compare_pixmap, QRectF(self._compare_pixmap.rect()))
            p.restore()
            p.save()
            p.setClipRect(QRectF(sx, 0, self.width() - sx, self.height()))
            p.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
            p.restore()
        else:
            p.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

        if scale >= PIXEL_GRID_ZOOM:
            self._paint_pixel_grid(p, rect, scale)
        self._paint_registration_marks(p, rect)
        if self._split_active():
            self._paint_split_handle(p, rect)

    def _paint_pixel_grid(self, p: QPainter, rect: QRectF, scale: float) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QPen(QColor(10, 15, 27, 120), 1))
        visible = rect.intersected(QRectF(self.rect()))
        x0 = math.floor((visible.left() - rect.left()) / scale)
        x1 = math.ceil((visible.right() - rect.left()) / scale)
        y0 = math.floor((visible.top() - rect.top()) / scale)
        y1 = math.ceil((visible.bottom() - rect.top()) / scale)
        for i in range(x0, x1 + 1):
            x = rect.left() + i * scale
            p.drawLine(QPointF(x, visible.top()), QPointF(x, visible.bottom()))
        for j in range(y0, y1 + 1):
            y = rect.top() + j * scale
            p.drawLine(QPointF(visible.left(), y), QPointF(visible.right(), y))
        if self._hover_px is not None:
            hx, hy = self._hover_px
            p.setPen(QPen(self._accent, 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(rect.left() + hx * scale, rect.top() + hy * scale, scale, scale))
        p.restore()

    def _paint_registration_marks(self, p: QPainter, rect: QRectF) -> None:
        """Görüntü köşelerinde baskı/mikroskop kayıt işaretleri."""
        p.save()
        p.setPen(QPen(self._accent, 1.4))
        L, g = 12.0, 5.0
        corners = (
            (rect.topLeft(), -1, -1),
            (rect.topRight(), 1, -1),
            (rect.bottomLeft(), -1, 1),
            (rect.bottomRight(), 1, 1),
        )
        for pt, sx, sy in corners:
            ox, oy = pt.x() + sx * g, pt.y() + sy * g
            p.drawLine(QPointF(ox, oy), QPointF(ox - sx * L, oy))
            p.drawLine(QPointF(ox, oy), QPointF(ox, oy - sy * L))
        p.restore()

    def _paint_split_handle(self, p: QPainter, rect: QRectF) -> None:
        sx = self._split_x()
        top, bottom = max(0.0, rect.top()), min(float(self.height()), rect.bottom())
        p.save()
        p.setPen(QPen(QColor(0, 0, 0, 120), 3))
        p.drawLine(QPointF(sx, top), QPointF(sx, bottom))
        p.setPen(QPen(self._accent, 1.5))
        p.drawLine(QPointF(sx, top), QPointF(sx, bottom))
        cy = (top + bottom) / 2
        diamond = QPainterPath()
        r = 9.0
        diamond.moveTo(sx, cy - r)
        diamond.lineTo(sx + r, cy)
        diamond.lineTo(sx, cy + r)
        diamond.lineTo(sx - r, cy)
        diamond.closeSubpath()
        p.setBrush(QColor(theme.BG_DEEP))
        p.drawPath(diamond)
        p.setPen(QPen(self._accent, 1.2))
        for d in (-1, 1):
            p.drawLine(QPointF(sx + d * 2.5, cy - 3), QPointF(sx + d * 5.5, cy))
            p.drawLine(QPointF(sx + d * 5.5, cy), QPointF(sx + d * 2.5, cy + 3))
        # Perde etiketleri
        p.setFont(theme.mono_font(8, QFont.Weight.DemiBold))
        for text, right in (("A  BEFORE", False), ("AFTER  B", True)):
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(text) + 12
            x = sx + 10 if right else sx - 10 - tw
            chip = QRectF(x, top + 10, tw, 20)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(10, 15, 27, 200))
            p.drawRect(chip)
            p.setPen(QColor(theme.TEXT))
            p.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
        p.restore()

    def _paint_frame(self, p: QPainter) -> None:
        p.save()
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(theme.LINE), 1))
        path = theme.chamfer_path(0.5, 0.5, self.width() - 1, self.height() - 1, 14, "tl,br")
        p.drawPath(path)
        # Pah kırılmış köşede vurgu çizgisi
        p.setPen(QPen(self._accent, 2))
        p.drawLine(QPointF(0.5, 14.5), QPointF(14.5, 0.5))
        p.restore()

    def _paint_labels(self, p: QPainter) -> None:
        p.save()
        # Sol üst: [A] SOURCE
        tag_rect = QRectF(20, 12, 20, 20)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._accent)
        p.drawRect(tag_rect)
        p.setPen(QColor(theme.BG_DEEP))
        p.setFont(theme.mono_font(9, QFont.Weight.Bold))
        p.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, self._tag)
        p.setPen(QColor(theme.TEXT))
        p.setFont(theme.display_font(9.5, QFont.Weight.DemiBold, 2.0))
        p.drawText(QRectF(48, 12, 300, 20), Qt.AlignmentFlag.AlignVCenter, self._title)

        # Sağ üst: boyut ve ölçek okuması
        if self._pixmap is not None:
            text = f"{self._pixmap.width()}×{self._pixmap.height()}  ·  {self.effective_scale() * 100:.0f}%"
            if self._fit:
                text += "  FIT"
            p.setFont(theme.mono_font(8.5))
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(text) + 14
            chip = QRectF(self.width() - tw - 14, 12, tw, 20)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(10, 15, 27, 190))
            p.drawRect(chip)
            p.setPen(QColor(theme.TEXT_DIM))
            p.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
        p.restore()

    def _paint_placeholder(self, p: QPainter) -> None:
        """"Sinyal yok" ekranı: tayf dalgasıyla nabız atan bir sensör matrisi."""
        t = self._phase_clock.elapsed() / 1000.0
        cx, cy = self.width() / 2, self.height() / 2 - 18
        n, cell, gap = 9, 9.0, 4.0
        span = n * cell + (n - 1) * gap
        x0, y0 = cx - span / 2, cy - span / 2

        p.save()
        # Dış nişangâh halkası ve eksenler
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        radius = span * 0.82
        p.drawEllipse(QPointF(cx, cy), radius, radius)
        p.setPen(QPen(QColor(theme.LINE), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(cx - radius - 18, cy), QPointF(cx + radius + 18, cy))
        p.drawLine(QPointF(cx, cy - radius - 18), QPointF(cx, cy + radius + 18))

        # Sensör hücreleri: köşegen boyunca ilerleyen tayf dalgası
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            for j in range(n):
                d = (i + j) / (2 * (n - 1))
                wave = 0.5 + 0.5 * math.sin(2 * math.pi * (d * 1.4 - t * 0.35))
                dist = math.hypot(i - (n - 1) / 2, j - (n - 1) / 2) / ((n - 1) / 2 * 1.42)
                strength = max(0.0, wave ** 3 * (1 - dist * 0.85))
                nm = theme.SPECTRUM_MAX_NM - d * (theme.SPECTRUM_MAX_NM - theme.SPECTRUM_MIN_NM)
                color = theme.wavelength_color(nm)
                base = QColor(theme.LINE_SOFT)
                mix = QColor(
                    round(base.red() + (color.red() - base.red()) * strength),
                    round(base.green() + (color.green() - base.green()) * strength),
                    round(base.blue() + (color.blue() - base.blue()) * strength),
                )
                p.setBrush(mix)
                p.drawRect(QRectF(x0 + i * (cell + gap), y0 + j * (cell + gap), cell, cell))

        # Yazılar
        p.setPen(QColor(theme.TEXT_DIM))
        p.setFont(theme.display_font(11, QFont.Weight.DemiBold, 4.0))
        p.drawText(QRectF(0, cy + radius + 26, self.width(), 22), Qt.AlignmentFlag.AlignHCenter, self._caption)
        p.setPen(QColor(theme.TEXT_FAINT))
        p.setFont(theme.mono_font(8.5))
        p.drawText(QRectF(0, cy + radius + 50, self.width(), 18), Qt.AlignmentFlag.AlignHCenter, self._empty_hint)
        if self._tagline:
            # İnce bir ayraç ve marka sloganı — kayıt işaretleriyle aynı vurgu rengi
            ty = cy + radius + 88
            p.fillRect(QRectF(self.width() / 2 - 14, ty - 10, 28, 1.5), self._accent)
            p.setPen(QColor(theme.TEXT_DIM))
            p.setFont(theme.display_font(8.5, QFont.Weight.DemiBold, 3.0))
            p.drawText(QRectF(0, ty, self.width(), 18), Qt.AlignmentFlag.AlignHCenter, self._tagline)
        p.restore()

    # ------------------------------------------------------------------
    # Olaylar
    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._pixmap is None:
            self._anim.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._anim.stop()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._clamp_center()
        self._emit_zoom()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(480, 400)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.set_zoom(self.effective_scale() * (ZOOM_STEP ** steps), event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._pixmap is None:
            return
        pos = event.position()
        if self._split_active() and abs(pos.x() - self._split_x()) < 10:
            self._dragging_split = True
        else:
            self._drag_origin = pos
            self._drag_center = QPointF(self._center)
            self._press_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._pixmap is None:
            return
        if self._dragging_split:
            rect = self._image_rect()
            left = max(0.0, rect.left())
            right = min(float(self.width()), rect.right())
            if right > left:
                self.set_split_position((pos.x() - left) / (right - left))
            return
        if self._drag_origin is not None:
            s = self.effective_scale()
            if not self._fit:
                dx = (pos.x() - self._drag_origin.x()) / (self._pixmap.width() * s)
                dy = (pos.y() - self._drag_origin.y()) / (self._pixmap.height() * s)
                self._center = QPointF(self._drag_center.x() - dx, self._drag_center.y() - dy)
                self._clamp_center()
                self._after_view_change()
            return
        if self._split_active():
            near = abs(pos.x() - self._split_x()) < 10
            self.setCursor(Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_hover(pos)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        press, self._press_pos = self._press_pos, None
        self._drag_origin = None
        self._dragging_split = False
        self.setCursor(Qt.CursorShape.CrossCursor)
        if press is None or event.button() != Qt.MouseButton.LeftButton or self._array is None:
            return
        pos = event.position()
        if abs(pos.x() - press.x()) + abs(pos.y() - press.y()) > CLICK_SLOP:
            return  # kaydırma (sürükleme) — tıklama değil
        pt = self._widget_to_image(pos)
        x, y = int(math.floor(pt.x())), int(math.floor(pt.y()))
        h, w = self._array.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.pixel_clicked.emit(x, y)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        if self._fit:
            self.set_zoom(1.0, event.position())
        else:
            self.fit_to_view()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_px = None
        self.cursor_info.emit("")
        self.update()

    def _update_hover(self, pos: QPointF) -> None:
        if self._array is None:
            return
        pt = self._widget_to_image(pos)
        x, y = int(math.floor(pt.x())), int(math.floor(pt.y()))
        value = pixel_value(self._array, x, y)
        new_hover = (x, y) if value is not None else None
        if new_hover != self._hover_px:
            self._hover_px = new_hover
            if self.effective_scale() >= PIXEL_GRID_ZOOM:
                self.update()
        if value is None:
            self.cursor_info.emit("")
            return
        if len(value) == 1:
            readout = f"L {value[0]:>3}"
        else:
            names = ("B", "G", "R", "A")
            readout = "  ".join(f"{n} {v:>3}" for n, v in zip(names, value))
        self.cursor_info.emit(f"x {x:>4}  y {y:>4}   {readout}")

    # --- sürükle-bırak ---------------------------------------------------
    @staticmethod
    def _dropped_image_path(mime) -> str | None:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if path and path.lower().endswith(SUPPORTED_EXTENSIONS):
                return path
        return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._dropped_image_path(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        path = self._dropped_image_path(event.mimeData())
        if path:
            event.acceptProposedAction()
            self.file_dropped.emit(path)
