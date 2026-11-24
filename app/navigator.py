"""Sol navigasyon: "Process Spectrum".

Yedi kategori, bir spektrometrenin emisyon çizgileri gibi görünür ışık
tayfına dizilir. Üstteki tayf skalası seçili kategorinin dalga boyunu bir
işaretçiyle gösterir; altında her kategori, kendi dalga boyu renginde bir
"bant" olarak yer alır. Seçilen bandın işlemleri hemen altında açılır.
"""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app import theme
from processing.registry import Algorithm, AlgorithmRegistry, Category


class _Animated(QAbstractButton):
    """Hover/seçim geçişleri için 0..1 arası animasyonlu bir ilerleme değeri."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._hover = 0.0
        self._anim = QPropertyAnimation(self, b"hover", self)
        self._anim.setDuration(170)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(True)

    def _get_hover(self) -> float:
        return self._hover

    def _set_hover(self, value: float) -> None:
        self._hover = value
        self.update()

    hover = Property(float, _get_hover, _set_hover)

    def _animate_to(self, target: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._hover)
        self._anim.setEndValue(target)
        self._anim.start()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._animate_to(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._animate_to(0.0)
        super().leaveEvent(event)


class SpectrumScale(QWidget):
    """380–720 nm tayf şeridi; kategori çizgileri ve kayan seçim işaretçisi."""

    def __init__(self, categories: list[Category], parent=None) -> None:
        super().__init__(parent)
        self._categories = categories
        self._marker = -1.0  # 0..1, <0 ise gizli
        self._anim = QPropertyAnimation(self, b"marker", self)
        self._anim.setDuration(420)
        self._anim.setEasingCurve(QEasingCurve.Type.OutQuint)
        self._selected_nm: int | None = None
        self.setFixedHeight(46)

    def _get_marker(self) -> float:
        return self._marker

    def _set_marker(self, value: float) -> None:
        self._marker = value
        self.update()

    marker = Property(float, _get_marker, _set_marker)

    def set_wavelength(self, nm: int | None) -> None:
        self._selected_nm = nm
        target = -1.0 if nm is None else theme.wavelength_to_unit(nm)
        if self._marker < 0 or target < 0:
            self._set_marker(target)
            return
        self._anim.stop()
        self._anim.setStartValue(self._marker)
        self._anim.setEndValue(target)
        self._anim.start()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0, x1 = 4.0, self.width() - 4.0
        band = QRectF(x0, 16, x1 - x0, 6)
        p.fillRect(band, theme.spectrum_gradient(x0, 0, x1, 0, alpha=150))

        # Kategori emisyon çizgileri
        for cat in self._categories:
            x = x0 + theme.wavelength_to_unit(cat.wavelength) * (x1 - x0)
            selected = cat.wavelength == self._selected_nm
            p.setPen(QPen(QColor(theme.TEXT) if selected else QColor(theme.BG_DEEP), 1.2))
            p.drawLine(QPointF(x, 14), QPointF(x, 24))

        # Ölçek yazıları
        p.setFont(theme.mono_font(7.5))
        p.setPen(QColor(theme.TEXT_FAINT))
        p.drawText(QRectF(x0, 28, 60, 14), Qt.AlignmentFlag.AlignLeft, f"{theme.SPECTRUM_MIN_NM}")
        p.drawText(QRectF(x1 - 60, 28, 60, 14), Qt.AlignmentFlag.AlignRight, f"{theme.SPECTRUM_MAX_NM} nm")

        # Kayan işaretçi
        if self._marker >= 0:
            x = x0 + self._marker * (x1 - x0)
            nm = theme.SPECTRUM_MIN_NM + self._marker * (theme.SPECTRUM_MAX_NM - theme.SPECTRUM_MIN_NM)
            color = theme.wavelength_color(nm)
            tri = QPainterPath()
            tri.moveTo(x - 5, 4)
            tri.lineTo(x + 5, 4)
            tri.lineTo(x, 11)
            tri.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawPath(tri)
            p.setPen(QColor(theme.TEXT))
            label = f"λ {nm:.0f}"
            p.drawText(QRectF(x - 30, 28, 60, 14), Qt.AlignmentFlag.AlignHCenter, label)
        p.end()


class CategoryTile(_Animated):
    """Bir kategori bandı: dalga boyu şeridi, sıra kodu, ad ve tayf okuması."""

    def __init__(self, category: Category, op_count: int, parent=None) -> None:
        super().__init__(parent)
        self.category = category
        self.color = theme.wavelength_color(category.wavelength)
        self._op_count = op_count
        self.setFixedHeight(54)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip(f"{category.title}\n{category.summary}")
        self.toggled.connect(lambda _c: self.update())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(240, 54)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        sel = self.isChecked()
        k = 1.0 if sel else self._hover

        # Zemin: seçili/hover iken bant renginden sağa doğru sönen ışık
        if k > 0:
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, theme.with_alpha(self.color, int(46 * k)))
            grad.setColorAt(0.65, theme.with_alpha(self.color, int(8 * k)))
            grad.setColorAt(1.0, theme.with_alpha(self.color, 0))
            p.fillRect(self.rect(), grad)

        # Dalga boyu bandı (hover'da genişler)
        band_w = 3 + 3 * k
        p.fillRect(QRectF(0, 6, band_w, h - 12), self.color)

        # Sıra kodu
        p.setFont(theme.mono_font(8.5, QFont.Weight.Bold))
        p.setPen(self.color if sel else QColor(theme.TEXT_FAINT))
        p.drawText(QRectF(16, 9, 26, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"{self.category.index:02d}")

        # Ad
        p.setFont(theme.display_font(10.5, QFont.Weight.DemiBold if sel else QFont.Weight.Normal, 0.3))
        p.setPen(QColor(theme.TEXT) if (sel or k > 0.5) else QColor("#B9C3D6"))
        title_rect = QRectF(42, 7, w - 42 - 32, 22)  # sağdaki dalga simgesine yer bırak
        title = p.fontMetrics().elidedText(self.category.title, Qt.TextElideMode.ElideRight, int(title_rect.width()))
        p.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

        # Alt satır: dalga boyu ve işlem sayısı
        p.setFont(theme.mono_font(7.5))
        p.setPen(QColor(theme.TEXT_DIM if sel else theme.TEXT_FAINT))
        p.drawText(QRectF(42, 29, w - 60, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"λ{self.category.wavelength} nm  ·  {self._op_count} ops")

        # Sağda "dalga formu" mini işareti — seçiliyken bant renginde
        p.setPen(QPen(self.color if sel else QColor(theme.LINE), 1.3))
        cx, cy = w - 18, h / 2
        path = QPainterPath(QPointF(cx - 6, cy))
        path.cubicTo(QPointF(cx - 3, cy - 6), QPointF(cx - 1, cy - 6), QPointF(cx, cy))
        path.cubicTo(QPointF(cx + 1, cy + 6), QPointF(cx + 3, cy + 6), QPointF(cx + 6, cy))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Alt ayraç
        p.setPen(QPen(QColor(theme.LINE_SOFT), 1))
        p.drawLine(QPointF(14, h - 0.5), QPointF(w, h - 0.5))
        p.end()


class OperationRow(_Animated):
    """Seçili kategorinin altında açılan işlem satırı."""

    def __init__(self, algorithm: Algorithm, color: QColor, parent=None) -> None:
        super().__init__(parent)
        self.algorithm = algorithm
        self.color = QColor(color)
        self.setFixedHeight(30)
        self.setToolTip(algorithm.summary)
        self.toggled.connect(lambda _c: self.update())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(220, 30)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        sel = self.isChecked()
        k = 1.0 if sel else self._hover

        # Akış çizgisi (kategori bandından inen ince hat)
        p.fillRect(QRectF(0, 0, 1, h), theme.with_alpha(self.color, 90))
        if k > 0:
            p.fillRect(QRectF(1, 2, w - 1, h - 4), theme.with_alpha(theme.PANEL_RAISED, int(255 * k)))

        # Piksel işaretçisi: boşken çerçeve, seçiliyken dolu kare
        px = QRectF(14, h / 2 - 3.5, 7, 7)
        if sel:
            p.fillRect(px, self.color)
        else:
            p.setPen(QPen(theme.with_alpha(self.color, 110 + int(145 * k)), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(px)

        p.setFont(theme.body_font(9.2, QFont.Weight.DemiBold if sel else QFont.Weight.Normal))
        p.setPen(QColor(theme.TEXT) if (sel or k > 0.4) else QColor(theme.TEXT_DIM))
        rect = QRectF(30, 0, w - 44, h)
        text = p.fontMetrics().elidedText(self.algorithm.title, Qt.TextElideMode.ElideRight, int(rect.width()))
        p.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)

        if sel:
            p.setPen(QPen(self.color, 1.4))
            x, y = w - 14, h / 2
            p.drawLine(QPointF(x - 4, y - 4), QPointF(x, y))
            p.drawLine(QPointF(x, y), QPointF(x - 4, y + 4))
        p.end()


class SpectralNavigator(QWidget):
    category_selected = Signal(str)
    algorithm_selected = Signal(str)

    def __init__(self, registry: AlgorithmRegistry, parent=None) -> None:
        super().__init__(parent)
        self._registry = registry
        self.setMinimumWidth(228)
        self.setMaximumWidth(340)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 6, 10)
        root.setSpacing(6)

        title = QLabel("PROCESS SPECTRUM")
        title.setObjectName("SectionTitle")
        root.addWidget(title)

        categories = registry.categories()
        self.scale = SpectrumScale(categories)
        root.addWidget(self.scale)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        holder = QWidget()
        self._list = QVBoxLayout(holder)
        self._list.setContentsMargins(0, 0, 4, 0)
        self._list.setSpacing(0)
        scroll.setWidget(holder)
        root.addWidget(scroll, 1)

        self._tile_group = QButtonGroup(self)
        self._tile_group.setExclusive(True)
        self.tiles: dict[str, CategoryTile] = {}
        for cat in categories:
            tile = CategoryTile(cat, len(registry.in_category(cat.key)))
            tile.clicked.connect(lambda _c=False, k=cat.key: self.select_category(k))
            self._tile_group.addButton(tile)
            self._list.addWidget(tile)
            self.tiles[cat.key] = tile
        self._list.addStretch(1)

        self._tray = QWidget()
        self._tray_layout = QVBoxLayout(self._tray)
        self._tray_layout.setContentsMargins(14, 2, 0, 8)
        self._tray_layout.setSpacing(0)
        self._tray.hide()
        self._op_group: QButtonGroup | None = None
        self.operation_rows: dict[str, OperationRow] = {}

        hint = QLabel("WHEEL zoom  ·  DRAG pan\nDOUBLE-CLICK fit ⇄ 1:1")
        hint.setObjectName("Faint")
        root.addWidget(hint)

        self._current_category: str | None = None
        self._current_algorithm: str | None = None

    # ------------------------------------------------------------------
    def current_category(self) -> str | None:
        return self._current_category

    def current_algorithm(self) -> str | None:
        return self._current_algorithm

    def select_category(self, key: str) -> None:
        cat = self._registry.category(key)
        tile = self.tiles[key]
        tile.setChecked(True)
        if key != self._current_category:
            self._current_category = key
            self._current_algorithm = None
            self._rebuild_tray(cat, tile)
            self.scale.set_wavelength(cat.wavelength)
            self.category_selected.emit(key)

    def select_algorithm(self, key: str) -> None:
        algo = self._registry.get(key)
        if algo.category != self._current_category:
            self.select_category(algo.category)
        row = self.operation_rows[key]
        row.setChecked(True)
        if key != self._current_algorithm:
            self._current_algorithm = key
            self.algorithm_selected.emit(key)

    def _rebuild_tray(self, category: Category, tile: CategoryTile) -> None:
        while self._tray_layout.count():
            item = self._tray_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.operation_rows = {}
        self._op_group = QButtonGroup(self._tray)
        self._op_group.setExclusive(True)
        for algo in self._registry.in_category(category.key):
            row = OperationRow(algo, tile.color)
            row.clicked.connect(lambda _c=False, k=algo.key: self.select_algorithm(k))
            self._op_group.addButton(row)
            self._tray_layout.addWidget(row)
            self.operation_rows[algo.key] = row
        # Tepsiyi seçili kategori bandının hemen altına taşı (akordeon).
        self._list.removeWidget(self._tray)
        self._list.insertWidget(self._list.indexOf(tile) + 1, self._tray)
        self._tray.show()
