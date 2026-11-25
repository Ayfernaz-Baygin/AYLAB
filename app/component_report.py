"""Bağlantılı bileşen raporu: sayı, alan, sınırlayıcı kutu ve ağırlık merkezi tablosu.

Sağ paneldeki analiz alanında, histogramın üstünde gösterilir; yalnızca rapor
üreten bir işlem (Connected Components) uygulandığında görünür. Sayılar
``processing.morphology.ComponentReport`` içinden gelir; bu modül yalnızca çizer.
Satırdaki renk örneği, etiket görüntüsündeki rengin aynısıdır.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import theme
from processing.morphology import ComponentReport, label_colors

MAX_ROWS = 5000  # çok parçalı (gürültülü) görüntülerde arayüzü kilitlememek için


class _NumericItem(QTableWidgetItem):
    """Sayısal sıralama: görünen metin yerine saklanan sayıya göre karşılaştırır."""

    def __init__(self, text: str, key: float) -> None:
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, key)
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other: QTableWidgetItem) -> bool:  # noqa: D105
        return self.data(Qt.ItemDataRole.UserRole) < other.data(Qt.ItemDataRole.UserRole)


class ComponentReportView(QWidget):
    HEADERS = ("#", "Area", "Box x,y w×h", "Centroid")
    TOOLTIPS = ("Label (background = 0)", "Area in pixels", "Bounding box: top-left x,y and width × height",
                "Centroid x, y (mean pixel position)")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        title = QLabel("COMPONENTS")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.count_label = QLabel("")
        self.count_label.setFont(theme.mono_font(8.5))
        self.count_label.setWordWrap(True)
        layout.addWidget(self.count_label)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.verticalHeader().hide()
        self.table.setFont(theme.mono_font(8))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.setWordWrap(False)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(19)
        for i, tip in enumerate(self.TOOLTIPS):
            self.table.horizontalHeaderItem(i).setToolTip(tip)
        header = self.table.horizontalHeader()
        header.setFont(theme.mono_font(7.5))
        header.setMinimumSectionSize(20)
        # İlk üç sütunun genişliği içerikten hesaplanır (_fit_columns); son sütun kalan alanı alır.
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(200)
        self.table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.table.setStyleSheet(
            f"QTableWidget {{ background: {theme.BG_DEEP}; color: {theme.TEXT}; border: 1px solid {theme.LINE_SOFT};"
            f" selection-background-color: {theme.LINE}; selection-color: {theme.TEXT}; }}"
            f"QHeaderView::section {{ background: {theme.PANEL_RAISED}; color: {theme.TEXT_DIM}; border: none;"
            f" border-bottom: 1px solid {theme.LINE}; padding: 3px 4px; }}"
        )
        layout.addWidget(self.table)
        self.note = QLabel("")
        self.note.setObjectName("Faint")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.report: ComponentReport | None = None
        self.set_report(None)

    def set_report(self, report: ComponentReport | None) -> None:
        self.report = report
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if report is None:
            self.hide()
            return
        self.count_label.setText(report.summary())
        palette = label_colors(report.count)
        rows = report.components[:MAX_ROWS]
        self.table.setRowCount(len(rows))
        for r, comp in enumerate(rows):
            b, g, rr = (int(v) for v in palette[comp.label])
            swatch = QPixmap(10, 10)
            swatch.fill(QColor(rr, g, b))
            first = _NumericItem(str(comp.label), comp.label)
            first.setIcon(QIcon(swatch))
            first.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            tip = (f"Label {comp.label} — area {comp.area} px\nBounding box x={comp.x}, y={comp.y}, "
                   f"w={comp.width}, h={comp.height}\nCentroid ({comp.cx:.2f}, {comp.cy:.2f})")
            cells = (first, _NumericItem(str(comp.area), comp.area),
                     QTableWidgetItem(f"{comp.x},{comp.y} {comp.width}×{comp.height}"),
                     QTableWidgetItem(f"{comp.cx:.1f}, {comp.cy:.1f}"))
            for col, item in enumerate(cells):
                item.setToolTip(tip)
                self.table.setItem(r, col, item)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicatorShown(False)   # ok işareti sütunu gereksiz genişletir
        self.table.sortItems(0, Qt.SortOrder.AscendingOrder)
        self._fit_columns()
        hidden = report.count - len(rows)
        self.note.setText(
            (f"Showing the first {len(rows)} of {report.count} components.  " if hidden else "")
            + "Background = label 0 (black). Click a header to sort."
        )
        self.show()

    def _fit_columns(self) -> None:
        """İlk üç sütunu en uzun metne (ve başlığa) göre ayarlar; ikon ve hücre boşluğu payı eklenir."""
        fm, hfm = self.table.fontMetrics(), self.table.horizontalHeader().fontMetrics()
        header = self.table.horizontalHeader()
        for col in range(3):
            texts = [self.table.item(r, col).text() for r in range(self.table.rowCount())]
            width = max([fm.horizontalAdvance(t) for t in texts] + [hfm.horizontalAdvance(self.HEADERS[col])])
            header.resizeSection(col, width + (26 if col == 0 else 12))

    def rows(self) -> list[tuple[str, ...]]:
        """Tablonun görünen satırları (test ve doğrulama için)."""
        return [tuple(self.table.item(r, c).text() for c in range(self.table.columnCount()))
                for r in range(self.table.rowCount())]
