"""Orijinal (A) ve işlenmiş (B) görüntünün karşılaştırıldığı merkez sahne.

* SIDE  — A ve B yan yana; dar pencerelerde otomatik olarak üst üste dizilir.
* SPLIT — tek tuval üzerinde sürüklenebilir önce/sonra perdesi.
* LINK  — açıkken iki tuvalin zoom ve kaydırması eşlenir.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import theme
from app.image_viewer import ImageViewer


class CompareMode:
    SIDE = "side"
    SPLIT = "split"


def _tool(text: str, tip: str, checkable: bool = False) -> QToolButton:
    b = QToolButton()
    b.setObjectName("StageTool")
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class CompareStage(QWidget):
    zoom_changed = Signal(float)
    cursor_info = Signal(str)
    file_dropped = Signal(str)
    mode_changed = Signal(str)
    pixel_clicked = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mode = CompareMode.SIDE
        self._original: np.ndarray | None = None

        self.original_viewer = ImageViewer("Source", tag="A", caption="NO SIGNAL",
                                           empty_hint="Drop an image here  ·  Ctrl+O",
                                           tagline=theme.BRAND_TAGLINE)
        self.result_viewer = ImageViewer("Output", tag="B", caption="AWAITING SOURCE",
                                         empty_hint="Processed result appears here")

        # --- araç şeridi ------------------------------------------------
        bar = QWidget()
        bar.setFixedHeight(34)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(2, 0, 2, 0)
        bl.setSpacing(4)

        title = QLabel("COMPARATOR")
        title.setObjectName("SectionTitle")
        bl.addWidget(title)
        bl.addSpacing(10)

        self.side_button = _tool("A │ B", "Side by side — source and output next to each other", True)
        self.split_button = _tool("SPLIT", "Before / after curtain — drag the divider", True)
        self.side_button.setChecked(True)
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.side_button)
        group.addButton(self.split_button)
        bl.addWidget(self.side_button)
        bl.addWidget(self.split_button)

        self.link_button = _tool("LINK", "Synchronise zoom and pan of A and B", True)
        self.link_button.setChecked(True)
        bl.addWidget(self.link_button)
        bl.addStretch(1)

        self.zoom_out_button = _tool("−", "Zoom out  (Ctrl+−)")
        self.zoom_label = QLabel("—")
        self.zoom_label.setFont(theme.mono_font(9))
        self.zoom_label.setFixedWidth(58)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setObjectName("Muted")
        self.zoom_in_button = _tool("+", "Zoom in  (Ctrl++)")
        self.fit_button = _tool("FIT", "Fit to view  (Ctrl+0)")
        self.actual_button = _tool("1:1", "Actual pixels  (Ctrl+1)")
        for w in (self.zoom_out_button, self.zoom_label, self.zoom_in_button, self.fit_button,
                  self.actual_button):
            bl.addWidget(w)

        # --- tuvaller ---------------------------------------------------
        self._viewers = QWidget()
        self._viewer_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self._viewers)
        self._viewer_layout.setContentsMargins(0, 0, 0, 0)
        self._viewer_layout.setSpacing(10)
        self._viewer_layout.addWidget(self.original_viewer)
        self._viewer_layout.addWidget(self.result_viewer)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        root.addWidget(bar)
        root.addWidget(self._viewers, 1)

        # --- bağlantılar ------------------------------------------------
        self.side_button.clicked.connect(lambda: self.set_mode(CompareMode.SIDE))
        self.split_button.clicked.connect(lambda: self.set_mode(CompareMode.SPLIT))
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.fit_button.clicked.connect(self.fit_to_view)
        self.actual_button.clicked.connect(self.actual_size)

        for src, dst in ((self.original_viewer, self.result_viewer),
                         (self.result_viewer, self.original_viewer)):
            src.view_changed.connect(lambda z, f, c, d=dst: self._sync(d, z, f, c))
            src.cursor_info.connect(self.cursor_info)
            src.file_dropped.connect(self.file_dropped)
            src.pixel_clicked.connect(self.pixel_clicked)
        self.result_viewer.zoom_changed.connect(self._on_zoom)
        self.original_viewer.zoom_changed.connect(self._on_zoom_from_original)
        self._set_tools_enabled(False)

    # ------------------------------------------------------------------
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in (CompareMode.SIDE, CompareMode.SPLIT):
            raise ValueError(mode)
        self._mode = mode
        split = mode == CompareMode.SPLIT
        self.side_button.setChecked(not split)
        self.split_button.setChecked(split)
        self.original_viewer.setVisible(not split)
        self.result_viewer.set_compare_image(self._original if split else None)
        self.result_viewer.set_split_enabled(split)
        self._update_direction()
        self.mode_changed.emit(mode)

    def set_accent(self, color: QColor) -> None:
        self.original_viewer.set_accent(color)
        self.result_viewer.set_accent(color)

    def set_original(self, image: np.ndarray) -> None:
        self._original = image
        self.original_viewer.set_image(image)
        if self._mode == CompareMode.SPLIT:
            self.result_viewer.set_compare_image(image)
        self._set_tools_enabled(True)

    def set_processed(self, image: np.ndarray) -> None:
        self.result_viewer.set_image(image)
        self._on_zoom(self.result_viewer.effective_scale())

    def clear(self) -> None:
        self._original = None
        self.original_viewer.clear_image()
        self.result_viewer.clear_image()
        self.result_viewer.set_compare_image(None)
        self._set_tools_enabled(False)
        self.zoom_label.setText("—")

    # --- yakınlaştırma (aktif tuval B; LINK açıksa A da izler) -----------
    def _active(self) -> ImageViewer:
        return self.result_viewer

    def zoom_in(self) -> None:
        self._active().zoom_in()

    def zoom_out(self) -> None:
        self._active().zoom_out()

    def fit_to_view(self) -> None:
        self._active().fit_to_view()

    def actual_size(self) -> None:
        self._active().actual_size()

    def _sync(self, target: ImageViewer, zoom: float, fit: bool, center: QPointF) -> None:
        if self.link_button.isChecked():
            target.apply_view(zoom, fit, center)

    def _on_zoom(self, scale: float) -> None:
        if self.result_viewer.has_image():
            self.zoom_label.setText(f"{scale * 100:.0f}%")
            self.zoom_changed.emit(scale)

    def _on_zoom_from_original(self, scale: float) -> None:
        if not self.link_button.isChecked() and self.original_viewer.has_image():
            self.zoom_changed.emit(scale)

    def _set_tools_enabled(self, enabled: bool) -> None:
        for b in (self.zoom_in_button, self.zoom_out_button, self.fit_button, self.actual_button):
            b.setEnabled(enabled)

    # --- duyarlı yerleşim -------------------------------------------------
    def _update_direction(self) -> None:
        vertical = self._mode == CompareMode.SIDE and self._viewers.width() < self._viewers.height() * 1.05
        direction = QBoxLayout.Direction.TopToBottom if vertical else QBoxLayout.Direction.LeftToRight
        if self._viewer_layout.direction() != direction:
            self._viewer_layout.setDirection(direction)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_direction()
