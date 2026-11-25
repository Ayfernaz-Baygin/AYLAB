"""Sağ panel: seçili algoritmanın dinamik parametre konsolu + analiz alanı.

Form, ``Algorithm.params`` tanımlarından çalışma anında üretilir. Panel
algoritmanın ne yaptığını bilmez; yalnızca ``apply_requested(key, params)``
sinyalini yayar. Uygulama mantığı ana pencerede / işleme katmanındadır.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app import theme
from app.analysis_panel import AnalysisPanel
from app.component_report import ComponentReportView
from app.controls import ParameterControl, create_control
from processing.registry import Algorithm, Category


class _ConsoleSurface(QWidget):
    """Panel zemini: pah kırılmış grafit yüzey ve vurgu renginde üst kenar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.accent = QColor(theme.NEUTRAL_ACCENT)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillPath(theme.chamfer_path(0, 0, w, h, 18, "tl"), QColor(theme.PANEL))
        p.setPen(QPen(self.accent, 2))
        p.drawLine(18, 1, int(w * 0.42), 1)
        p.drawLine(1, 18, 18, 1)
        p.end()


class _Divider(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {theme.LINE_SOFT};")


class _SpectralBars(QWidget):
    """Yedi tayf çizgisi — sol paneldeki kategorilere görsel gönderme."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(66, 42)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        for i, (_nm, hex_color) in enumerate(reversed(theme.SPECTRAL_STOPS)):
            height = 34 - abs(3 - i) * 4
            p.fillRect(QRectF(4 + i * 9, 4 + (34 - height) / 2, 3, height), QColor(hex_color))
        p.end()


class _EmptyState(QWidget):
    """Algoritma seçilmeden önce gösterilen küçük kılavuz."""

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        layout.addWidget(_SpectralBars(), 0, Qt.AlignmentFlag.AlignTop)
        text = QLabel("Choose a band in the Process Spectrum, then an operation — "
                      "its parameters are tuned here.")
        text.setObjectName("Muted")
        text.setWordWrap(True)
        layout.addWidget(text, 1)


class ParameterPanel(QScrollArea):
    apply_requested = Signal(str, dict)
    parameters_changed = Signal(dict)
    defaults_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(286)
        self.setMaximumWidth(400)

        self._surface = _ConsoleSurface()
        self.setWidget(self._surface)
        root = QVBoxLayout(self._surface)
        root.setContentsMargins(18, 16, 16, 16)
        root.setSpacing(10)

        # --- başlık -----------------------------------------------------
        self.eyebrow = QLabel("PARAMETERS")
        self.eyebrow.setObjectName("Eyebrow")
        self.title = QLabel("No operation selected")
        self.title.setFont(theme.display_font(15, QFont.Weight.DemiBold))
        self.title.setWordWrap(True)
        self.summary = QLabel("")
        self.summary.setObjectName("Muted")
        self.summary.setWordWrap(True)
        root.addWidget(self.eyebrow)
        root.addWidget(self.title)
        root.addWidget(self.summary)

        self.empty_state = _EmptyState()
        root.addWidget(self.empty_state)

        # --- dinamik form -----------------------------------------------
        self._form_host = QWidget()
        self._form = QVBoxLayout(self._form_host)
        self._form.setContentsMargins(0, 6, 0, 0)
        self._form.setSpacing(14)
        root.addWidget(self._form_host)
        self.controls: dict[str, ParameterControl] = {}

        # --- eylemler ---------------------------------------------------
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.apply_button = QPushButton("APPLY  ▸")
        self.apply_button.setObjectName("PrimaryAction")
        self.apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_button.setToolTip("Run the operation on the output image  (Ctrl+Enter)")
        self.defaults_button = QPushButton("DEFAULTS")
        self.defaults_button.setObjectName("GhostAction")
        self.defaults_button.setCursor(Qt.CursorShape.PointingHandCursor)
        actions.addWidget(self.apply_button, 1)
        actions.addWidget(self.defaults_button)
        root.addLayout(actions)

        self.notice = QLabel("")
        self.notice.setObjectName("Faint")
        self.notice.setWordWrap(True)
        root.addWidget(self.notice)

        root.addSpacing(4)
        root.addWidget(_Divider())
        root.addSpacing(2)

        # --- rapor (yalnızca rapor üreten işlemlerde görünür) --------------
        self.report_view = ComponentReportView()
        root.addWidget(self.report_view)

        # --- analiz -----------------------------------------------------
        self.analysis = AnalysisPanel()
        root.addWidget(self.analysis)
        root.addStretch(1)

        self.apply_button.clicked.connect(self.request_apply)
        self.defaults_button.clicked.connect(self.reset_defaults)

        self._algorithm: Algorithm | None = None
        self._image_loaded = False
        self.set_algorithm(None)

    # ------------------------------------------------------------------
    def algorithm(self) -> Algorithm | None:
        return self._algorithm

    def set_accent(self, color: QColor) -> None:
        self._surface.accent = QColor(color)
        self._surface.update()
        for c in self.controls.values():
            if hasattr(c, "set_accent"):      # anahtar ve SE ızgarası vurgu rengini izler
                c.set_accent(color)

    def set_category(self, category: Category | None) -> None:
        if category is None:
            self.eyebrow.setText("PARAMETERS")
        else:
            self.eyebrow.setText(f"{category.index:02d} · {category.title.upper()}  ·  λ{category.wavelength}")

    def set_algorithm(self, algorithm: Algorithm | None) -> None:
        self._algorithm = algorithm
        self._clear_form()
        has = algorithm is not None
        self.empty_state.setVisible(not has)
        self._form_host.setVisible(has)
        self.defaults_button.setEnabled(has and bool(algorithm.params))

        if algorithm is None:
            self.title.setText("No operation selected")
            self.summary.setText("")
            self.summary.hide()
        else:
            self.title.setText(algorithm.title)
            self.summary.setText(algorithm.summary)
            self.summary.show()
            if not algorithm.params:
                note = QLabel("This operation has no parameters.")
                note.setObjectName("Faint")
                self._form.addWidget(note)
            for spec in algorithm.params:
                control = create_control(spec)
                control.value_changed.connect(self._on_value_changed)
                self._form.addWidget(control)
                self.controls[spec.key] = control
            self.set_accent(self._surface.accent)
        self._refresh_state()

    def set_image_loaded(self, loaded: bool) -> None:
        self._image_loaded = loaded
        self._refresh_state()

    def values(self) -> dict[str, Any]:
        return {k: c.value() for k, c in self.controls.items()}

    def reset_defaults(self) -> None:
        for c in self.controls.values():
            c.reset()
        self.defaults_requested.emit()

    def request_apply(self) -> None:
        if self._algorithm is not None and self.apply_button.isEnabled():
            self.apply_requested.emit(self._algorithm.key, self.values())

    def update_analysis(self, current: np.ndarray | None, reference: np.ndarray | None) -> None:
        self.analysis.update_images(current, reference)

    def set_report(self, report) -> None:
        """Son işlemin sayısal raporu (şimdilik ``ComponentReport``) ya da None."""
        self.report_view.set_report(report)

    # ------------------------------------------------------------------
    def _clear_form(self) -> None:
        self.controls = {}
        while self._form.count():
            item = self._form.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_value_changed(self, _key: str, _value: object) -> None:
        if self._algorithm is not None and len(self.controls) == len(self._algorithm.params):
            self.parameters_changed.emit(self.values())

    def _refresh_state(self) -> None:
        algo = self._algorithm
        enabled = algo is not None and self._image_loaded
        self.apply_button.setEnabled(enabled)
        if algo is None:
            self.notice.setText("")
        elif not algo.implemented:
            self.notice.setText("◇ MODULE PENDING — parameters are live, the algorithm "
                                "will be connected in the next stage.")
        elif not self._image_loaded:
            self.notice.setText("Load an image to apply this operation.")
        else:
            self.notice.setText("")
