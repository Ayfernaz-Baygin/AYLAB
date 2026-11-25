"""AYLAB ana penceresi: bileşenleri birbirine bağlar ve uygulama durumunu tutar.

Durum (orijinal/işlenmiş görüntü, geçmiş) burada; görüntü işleme mantığı
``processing`` paketinde; görsel bileşenler ``app`` altındaki ayrı modüllerde.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app import theme
from app.about_dialog import APP_NAME, AboutDialog
from app.command_bar import CommandBar
from app.compare_stage import CompareStage
from app.history import ImageHistory
from app.navigator import SpectralNavigator
from app.parameter_panel import ParameterPanel
from app.status_panel import Telemetry
from processing.analysis import describe
from processing.catalog import REGISTRY
from processing.image_io import SUPPORTED_EXTENSIONS, ImageIOError, read_image, save_image
from processing.registry import AlgorithmNotImplemented, AlgorithmParameterError, AlgorithmRegistry

_IMAGE_FILTER = (
    "Images (" + " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS) + ");;All files (*)"
)
WINDOW_TITLE = theme.BRAND_TITLE  # "AYLAB — Image Processing Laboratory"


class MainWindow(QMainWindow):
    def __init__(self, registry: AlgorithmRegistry | None = None) -> None:
        super().__init__()
        self.registry = registry or REGISTRY
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(1024, 640)
        self.resize(1440, 880)

        self.original_image: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.current_path: Path | None = None
        self.history = ImageHistory(limit=20)
        self.accent = QColor(theme.NEUTRAL_ACCENT)

        self._create_actions()

        # --- bileşenler ---------------------------------------------------
        self.command_bar = CommandBar(self.actions_by_key)
        self.navigator = SpectralNavigator(self.registry)
        self.stage = CompareStage()
        self.parameters = ParameterPanel()
        self.original_viewer = self.stage.original_viewer
        self.result_viewer = self.stage.result_viewer
        # Geriye dönük uyumluluk: eski kod/testler dosya düğmelerine buradan erişir.
        self.controls = self.command_bar

        stage_host = QWidget()
        sl = QVBoxLayout(stage_host)
        sl.setContentsMargins(10, 8, 10, 10)
        sl.addWidget(self.stage)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.navigator)
        self.splitter.addWidget(stage_host)
        self.splitter.addWidget(self.parameters)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([262, 860, 318])

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.command_bar)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 8, 8)
        body.addWidget(self.splitter)
        root.addLayout(body, 1)
        self.setCentralWidget(central)

        self.telemetry = Telemetry(self.statusBar())
        self.statusBar().setSizeGripEnabled(False)

        # --- bağlantılar --------------------------------------------------
        self.navigator.category_selected.connect(self._on_category_selected)
        self.navigator.algorithm_selected.connect(self._on_algorithm_selected)
        self.parameters.apply_requested.connect(self.apply_algorithm)
        self.stage.zoom_changed.connect(self.telemetry.set_zoom)
        self.stage.cursor_info.connect(self.telemetry.set_cursor)
        self.stage.file_dropped.connect(self.load_image)

        self._apply_accent(self.accent, None)
        self._set_image_loaded(False)
        self._refresh_history()
        self.telemetry.set_image_info(None)
        self.telemetry.set_algorithm(None)
        self.statusBar().showMessage(f"Welcome to {APP_NAME}  —  open an image with Ctrl+O or drag & drop.")

    # ------------------------------------------------------------------
    # Eylemler (komut şeridi + klavye kısayolları)
    # ------------------------------------------------------------------
    def _create_actions(self) -> None:
        def make(key: str, text: str, slot, shortcut=None, tip: str = "") -> QAction:
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.setToolTip(tip or text)
            action.triggered.connect(slot)
            self.addAction(action)  # menü çubuğu olmadan da kısayollar çalışsın
            return action

        self.open_action = make("open", "Open Image…", self.open_image_dialog,
                                QKeySequence.StandardKey.Open, "Open an image  (Ctrl+O)")
        self.save_action = make("save", "Save Image…", self.save_image_dialog,
                                QKeySequence.StandardKey.Save, "Save the output image  (Ctrl+S)")
        self.reset_action = make("reset", "Reset", self.reset_image, QKeySequence("Ctrl+R"),
                                 "Discard processing and return to the source  (Ctrl+R)")
        self.undo_action = make("undo", "Undo", self.undo, QKeySequence.StandardKey.Undo,
                                "Undo the last operation  (Ctrl+Z)")
        self.about_action = make("about", "About", self.show_about, QKeySequence("F1"),
                                 f"About {APP_NAME}  (F1)")
        self.exit_action = make("exit", "Exit", self.close, QKeySequence.StandardKey.Quit)
        make("zoom_in", "Zoom in", lambda: self.stage.zoom_in(), QKeySequence.StandardKey.ZoomIn)
        make("zoom_out", "Zoom out", lambda: self.stage.zoom_out(), QKeySequence.StandardKey.ZoomOut)
        make("fit", "Fit", lambda: self.stage.fit_to_view(), QKeySequence("Ctrl+0"))
        make("actual", "Actual size", lambda: self.stage.actual_size(), QKeySequence("Ctrl+1"))
        make("apply", "Apply", lambda: self.parameters.request_apply(), QKeySequence("Ctrl+Return"))

        self.actions_by_key = {
            "open": self.open_action,
            "save": self.save_action,
            "reset": self.reset_action,
            "undo": self.undo_action,
            "about": self.about_action,
        }

    # ------------------------------------------------------------------
    # Dosya işlemleri
    # ------------------------------------------------------------------
    def open_image_dialog(self) -> None:
        start_dir = str(self.current_path.parent) if self.current_path else str(Path.home())
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", start_dir, _IMAGE_FILTER)
        if file_name:
            self.load_image(file_name)

    def load_image(self, path: str | Path) -> bool:
        """Görüntüyü yükler ve gösterir. Başarılıysa True döner."""
        try:
            image = read_image(path)
        except ImageIOError as exc:
            self.telemetry.set_state("ERROR")
            QMessageBox.critical(self, "Could not open image", str(exc))
            return False

        self.current_path = Path(path)
        self.original_image = image
        self.current_image = image.copy()
        self.history.clear()
        self.stage.set_original(self.original_image)
        self.stage.set_processed(self.current_image)
        self._set_image_loaded(True)
        self._refresh_image_info()
        self._refresh_history()
        self.telemetry.set_state("READY")
        self.command_bar.set_file_text(self.current_path.name)
        self.setWindowTitle(f"{self.current_path.name} — {WINDOW_TITLE}")

        info = describe(image)
        self.statusBar().showMessage(
            f"Loaded {self.current_path.name} — {info.width}×{info.height}, {info.channels} channel(s)"
        )
        return True

    def save_image_dialog(self) -> None:
        if self.current_image is None:
            return
        suggested = (
            str(self.current_path.with_stem(self.current_path.stem + "_islenmis")) if self.current_path else ""
        )
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", suggested, _IMAGE_FILTER)
        if not file_name:
            return
        try:
            save_image(file_name, self.current_image)
        except ImageIOError as exc:
            self.telemetry.set_state("ERROR")
            QMessageBox.critical(self, "Could not save image", str(exc))
            return
        self.statusBar().showMessage(f"Saved: {file_name}")

    # ------------------------------------------------------------------
    # Görüntü durumu
    # ------------------------------------------------------------------
    def set_processed_image(self, image: np.ndarray, record_history: bool = True) -> None:
        """İşlenmiş görüntüyü günceller; istenirse önceki hâli geçmişe ekler."""
        if record_history and self.current_image is not None:
            self.history.push(self.current_image)
        self.current_image = image
        self.stage.set_processed(image)
        self._refresh_image_info()
        self._refresh_history()

    def reset_image(self) -> None:
        if self.original_image is None:
            return
        if self.current_image is not None and not np.array_equal(self.current_image, self.original_image):
            self.history.push(self.current_image)
        self.current_image = self.original_image.copy()
        self.stage.set_processed(self.current_image)
        self._refresh_image_info()
        self._refresh_history()
        self.telemetry.set_state("READY")
        self.statusBar().showMessage("Reset to source image.")

    def undo(self) -> None:
        previous = self.history.undo()
        if previous is None:
            return
        self.current_image = previous
        self.stage.set_processed(previous)
        self._refresh_image_info()
        self._refresh_history()
        self.statusBar().showMessage("Undid last operation.")

    def apply_algorithm(self, key: str, params: dict) -> bool:
        """Seçili algoritmayı işlenmiş görüntüye uygular. Başarılıysa True."""
        if self.current_image is None:
            return False
        algorithm = self.registry.get(key)
        resolved = algorithm.resolve_params(params)
        self.telemetry.set_state("PROCESSING")
        started = time.perf_counter()
        try:
            result = algorithm.run(self.current_image, resolved)
        except AlgorithmNotImplemented:
            self.telemetry.set_state("PENDING")
            self.statusBar().showMessage(f"{algorithm.title}: algorithm module not connected yet.", 6000)
            return False
        except AlgorithmParameterError as exc:
            self.telemetry.set_state("ERROR")
            self.statusBar().showMessage(f"{algorithm.title}: invalid parameters — {exc}", 8000)
            QMessageBox.warning(self, "Invalid parameters", f"{algorithm.title}\n\n{exc}")
            return False
        except Exception as exc:  # Algoritma hatası arayüzü çökertmesin
            self.telemetry.set_state("ERROR")
            QMessageBox.warning(self, f"{algorithm.title} failed", str(exc))
            return False
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.set_processed_image(result)
        self.telemetry.set_state("DONE")
        summary = ", ".join(f"{k}={v}" for k, v in resolved.items())
        self.statusBar().showMessage(
            f"Applied {algorithm.title}" + (f" ({summary})" if summary else "") + f" in {elapsed_ms:.1f} ms"
        )
        return True

    # ------------------------------------------------------------------
    # Navigasyon ve vurgu rengi
    # ------------------------------------------------------------------
    def _on_category_selected(self, key: str) -> None:
        category = self.registry.category(key)
        self.parameters.set_category(category)
        self.parameters.set_algorithm(None)
        self.telemetry.set_algorithm(None)
        self._apply_accent(theme.wavelength_color(category.wavelength), category.wavelength)

    def _on_algorithm_selected(self, key: str) -> None:
        algorithm = self.registry.get(key)
        self.parameters.set_category(self.registry.category(algorithm.category))
        self.parameters.set_algorithm(algorithm)
        self.telemetry.set_algorithm(algorithm.title)
        if self.current_image is not None:
            self.telemetry.set_state("READY" if algorithm.implemented else "PENDING")

    def _apply_accent(self, color: QColor, wavelength: int | None) -> None:
        """Arayüzü seçili kategorinin dalga boyuna 'ayarlar'."""
        self.accent = QColor(color)
        self.setStyleSheet(theme.build_stylesheet(self.accent))
        self.command_bar.set_accent(self.accent, wavelength)
        self.stage.set_accent(self.accent)
        self.parameters.set_accent(self.accent)

    # ------------------------------------------------------------------
    def show_about(self) -> None:
        dialog = AboutDialog(self)
        dialog.exec()

    def _refresh_image_info(self) -> None:
        if self.current_image is None:
            self.telemetry.set_image_info(None)
            self.parameters.update_analysis(None, None)
            return
        self.telemetry.set_image_info(describe(self.current_image))
        self.parameters.update_analysis(self.current_image, self.original_image)

    def _refresh_history(self) -> None:
        self.undo_action.setEnabled(self.history.can_undo())
        self.telemetry.set_history(len(self.history), self.history.limit)

    def _set_image_loaded(self, loaded: bool) -> None:
        self.save_action.setEnabled(loaded)
        self.reset_action.setEnabled(loaded)
        self.parameters.set_image_loaded(loaded)
