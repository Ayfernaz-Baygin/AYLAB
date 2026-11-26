"""Segmentation / Edge-based paketinin AYLAB arayüzüyle entegrasyonu (offscreen)."""

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from app.controls import ChoiceControl
from app.main_window import MainWindow
from processing import segmentation as S
from processing.analysis import histograms
from processing.image_io import read_image, save_image

EDGE_KEYS = ["edge_discontinuity", "edge_point", "edge_line", "edge_gradient", "edge_roberts", "edge_sobel",
             "edge_prewitt", "edge_laplacian", "edge_log", "edge_canny", "edge_linking", "edge_boundary", "edge_hough"]


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def scene():
    """Sentetik renkli sahne: dikdörtgen, daire, çizgiler, tek nokta, hafif gürültü."""
    rng = np.random.default_rng(2)
    f = np.full((60, 80), 50.0)
    f[10:30, 8:30] = 190
    cv2.circle(f, (58, 38), 12, 25, -1)
    cv2.line(f, (5, 50), (75, 50), 230, 1)
    cv2.line(f, (40, 2), (40, 28), 230, 1)
    f[6, 66] = 255
    f = f + rng.normal(0, 3, f.shape)
    img = np.dstack([f * 0.9, f, f * 1.05])
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.fixture
def loaded(window, tmp_path):
    img = scene()
    path = tmp_path / "Kenar" / "sahne_ışık.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


CASES = [
    ("edge_discontinuity", {"order": "second", "axis": "vertical", "display": "abs"}, lambda im: S.discontinuity(im, "second", "vertical", "abs")),
    ("edge_point", {"t": 0.8}, lambda im: S.point_detection(im, 0.8)),
    ("edge_line", {"direction": "horizontal", "t": 0.5}, lambda im: S.line_detection(im, "horizontal", 0.5)),
    ("edge_gradient", {"display": "direction"}, lambda im: S.gradient(im, display="direction")),
    ("edge_roberts", {"display": "gx"}, lambda im: S.roberts(im, display="gx")),
    ("edge_sobel", {"display": "edges", "t": 0.35, "smoothing": "gauss5"}, lambda im: S.sobel(im, display="edges", t=0.35, smoothing="gauss5")),
    ("edge_prewitt", {"magnitude": "abs"}, lambda im: S.prewitt(im, magnitude="abs")),
    ("edge_laplacian", {"kernel": "4", "display": "zero_crossings", "t": 0.2}, lambda im: S.laplacian_edges(im, "4", "zero_crossings", 0.2)),
    ("edge_log", {"sigma": 1.5}, lambda im: S.laplacian_of_gaussian(im, 1.5)),
    ("edge_canny", {"sigma": 1.0, "low": 15, "high": 60, "display": "double"}, lambda im: S.canny(im, 1.0, 15, 60, "double")),
    ("edge_linking", {"k": 10, "display": "overlay"}, lambda im: S.edge_linking(im, k=10, display="overlay")),
    ("edge_linking", {"orientation": "angle", "angle": 135, "ta": 20, "k": 12},
     lambda im: S.edge_linking(im, ta=20, k=12, orientation="angle", angle=135)),
    ("edge_boundary", {}, lambda im: S.boundary_extraction(im)),
    ("edge_hough", {"frac": 0.6, "max_lines": 6}, lambda im: S.hough_lines(im, frac=0.6, max_lines=6)),
]


def test_edge_operations_listed_after_thresholding(window):
    window.navigator.tiles["segmentation"].click()
    rows = list(window.navigator.operation_rows)
    assert rows[8:21] == EDGE_KEYS and rows[-1] == "kmeans" and len(rows) == 26   # Paket 3: 4 bölge işlemi arada
    assert all(k.startswith("region_") for k in rows[21:25])
    assert len(window.navigator.tiles) == 7


@pytest.mark.parametrize("key,values,reference", CASES, ids=[c[0] for c in CASES])
def test_apply_from_panel(loaded, key, values, reference):
    window, img = loaded
    window.navigator.select_algorithm(key)
    for k, v in values.items():
        window.parameters.controls[k].set_value(v)
    window.parameters.apply_button.click()
    expected = reference(img)
    np.testing.assert_array_equal(window.current_image, expected.image)
    np.testing.assert_array_equal(window.original_image, img)
    assert window.telemetry.state.state() == "DONE"
    assert expected.report.summary()[:60] in window.statusBar().currentMessage()
    assert window.parameters.report_view.isVisible() == (key == "edge_boundary")   # ComponentReport tablosu
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_controls_defaults_and_display_choices(window):
    for key in EDGE_KEYS:
        window.navigator.select_algorithm(key)
        algo = window.registry.get(key)
        assert list(window.parameters.controls) == [p.key for p in algo.params]
        assert window.parameters.values() == algo.default_params()
        if "display" in window.parameters.controls:
            assert isinstance(window.parameters.controls["display"], ChoiceControl)


def test_invalid_canny_thresholds_are_reported_not_applied(loaded, monkeypatch):
    window, img = loaded
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
    window.navigator.select_algorithm("edge_canny")
    window.parameters.controls["low"].set_value(80)
    window.parameters.controls["high"].set_value(40)
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, img)          # görüntü değişmedi
    assert window.telemetry.state.state() != "DONE"
    assert window.telemetry.state.state() == "ERROR" and "smaller than T₂" in window.statusBar().currentMessage()
    assert shown and shown[0][0] == "Invalid parameters" and "Canny" in shown[0][1]
    assert not window.history.can_undo()


def test_chain_undo_reset_compare_and_save(loaded, qapp, tmp_path, monkeypatch):
    window, img = loaded
    window.navigator.select_algorithm("edge_canny")
    window.parameters.apply_button.click()
    edges = window.current_image.copy()
    window.navigator.select_algorithm("edge_hough")
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, S.hough_lines(edges).image)   # zincir: kenar haritasında Hough
    window.navigator.select_algorithm("edge_boundary")
    window.parameters.apply_button.click()
    assert window.parameters.report_view.isVisible()
    window.undo_action.trigger()
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, edges)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, edges)
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")
    out = tmp_path / "çıktı" / "kenarlar.png"
    out.parent.mkdir()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window.save_image_dialog()
    np.testing.assert_array_equal(read_image(out), edges)               # kaydedilen = mevcut B (0/255)
    assert window.load_image(out)                                       # kaydedilen kenar haritası yeniden açılır
    np.testing.assert_array_equal(window.original_image, edges)


def test_hough_accumulator_has_its_own_size(loaded):
    window, img = loaded
    window.navigator.select_algorithm("edge_hough")
    window.parameters.controls["display"].set_value("accumulator")
    window.parameters.apply_button.click()
    assert window.current_image.shape[1] == 180 and window.telemetry.state.state() == "DONE"
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
