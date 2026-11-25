"""Morfoloji kategorisinin AYLAB arayüzüyle entegrasyonu (offscreen)."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from app import theme
from app.controls import ChoiceControl, SliderControl
from app.main_window import MainWindow
from processing import morphology as M
from processing.analysis import histograms
from processing.image_io import save_image

KEYS = ["erode", "dilate", "opening", "closing", "gradient", "hit_or_miss"]


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


@pytest.fixture
def loaded(window, tmp_path):
    img = np.full((40, 56, 3), 30, np.uint8)
    img[5:20, 6:26] = (200, 210, 190)      # dikdörtgen
    img[25:35, 30:50] = (180, 220, 200)
    img[10, 40] = (255, 255, 255)          # leke
    img[29, 38] = (30, 30, 30)             # delik
    path = tmp_path / "Morfoloji" / "ikili_şekiller.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


def _select(window, key, **values):
    window.navigator.select_algorithm(key)
    for name, value in values.items():
        window.parameters.controls[name].set_value(value)


CASES = [
    ("erode", {"shape": "rect", "width": 3, "height": 3}, lambda im: M.erode(im)),
    ("dilate", {"shape": "cross", "width": 5, "height": 5, "iterations": 2},
     lambda im: M.dilate(im, shape="cross", width=5, height=5, iterations=2)),
    ("opening", {"shape": "ellipse", "width": 5, "height": 3},
     lambda im: M.opening(im, shape="ellipse", width=5, height=3)),
    ("closing", {"binarize": "manual", "thresh": 100}, lambda im: M.closing(im, binarize="manual", thresh=100)),
    ("gradient", {"anchor_x": 0, "anchor_y": 2}, lambda im: M.gradient(im, anchor_x=0, anchor_y=2)),
    ("hit_or_miss", {"pattern": "corner", "rotation": "90", "display": "overlay"},
     lambda im: M.hit_or_miss(im, pattern="corner", rotation="90", display="overlay")),
]


def test_morphology_category_lists_six_operations(window):
    window.navigator.tiles["morphology"].click()
    assert list(window.navigator.operation_rows) == KEYS
    titles = [window.navigator.operation_rows[k].algorithm.title for k in KEYS]
    assert titles == ["Erosion", "Dilation", "Opening", "Closing", "Morphological Gradient", "Hit-or-Miss Transform"]
    category = window.registry.category("morphology")
    assert window.accent == theme.wavelength_color(category.wavelength)   # mevcut λ535 renk sistemi


@pytest.mark.parametrize("key,values,reference", CASES, ids=[c[0] for c in CASES])
def test_apply_from_panel(loaded, key, values, reference):
    window, img = loaded
    _select(window, key, **values)
    assert window.telemetry.state.state() == "READY"
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, reference(img))
    np.testing.assert_array_equal(window.original_image, img)
    assert window.telemetry.state.state() == "DONE"
    assert window.telemetry.size.value() == "56×40"
    assert window.statusBar().currentMessage().startswith("Applied ")
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_parameter_controls(window):
    window.navigator.select_algorithm("erode")
    c = window.parameters.controls
    assert list(c) == ["binarize", "thresh", "foreground", "shape", "width", "height",
                       "anchor_x", "anchor_y", "iterations"]
    assert isinstance(c["shape"], ChoiceControl) and set(c["shape"].buttons) == {"rect", "ellipse", "cross"}
    assert isinstance(c["width"], SliderControl)
    c["width"].field.setValue(4)
    assert c["width"].value() == 5           # tek sayıya düzeltilir
    assert c["anchor_x"].value() == -1
    window.navigator.select_algorithm("hit_or_miss")
    c = window.parameters.controls
    assert c["pattern"].combo is not None and c["pattern"].combo.count() == len(M.HMT_PATTERNS)
    assert c["rotation"].combo is not None and c["rotation"].combo.count() == 4
    assert set(c["display"].buttons) == {"matches", "overlay"}


def test_opening_removes_speck_closing_fills_hole_in_app(loaded):
    window, img = loaded
    _select(window, "opening")
    window.parameters.apply_button.click()
    assert window.current_image[10, 40, 0] == 0         # leke silindi
    window.reset_action.trigger()
    _select(window, "closing")
    window.parameters.apply_button.click()
    assert window.current_image[29, 38, 0] == 255       # delik doldu


def test_chained_operations_undo_and_reset(loaded):
    window, img = loaded
    _select(window, "erode")
    window.parameters.apply_button.click()
    eroded = window.current_image.copy()
    _select(window, "dilate")
    window.parameters.apply_button.click()
    # İkinci adım birinci adımın (zaten ikili) çıktısına uygulanır: erosion → dilation = opening
    np.testing.assert_array_equal(window.current_image, M.opening(img))
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, eroded)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, eroded)
    np.testing.assert_array_equal(window.original_image, img)


def test_bgra_file_in_app(window, tmp_path):
    """Uygulama dosyaları (image_io.read_image, IMREAD_COLOR) BGR olarak yükler: alfa yüklemede atılır.

    Saydam piksellerin rengi ikili görüntüde aynen kullanılır; BGRA dizilerde alfanın
    korunması işleme katmanında test edilir (test_morphology.py).
    """
    img = np.zeros((20, 24, 4), np.uint8)
    img[4:16, 4:20, :3] = 220
    img[..., 3] = np.linspace(0, 255, 24, dtype=np.uint8)[None, :]
    save_image(tmp_path / "saydam.png", img)
    window.load_image(tmp_path / "saydam.png")
    assert window.original_image.shape == (20, 24, 3)
    _select(window, "gradient")
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, M.gradient(img)[..., :3])


def test_anchor_outside_se_uses_existing_error_dialog(loaded, monkeypatch):
    window, img = loaded
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
    _select(window, "dilate", width=3, anchor_x=5)
    window.parameters.apply_button.click()
    assert shown and shown[0][0] == "Invalid parameters" and "Anchor X" in shown[0][1]
    assert window.telemetry.state.state() == "ERROR"
    np.testing.assert_array_equal(window.current_image, img)
    assert len(window.history) == 0


def test_apply_disabled_without_image(window):
    window.navigator.select_algorithm("hit_or_miss")
    assert not window.parameters.apply_button.isEnabled()
    assert window.apply_algorithm("hit_or_miss", {}) is False
