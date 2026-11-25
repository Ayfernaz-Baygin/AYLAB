"""Top-hat / black-hat / birleşik düzeltmenin AYLAB arayüzüyle entegrasyonu (offscreen)."""

import numpy as np
import pytest

from app import theme
from app.controls import ChoiceControl, SliderControl
from app.main_window import MainWindow
from processing import morphology as M
from processing.analysis import histograms
from processing.image_io import save_image

HAT_KEYS = ["top_hat", "black_hat", "hat_correction"]


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def scene():
    """Eğik aydınlatma + küçük parlak ve koyu ayrıntılar (renkli)."""
    h, w = 48, 64
    base = np.add.outer(np.linspace(60, 150, h), np.linspace(0, 40, w))
    img = np.dstack([base * 0.9, base, base * 1.1])
    img[10:13, 10:13] += 70
    img[30:32, 40:43] -= 45
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.fixture
def loaded(window, tmp_path):
    img = scene()
    path = tmp_path / "Morfoloji" / "eğik_ışık.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


CASES = [
    ("top_hat", {}, lambda im: M.top_hat(im)),
    ("black_hat", {"shape": "ellipse", "width": 9, "height": 7}, lambda im: M.black_hat(im, shape="ellipse", width=9, height=7)),
    ("hat_correction", {"width": 11, "height": 11, "anchor_x": 0, "anchor_y": 10},
     lambda im: M.hat_correction(im, width=11, height=11, anchor_x=0, anchor_y=10)),
]


def test_hat_operations_listed_in_existing_category(window):
    window.navigator.tiles["morphology"].click()
    rows = list(window.navigator.operation_rows)
    assert rows[6:9] == HAT_KEYS and len(rows) == 12   # Aşama 3–5: boundary, hole_filling, connected_components
    assert [window.navigator.operation_rows[k].algorithm.title for k in HAT_KEYS] == [
        "White Top-Hat", "Black-Hat", "Illumination / Contrast Correction"]
    assert len(window.navigator.tiles) == 7                     # yeni kategori yok
    assert window.accent == theme.wavelength_color(window.registry.category("morphology").wavelength)


@pytest.mark.parametrize("key,values,reference", CASES, ids=[c[0] for c in CASES])
def test_apply_from_panel(loaded, key, values, reference):
    window, img = loaded
    window.navigator.select_algorithm(key)
    for k, v in values.items():
        window.parameters.controls[k].set_value(v)
    assert window.telemetry.state.state() == "READY"
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, reference(img))
    np.testing.assert_array_equal(window.original_image, img)
    assert window.telemetry.state.state() == "DONE"
    assert window.telemetry.size.value() == "64×48"
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_controls_and_defaults(window):
    for key in HAT_KEYS:
        window.navigator.select_algorithm(key)
        c = window.parameters.controls
        assert list(c) == ["shape", "width", "height", "anchor_x", "anchor_y"]
        assert isinstance(c["shape"], ChoiceControl) and isinstance(c["width"], SliderControl)
        assert window.parameters.values() == dict(shape="rect", width=15, height=15, anchor_x=-1, anchor_y=-1)
    c["width"].field.setValue(20)
    assert c["width"].value() == 21


def test_hats_show_the_details_in_app(loaded):
    window, img = loaded
    window.navigator.select_algorithm("top_hat")
    window.parameters.apply_button.click()
    th = window.current_image[..., 0]
    assert th[10:13, 10:13].min() > 50 and th[30:32, 40:43].max() == 0
    window.reset_action.trigger()
    window.navigator.select_algorithm("black_hat")
    window.parameters.apply_button.click()
    bh = window.current_image[..., 0]
    assert bh[30:32, 40:43].min() > 30 and bh[10:13, 10:13].max() == 0


def test_undo_and_reset_with_hat_operations(loaded):
    window, img = loaded
    window.navigator.select_algorithm("hat_correction")
    window.parameters.apply_button.click()
    corrected = window.current_image.copy()
    window.navigator.select_algorithm("top_hat")
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, M.top_hat(corrected))   # zincir: önceki çıktıya uygulanır
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, corrected)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, corrected)
    np.testing.assert_array_equal(window.original_image, img)
