"""Uzamsal filtrelerin AYLAB arayüzüyle entegrasyonu (offscreen)."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from app.controls import ChoiceControl, SliderControl
from app.main_window import MainWindow
from processing import filters as F
from processing.analysis import histograms
from processing.image_io import save_image


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
    rng = np.random.default_rng(5)
    img = np.clip(rng.normal(120, 35, (40, 56, 3)), 0, 255).astype(np.uint8)
    img[:, 28:] = np.clip(img[:, 28:].astype(int) + 60, 0, 255)  # belirgin bir kenar
    path = tmp_path / "Uzamsal_süzgeç" / "gürültülü_görüntü.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


def _select(window, key, **values):
    window.navigator.select_algorithm(key)
    for name, value in values.items():
        window.parameters.controls[name].set_value(value)


def _apply(window):
    assert window.parameters.apply_button.isEnabled()
    window.parameters.apply_button.click()


CASES = [
    ("mean_blur", {"ksize": 5}, lambda im: F.mean_filter(im, 5)),
    ("gaussian_blur", {"ksize": 7, "sigma_x": 1.6}, lambda im: F.gaussian_filter(im, 7, 1.6)),
    ("median_blur", {"ksize": 5}, lambda im: F.median_filter(im, 5)),
    ("sobel_x", {"ksize": 3, "display": "abs"}, lambda im: F.sobel_x(im, 3, "abs")),
    ("sobel_y", {"ksize": 5, "display": "signed"}, lambda im: F.sobel_y(im, 5, "signed")),
    ("laplacian", {"neighbours": "8", "display": "abs"}, lambda im: F.laplacian(im, "8", "abs")),
]


def test_spatial_category_lists_all_six(window):
    window.navigator.tiles["spatial"].click()
    rows = list(window.navigator.operation_rows)
    assert rows[:6] == ["mean_blur", "gaussian_blur", "median_blur", "sobel_x", "sobel_y", "laplacian"]
    titles = [window.navigator.operation_rows[k].algorithm.title for k in rows[:6]]
    assert titles == ["Mean Filter", "Gaussian Filter", "Median Filter", "Sobel X", "Sobel Y", "Laplacian Filter"]


@pytest.mark.parametrize("key,values,reference", CASES, ids=[c[0] for c in CASES])
def test_apply_from_panel(loaded, key, values, reference):
    window, img = loaded
    _select(window, key, **values)
    assert window.telemetry.state.state() == "READY"
    _apply(window)
    np.testing.assert_array_equal(window.current_image, reference(img))
    np.testing.assert_array_equal(window.original_image, img)  # A değişmez
    assert window.original_viewer.image_array() is window.original_image
    assert window.result_viewer.image_array() is window.current_image
    assert window.telemetry.state.state() == "DONE"
    assert window.telemetry.size.value() == "56×40"
    assert window.statusBar().currentMessage().startswith("Applied ")
    # Histogram yeni çıktıya göre güncellenir
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_parameter_controls_match_definitions(window):
    window.navigator.select_algorithm("gaussian_blur")
    c = window.parameters.controls
    assert isinstance(c["ksize"], SliderControl) and isinstance(c["sigma_x"], SliderControl)
    c["ksize"].field.setValue(6)  # çift değer → tek sayıya düzeltilir
    assert c["ksize"].value() == 7
    window.navigator.select_algorithm("sobel_x")
    assert isinstance(window.parameters.controls["display"], ChoiceControl)
    assert set(window.parameters.controls["display"].buttons) == {"abs", "signed"}
    window.navigator.select_algorithm("median_blur")
    assert window.parameters.controls["ksize"].slider.minimum() == 0
    assert window.parameters.controls["ksize"].value() == 3


def test_sobel_on_color_keeps_three_equal_channels(loaded):
    window, img = loaded
    _select(window, "sobel_x")
    _apply(window)
    out = window.current_image
    assert out.shape == img.shape
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert window.telemetry.channels.value() == "3 · BGR"


def test_undo_and_reset_after_filters(loaded):
    window, img = loaded
    _select(window, "gaussian_blur")
    _apply(window)
    blurred = window.current_image.copy()
    _select(window, "laplacian")
    _apply(window)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, blurred)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, blurred)


def test_out_of_range_values_are_clamped_before_execution(loaded):
    """Panel/registry aralık dışı değerleri tanımlı sınırlara çeker; filtre hiç geçersiz değer görmez."""
    window, img = loaded
    assert window.apply_algorithm("gaussian_blur", {"ksize": 4, "sigma_x": 0.0})
    np.testing.assert_array_equal(window.current_image, F.gaussian_filter(img, 5, 0.1))
    assert "ksize=5" in window.statusBar().currentMessage()


def test_filter_validation_error_uses_existing_dialog(qapp, tmp_path, monkeypatch):
    """Fonksiyon içindeki doğrulama hatası, mevcut 'Invalid parameters' penceresiyle gösterilir.

    Registry tanımı geniş tutulmuş bir kayıtla (median için ksize = 1'e izin veren)
    fonksiyon düzeyindeki kontrolün gerçekten devreye girdiği doğrulanır.
    """
    from processing.catalog import build_registry
    from processing.registry import Algorithm, AlgorithmRegistry, IntParam

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
    base = build_registry()
    registry = AlgorithmRegistry()
    for cat in base.categories():
        registry.add_category(cat)
    registry.add(Algorithm("median_blur", "Median Filter", "spatial", "test",
                           (IntParam("ksize", "Kernel size", 1, 1, 31, odd_only=True),), F.median_filter))
    window = MainWindow(registry)
    window.show()
    img = np.full((10, 10, 3), 90, np.uint8)
    save_image(tmp_path / "sabit.png", img)
    window.load_image(tmp_path / "sabit.png")

    assert window.apply_algorithm("median_blur", {"ksize": 1}) is False
    assert shown and shown[0][0] == "Invalid parameters" and "≥ 3" in shown[0][1]
    assert window.telemetry.state.state() == "ERROR"
    np.testing.assert_array_equal(window.current_image, img)
    assert len(window.history) == 0
    window.close()


def test_apply_disabled_without_image(window):
    window.navigator.select_algorithm("sobel_y")
    assert not window.parameters.apply_button.isEnabled()
    assert window.apply_algorithm("sobel_y", {}) is False
