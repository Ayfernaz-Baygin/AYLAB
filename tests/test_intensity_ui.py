"""Yoğunluk dönüşümlerinin AYLAB arayüzüyle entegrasyonu (offscreen)."""

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from app.main_window import MainWindow
from processing import intensity
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
    """Düşük kontrastlı renkli bir görüntü, Türkçe karakterli yoldan yüklenir."""
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(110, 14, (48, 64, 3)), 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2].astype(int) + 25, 0, 255)  # hafif sıcak ton
    path = tmp_path / "Düşük_kontrast" / "örnek_görüntü.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


def _select(window, key, **values):
    window.navigator.select_algorithm(key)
    for name, value in values.items():
        window.parameters.controls[name].set_value(value)


def _apply_via_button(window):
    assert window.parameters.apply_button.isEnabled()
    window.parameters.apply_button.click()


CASES = [
    ("negative", {}, lambda img: intensity.negative(img)),
    ("gamma", {"gamma": 0.5, "c": 1.0}, lambda img: intensity.gamma_correction(img, 0.5, 1.0)),
    ("contrast_stretch", {"mode": "auto", "saturate": 1.0},
     lambda img: intensity.contrast_stretch(img, "auto", saturate=1.0)),
    ("contrast_stretch", {"mode": "manual", "low": 80, "high": 160},
     lambda img: intensity.contrast_stretch(img, "manual", 80, 160)),
    ("hist_eq", {"method": "global", "space": "lab"},
     lambda img: intensity.equalize_histogram(img, "global", "lab")),
    ("hist_eq", {"method": "clahe", "clip": 3.0, "tiles": 4},
     lambda img: intensity.equalize_histogram(img, "clahe", "ycrcb", 3.0, 4)),
]


@pytest.mark.parametrize("key,values,reference", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_apply_from_panel_processes_output_only(loaded, key, values, reference):
    window, img = loaded
    _select(window, key, **values)
    assert window.telemetry.state.state() == "READY"  # bağlı algoritma → PENDING değil
    _apply_via_button(window)

    expected = reference(img)
    np.testing.assert_array_equal(window.current_image, expected)
    np.testing.assert_array_equal(window.original_image, img)  # orijinal korunur
    assert window.result_viewer.image_array() is window.current_image
    assert window.original_viewer.image_array() is window.original_image
    assert window.telemetry.state.state() == "DONE"
    assert window.statusBar().currentMessage().startswith(f"Applied {window.registry.get(key).title}")
    assert len(window.history) == 1 and window.undo_action.isEnabled()


def test_histogram_and_stats_follow_processed_image(loaded):
    window, img = loaded
    scope = window.parameters.analysis.scope
    before = {k: v.copy() for k, v in scope._current.items()}
    _select(window, "hist_eq")
    _apply_via_button(window)
    after = scope._current
    expected = histograms(window.current_image)
    for name in expected:
        np.testing.assert_array_equal(after[name], expected[name])
        assert not np.array_equal(after[name], before[name])
    # Kesikli "hayalet" eğri orijinal görüntünün histogramı olarak kalır
    for name, values in histograms(img).items():
        np.testing.assert_array_equal(scope._reference[name], values)
    # Kanal ölçerleri yeni görüntünün istatistiklerini gösterir
    meters = {m.name: m for m in window.parameters.analysis.meters}
    assert meters["R"].mean == pytest.approx(window.current_image[..., 2].mean())


def test_chained_operations_undo_and_reset(loaded):
    window, img = loaded
    _select(window, "negative")
    _apply_via_button(window)
    after_negative = window.current_image.copy()
    _select(window, "gamma", gamma=2.0)
    _apply_via_button(window)
    np.testing.assert_array_equal(window.current_image, intensity.gamma_correction(after_negative, 2.0))
    assert window.telemetry.history.value() == "2/20"

    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, after_negative)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    assert not window.undo_action.isEnabled()

    _select(window, "contrast_stretch")
    _apply_via_button(window)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.undo_action.trigger()  # Reset de geri alınabilir
    np.testing.assert_array_equal(window.current_image, intensity.contrast_stretch(img))
    np.testing.assert_array_equal(window.original_image, img)


def test_invalid_parameters_show_clear_message_and_keep_image(loaded, monkeypatch):
    window, img = loaded
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
    _select(window, "contrast_stretch", mode="manual", low=200, high=100)
    _apply_via_button(window)

    assert shown and shown[0][0] == "Invalid parameters"
    assert "greater than input low" in shown[0][1]
    assert window.telemetry.state.state() == "ERROR"
    assert "invalid parameters" in window.statusBar().currentMessage()
    np.testing.assert_array_equal(window.current_image, img)
    assert len(window.history) == 0  # başarısız işlem geçmişe yazılmaz


def test_apply_impossible_without_image(window):
    window.navigator.select_algorithm("gamma")
    assert not window.parameters.apply_button.isEnabled()
    window.parameters.request_apply()  # Ctrl+Enter yolu da etkisiz
    assert window.current_image is None
    assert window.apply_algorithm("gamma", {}) is False


def test_gamma_panel_offers_slider_and_numeric_field(window):
    window.navigator.select_algorithm("gamma")
    control = window.parameters.controls["gamma"]
    control.field.setValue(0.4)
    assert control.value() == pytest.approx(0.4)
    assert control.slider.value() == round((0.4 - 0.05) / 0.05)
    control.slider.setValue(control.slider.maximum())
    assert control.value() == pytest.approx(10.0)


def test_grayscale_image_pipeline(window, tmp_path):
    gray = np.tile(np.linspace(60, 140, 50).astype(np.uint8), (30, 1))
    path = tmp_path / "gri_ölçek.png"
    save_image(path, gray)
    window.load_image(path)
    # read_image renkli (BGR) okur; gri bir görüntüde üç kanal eşit kalmalı
    for key in ("negative", "gamma", "contrast_stretch", "hist_eq"):
        window.reset_image()
        _select(window, key)
        _apply_via_button(window)
        out = window.current_image
        assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2]), key
