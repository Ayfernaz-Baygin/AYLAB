"""Color Processing / Paket 1 — AYLAB arayüzü entegrasyonu (offscreen)."""

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from app.main_window import MainWindow
from processing import color_processing as C
from processing.analysis import histograms
from processing.image_io import save_image

COLOR_KEYS = ["color_space", "channel", "color_histogram"]


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def scene():
    """Sentetik renk sahnesi: ana ve ara renk blokları, gri rampa, ton rampası."""
    h, w = 48, 96
    img = np.zeros((h, w, 3), np.uint8)
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]   # BGR
    for k, c in enumerate(cols):
        img[:16, k * 16:(k + 1) * 16] = c
    img[16:32] = np.linspace(0, 255, w).astype(np.uint8)[None, :, None]
    hue = np.linspace(0, 179, w).astype(np.uint8)
    hsv = np.dstack([np.repeat(hue[None], 16, 0), np.full((16, w), 220, np.uint8), np.full((16, w), 230, np.uint8)])
    img[32:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


@pytest.fixture
def loaded(window, tmp_path):
    img = scene()
    path = tmp_path / "Renk" / "sahne_ç.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


CASES = [
    ("color_space", {}, lambda im: C.color_space_conversion(im)),
    ("color_space", {"model": "lab", "output": "roundtrip8"}, lambda im: C.color_space_conversion(im, "lab", "roundtrip8")),
    ("color_space", {"model": "cmyk", "output": "roundtrip"}, lambda im: C.color_space_conversion(im, "cmyk", "roundtrip")),
    ("channel", {"model": "hsi", "channel": "1"}, lambda im: C.channel_visualization(im, "hsi", "1")),
    ("channel", {"model": "cmyk", "channel": "4", "style": "tint"}, lambda im: C.channel_visualization(im, "cmyk", "4", "tint")),
    ("channel", {"model": "ycbcr", "channel": "3"}, lambda im: C.channel_visualization(im, "ycbcr", "3")),
    ("color_histogram", {}, lambda im: C.color_histogram(im)),
    ("color_histogram", {"model": "hsv", "bins": "64", "hue_mask": True}, lambda im: C.color_histogram(im, "hsv", "64", True)),
]


def test_color_operations_listed(window):
    window.navigator.tiles["color"].click()
    rows = list(window.navigator.operation_rows)
    assert rows[:3] == COLOR_KEYS and rows[3:] == ["intensity_slicing", "pseudocolor",   # + Paket 2
                                         "color_transform", "color_slicing", "tone_correction", "color_balance", "hsi_equalization",   # + Paket 3
                                         "color_smooth_sharpen"]   # + Paket 4


@pytest.mark.parametrize("key,values,reference", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
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
    view = window.parameters.color_histogram_view
    assert view.isVisible() == (key == "color_histogram")
    assert not window.parameters.report_view.isVisible()
    if key == "color_histogram":
        rep = view.report
        assert [c.name for c in rep.channels] == [c.name for c in expected.report.channels]
        for got, exp in zip(rep.channels, expected.report.channels):
            np.testing.assert_array_equal(got.counts, exp.counts)
        assert view.plot.height() == 62 * len(rep.channels)
        assert str(img.shape[0] * img.shape[1]) in view.info.text()
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_controls_and_defaults(window):
    for key in COLOR_KEYS:
        window.navigator.select_algorithm(key)
        algo = window.registry.get(key)
        assert list(window.parameters.controls) == [p.key for p in algo.params]
        assert window.parameters.values() == algo.default_params()


def test_invalid_channel_is_reported_not_applied(loaded, monkeypatch):
    window, img = loaded
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
    window.navigator.select_algorithm("channel")
    window.parameters.controls["model"].set_value("hsv")
    window.parameters.controls["channel"].set_value("4")
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, img)
    assert window.telemetry.state.state() == "ERROR" and "has 3 channels" in window.statusBar().currentMessage()
    assert shown and "Channel Visualization" in shown[0][1]
    assert not window.history.can_undo()


def test_histogram_view_hides_after_other_operations_and_undo_reset_compare(loaded, qapp):
    window, img = loaded
    window.navigator.select_algorithm("color_space")
    window.parameters.controls["output"].set_value("roundtrip")
    window.parameters.apply_button.click()
    rt = window.current_image.copy()
    np.testing.assert_array_equal(rt, img)                                    # gidiş-dönüş hatasız
    window.navigator.select_algorithm("color_histogram")
    window.parameters.apply_button.click()
    assert window.parameters.color_histogram_view.isVisible()
    window.navigator.select_algorithm("channel")
    window.parameters.apply_button.click()
    assert not window.parameters.color_histogram_view.isVisible()
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, rt)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")
    window.navigator.select_algorithm("region_watershed")                    # bileşen tablosu ile çakışmaz
    window.parameters.apply_button.click()
    assert window.parameters.report_view.isVisible() and not window.parameters.color_histogram_view.isVisible()
