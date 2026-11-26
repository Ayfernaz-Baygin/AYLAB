"""Color Processing / Paket 3 — full-color arayüz entegrasyonu: panel, TRANSFER CURVES, tıklayarak renk seçimi,
Undo / Reset / A-B, analiz işlemlerinin geçmişe adım eklememesi."""

import numpy as np
import pytest

from app.main_window import MainWindow
from processing import color_processing as C
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


def color_scene():
    """Sentetik renkli sahne (sıcak kaymalı): kırmızı daireler, yeşil alan, gri şerit, mavi gökyüzü."""
    yy, xx = np.mgrid[0:60, 0:90]
    rgb = np.zeros((60, 90, 3), float)
    rgb[:] = (70, 110, 200)
    rgb[30:] = (60, 150, 60)
    rgb[np.hypot(xx - 25, yy - 40) < 10] = (200, 30, 45)
    rgb[np.hypot(xx - 60, yy - 42) < 8] = (170, 20, 35)
    rgb[:6] = np.linspace(0, 255, 90)[None, :, None]
    rgb = rgb * np.array([1.1, 0.95, 0.7])                                       # laboratuvar 'warm' kayması
    return np.ascontiguousarray(C.c8(rgb)[..., ::-1])                            # BGR


@pytest.fixture
def loaded(window, tmp_path):
    path = tmp_path / "Tam renk" / "sahne.png"
    save_image(path, color_scene())
    assert window.load_image(path)
    return window, window.original_image.copy()


CASES = [
    ("color_transform", {}, lambda im: C.color_transformation(im)),
    ("color_transform", {"model": "hsi", "k": 0.4}, lambda im: C.color_transformation(im, "hsi", 0.4)),
    ("color_slicing", {"shape": "sphere", "radius": 60}, lambda im: C.color_slicing(im, "sphere", 110, 60)),
    ("tone_correction", {"curve": "gamma", "gamma": 0.45}, lambda im: C.tone_correction(im, "gamma", 9.0, 0.45)),
    ("color_balance", {"method": "gray"}, lambda im: C.color_balance(im, "gray")),
    ("color_balance", {"method": "curves", "gamma_b": 0.6}, lambda im: C.color_balance(im, "curves", gamma_b=0.6)),
    ("hsi_equalization", {"saturation": 1.3}, lambda im: C.hsi_equalization(im, "hsi", 1.3)),
    ("hsi_equalization", {"method": "rgb"}, lambda im: C.hsi_equalization(im, "rgb")),
]


@pytest.mark.parametrize("key,values,reference", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_apply_from_panel_with_transfer_curves(loaded, key, values, reference):
    window, img = loaded
    window.navigator.select_algorithm(key)
    for k, v in values.items():
        window.parameters.controls[k].set_value(v)
    window.parameters.apply_button.click()
    expected = reference(img)
    np.testing.assert_array_equal(window.current_image, expected.image)
    assert window.current_image.shape == img.shape and window.current_image.dtype == np.uint8
    assert window.telemetry.state.state() == "DONE"
    assert expected.report.summary()[:60] in window.statusBar().currentMessage()
    view = window.parameters.transfer_curve_view
    assert view.isVisible() and view.report.summary() == expected.report.summary()
    for shown, exp in zip(view.plot.curves, expected.report.curves):
        assert shown.name == exp.name and np.array_equal(shown.values, exp.values)
    assert len(view.plot.curves) == len(expected.report.curves) and view.plot.isVisible() == bool(expected.report.curves)
    assert view.rows() == list(expected.report.rows)
    assert not window.parameters.color_legend_view.isVisible() and not window.parameters.color_histogram_view.isVisible()
    assert len(window.history._stack) == 1
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_click_picks_the_prototype_color_from_b(loaded):
    window, img = loaded
    window.navigator.select_algorithm("color_slicing")
    window.stage.pixel_clicked.emit(25, 40)                                       # kırmızı dairenin merkezi
    b, g, r = (int(v) for v in img[40, 25])
    assert [window.parameters.controls[f"proto_{c}"].value() for c in "rgb"] == [r, g, b]   # BGR → RGB
    assert f"RGB {r}, {g}, {b}" in window.statusBar().currentMessage()
    window.parameters.controls["width"].set_value(60)
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, C.color_slicing(img, "cube", 60, 80, r, g, b).image)
    out = window.current_image
    assert (out[40, 25] == img[40, 25]).all() and (out[15, 45] == 128).all()      # daire kalır, gökyüzü gri
    assert window.parameters.transfer_curve_view.report.swatch == (r, g, b)


def test_click_picks_the_neutral_pixel_and_switches_method(loaded):
    window, img = loaded
    window.navigator.select_algorithm("color_balance")
    assert window.parameters.controls["method"].value() == "white"
    window.stage.pixel_clicked.emit(45, 3)                                        # gri şerit (sıcak kaymalı)
    b, g, r = (int(v) for v in img[3, 45])
    assert window.parameters.controls["method"].value() == "neutral"
    assert [window.parameters.controls[f"neutral_{c}"].value() for c in "rgb"] == [r, g, b]
    window.parameters.apply_button.click()
    fixed = window.current_image[3, 45].astype(int)
    assert np.ptp(fixed) <= 1                                                     # tıklanan piksel nötr olur
    window.stage.pixel_clicked.emit(500, 500)                                     # görüntü dışı: değişmez
    assert [window.parameters.controls[f"neutral_{c}"].value() for c in "rgb"] == [r, g, b]


def test_seed_click_still_drives_region_growing(loaded):
    window, _ = loaded
    window.navigator.select_algorithm("region_growing")
    window.stage.pixel_clicked.emit(12, 34)
    assert (window.parameters.controls["seed_x"].value(), window.parameters.controls["seed_y"].value()) == (12, 34)


def test_gray_input_click_and_operations(window, tmp_path, qapp):
    gray = np.tile(np.arange(20, 200, 2, dtype=np.uint8), (30, 1))
    path = tmp_path / "gri.png"
    save_image(path, gray)
    assert window.load_image(path)
    a = window.original_image.copy()
    window.navigator.select_algorithm("color_slicing")
    window.stage.pixel_clicked.emit(10, 5)
    v = int(a[5, 10]) if a.ndim == 2 else int(a[5, 10, 0])
    assert [window.parameters.controls[f"proto_{c}"].value() for c in "rgb"] == [v, v, v]
    for key in ("color_transform", "tone_correction", "hsi_equalization"):
        window.navigator.select_algorithm(key)
        window.parameters.apply_button.click()
        out = window.current_image
        assert out.ndim == 3 and out.shape[2] == 3 and (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()


def test_undo_reset_defaults_and_analysis_history(loaded, qapp):
    window, img = loaded
    window.navigator.select_algorithm("color_balance")
    window.parameters.apply_button.click()
    balanced = window.current_image.copy()
    window.navigator.select_algorithm("hsi_equalization")                         # zincir: dengelenmiş → HSI eşitleme
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, C.hsi_equalization(balanced).image)
    depth = len(window.history._stack)
    window.navigator.select_algorithm("color_histogram")                          # analiz: geçmişe adım eklemez
    window.parameters.controls["model"].set_value("hsi")
    window.parameters.apply_button.click()
    assert len(window.history._stack) == depth and window.parameters.color_histogram_view.isVisible()
    assert not window.parameters.transfer_curve_view.isVisible()
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, balanced)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    assert not window.parameters.transfer_curve_view.isVisible()
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")
    for key in ("color_transform", "color_slicing", "tone_correction", "color_balance", "hsi_equalization"):
        window.navigator.select_algorithm(key)
        window.parameters.reset_defaults()
        assert window.parameters.values() == window.registry.get(key).default_params()
