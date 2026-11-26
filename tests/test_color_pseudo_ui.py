"""Color Processing / Paket 2 — pseudo-color arayüz entegrasyonu, dinamik kanal seçici, histogram geçmişi."""

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


def gray_scene():
    """Sentetik yükseklik haritası: iki tepe + yatay rampa (lisans sorunu yok)."""
    yy, xx = np.mgrid[0:60, 0:90]
    f = 40 + 120 * np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 300) + 90 * np.exp(-((xx - 66) ** 2 + (yy - 24) ** 2) / 150)
    f[50:] = np.linspace(0, 255, 90)[None]
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


@pytest.fixture
def loaded(window, tmp_path):
    img = gray_scene()
    path = tmp_path / "Sözde renk" / "yükseklik.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, window.original_image.copy()


CASES = [
    ("intensity_slicing", {}, lambda im: C.intensity_slicing(im)),
    ("intensity_slicing", {"colors": "8"}, lambda im: C.intensity_slicing(im, "8")),
    ("intensity_slicing", {"colors": "2", "plane": 90, "palette": "hot"}, lambda im: C.intensity_slicing(im, "2", 90, "hot")),
    ("pseudocolor", {}, lambda im: C.color_coding(im)),
    ("pseudocolor", {"cmap": "parula"}, lambda im: C.color_coding(im, "parula")),
]


@pytest.mark.parametrize("key,values,reference", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_apply_from_panel_with_legend(loaded, key, values, reference):
    window, img = loaded
    window.navigator.select_algorithm(key)
    for k, v in values.items():
        window.parameters.controls[k].set_value(v)
    window.parameters.apply_button.click()
    expected = reference(img)
    np.testing.assert_array_equal(window.current_image, expected.image)
    assert window.current_image.ndim == 3 and window.current_image.shape[2] == 3
    assert window.telemetry.state.state() == "DONE"
    assert expected.report.summary()[:60] in window.statusBar().currentMessage()
    legend = window.parameters.color_legend_view
    assert legend.isVisible() and not window.parameters.color_histogram_view.isVisible()
    np.testing.assert_array_equal(legend.report.lut, expected.report.lut)
    assert len(legend.rows()) == len(expected.report.slices)
    if expected.report.slices:
        lo, hi, rgb = expected.report.slices[0]
        assert legend.rows()[0] == f"c1  [{lo}, {hi})  RGB {rgb[0]}, {rgb[1]}, {rgb[2]}"
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_changing_the_color_map_updates_the_result(loaded):
    window, img = loaded
    window.navigator.select_algorithm("pseudocolor")
    outs = {}
    for name in C.COLORMAPS:
        window.reset_action.trigger()
        window.parameters.controls["cmap"].set_value(name)
        window.parameters.apply_button.click()
        outs[name] = window.current_image.copy()
        np.testing.assert_array_equal(outs[name], C.color_coding(img, name).image)
    assert len({o.tobytes() for o in outs.values()}) == len(C.COLORMAPS)


def test_channel_selector_follows_the_color_model(window):
    window.navigator.select_algorithm("channel")
    ch, model = window.parameters.controls["channel"], window.parameters.controls["model"]
    labels = lambda: [ch.combo.itemText(i) for i in range(ch.combo.count())]
    assert model.value() == "hsv" and labels() == ["1 · H", "2 · S", "3 · V"]
    model.set_value("cmyk")
    assert labels() == ["1 · C", "2 · M", "3 · Y", "4 · K"]
    ch.set_value("4")
    model.set_value("lab")                                                       # 4. kanal yok → ilk kanala döner
    assert labels() == ["1 · L*", "2 · a*", "3 · b*"] and ch.value() == "1"
    ch.set_value("3")
    model.set_value("ycbcr")                                                     # geçerli seçim korunur
    assert labels() == ["1 · Y", "2 · Cb", "3 · Cr"] and ch.value() == "3"
    window.parameters.reset_defaults()
    assert labels() == ["1 · H", "2 · S", "3 · V"] and window.parameters.values() == window.registry.get("channel").default_params()


def test_color_histogram_adds_no_undo_step_and_keeps_b(loaded):
    window, img = loaded
    window.navigator.select_algorithm("pseudocolor")
    window.parameters.apply_button.click()
    coded = window.current_image.copy()
    assert len(window.history._stack) == 1
    for model in ("rgb", "hsv", "lab"):
        window.navigator.select_algorithm("color_histogram")
        window.parameters.controls["model"].set_value(model)
        window.parameters.apply_button.click()
        assert window.parameters.color_histogram_view.isVisible()
        assert window.telemetry.state.state() == "DONE"
        np.testing.assert_array_equal(window.current_image, coded)              # B aynı
        assert len(window.history._stack) == 1                                   # geçmişe adım eklenmedi
    window.undo_action.trigger()                                                 # tek undo → doğrudan kaynağa
    np.testing.assert_array_equal(window.current_image, img)
    assert not window.history.can_undo()


def test_undo_reset_compare_and_chain(loaded, qapp):
    window, img = loaded
    window.navigator.select_algorithm("intensity_slicing")
    window.parameters.controls["colors"].set_value("4")
    window.parameters.apply_button.click()
    sliced = window.current_image.copy()
    window.navigator.select_algorithm("pseudocolor")                             # zincir: dilimlenmiş BGR → Y → LUT
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, C.color_coding(sliced).image)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, sliced)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    assert not window.parameters.color_legend_view.isVisible()                   # rapor reset ile kalkar
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")
