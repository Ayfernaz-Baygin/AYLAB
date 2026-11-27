"""Segmentation / Thresholding paketinin AYLAB arayüzüyle entegrasyonu (offscreen)."""

import numpy as np
import pytest

from app import theme
from app.controls import ChoiceControl, SliderControl
from app.main_window import MainWindow
from processing import morphology as M
from processing import segmentation as S
from processing.analysis import histograms
from processing.image_io import save_image

KEYS = ["threshold", "iterative_threshold", "otsu", "adaptive_mean", "adaptive_gaussian",
        "moving_average", "niblack", "sauvola"]


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def scene():
    """Sentetik renkli 'belge': eğik aydınlatma + koyu yazı benzeri çizgiler."""
    h, w = 48, 64
    base = np.add.outer(np.linspace(90, 215, h), np.linspace(-25, 25, w))
    for r in range(6, h - 6, 10):
        base[r:r + 2, 5:w - 5:4] -= 75
    img = np.dstack([base * 0.95, base, base * 1.05])
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.fixture
def loaded(window, tmp_path):
    img = scene()
    path = tmp_path / "Bölütleme" / "belge_ışık.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


CASES = [
    ("threshold", {"thresh": 150, "foreground": "dark"}, lambda im: S.manual_threshold(im, 150, "dark")),
    ("iterative_threshold", {"init": "custom", "t0": 40, "delta": 0.2},
     lambda im: S.automatic_threshold(im, "custom", 40, 0.2)),
    ("otsu", {}, lambda im: S.otsu_threshold(im)),
    ("adaptive_mean", {"block": 11, "c": 12, "foreground": "dark"}, lambda im: S.adaptive_mean_threshold(im, 11, 12, "dark")),
    ("adaptive_gaussian", {"block": 21, "c": 5}, lambda im: S.adaptive_gaussian_threshold(im, 21, 5)),
    ("moving_average", {"n": 30, "c": 0.8}, lambda im: S.moving_average_threshold(im, 30, 0.8)),
    ("niblack", {"window": 25, "k": 0.35, "foreground": "dark"}, lambda im: S.niblack_threshold(im, 25, 0.35, "dark")),
    ("sauvola", {"window": 9, "k": 0.4, "r": 64}, lambda im: S.sauvola_threshold(im, 9, 0.4, 64)),
]


def test_thresholding_listed_in_segmentation_category(window):
    window.navigator.tiles["segmentation"].click()
    rows = list(window.navigator.operation_rows)
    assert rows[:8] == KEYS and rows[-1] == "region_marker_watershed" and len(rows) == 25   # kmeans yer tutucusu kaldırıldı (Chapter 6 k-means: Color)
    assert len(window.navigator.tiles) == 7                     # yeni kategori yok
    assert window.accent == theme.wavelength_color(window.registry.category("segmentation").wavelength)
    assert "SEGMENTATION" in window.parameters.eyebrow.text()


@pytest.mark.parametrize("key,values,reference", CASES, ids=[c[0] for c in CASES])
def test_apply_from_panel(loaded, key, values, reference):
    window, img = loaded
    window.navigator.select_algorithm(key)
    for k, v in values.items():
        window.parameters.controls[k].set_value(v)
    assert window.telemetry.state.state() == "READY"
    window.parameters.apply_button.click()
    expected = reference(img)
    np.testing.assert_array_equal(window.current_image, expected.image)
    assert set(np.unique(window.current_image)) <= {0, 255} and window.current_image.dtype == np.uint8
    np.testing.assert_array_equal(window.original_image, img)
    assert window.telemetry.state.state() == "DONE"
    assert expected.report.summary() in window.statusBar().currentMessage()
    assert not window.parameters.report_view.isVisible()       # bileşen tablosu yalnızca CC için
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_controls_types_and_defaults(window):
    for key in KEYS:
        window.navigator.select_algorithm(key)
        algo = window.registry.get(key)
        assert list(window.parameters.controls) == [p.key for p in algo.params]
        assert window.parameters.values() == algo.default_params()
        assert isinstance(window.parameters.controls["foreground"], ChoiceControl)
        assert window.parameters.summary.text() == algo.summary
    window.navigator.select_algorithm("adaptive_mean")
    block = window.parameters.controls["block"]
    assert isinstance(block, SliderControl)
    block.set_value(20)
    assert window.parameters.values()["block"] % 2 == 1         # tek sayıya çekilir
    window.parameters.controls["c"].set_value(25)
    window.parameters.reset_defaults()
    assert window.parameters.values() == {"block": 15, "c": 7, "foreground": "bright"}


def test_chain_undo_reset_and_compare(loaded, qapp):
    window, img = loaded
    window.navigator.select_algorithm("sauvola")
    window.parameters.controls["foreground"].set_value("dark")
    window.parameters.apply_button.click()
    mask = window.current_image.copy()
    np.testing.assert_array_equal(mask, S.sauvola_threshold(img, foreground="dark").image)
    # zincir: eşiklenen maske morfolojiye girdi olur
    window.navigator.select_algorithm("connected_components")
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, M.connected_components(mask).image)
    assert window.parameters.report_view.isVisible()
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, mask)
    assert not window.parameters.report_view.isVisible()
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, mask)
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")
    np.testing.assert_array_equal(window.original_image, img)


def test_gray_and_alpha_images(window, tmp_path):
    rng = np.random.default_rng(0)
    for name, img in (("gri.png", rng.integers(0, 256, (30, 40), dtype=np.uint8)),
                      ("alfa.png", rng.integers(0, 256, (30, 40, 4), dtype=np.uint8))):
        path = tmp_path / name
        save_image(path, img)
        assert window.load_image(path)
        loaded = window.original_image.copy()
        for key in KEYS:
            window.reset_action.trigger()
            window.navigator.select_algorithm(key)
            window.parameters.apply_button.click()
            assert window.telemetry.state.state() == "DONE"
            np.testing.assert_array_equal(window.current_image, window.registry.get(key).run(loaded))
            if loaded.ndim == 3 and loaded.shape[2] == 4:
                np.testing.assert_array_equal(window.current_image[..., 3], loaded[..., 3])
