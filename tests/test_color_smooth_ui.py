"""Color Processing / Paket 4 — renkli yumuşatma / keskinleştirme arayüz entegrasyonu: panel, rapor görünümü,
Display (sonuç / fark × 8), Undo / Reset / A-B, DEFAULTS."""

import numpy as np
import pytest

from app.main_window import MainWindow
from processing import color_processing as C
from processing.analysis import histograms
from processing.image_io import save_image
from tests.test_color_smooth import patches


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
    path = tmp_path / "Renk yamaları" / "yamalar.png"
    save_image(path, np.ascontiguousarray(patches()[..., ::-1]))
    assert window.load_image(path)
    return window, window.original_image.copy()


CASES = [{}, {"kernel": "5"}, {"process": "hsi"}, {"kernel": "5", "process": "hsi"}, {"operation": "sharpen"},
         {"operation": "sharpen", "process": "hsi"}, {"display": "difference", "kernel": "5"},
         {"operation": "sharpen", "display": "difference"}]


@pytest.mark.parametrize("values", CASES, ids=[str(i) for i in range(len(CASES))])
def test_apply_from_panel(loaded, values):
    window, img = loaded
    window.navigator.select_algorithm("color_smooth_sharpen")
    for k, v in values.items():
        window.parameters.controls[k].set_value(v)
    window.parameters.apply_button.click()
    full = {**REGISTRY_DEFAULTS, **values}
    expected = C.color_smoothing_sharpening(img, **full)
    np.testing.assert_array_equal(window.current_image, expected.image)
    assert window.current_image.shape == img.shape and window.current_image.dtype == np.uint8
    assert window.telemetry.state.state() == "DONE" and len(window.history._stack) == 1
    assert expected.report.summary()[:60] in window.statusBar().currentMessage()
    view = window.parameters.transfer_curve_view
    assert view.isVisible() and view.title.text().startswith("SMOOTHING & SHARPENING · ")
    assert not view.plot.isVisible() and view.rows() == list(expected.report.rows)
    assert not window.parameters.color_legend_view.isVisible() and not window.parameters.color_histogram_view.isVisible()
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


REGISTRY_DEFAULTS = {"operation": "smooth", "kernel": "3", "process": "rgb", "display": "result"}


def test_segmented_controls_and_defaults(window):
    window.navigator.select_algorithm("color_smooth_sharpen")
    c = window.parameters.controls
    assert set(c) == {"operation", "kernel", "process", "display"}
    assert all(c[k].combo is None and len(c[k].buttons) == 2 for k in c)             # 2 seçenek → segmentli düğmeler
    assert window.parameters.values() == window.registry.get("color_smooth_sharpen").default_params() == REGISTRY_DEFAULTS
    c["operation"].set_value("sharpen"); c["process"].set_value("hsi")
    window.parameters.reset_defaults()
    assert window.parameters.values() == REGISTRY_DEFAULTS


def test_undo_reset_compare_and_chain(loaded, qapp):
    window, img = loaded
    window.navigator.select_algorithm("color_smooth_sharpen")
    window.parameters.controls["kernel"].set_value("5")
    window.parameters.apply_button.click()
    smoothed = window.current_image.copy()
    window.parameters.controls["operation"].set_value("sharpen")                      # zincir: yumuşat → keskinleştir
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, C.color_smoothing_sharpening(smoothed, "sharpen").image)
    assert len(window.history._stack) == 2
    window.navigator.select_algorithm("color_histogram")                              # analiz: adım eklemez
    window.parameters.apply_button.click()
    assert len(window.history._stack) == 2 and not window.parameters.transfer_curve_view.isVisible()
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, smoothed)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    assert not window.parameters.transfer_curve_view.isVisible()
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")


def _shown(window, key):
    row = window.navigator.operation_rows[key]
    return not row.visibleRegion().isEmpty() and row.visibleRegion().boundingRect().height() == row.height()


def test_navigator_scrolls_to_the_selected_operation(window, qapp):
    window.resize(1400, 520)                                                       # kısa pencere: liste kaymalı
    qapp.processEvents()
    nav = window.navigator
    bar = nav._scroll.verticalScrollBar()
    nav.select_algorithm("color_space")
    qapp.processEvents()
    top = bar.value()
    assert _shown(window, "color_space")
    nav.select_algorithm("color_smooth_sharpen")                                   # 11. işlem: alt kısımda
    qapp.processEvents()
    assert bar.value() > top and _shown(window, "color_smooth_sharpen")
    after = bar.value()
    nav.select_algorithm("hsi_equalization")                                       # zaten görünür → kımıldamaz
    qapp.processEvents()
    assert bar.value() == after and _shown(window, "hsi_equalization")
    bar.setValue(0)                                                                # kullanıcı listeyi en üste kaydırdı
    qapp.processEvents()
    assert not _shown(window, "color_smooth_sharpen")
    nav.select_algorithm("color_smooth_sharpen")                                   # aynı satır yeniden seçilince de görünür
    qapp.processEvents()
    assert bar.value() > 0 and _shown(window, "color_smooth_sharpen")
    nav.select_algorithm("erode")                                                  # başka kategori: tepsi yeniden kurulur
    qapp.processEvents()
    assert _shown(window, "erode") and nav.current_algorithm() == "erode"
