"""Color Processing / Paket 5 — renkli bölütleme arayüz entegrasyonu: panel, rapor görünümü, Display seçenekleri,
navigatör, Undo / Reset / A-B, DEFAULTS."""

import numpy as np
import pytest

from app.main_window import MainWindow
from processing import color_processing as C
from processing.analysis import histograms
from processing.image_io import save_image
from tests.test_color_segmentation import SEG_KEYS
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
    path = tmp_path / "Renkli bölütleme" / "yamalar.png"
    save_image(path, np.ascontiguousarray(patches()[..., ::-1]))
    assert window.load_image(path)
    return window, window.original_image.copy()


FUNCS = dict(zip(SEG_KEYS, (C.hue_thresholding, C.kmeans_lab, C.skin_regions, C.color_edge_detection)))
CASES = [("color_hue_threshold", {}), ("color_hue_threshold", {"target": "red", "display": "mask"}),
         ("color_hue_threshold", {"target": "custom", "h_min": 170, "h_max": 10, "s_min": 30}),
         ("color_kmeans", {}), ("color_kmeans", {"k": 3, "features": "ab", "seed": 9}),
         ("color_skin", {}), ("color_skin", {"cr_min": 140, "display": "mask"}),
         ("color_edges", {}), ("color_edges", {"method": "dizenzo", "t": 60}),
         ("color_edges", {"method": "rgb", "display": "magnitude"})]


@pytest.mark.parametrize("key,values", CASES, ids=[f"{k}-{i}" for i, (k, _) in enumerate(CASES)])
def test_apply_from_panel(loaded, key, values):
    window, img = loaded
    window.navigator.select_algorithm(key)
    for k, v in values.items():
        window.parameters.controls[k].set_value(v)
    window.parameters.apply_button.click()
    expected = FUNCS[key](img, **window.parameters.values())
    np.testing.assert_array_equal(window.current_image, expected.image)
    assert window.current_image.shape == img.shape and window.current_image.dtype == np.uint8
    assert window.telemetry.state.state() == "DONE" and len(window.history._stack) == 1
    assert expected.report.summary()[:60] in window.statusBar().currentMessage()
    view = window.parameters.transfer_curve_view
    assert view.isVisible() and view.title.text().startswith("COLOR SEGMENTATION · ")
    assert view.rows() == list(expected.report.rows) and not view.plot.isVisible()
    assert not window.parameters.color_legend_view.isVisible() and not window.parameters.color_histogram_view.isVisible()
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_controls_defaults_and_navigator(window, qapp):
    nav = window.navigator
    for key in SEG_KEYS:
        nav.select_algorithm(key)
        qapp.processEvents()
        row = nav.operation_rows[key]
        assert row.visibleRegion().boundingRect().height() == row.height()        # 12.–15. satır görünür kaydırılır
        assert window.parameters.values() == window.registry.get(key).default_params()
    c = window.parameters.controls
    assert c["method"].combo is not None and c["method"].combo.count() == 4        # 4 seçenek → açılır liste
    c["method"].set_value("dizenzo"); c["t"].set_value(300)
    window.parameters.reset_defaults()
    assert window.parameters.values() == {"method": "gray", "t": 120, "display": "edges"}
    nav.select_algorithm("color_kmeans")
    assert window.parameters.controls["features"].combo is None and len(window.parameters.controls["features"].buttons) == 3
    nav.select_algorithm("color_hue_threshold")
    assert window.parameters.controls["target"].combo.count() == 5


def test_undo_reset_compare_and_chain(loaded, qapp):
    window, img = loaded
    window.navigator.select_algorithm("color_kmeans")
    window.parameters.controls["k"].set_value(5)
    window.parameters.apply_button.click()
    clustered = window.current_image.copy()
    window.navigator.select_algorithm("color_edges")                                 # zincir: kümeler → kenarlar
    window.parameters.controls["method"].set_value("dizenzo")
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, C.color_edge_detection(clustered, "dizenzo").image)
    assert len(window.history._stack) == 2
    window.navigator.select_algorithm("color_histogram")                             # analiz: geçmişe adım eklemez
    window.parameters.apply_button.click()
    assert len(window.history._stack) == 2
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, clustered)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    assert not window.parameters.transfer_curve_view.isVisible()
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")
