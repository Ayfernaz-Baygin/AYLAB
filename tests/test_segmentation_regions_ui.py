"""Segmentation / Region-based paketinin AYLAB arayüzüyle entegrasyonu (offscreen)."""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from app.controls import ChoiceControl, SliderControl
from app.main_window import MainWindow
from processing import segmentation as S
from processing.analysis import histograms
from processing.image_io import save_image

REGION_KEYS = ["region_growing", "region_split_merge", "region_watershed", "region_marker_watershed"]


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def scene():
    """Sentetik renkli sahne: iki değen disk, küçük bir disk, hafif gürültü (lisans sorunu yok)."""
    rng = np.random.default_rng(3)
    h, w = 72, 110
    f = np.full((h, w), 45.0)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, r in ((32, 36, 20), (67, 36, 17), (94, 58, 8)):
        f[np.hypot(xx - cx, yy - cy) < r] = 190
    f = f + rng.normal(0, 4, f.shape)
    img = np.dstack([f * 0.92, f, f * 1.04])
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


@pytest.fixture
def loaded(window, tmp_path):
    img = scene()
    path = tmp_path / "Bölge" / "sahne_ğüş.png"
    save_image(path, img)
    assert window.load_image(path)
    return window, img


def widget_pos(viewer, x, y):
    rect = viewer._image_rect()
    s = rect.width() / viewer.image_array().shape[1]
    return QPoint(int(rect.left() + (x + 0.5) * s), int(rect.top() + (y + 0.5) * s))


CASES = [
    ("region_growing", {"seed_x": 32, "seed_y": 36, "t": 50}, lambda im: S.region_growing(im, seed_x=32, seed_y=36, t=50)),
    ("region_growing", {"seeds": "brightest", "q": 1.0, "t": 30, "display": "labels"},
     lambda im: S.region_growing(im, seeds="brightest", q=1.0, t=30, display="labels")),
    ("region_split_merge", {}, lambda im: S.split_and_merge(im)),
    ("region_split_merge", {"predicate": "gonzalez", "a": 5, "min_size": "2", "display": "split"},
     lambda im: S.split_and_merge(im, "gonzalez", 5, min_size="2", display="split")),
    ("region_watershed", {}, lambda im: S.watershed(im)),
    ("region_watershed", {"surface": "inverted", "sigma": 1.5, "display": "minima"},
     lambda im: S.watershed(im, "inverted", 1.5, "minima")),
    ("region_marker_watershed", {}, lambda im: S.marker_controlled_watershed(im)),
    ("region_marker_watershed", {"surface": "distance", "alpha": 0.6, "display": "markers"},
     lambda im: S.marker_controlled_watershed(im, alpha=0.6, surface="distance", display="markers")),
]


def test_region_operations_listed_after_edge_operations(window):
    window.navigator.tiles["segmentation"].click()
    rows = list(window.navigator.operation_rows)
    assert rows[21:] == REGION_KEYS and len(rows) == 25   # kmeans yer tutucusu kaldırıldı (Chapter 6 k-means: Color)


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
    view = window.parameters.report_view
    assert view.isVisible() and view.report is not None                       # ComponentReport tablosu
    assert len(view.rows()) == expected.report.count
    assert view.count_label.text() == expected.report.summary()
    for name, counts in histograms(window.current_image).items():
        np.testing.assert_array_equal(window.parameters.analysis.scope._current[name], counts)


def test_controls_and_defaults(window):
    for key in REGION_KEYS:
        window.navigator.select_algorithm(key)
        algo = window.registry.get(key)
        assert list(window.parameters.controls) == [p.key for p in algo.params]
        assert window.parameters.values() == algo.default_params()
        assert isinstance(window.parameters.controls["display"], ChoiceControl)
    window.navigator.select_algorithm("region_growing")
    assert isinstance(window.parameters.controls["seed_x"], SliderControl)
    window.parameters.controls["seed_x"].set_value(20000)
    assert window.parameters.values()["seed_x"] == S.SEED_COORD_MAX                # kontrol aralığa çeker


@pytest.mark.parametrize("viewer_name", ["original_viewer", "result_viewer"])
def test_click_on_the_image_places_the_seed(loaded, qapp, viewer_name):
    window, img = loaded
    window.navigator.select_algorithm("region_growing")
    window.parameters.controls["seeds"].set_value("brightest")
    viewer = getattr(window, viewer_name)
    QTest.mouseClick(viewer, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, widget_pos(viewer, 67, 36))
    qapp.processEvents()
    values = window.parameters.values()
    assert (values["seed_x"], values["seed_y"], values["seeds"]) == (67, 36, "point")
    assert "Seed set to (67, 36)" in window.statusBar().currentMessage()
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, S.region_growing(img, seed_x=67, seed_y=36).image)


def test_drag_and_other_operations_do_not_change_the_seed(loaded, qapp):
    window, _ = loaded
    viewer = window.original_viewer
    window.navigator.select_algorithm("region_growing")
    p0 = widget_pos(viewer, 10, 10)
    QTest.mousePress(viewer, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
    QTest.mouseRelease(viewer, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0 + QPoint(25, 18))
    qapp.processEvents()
    assert (window.parameters.values()["seed_x"], window.parameters.values()["seed_y"]) == (-1, -1)   # sürükleme
    window.navigator.select_algorithm("region_watershed")
    before = window.parameters.values()
    QTest.mouseClick(viewer, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, widget_pos(viewer, 5, 5))
    qapp.processEvents()
    assert window.parameters.values() == before and "Seed" not in window.statusBar().currentMessage()


def test_seed_outside_the_image_is_reported_not_applied(loaded, monkeypatch):
    window, img = loaded
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
    window.navigator.select_algorithm("region_growing")
    window.parameters.controls["seed_x"].set_value(500)
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, img)
    assert window.telemetry.state.state() == "ERROR" and "outside" in window.statusBar().currentMessage()
    assert shown and shown[0][0] == "Invalid parameters" and "Region Growing" in shown[0][1]
    assert not window.history.can_undo()


def test_chain_undo_reset_and_compare(loaded, qapp):
    window, img = loaded
    window.navigator.select_algorithm("region_watershed")
    window.parameters.controls["display"].set_value("surface")
    window.parameters.apply_button.click()
    surface = window.current_image.copy()
    window.navigator.select_algorithm("region_marker_watershed")
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, S.marker_controlled_watershed(surface).image)   # zincir
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, surface)
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, img)
    window.stage.set_mode("split")
    qapp.processEvents()
    assert window.stage.mode() == "split"
    window.stage.set_mode("side")
    window.navigator.select_algorithm("edge_canny")                                  # rapor üretmeyen işlem → tablo gizlenir
    window.parameters.apply_button.click()
    assert not window.parameters.report_view.isVisible()
