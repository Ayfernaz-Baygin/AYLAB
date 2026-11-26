"""AYLAB arayüz etkileşim testleri (offscreen)."""

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from app import theme
from app.controls import ChoiceControl, SliderControl, ToggleControl, create_control
from app.main_window import MainWindow
from processing.catalog import build_registry
from processing.image_io import read_image, save_image
from processing.registry import ChoiceParam, FloatParam, IntParam


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


@pytest.fixture
def image_path(tmp_path):
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:, :40] = (30, 90, 200)
    img[:, 40:] = (200, 60, 20)
    path = tmp_path / "Görseller" / "deneme_ışık.png"
    save_image(path, img)
    return path


def test_initial_state(window):
    assert window.windowTitle() == "AYLAB — Image Processing Laboratory"
    assert "SPECTRA" not in window.windowTitle()
    assert window.statusBar().currentMessage().startswith("Welcome to AYLAB")
    assert not window.save_action.isEnabled()
    assert not window.undo_action.isEnabled()
    assert not window.controls.save_button.isEnabled()
    assert not window.original_viewer.has_image()
    assert window.telemetry.state.state() == "NO IMAGE"
    assert len(window.navigator.tiles) == 7


def test_category_navigation_retunes_accent_and_lists_operations(window):
    window.navigator.tiles["morphology"].click()
    assert window.navigator.current_category() == "morphology"
    category = window.registry.category("morphology")
    assert window.accent == theme.wavelength_color(category.wavelength)
    expected = [a.key for a in window.registry.in_category("morphology")]
    assert list(window.navigator.operation_rows) == expected
    assert "MORPHOLOGICAL" in window.parameters.eyebrow.text()


def test_selecting_algorithm_builds_dynamic_form(window):
    window.navigator.select_category("spatial")
    window.navigator.operation_rows["gaussian_blur"].click()
    algo = window.registry.get("gaussian_blur")
    assert window.parameters.algorithm() is algo
    assert list(window.parameters.controls) == [p.key for p in algo.params]
    assert window.parameters.values() == algo.default_params()
    assert window.telemetry.algorithm.value() == algo.title

    # Başka algoritmaya geçince form yeniden kurulur
    window.navigator.select_algorithm("adaptive_mean")
    assert list(window.parameters.controls) == ["block", "c", "foreground"]
    assert window.navigator.current_category() == "segmentation"


def test_apply_pending_algorithm_keeps_image_and_reports(window, image_path):
    window.load_image(image_path)
    before = window.current_image.copy()
    window.navigator.select_algorithm("log")  # henüz bağlanmamış bir algoritma
    assert window.parameters.apply_button.isEnabled()
    window.parameters.apply_button.click()
    np.testing.assert_array_equal(window.current_image, before)
    assert window.telemetry.state.state() == "PENDING"
    assert not window.history.can_undo()


def test_apply_bound_algorithm_then_undo(qapp, image_path):
    # Paylaşılan REGISTRY'yi değiştirmemek için teste özel bir kayıt kullanılır.
    registry = build_registry()
    registry.bind("negative", lambda image: 255 - image)
    window = MainWindow(registry)
    window.show()
    window.load_image(image_path)
    original = window.current_image.copy()
    window.navigator.select_algorithm("negative")
    window.parameters.request_apply()
    np.testing.assert_array_equal(window.current_image, 255 - original)
    assert window.undo_action.isEnabled()
    assert window.telemetry.state.state() == "DONE"
    window.undo_action.trigger()
    np.testing.assert_array_equal(window.current_image, original)
    assert not window.undo_action.isEnabled()
    window.close()


def test_reset_is_undoable(window, image_path):
    window.load_image(image_path)
    window.set_processed_image(np.zeros_like(window.current_image))
    window.reset_action.trigger()
    np.testing.assert_array_equal(window.current_image, window.original_image)
    window.undo()
    assert window.current_image.max() == 0


def test_load_updates_telemetry_and_analysis(window, image_path):
    assert window.load_image(image_path)
    assert window.telemetry.size.value() == "80×60"
    assert window.telemetry.channels.value() == "3 · BGR"
    assert window.telemetry.state.state() == "READY"
    assert window.parameters.analysis.scope.has_data()
    assert [m.name for m in window.parameters.analysis.meters] == ["R", "G", "B"]
    assert window.command_bar.file_label.text() == "deneme_ışık.png"
    assert window.save_action.isEnabled()


def test_open_and_save_through_dialogs(window, image_path, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(image_path), ""))
    window.open_action.trigger()
    assert window.original_viewer.has_image()

    target = tmp_path / "çıktı klasörü" / "sonuç_görüntü.jpg"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.save_action.trigger()
    assert target.is_file()
    assert read_image(target).shape == window.current_image.shape
    assert "Saved" in window.statusBar().currentMessage()


def test_zoom_fit_and_linked_views(window, image_path):
    window.load_image(image_path)
    viewer_a, viewer_b = window.original_viewer, window.result_viewer
    assert viewer_b.is_fit()
    window.stage.actual_size()
    assert viewer_b.effective_scale() == pytest.approx(1.0)
    window.stage.zoom_in()
    assert viewer_b.effective_scale() == pytest.approx(1.25)
    assert viewer_a.effective_scale() == pytest.approx(1.25)  # LINK açık
    assert window.telemetry.zoom.value() == "125%"
    window.stage.fit_to_view()
    assert viewer_b.is_fit() and viewer_a.is_fit()

    window.stage.link_button.setChecked(False)
    window.stage.zoom_in()
    assert not viewer_b.is_fit() and viewer_a.is_fit()


def test_wheel_zoom_and_cursor_readout(window, image_path, qapp):
    window.load_image(image_path)
    viewer = window.result_viewer
    readouts = []
    viewer.cursor_info.connect(readouts.append)
    rect = viewer._image_rect()
    QTest.mouseMove(viewer, rect.center().toPoint())
    qapp.processEvents()
    assert any(r.startswith("x") and "R" in r for r in readouts)
    scale = viewer.effective_scale()
    viewer.set_zoom(scale * 2, QPointF(rect.center()))
    assert viewer.effective_scale() == pytest.approx(scale * 2)


def test_split_mode(window, image_path):
    window.load_image(image_path)
    window.stage.split_button.click()
    assert window.stage.mode() == "split"
    assert window.original_viewer.isHidden()
    assert window.result_viewer.split_enabled()
    window.result_viewer.set_split_position(1.7)
    assert window.result_viewer.split_position() == 1.0
    window.stage.side_button.click()
    assert window.stage.mode() == "side"
    assert not window.original_viewer.isHidden()


def test_window_resizes_without_errors(window, image_path, qapp):
    window.load_image(image_path)
    for size in ((1024, 640), (1920, 1080), (1100, 900)):
        window.resize(*size)
        qapp.processEvents()
        window.grab()  # tüm bileşenleri çizdir
    assert window.result_viewer.is_fit()


def test_invalid_file_shows_error_state(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(a))
    bad = tmp_path / "bozuk_dosya.png"
    bad.write_bytes(b"not an image")
    assert window.load_image(bad) is False
    assert shown and window.telemetry.state.state() == "ERROR"


def test_about_dialog_renders(window):
    from app.about_dialog import APP_NAME, APP_TAGLINE, AboutDialog

    assert (APP_NAME, APP_TAGLINE) == ("AYLAB", "BEYOND THE PIXEL.")
    dialog = AboutDialog(window)
    assert dialog.windowTitle() == "About AYLAB"
    assert not dialog.grab().isNull()
    dialog.close()


# --- parametre kontrolleri ------------------------------------------------

def test_slider_control_odd_kernel(qapp):
    c = create_control(IntParam("ksize", "Kernel", 3, 1, 31, odd_only=True))
    assert isinstance(c, SliderControl)
    seen = []
    c.value_changed.connect(lambda k, v: seen.append((k, v)))
    c.field.setValue(8)  # çift değer yazılırsa tek sayıya düzeltilir
    assert c.value() == 9
    c.slider.setValue(c.slider.maximum())
    assert c.value() == 31
    c.reset()
    assert c.value() == 3 and seen[-1] == ("ksize", 3)


def test_float_slider_and_choice_and_toggle(qapp):
    f = create_control(FloatParam("g", "Gamma", 1.0, 0.1, 5.0, step=0.05))
    f.set_value(2.333)
    assert f.value() == pytest.approx(2.33)
    assert f.slider.value() == round((2.33 - 0.1) / 0.05)

    seg = create_control(ChoiceParam("m", "Mode", (("a", "A"), ("b", "B")), "a"))
    assert isinstance(seg, ChoiceControl) and seg.combo is None
    seg.buttons["b"].click()
    assert seg.value() == "b"

    many = create_control(ChoiceParam("c", "Map", tuple((str(i), str(i)) for i in range(6)), "0"))
    assert many.combo is not None
    many.combo.setCurrentIndex(4)
    assert many.value() == "4"

    from processing.registry import BoolParam

    t = create_control(BoolParam("inv", "Invert"))
    assert isinstance(t, ToggleControl) and t.value() is False
    QTest.mouseClick(t.switch, Qt.MouseButton.LeftButton)
    assert t.value() is True
