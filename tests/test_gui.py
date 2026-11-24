"""Ana pencerenin ekran göstermeden (offscreen) duman testi."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.image_viewer import ndarray_to_qimage
from app.main_window import MainWindow
from processing.image_io import save_image


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_window_loads_image_from_turkish_path(qapp, tmp_path):
    img = np.full((30, 50, 3), (10, 20, 200), dtype=np.uint8)
    path = tmp_path / "Masaüstü" / "kırmızı_görsel.png"
    save_image(path, img)

    window = MainWindow()
    window.show()
    assert not window.controls.save_button.isEnabled()

    assert window.load_image(path) is True
    np.testing.assert_array_equal(window.original_image, img)
    assert window.original_viewer.has_image()
    assert window.result_viewer.has_image()
    assert window.controls.save_button.isEnabled()
    assert "50×30" in window.statusBar().currentMessage()

    window.current_image[:] = 0
    window.reset_image()
    np.testing.assert_array_equal(window.current_image, img)
    window.close()


@pytest.mark.parametrize("shape", [(10, 12), (10, 12, 3), (10, 12, 4)])
def test_ndarray_to_qimage(qapp, shape):
    qimg = ndarray_to_qimage(np.zeros(shape, dtype=np.uint8))
    assert (qimg.width(), qimg.height()) == (12, 10)
