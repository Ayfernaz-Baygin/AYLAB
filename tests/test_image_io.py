"""Türkçe karakterli yollarda okuma/yazma testleri."""

import numpy as np
import pytest

from processing.image_io import ImageIOError, read_image, save_image


@pytest.fixture
def sample_image() -> np.ndarray:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :20] = (255, 0, 0)
    img[:, 20:40] = (0, 255, 0)
    img[:, 40:] = (0, 0, 255)
    return img


@pytest.mark.parametrize("ext", [".png", ".bmp", ".tiff"])
def test_roundtrip_turkish_path(tmp_path, sample_image, ext):
    path = tmp_path / "Görüntü_çğıöşü_İĞŞ" / f"şekil_ığdır{ext}"
    save_image(path, sample_image)
    assert path.is_file()
    loaded = read_image(path)
    np.testing.assert_array_equal(loaded, sample_image)


def test_jpeg_turkish_path(tmp_path, sample_image):
    path = tmp_path / "fotoğraf_ünlü.jpg"
    save_image(path, sample_image)
    loaded = read_image(path)
    assert loaded.shape == sample_image.shape


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageIOError):
        read_image(tmp_path / "yok_böyle_dosya.png")


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "bozuk.png"
    path.write_bytes(b"resim degil")
    with pytest.raises(ImageIOError):
        read_image(path)


def test_unsupported_extension_raises(tmp_path, sample_image):
    with pytest.raises(ImageIOError):
        save_image(tmp_path / "dosya.xyz", sample_image)
