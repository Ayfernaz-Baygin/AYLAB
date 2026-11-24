"""Unicode (Türkçe karakter) içeren yollarla güvenli görüntü okuma/yazma.

cv2.imread / cv2.imwrite Windows'ta ASCII dışı karakter içeren yollarda
(ör. C:\\Users\\Vıctus\\...) sessizce başarısız olur. Bu yüzden dosyayı
NumPy ile bayt olarak okuyup cv2.imdecode ile çözüyoruz; yazarken de
cv2.imencode ile kodlayıp NumPy ile diske yazıyoruz.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class ImageIOError(Exception):
    """Görüntü okunamadığında veya kaydedilemediğinde fırlatılır."""


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Görüntüyü BGR (veya flags'e göre gri) NumPy dizisi olarak okur."""
    path = Path(path)
    if not path.is_file():
        raise ImageIOError(f"Dosya bulunamadı: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ImageIOError(f"Görüntü çözümlenemedi (desteklenmeyen/bozuk dosya): {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Görüntüyü uzantısına uygun formatta kaydeder."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageIOError(f"Desteklenmeyen uzantı: {ext or '(yok)'}")
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise ImageIOError(f"Görüntü kodlanamadı: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))
