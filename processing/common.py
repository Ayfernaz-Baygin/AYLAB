"""İşleme modüllerinin paylaştığı görüntü doğrulama ve kanal yardımcıları.

AYLAB işleme hattının kuralları:
* Görüntüler 8-bit (uint8): gri (H×W, H×W×1), BGR (H×W×3) veya BGRA (H×W×4).
* Alfa kanalı bir yoğunluk değildir; işlenmez, aynen korunur.
* Girdi dizisi değiştirilmez; her işlem yeni bir dizi döndürür.
"""

from __future__ import annotations

import numpy as np

from processing.registry import AlgorithmParameterError

# ITU-R BT.601 luma ağırlıkları, OpenCV'nin BGR düzeninde (B, G, R)
LUMA_WEIGHTS_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float64)


def check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise AlgorithmParameterError("Input must be a NumPy image array.")
    if image.dtype != np.uint8:
        raise AlgorithmParameterError(f"Expected an 8-bit image (uint8), got {image.dtype}.")
    if image.size == 0:
        raise AlgorithmParameterError("The image is empty.")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3, 4))):
        raise AlgorithmParameterError(f"Unsupported image shape {image.shape}; expected gray, BGR or BGRA.")


def split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image, None


def merge_alpha(color: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    return color if alpha is None else np.dstack((color, alpha))


def round_to_uint8(values: np.ndarray) -> np.ndarray:
    """Gerçel sonuçları en yakın tamsayıya yuvarlayıp [0, 255]'e kırpar."""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def luminance(image: np.ndarray) -> np.ndarray:
    """Görüntünün parlaklığını float64 olarak döndürür (H×W).

    Gri görüntüde değerlerin kendisi; renkli görüntüde BT.601 luma
    Y = 0.299·R + 0.587·G + 0.114·B. Hesap kayan noktada yapılır; ara
    sonucu uint8'e yuvarlayarak türev hesabına nicemleme hatası eklenmez.
    """
    check_image(image)
    color, _ = split_alpha(image)
    if color.ndim == 2:
        return color.astype(np.float64)
    if color.shape[2] == 1:
        return color[:, :, 0].astype(np.float64)
    return color.astype(np.float64) @ LUMA_WEIGHTS_BGR


def match_layout(gray: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Tek kanallı uint8 sonucu girdinin kanal düzenine getirir.

    Gri → gri, H×W×1 → H×W×1, BGR → üç eşit kanal, BGRA → üç eşit kanal + orijinal alfa.
    Böylece işlem hattındaki sonraki adımlar aynı kanal sayısıyla çalışır.
    """
    color, alpha = split_alpha(like)
    if color.ndim == 2:
        return gray
    if color.shape[2] == 1:
        return gray[:, :, None]
    return merge_alpha(np.repeat(gray[:, :, None], 3, axis=2), alpha)
