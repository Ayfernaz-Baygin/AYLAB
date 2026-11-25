"""Yoğunluk dönüşümleri (Gonzalez & Woods, Chapter 3).

Tüm fonksiyonlar 8-bit (uint8) görüntülerle çalışır: L = 256 gri seviye,
r, s ∈ {0, …, L−1}. Girdi dizisi hiçbir zaman değiştirilmez; her fonksiyon
yeni bir dizi döndürür.

Nokta işlemleri (negative, gamma, contrast stretching) yalnızca piksel
değerine bağlı olduğu için 256 girişli bir arama tablosu (LUT) ile
uygulanır: T(r) bir kez hesaplanır, sonra tüm görüntüye vektörel olarak
eşlenir. Bu hem hızlıdır hem de dönüşümü sayısal olarak tam belirler
(her r için tek bir s).

Renk ve alfa kuralları
----------------------
* Gri (H×W veya H×W×1): tek kanal işlenir.
* BGR: nokta işlemleri her kanala *aynı* T(r) ile uygulanır (doğrusal/monoton
  ortak eşleme renk tonunu kaydırmaz). Histogram eşitleme ise *yalnızca
  parlaklık kanalında* yapılır (bkz. ``equalize_histogram``).
* BGRA: alfa (saydamlık) bir yoğunluk değildir; aynen korunur.
"""

from __future__ import annotations

import cv2
import numpy as np

from processing.registry import AlgorithmParameterError

L = 256  # 8-bit gri seviye sayısı
_LEVELS = np.arange(L, dtype=np.float64)

# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------


def _check_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise AlgorithmParameterError("Input must be a NumPy image array.")
    if image.dtype != np.uint8:
        raise AlgorithmParameterError(
            f"Intensity transformations expect an 8-bit image (uint8), got {image.dtype}."
        )
    if image.size == 0:
        raise AlgorithmParameterError("The image is empty.")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3, 4))):
        raise AlgorithmParameterError(f"Unsupported image shape {image.shape}; expected gray, BGR or BGRA.")


def _split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image, None


def _merge_alpha(color: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return color
    return np.dstack((color, alpha))


def _to_lut(values: np.ndarray) -> np.ndarray:
    """Gerçel değerli T(r)'yi [0, 255] aralığına kırpıp en yakın tamsayıya yuvarlar."""
    return np.clip(np.rint(values), 0, L - 1).astype(np.uint8)


def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """256 girişli LUT'u renk kanallarına uygular (alfa korunur). Yeni dizi döner."""
    _check_image(image)
    if lut.shape != (L,) or lut.dtype != np.uint8:
        raise ValueError("LUT must be a uint8 array of length 256")
    color, alpha = _split_alpha(image)
    return _merge_alpha(np.take(lut, color), alpha)


# --------------------------------------------------------------------------
# A. Negative
# --------------------------------------------------------------------------


def negative_lut() -> np.ndarray:
    """s = (L − 1) − r"""
    return _to_lut((L - 1) - _LEVELS)


def negative(image: np.ndarray) -> np.ndarray:
    """Görüntü negatifi: s = 255 − r (her renk kanalında; alfa korunur)."""
    return apply_lut(image, negative_lut())


# --------------------------------------------------------------------------
# B. Gamma (power-law) correction
# --------------------------------------------------------------------------


def gamma_lut(gamma: float, c: float = 1.0) -> np.ndarray:
    """s = (L − 1) · c · (r / (L − 1))^γ

    Hesap [0, 1]'e normalize edilmiş r üzerinde float64 ile yapılır; c > 1
    durumunda oluşabilecek taşma 255'e kırpılır (uint8 sarmalanması olmaz).
    """
    if gamma is None or c is None:
        raise AlgorithmParameterError("Gamma and c must be provided.")
    gamma, c = float(gamma), float(c)
    if not np.isfinite(gamma) or gamma <= 0:
        raise AlgorithmParameterError(f"Gamma must be a positive number (got {gamma}).")
    if not np.isfinite(c) or c <= 0:
        raise AlgorithmParameterError(f"Constant c must be a positive number (got {c}).")
    r = _LEVELS / (L - 1)
    return _to_lut((L - 1) * c * np.power(r, gamma))


def gamma_correction(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """Güç yasası dönüşümü. γ < 1 koyu tonları açar, γ > 1 açık tonları koyulaştırır."""
    return apply_lut(image, gamma_lut(gamma, c))


# --------------------------------------------------------------------------
# C. Contrast stretching
# --------------------------------------------------------------------------


def stretch_lut(r_min: float, r_max: float) -> np.ndarray:
    """s = (r − r_min) · (L − 1) / (r_max − r_min), [0, L − 1]'e kırpılmış."""
    if not r_max > r_min:
        raise AlgorithmParameterError(
            f"Input high ({r_max:g}) must be greater than input low ({r_min:g})."
        )
    return _to_lut((_LEVELS - r_min) * (L - 1) / (r_max - r_min))


def stretch_limits(image: np.ndarray, saturate: float = 0.0) -> tuple[float, float]:
    """Otomatik germe sınırları: min/maks veya her iki uçtan %saturate kırpılmış yüzdelikler.

    Renkli görüntüde sınırlar tüm renk kanallarından *ortak* hesaplanır.
    """
    _check_image(image)
    if not 0.0 <= saturate < 50.0:
        raise AlgorithmParameterError(f"Saturation must be in [0, 50) percent (got {saturate}).")
    color, _ = _split_alpha(image)
    if saturate > 0:
        lo, hi = np.percentile(color, (saturate, 100.0 - saturate))
        if hi > lo:
            return float(lo), float(hi)
    return float(color.min()), float(color.max())


def contrast_stretch(image: np.ndarray, mode: str = "auto", low: int = 0, high: int = 255,
                     saturate: float = 0.0) -> np.ndarray:
    """Doğrusal kontrast germe (min–maks normalizasyonu).

    * ``auto``   — r_min / r_max görüntüden alınır (isteğe bağlı yüzdelik kırpma ile).
    * ``manual`` — r_min = low, r_max = high; aralık dışı değerler 0 / 255'e doyar.

    Sabit yoğunluklu bir görüntüde (r_max = r_min) germe tanımsızdır; sıfıra
    bölmek yerine görüntü değiştirilmeden (kopya olarak) döndürülür.
    """
    _check_image(image)
    if mode == "manual":
        if low is None or high is None:
            raise AlgorithmParameterError("Manual mode needs both input low and input high.")
        lo, hi = float(low), float(high)
        if not (0 <= lo <= L - 1 and 0 <= hi <= L - 1):
            raise AlgorithmParameterError("Input low/high must lie within [0, 255].")
        return apply_lut(image, stretch_lut(lo, hi))
    if mode != "auto":
        raise AlgorithmParameterError(f"Unknown contrast stretching mode: {mode!r}.")
    lo, hi = stretch_limits(image, saturate)
    if hi <= lo:  # sabit görüntü — gerilecek aralık yok
        return image.copy()
    return apply_lut(image, stretch_lut(lo, hi))


# --------------------------------------------------------------------------
# D. Histogram equalization
# --------------------------------------------------------------------------


def equalization_lut(channel: np.ndarray) -> np.ndarray:
    """Tek kanal için histogram eşitleme dönüşümü T(r).

    p(r_k) = n_k / N  ve  CDF(r_k) = Σ_{j≤k} n_j  olmak üzere

        s_k = round( (L − 1) · (CDF(r_k) − CDF_min) / (N − CDF_min) )

    CDF_min, görüntüde bulunan en küçük seviyenin birikimli sayısıdır. Bu,
    Gonzalez'in s_k = (L − 1)·Σ p(r_j) ifadesinin, en koyu mevcut seviyeyi
    0'a, en açığı 255'e eşleyen (tam aralığı kullanan) normalize biçimidir ve
    OpenCV ``equalizeHist`` ile aynı sonucu verir (float32 ölçek, yuvarlama
    en yakına). Sabit kanalda (N = CDF_min) birim dönüşüm döndürülür.
    """
    hist = np.bincount(channel.ravel(), minlength=L)
    cdf = np.cumsum(hist)
    total = int(cdf[-1])
    cdf_min = int(cdf[np.flatnonzero(hist)[0]])
    if total == cdf_min:  # tek seviye: eşitlenecek dağılım yok
        return np.arange(L, dtype=np.uint8)
    scale = np.float32(L - 1) / np.float32(total - cdf_min)
    lut = (cdf - cdf_min).astype(np.float32) * scale
    return _to_lut(lut.astype(np.float64))


def _equalize_channel(channel: np.ndarray, method: str, clip: float, tiles: int) -> np.ndarray:
    if method == "clahe":
        if not clip > 0:
            raise AlgorithmParameterError(f"CLAHE clip limit must be positive (got {clip}).")
        if not 1 <= int(tiles) <= 64:
            raise AlgorithmParameterError(f"CLAHE tile grid must be between 1 and 64 (got {tiles}).")
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
        return clahe.apply(np.ascontiguousarray(channel))
    if method != "global":
        raise AlgorithmParameterError(f"Unknown equalization method: {method!r}.")
    return np.take(equalization_lut(channel), channel)


# Renkli görüntüde eşitlenecek parlaklık kanalı: (ileri dönüşüm, geri dönüşüm, kanal indeksi)
_LUMA_SPACES = {
    "ycrcb": (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR, 0),  # Y  — luma
    "lab": (cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, 0),  # L* — algısal açıklık
    "hsv": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, 2),  # V  — maks(R, G, B)
}


def equalize_histogram(image: np.ndarray, method: str = "global", space: str = "ycrcb",
                       clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Histogram eşitleme (global) veya CLAHE.

    Gri görüntüde doğrudan uygulanır. Renkli görüntüde B, G, R kanallarını
    *ayrı ayrı* eşitlemek her kanala farklı bir T(r) uygular ve renk dengesini
    bozar (ör. ten rengi morarır). Bunun yerine görüntü, parlaklığı renk
    bilgisinden ayıran bir uzaya (YCrCb / Lab / HSV) çevrilir, *yalnızca*
    parlaklık kanalı eşitlenir, renk kanalları (Cr–Cb, a*–b*, H–S) olduğu gibi
    bırakılıp BGR'ye geri dönülür.
    """
    _check_image(image)
    color, alpha = _split_alpha(image)

    if color.ndim == 2 or color.shape[2] == 1:
        gray = color.reshape(color.shape[:2])
        out = _equalize_channel(gray, method, clip, tiles).reshape(color.shape)
        return _merge_alpha(out, alpha)

    if space not in _LUMA_SPACES:
        raise AlgorithmParameterError(f"Unknown color space for equalization: {space!r}.")
    forward, backward, idx = _LUMA_SPACES[space]
    converted = cv2.cvtColor(np.ascontiguousarray(color), forward)
    luma = converted[:, :, idx]
    equalized = _equalize_channel(luma, method, clip, tiles)
    if np.array_equal(equalized, luma):
        # Parlaklık değişmedi (ör. sabit görüntü): renk uzayı gidiş-dönüşünün
        # ±1'lik yuvarlama hatasını eklememek için girdi aynen döndürülür.
        return image.copy()
    converted[:, :, idx] = equalized
    # Not: yükseltilen Y ile korunan Cr/Cb, çok doygun renklerde RGB küpünün
    # dışına düşebilir; OpenCV bu değerleri [0, 255]'e kırpar (gamut sınırı).
    return _merge_alpha(cv2.cvtColor(converted, backward), alpha)
