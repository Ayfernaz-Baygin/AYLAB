"""Morfolojik işlemler — Gonzalez & Woods / COMP 4360 Chapter 4.

İkili işlemler (erosion, dilation, opening, closing, gradient, hit-or-miss) ve
gri ton top-hat / black-hat (dosyanın sonundaki bölüm; eşikleme yapılmaz).

Görüntü bir küme olarak ele alınır: A = { (x, y) | f(x, y) = ön plan }.
Yapı elemanı (SE) B, köşe/merkez fark etmeksizin seçilebilen bir orijine
(anchor) göre tanımlı ofsetler kümesidir.

    Erosion   A ⊖ B = { z | (B)_z ⊆ A }                     — B tamamen sığmalı (fit)
    Dilation  A ⊕ B = { z | (B̂)_z ∩ A ≠ ∅ } = ∪_{b∈B} A_b   — B̂ en az bir pikselde değmeli (hit)
    Opening   A ∘ B = (A ⊖ B) ⊕ B
    Closing   A • B = (A ⊕ B) ⊖ B
    Gradient  (A ⊕ B) − (A ⊖ B)
    Hit-or-miss  A ⊛ B = (A ⊖ B₁) ∩ (Aᶜ ⊖ B₂)

Uygulama kuralları
------------------
* **Sınır:** görüntünün dışındaki pikseller *arka plandır* (dersin laboratuvar
  sayfasındaki kural). İşlemler, görüntü SE'nin etki alanı kadar arka planla
  genişletilmiş bir tuvalde yapılıp sonra kırpılır. Böylece sonuç, sonsuz
  düzlemde aynı tanımın birebir kırpılmış halidir: closing kapsayıcıdır
  (A ⊆ A • B), opening ve closing idempotenttir, yinelenen dilation sınırda
  bilgi kaybetmez.
* **Yansıma:** OpenCV ``dilate`` çekirdeği yansıtmadan uygular
  (z + o ∈ A koşulu, yani A ⊕ B̂). Ders kitabı tanımı için çekirdek ve anchor
  merkez etrafında 180° döndürülerek verilir. Simetrik SE + merkez anchor için
  iki tanım aynıdır; fark yalnızca asimetrik durumda ortaya çıkar.
* **İkili görüntü:** renkli girdide BT.601 parlaklığı Y (0–255 tamsayıya
  yuvarlanmış), gri girdide değerin kendisi kullanılır. Ön plan BRIGHT için
  Y > T, DARK için Y ≤ T'dir (aynı bölümleme, roller yer değiştirir). T elle
  veya Otsu yöntemiyle seçilir.
* **Çıktı:** ön plan 255, arka plan 0; girdinin kanal düzeninde (BGR → üç eşit
  kanal). **Alfa** işlenmez ve ikili görüntüye katılmaz; orijinal alfa aynen
  geri eklenir (diğer AYLAB modülleriyle tutarlı). Girdi dizisi değiştirilmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import cv2
import numpy as np

from processing.common import (
    check_image,
    luminance,
    match_layout,
    merge_alpha,
    round_to_uint8,
    split_alpha,
)
from processing.filters import validate_ksize
from processing.registry import AlgorithmParameterError

SE_SHAPES = ("rect", "ellipse", "cross")
BINARIZE_MODES = ("otsu", "manual")
FOREGROUND_MODES = ("bright", "dark")

# --------------------------------------------------------------------------
# Yapı elemanı
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuringElement:
    """İkili maske (H×W, 0/1) ve orijin ``anchor = (x, y)`` = (sütun, satır)."""

    mask: np.ndarray
    anchor: tuple[int, int]

    @property
    def offsets(self) -> list[tuple[int, int]]:
        """Orijine göre üye ofsetleri (dy, dx)."""
        ax, ay = self.anchor
        return [(int(r) - ay, int(c) - ax) for r, c in zip(*np.nonzero(self.mask))]

    def reflected(self) -> "StructuringElement":
        """B̂ = { −b | b ∈ B }: maske ve orijin 180° döner."""
        h, w = self.mask.shape
        ax, ay = self.anchor
        return StructuringElement(np.ascontiguousarray(self.mask[::-1, ::-1]), (w - 1 - ax, h - 1 - ay))

    @property
    def reach(self) -> int:
        """Orijinden en uzak SE hücresinin satranç tahtası uzaklığı (dolgu payı için)."""
        h, w = self.mask.shape
        ax, ay = self.anchor
        return max(ax, w - 1 - ax, ay, h - 1 - ay)


def _validate_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, float)):
        raise AlgorithmParameterError(f"{name} must be an integer (got {value!r}).")
    if isinstance(value, float) and not value.is_integer():
        raise AlgorithmParameterError(f"{name} must be an integer (got {value!r}).")
    return int(value)


def _validate_anchor(value, size: int, name: str) -> int:
    a = _validate_int(value, name)
    if a == -1:
        return size // 2
    if not 0 <= a < size:
        raise AlgorithmParameterError(f"{name} must be −1 (centre) or 0…{size - 1} for this SE size (got {a}).")
    return a


def structuring_element(shape: str = "rect", width: int = 3, height: int = 3,
                        anchor_x: int = -1, anchor_y: int = -1) -> StructuringElement:
    """Dikdörtgen/kare, elips veya artı biçimli SE ve orijini.

    * rect    — tüm hücreler 1 (width = height → kare).
    * ellipse — kutuya içten teğet dijital elips (OpenCV ``MORPH_ELLIPSE`` ile
      aynı; OpenCV'nin tek satırlı elipste tek piksel döndürdüğü dejenere durum
      düzeltilir: 1×k veya k×1 elips bir çizgidir).
    * cross   — kutunun *geometrik* merkez satırı ve sütunu. Şekil orijinden
      bağımsızdır; OpenCV ``MORPH_CROSS`` ise artıyı anchor'a taşır.

    Orijin −1 ise merkezdedir; aksi halde kutunun içindeki herhangi bir
    hücre olabilir (üye olması gerekmez).
    """
    if shape not in SE_SHAPES:
        raise AlgorithmParameterError(f"SE shape must be one of {', '.join(SE_SHAPES)} (got {shape!r}).")
    w = validate_ksize(width, name="SE width")
    h = validate_ksize(height, name="SE height")
    anchor = (_validate_anchor(anchor_x, w, "Anchor X"), _validate_anchor(anchor_y, h, "Anchor Y"))
    if shape == "rect" or (shape == "ellipse" and min(w, h) == 1):
        mask = np.ones((h, w), np.uint8)
    elif shape == "ellipse":
        mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))
    else:
        mask = np.zeros((h, w), np.uint8)
        mask[h // 2, :] = 1
        mask[:, w // 2] = 1
    return StructuringElement(mask.astype(np.uint8), anchor)


# --------------------------------------------------------------------------
# İkili görüntü
# --------------------------------------------------------------------------


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu eşiği (sınıflar arası varyansı en büyük yapan T; ön plan = Y > T)."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def to_binary(image: np.ndarray, binarize: str = "otsu", thresh: int = 127,
              foreground: str = "bright") -> np.ndarray:
    """Görüntüyü ön plan maskesine (bool, H×W) çevirir. Alfa dikkate alınmaz."""
    check_image(image)
    if binarize not in BINARIZE_MODES:
        raise AlgorithmParameterError(f"Binarization must be 'otsu' or 'manual' (got {binarize!r}).")
    if foreground not in FOREGROUND_MODES:
        raise AlgorithmParameterError(f"Foreground must be 'bright' or 'dark' (got {foreground!r}).")
    y = round_to_uint8(luminance(image))
    if binarize == "otsu":
        t = otsu_threshold(y)
    else:
        t = _validate_int(thresh, "Threshold")
        if not 0 <= t <= 255:
            raise AlgorithmParameterError(f"Threshold must be in 0…255 (got {t}).")
    bright = y > t
    return bright if foreground == "bright" else ~bright


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        raise AlgorithmParameterError("A binary image must be a non-empty 2-D array.")
    return mask.astype(bool)


def _validate_iterations(iterations) -> int:
    n = _validate_int(iterations, "Iterations")
    if n < 1:
        raise AlgorithmParameterError(f"Iterations must be ≥ 1 (got {n}).")
    return n


def _pad(mask: np.ndarray, p: int, value: int = 0) -> np.ndarray:
    return np.pad(mask.astype(np.uint8), p, mode="constant", constant_values=value)


def _crop(padded: np.ndarray, p: int, shape: tuple[int, int]) -> np.ndarray:
    return padded[p:p + shape[0], p:p + shape[1]].astype(bool)


def _erode_raw(m: np.ndarray, se: StructuringElement, iterations: int = 1) -> np.ndarray:
    # cv2.erode: dst(z) = min_{o∈B} src(z + o)  → (B)_z ⊆ A (ders kitabı tanımı, yansıma yok)
    return cv2.erode(m, se.mask, anchor=se.anchor, iterations=iterations,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _dilate_raw(m: np.ndarray, se: StructuringElement, iterations: int = 1) -> np.ndarray:
    # cv2.dilate: dst(z) = max_{o∈K} src(z + o). K = B̂ verildiğinde: ∃ b ∈ B, z − b ∈ A  → A ⊕ B
    r = se.reflected()
    return cv2.dilate(m, r.mask, anchor=r.anchor, iterations=iterations,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)


def binary_erosion(mask: np.ndarray, se: StructuringElement, iterations: int = 1) -> np.ndarray:
    """A ⊖ B (n kez). Görüntü dışı arka plandır."""
    a, n = _validate_mask(mask), _validate_iterations(iterations)
    p = se.reach * n + 1
    return _crop(_erode_raw(_pad(a, p), se, n), p, a.shape)


def binary_dilation(mask: np.ndarray, se: StructuringElement, iterations: int = 1) -> np.ndarray:
    """A ⊕ B (n kez). Tuval, dilation'ın görüntü dışına taşan kısmı kaybolmayacak kadar genişletilir."""
    a, n = _validate_mask(mask), _validate_iterations(iterations)
    p = se.reach * n + 1
    return _crop(_dilate_raw(_pad(a, p), se, n), p, a.shape)


def binary_opening(mask: np.ndarray, se: StructuringElement) -> np.ndarray:
    """A ∘ B = (A ⊖ B) ⊕ B — B'nin A'ya sığan tüm ötelemelerinin birleşimi."""
    a = _validate_mask(mask)
    p = 2 * se.reach + 1
    return _crop(_dilate_raw(_erode_raw(_pad(a, p), se), se), p, a.shape)


def binary_closing(mask: np.ndarray, se: StructuringElement) -> np.ndarray:
    """A • B = (A ⊕ B) ⊖ B — kapsayıcı: A ⊆ A • B."""
    a = _validate_mask(mask)
    p = 2 * se.reach + 1
    return _crop(_erode_raw(_dilate_raw(_pad(a, p), se), se), p, a.shape)


def binary_gradient(mask: np.ndarray, se: StructuringElement) -> np.ndarray:
    """(A ⊕ B) − (A ⊖ B): nesne sınırının iç ve dış kuşağı."""
    return binary_dilation(mask, se) & ~binary_erosion(mask, se)


# --------------------------------------------------------------------------
# Hit-or-miss
# --------------------------------------------------------------------------

# Slayt 38–39 ve dersin laboratuvar sayfasındaki çekirdekler (1 = ön plan, −1 = arka plan, 0 = önemsiz)
HMT_PATTERNS: dict[str, np.ndarray] = {
    "hole": np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]]),
    "corner": np.array([[-1, -1, -1], [1, 1, -1], [1, 1, -1]]),
    "edge": np.array([[0, 0, -1], [1, 1, -1], [0, 0, -1]]),
    "cv_example": np.array([[1, 1, -1], [1, 1, -1], [-1, -1, -1]]),
    "isolated": np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),
    "end_point": np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]]),
}
HMT_ROTATIONS = ("0", "90", "180", "270")
HMT_DISPLAYS = ("matches", "overlay")
OVERLAY_SHAPE_LEVEL = 70  # "overlay" gösteriminde ön plan şeklinin gri düzeyi


def hmt_kernel(pattern: str = "corner", rotation: str = "0") -> np.ndarray:
    """Hazır deseni saat yönünün tersine 0/90/180/270° döndürür (np.rot90)."""
    if pattern not in HMT_PATTERNS:
        raise AlgorithmParameterError(f"Unknown hit-or-miss pattern {pattern!r}.")
    if rotation not in HMT_ROTATIONS:
        raise AlgorithmParameterError(f"Rotation must be one of 0, 90, 180, 270 (got {rotation!r}).")
    return np.rot90(HMT_PATTERNS[pattern], int(rotation) // 90).copy()


def hit_or_miss_mask(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """A ⊛ B = (A ⊖ B₁) ∩ (Aᶜ ⊖ B₂); B₁ = {1 hücreleri}, B₂ = {−1 hücreleri}, 0 = önemsiz.

    Orijin çekirdeğin merkezidir. Görüntü dışı arka plan olduğundan, dışarıda
    kalan −1 hücreleri sağlanmış, +1 hücreleri sağlanmamış sayılır.
    """
    a = _validate_mask(mask)
    k = np.asarray(kernel)
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise AlgorithmParameterError("Hit-or-miss kernel must be a 2-D array with odd sides.")
    if not np.isin(k, (-1, 0, 1)).all():
        raise AlgorithmParameterError("Hit-or-miss kernel values must be 1 (hit), −1 (miss) or 0 (don't care).")
    anchor = (k.shape[1] // 2, k.shape[0] // 2)
    p = max(k.shape) + 1
    result = np.ones(a.shape, bool)
    b1, b2 = (k == 1).astype(np.uint8), (k == -1).astype(np.uint8)
    if b1.any():  # boş B₁ ile erosion tüm düzlemdir
        result &= _crop(_erode_raw(_pad(a, p, 0), StructuringElement(b1, anchor)), p, a.shape)
    if b2.any():  # Aᶜ: görüntü dışı arka plan → tümleyende ön plan
        result &= _crop(_erode_raw(_pad(~a, p, 1), StructuringElement(b2, anchor)), p, a.shape)
    return result


# --------------------------------------------------------------------------
# Gri ton morfoloji (düz SE) — Top-hat / Black-hat (slayt 43–47, 82–85)
# --------------------------------------------------------------------------
#
#   [f ⊖ b](z) = min_{s∈b} f(z + s)        [f ⊕ b](z) = max_{s∈b} f(z − s)
#   f ∘ b = (f ⊖ b) ⊕ b                     f • b = (f ⊕ b) ⊖ b
#   T_hat(f) = f − (f ∘ b)                  B_hat(f) = (f • b) − f
#
# Sınır (dersin laboratuvar sayfası): görüntü dışındaki pikseller *yok sayılır*;
# min/max yalnızca görüntü içindeki komşular üzerinden alınır (OpenCV'nin gri
# erode/dilate varsayılan sınırı). Hesap uint8 kafesinde [0, 255] yapılır:
# komşuluk tamamen dışarıda kalırsa inf ∅ = 255, sup ∅ = 0 olur. Bu kural ile
# f ∘ b ≤ f ≤ f • b her pikselde sağlanır; iki dönüşüm de tanım gereği negatif
# olamaz. (float'ta boş küme ±DBL_MAX üretir; bu yüzden float kullanılmaz.)


def _validate_gray(f: np.ndarray) -> np.ndarray:
    if not isinstance(f, np.ndarray) or f.dtype != np.uint8 or f.ndim != 2 or f.size == 0:
        raise AlgorithmParameterError("Grayscale morphology expects a non-empty 2-D uint8 array.")
    return f


def gray_erosion(f: np.ndarray, se: StructuringElement) -> np.ndarray:
    """f ⊖ b: düz SE altındaki minimum (görüntü dışı yok sayılır)."""
    return cv2.erode(_validate_gray(f), se.mask, anchor=se.anchor)


def gray_dilation(f: np.ndarray, se: StructuringElement) -> np.ndarray:
    """f ⊕ b: yansıtılmış SE altındaki maksimum (ikili dilation ile aynı yansıma kuralı)."""
    r = se.reflected()
    return cv2.dilate(_validate_gray(f), r.mask, anchor=r.anchor)


def gray_opening(f: np.ndarray, se: StructuringElement) -> np.ndarray:
    return gray_dilation(gray_erosion(f, se), se)


def gray_closing(f: np.ndarray, se: StructuringElement) -> np.ndarray:
    return gray_erosion(gray_dilation(f, se), se)


def white_top_hat_values(f: np.ndarray, se: StructuringElement) -> np.ndarray:
    """T_hat(f) = f − (f ∘ b) (int32, ≥ 0)."""
    return f.astype(np.int32) - gray_opening(f, se).astype(np.int32)


def black_hat_values(f: np.ndarray, se: StructuringElement) -> np.ndarray:
    """B_hat(f) = (f • b) − f (int32, ≥ 0)."""
    return gray_closing(f, se).astype(np.int32) - f.astype(np.int32)


def intensity_channel(image: np.ndarray) -> np.ndarray:
    """Gri ton morfolojinin uygulandığı kanal (uint8, H×W).

    Gri girdide değerlerin kendisi (eşikleme yok). Renkli girdide BT.601
    parlaklığı Y = 0.299R + 0.587G + 0.114B, en yakın tamsayıya yuvarlanmış.
    """
    check_image(image)
    return round_to_uint8(luminance(image))


def _hat_se(shape, width, height, anchor_x, anchor_y) -> StructuringElement:
    return structuring_element(shape, width, height, anchor_x, anchor_y)


def top_hat(image: np.ndarray, shape: str = "rect", width: int = 15, height: int = 15,
            anchor_x: int = -1, anchor_y: int = -1) -> np.ndarray:
    """White top-hat T_hat(f) = f − (f ∘ b): b'den küçük PARLAK ayrıntılar.

    Değerler ölçeklenmeden gösterilir (gerçek yoğunluk farkı). Renkli girdide
    parlaklık üzerinde hesaplanır; sonuç üç eşit kanal, alfa korunur.
    """
    se = _hat_se(shape, width, height, anchor_x, anchor_y)
    f = intensity_channel(image)
    return match_layout(white_top_hat_values(f, se).astype(np.uint8), image)


def black_hat(image: np.ndarray, shape: str = "rect", width: int = 15, height: int = 15,
              anchor_x: int = -1, anchor_y: int = -1) -> np.ndarray:
    """Black-hat B_hat(f) = (f • b) − f: b'den küçük KOYU ayrıntılar (ölçeklenmeden)."""
    se = _hat_se(shape, width, height, anchor_x, anchor_y)
    f = intensity_channel(image)
    return match_layout(black_hat_values(f, se).astype(np.uint8), image)


def hat_correction(image: np.ndarray, shape: str = "rect", width: int = 15, height: int = 15,
                   anchor_x: int = -1, anchor_y: int = -1) -> np.ndarray:
    """corrected = img + top_hat − black_hat (slayt 47), int32'de hesaplanıp [0, 255]'e kırpılır.

    Aynı SE ile: parlak ayrıntılar daha parlak, koyu ayrıntılar daha koyu olur
    (yerel kontrast artar). Renkli girdide T_hat ve B_hat parlaklık Y üzerinden
    hesaplanır ve Δ = T_hat − B_hat her kanala eşit eklenir: ağırlıkların toplamı
    1 olduğundan Y' = Y + Δ olur ve renklilik (B − Y, R − Y) değişmez — YCbCr'de
    yalnızca Y'yi değiştirmekle aynıdır. Kırpma, doygun piksellerde tonu az
    miktarda kaydırabilir. Alfa korunur.
    """
    se = _hat_se(shape, width, height, anchor_x, anchor_y)
    f = intensity_channel(image)
    delta = white_top_hat_values(f, se) - black_hat_values(f, se)
    color, alpha = split_alpha(image)
    if color.ndim == 2:
        out = f.astype(np.int32) + delta
    elif color.shape[2] == 1:
        out = (f.astype(np.int32) + delta)[:, :, None]
    else:
        out = color.astype(np.int32) + delta[:, :, None]
    return merge_alpha(np.clip(out, 0, 255).astype(np.uint8), alpha)


# --------------------------------------------------------------------------
# Kayıtlı işlevler (AYLAB parametre paneliyle aynı imzalar)
# --------------------------------------------------------------------------


def _to_output(mask: np.ndarray, like: np.ndarray) -> np.ndarray:
    return match_layout(np.where(mask, 255, 0).astype(np.uint8), like)


def _prepare(image, binarize, thresh, foreground, shape, width, height, anchor_x, anchor_y):
    se = structuring_element(shape, width, height, anchor_x, anchor_y)
    return to_binary(image, binarize, thresh, foreground), se


def erode(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
          shape: str = "rect", width: int = 3, height: int = 3, anchor_x: int = -1, anchor_y: int = -1,
          iterations: int = 1) -> np.ndarray:
    a, se = _prepare(image, binarize, thresh, foreground, shape, width, height, anchor_x, anchor_y)
    return _to_output(binary_erosion(a, se, iterations), image)


def dilate(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
           shape: str = "rect", width: int = 3, height: int = 3, anchor_x: int = -1, anchor_y: int = -1,
           iterations: int = 1) -> np.ndarray:
    a, se = _prepare(image, binarize, thresh, foreground, shape, width, height, anchor_x, anchor_y)
    return _to_output(binary_dilation(a, se, iterations), image)


def opening(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
            shape: str = "rect", width: int = 3, height: int = 3, anchor_x: int = -1,
            anchor_y: int = -1) -> np.ndarray:
    a, se = _prepare(image, binarize, thresh, foreground, shape, width, height, anchor_x, anchor_y)
    return _to_output(binary_opening(a, se), image)


def closing(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
            shape: str = "rect", width: int = 3, height: int = 3, anchor_x: int = -1,
            anchor_y: int = -1) -> np.ndarray:
    a, se = _prepare(image, binarize, thresh, foreground, shape, width, height, anchor_x, anchor_y)
    return _to_output(binary_closing(a, se), image)


def gradient(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
             shape: str = "rect", width: int = 3, height: int = 3, anchor_x: int = -1,
             anchor_y: int = -1) -> np.ndarray:
    a, se = _prepare(image, binarize, thresh, foreground, shape, width, height, anchor_x, anchor_y)
    return _to_output(binary_gradient(a, se), image)


def hit_or_miss(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
                pattern: str = "corner", rotation: str = "0", display: str = "matches") -> np.ndarray:
    """Hit-or-miss dönüşümü. ``matches``: yalnızca eşleşen konumlar (255);
    ``overlay``: eşleşmeler 255, ön plan şekli koyu gri (70), arka plan 0."""
    if display not in HMT_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be 'matches' or 'overlay' (got {display!r}).")
    kernel = hmt_kernel(pattern, rotation)
    a = to_binary(image, binarize, thresh, foreground)
    hits = hit_or_miss_mask(a, kernel)
    if display == "matches":
        return _to_output(hits, image)
    out = np.where(hits, 255, np.where(a, OVERLAY_SHAPE_LEVEL, 0)).astype(np.uint8)
    return match_layout(out, image)
