"""Uzamsal filtreleme (Gonzalez & Woods, Chapter 3 — smoothing ve sharpening filtreleri).

Bir uzamsal filtre, her pikselin yeni değerini çevresindeki k×k komşuluktan
hesaplar. Doğrusal filtrelerde bu bir korelasyondur:

    g(x, y) = Σ_s Σ_t  w(s, t) · f(x + s, y + t)

(Burada kullanılan tüm çekirdekler simetrik veya antisimetrik olduğundan
korelasyon ile konvolüsyon aynı / yalnızca işaret farkıyla aynı sonucu verir;
türevlerde işaret kuralı aşağıda açıkça belirtilmiştir.)

Ortak kurallar
--------------
* Kenarlar: görüntü sınırında komşuluk, sınır pikseli tekrarlanmadan aynalanır
  (``BORDER_REFLECT_101``: …c b | a b c…). Median filtre OpenCV gereği sınır
  pikselini tekrarlar (``BORDER_REPLICATE``).
* Yumuşatma filtreleri (mean, Gaussian, median) her renk kanalına ayrı uygulanır;
  doğrusal bir ortalama renk tonunu kaydırmaz. Ara sonuç float64'tür ve en sona
  tek bir yuvarlama yapılır.
* Türev filtreleri (Sobel X/Y, Laplacian) renk kanallarına ayrı ayrı *uygulanmaz*:
  kanal başına kenar haritaları, parlaklığı aynı ama tonu farklı bölgeler arasında
  sahte kenarlar üretir ve kanalları birleştirmenin tek doğru yolu yoktur. Bunun
  yerine BT.601 parlaklığı (luma) üzerinde türev alınır.
* Türevler işaretlidir. Yanıt float64'te hesaplanır ve uint8'e doğrudan
  dönüştürülmez (negatif değerler yok olurdu); ``display_response`` ile belgeli
  biçimde görüntülenebilir hale getirilir.
* Alfa kanalı korunur, boyutlar korunur, girdi dizisi değiştirilmez.
"""

from __future__ import annotations

from numbers import Integral, Real

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
from processing.registry import AlgorithmParameterError

BORDER = cv2.BORDER_REFLECT_101
SOBEL_APERTURES = (1, 3, 5, 7)  # cv2.Sobel'in desteklediği çekirdek boyutları

# Gonzalez Fig. 3.45: Laplacian çekirdekleri (merkez negatif)
LAPLACIAN_KERNELS = {
    "4": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64),
    "8": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64),
}

# --------------------------------------------------------------------------
# Parametre doğrulama
# --------------------------------------------------------------------------


def validate_ksize(ksize, name: str = "Kernel size", minimum: int = 1,
                   allowed: tuple[int, ...] | None = None) -> int:
    """Pozitif tek tamsayı çekirdek boyutunu doğrular ve int olarak döndürür."""
    if isinstance(ksize, bool) or ksize is None:
        raise AlgorithmParameterError(f"{name} must be an odd integer (got {ksize!r}).")
    if isinstance(ksize, Real) and not isinstance(ksize, Integral):
        if not float(ksize).is_integer():
            raise AlgorithmParameterError(f"{name} must be an odd integer (got {ksize!r}).")
        ksize = int(ksize)
    if not isinstance(ksize, Integral):
        raise AlgorithmParameterError(f"{name} must be an odd integer (got {ksize!r}).")
    ksize = int(ksize)
    if allowed is not None and ksize not in allowed:
        raise AlgorithmParameterError(f"{name} must be one of {', '.join(map(str, allowed))} (got {ksize}).")
    if ksize < minimum or ksize % 2 == 0:
        raise AlgorithmParameterError(f"{name} must be an odd integer ≥ {minimum} (got {ksize}).")
    return ksize


def validate_sigma(sigma, name: str = "Sigma X") -> float:
    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        raise AlgorithmParameterError(f"{name} must be a positive number (got {sigma!r}).")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise AlgorithmParameterError(f"{name} must be a positive number (got {sigma}).")
    return sigma


# --------------------------------------------------------------------------
# Yumuşatma (smoothing) filtreleri
# --------------------------------------------------------------------------


def _smooth_linear(image: np.ndarray, op) -> np.ndarray:
    """Doğrusal filtreyi float64'te renk kanallarına uygular, sonda bir kez yuvarlar."""
    check_image(image)
    color, alpha = split_alpha(image)
    filtered = op(color.astype(np.float64))
    return merge_alpha(round_to_uint8(filtered).reshape(color.shape), alpha)


def mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Normalize kutu (ortalama) filtresi: w(s, t) = 1 / k² olan k×k çekirdek.

    Her piksel, k×k komşuluğunun aritmetik ortalaması olur. k büyüdükçe
    yumuşatma güçlenir; kenarlar da aynı oranda bulanıklaşır.
    """
    k = validate_ksize(ksize)
    return _smooth_linear(image, lambda f: cv2.blur(f, (k, k), borderType=BORDER))


def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """1-B Gauss çekirdeği: w(i) ∝ exp(−i² / 2σ²), i = −(k−1)/2 … (k−1)/2, Σ w = 1."""
    k, s = validate_ksize(ksize), validate_sigma(sigma)
    return cv2.getGaussianKernel(k, s, cv2.CV_64F).ravel()


def gaussian_filter(image: np.ndarray, ksize: int = 5, sigma_x: float = 1.0) -> np.ndarray:
    """Gauss konvolüsyonu: w(s, t) = K · exp(−(s² + t²) / 2σ²), Σ w = 1.

    Çekirdek ayrılabilir (2-B = 1-B ⊗ 1-B) ve izotropiktir (σ_y = σ_x).
    k, ±3σ'yı kapsamıyorsa (k < 6σ + 1) çekirdek kesilir ve yeniden normalize
    edilir; sonuç yine geçerli bir ağırlıklı ortalamadır.
    """
    k, s = validate_ksize(ksize), validate_sigma(sigma_x)
    return _smooth_linear(image, lambda f: cv2.GaussianBlur(f, (k, k), sigmaX=s, sigmaY=s, borderType=BORDER))


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Doğrusal olmayan sıra istatistiği filtresi: g(x, y) = median{ f(s, t) : (s, t) ∈ S_xy }.

    Tuz-biber gürültüsünde etkilidir: tekil uç değerler sıralamanın uçlarında
    kalır ve medyana ulaşamaz. OpenCV ``medianBlur`` 8-bit görüntüde tek ve
    ≥ 3 çekirdek boyutu gerektirir; renkli görüntüde her kanala ayrı uygulanır.
    """
    check_image(image)
    k = validate_ksize(ksize, minimum=3)
    color, alpha = split_alpha(image)
    filtered = cv2.medianBlur(np.ascontiguousarray(color), k)
    return merge_alpha(filtered.reshape(color.shape), alpha)


# --------------------------------------------------------------------------
# Türev (sharpening) filtreleri — işaretli yanıtlar
# --------------------------------------------------------------------------


def sobel_response(image: np.ndarray, axis: str, ksize: int = 3) -> np.ndarray:
    """Parlaklığın Sobel türevi (float64, işaretli).

    axis="x": G_x ≈ ∂f/∂x — soldan sağa parlaklık artışında pozitif; dikey kenarlara duyarlı.
    axis="y": G_y ≈ ∂f/∂y — yukarıdan aşağıya artışta pozitif; yatay kenarlara duyarlı.
    3×3 çekirdek (x): [[−1, 0, 1], [−2, 0, 2], [−1, 0, 1]]; y bunun transpozudur.
    """
    if axis not in ("x", "y"):
        raise AlgorithmParameterError(f"Sobel axis must be 'x' or 'y' (got {axis!r}).")
    k = validate_ksize(ksize, name="Aperture", allowed=SOBEL_APERTURES)
    dx, dy = (1, 0) if axis == "x" else (0, 1)
    return cv2.Sobel(luminance(image), cv2.CV_64F, dx, dy, ksize=k, borderType=BORDER)


def laplacian_response(image: np.ndarray, neighbours: str = "4") -> np.ndarray:
    """Parlaklığın Laplasyeni ∇²f = ∂²f/∂x² + ∂²f/∂y² (float64, işaretli).

    İkinci türev izotropiktir: her yöndeki yoğunluk değişimine yanıt verir.
    4-komşu çekirdek eksenleri, 8-komşu çekirdek köşegenleri de kapsar.
    Merkez negatif olduğundan bir basamak kenarın koyu tarafında pozitif,
    açık tarafında negatif yanıt oluşur; kenar sıfır geçişindedir.
    """
    if neighbours not in LAPLACIAN_KERNELS:
        raise AlgorithmParameterError(f"Laplacian kernel must be '4' or '8' neighbours (got {neighbours!r}).")
    return cv2.filter2D(luminance(image), cv2.CV_64F, LAPLACIAN_KERNELS[neighbours], borderType=BORDER)


DISPLAY_MODES = ("abs", "signed")


def display_response(response: np.ndarray, mode: str = "abs") -> np.ndarray:
    """İşaretli türev yanıtını görüntülenebilir uint8'e çevirir.

    * ``abs``    — |R| · 255 / max|R|: büyüklük, en güçlü yanıt 255 olacak şekilde
                   doğrusal ölçeklenir (işaret atılır, ama kaybolmaz: ±R aynı parlaklıkta).
    * ``signed`` — 127.5 + 127.5 · R / max|R|: sıfır yanıt orta gri (128), pozitif
                   açık, negatif koyu. Laplasyende kenarlar sıfır geçişleri olarak görünür.

    Ölçek görüntüye görelidir (max|R|), yani farklı görüntülerin parlaklıkları
    doğrudan karşılaştırılamaz. Yanıt her yerde sıfırsa (sabit görüntü) bölme
    yapılmaz: ``abs`` siyah, ``signed`` düz orta gri döner.
    """
    if mode not in DISPLAY_MODES:
        raise AlgorithmParameterError(f"Display mode must be 'abs' or 'signed' (got {mode!r}).")
    response = np.asarray(response, dtype=np.float64)
    peak = float(np.max(np.abs(response))) if response.size else 0.0
    if peak <= 1e-12:
        return np.full(response.shape, 0 if mode == "abs" else 128, dtype=np.uint8)
    # Önce çarp, sonra böl: 255/peak gibi ikili sistemde tam ifade edilemeyen bir
    # oranı önceden hesaplamak, .5 sınırındaki değerleri yanlış tarafa yuvarlayabilir.
    if mode == "abs":
        return round_to_uint8(np.abs(response) * 255.0 / peak)
    return round_to_uint8(127.5 + response * 127.5 / peak)


def sobel_x(image: np.ndarray, ksize: int = 3, display: str = "abs") -> np.ndarray:
    """Yatay türev G_x (dikey kenarlar), görüntülenebilir biçimde; girdinin kanal düzeninde."""
    return match_layout(display_response(sobel_response(image, "x", ksize), display), image)


def sobel_y(image: np.ndarray, ksize: int = 3, display: str = "abs") -> np.ndarray:
    """Dikey türev G_y (yatay kenarlar), görüntülenebilir biçimde; girdinin kanal düzeninde."""
    return match_layout(display_response(sobel_response(image, "y", ksize), display), image)


def laplacian(image: np.ndarray, neighbours: str = "4", display: str = "abs") -> np.ndarray:
    """Laplasyen yanıtı ∇²f, görüntülenebilir biçimde; girdinin kanal düzeninde."""
    return match_layout(display_response(laplacian_response(image, neighbours), display), image)
