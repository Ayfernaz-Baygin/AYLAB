"""Görüntü bölütleme — COMP 4360 Chapter 5 (slayt 7–17) ve Chapter 5 laboratuvarı.

Paket 1: eşikleme (thresholding). Sıra ve terimler dersin sırasını izler:

    Manual / fixed threshold        g = 1  ⇔  f(x, y) > T                    (slayt 7, 10)
    Automatic global threshold      T_k+1 = (μ₁ + μ₂) / 2,  |ΔT| < ΔT₀ olana dek   (slayt 11)
    Otsu's method                   T* = argmax_k σ²_B(k)                      (slayt 12–13)
    Adaptive mean                   T(x, y) = b×b bloğun ortalaması − C         (slayt 14–15)
    Adaptive Gaussian               T(x, y) = Gauss ağırlıklı ortalama − C      (slayt 14–15)
    Moving averages                 g = 1  ⇔  f > c · m,  m = zikzak taramada son n pikselin ortalaması (slayt 16)
    Niblack                         T(x, y) = m − k · s                        (slayt 17)
    Sauvola                         T(x, y) = m · (1 + k · (s / R − 1))         (slayt 17)

Formüller, varsayılanlar ve sınır kuralı laboratuvar sayfasının (Chapter 5.html)
kendi hesabından alınmıştır; slaytlar bu yöntemler için yalnızca örnek görüntü verir.

Paket 2: kenar tabanlı bölütleme (slayt 18–28; dosyanın ikinci bölümü) — discontinuity
(f′, f″), point / line detection, gradient, Roberts, Sobel, Prewitt, Laplacian, LoG, Canny,
edge linking (Gonzalez yerel işleme), boundary extraction (Moore sınır izleme), Hough.

Paket 3: bölge tabanlı bölütleme (slayt 29–38; dosyanın üçüncü bölümü) — region growing,
region splitting & merging (quadtree), watershed, marker-controlled watershed.

Uygulama kuralları
------------------
* **Giriş:** gri girdide değerin kendisi; renkli girdide BT.601 parlaklığı
  Y = 0.299R + 0.587G + 0.114B, en yakın tamsayıya yuvarlanmış (diğer AYLAB
  modülleriyle aynı). Alfa eşiklemeye katılmaz; çıktıya aynen geri eklenir.
* **Karar:** laboratuvardaki gibi ``f > T`` → ön plan. Ön plan DARK seçilirse
  sonuç tümleyendir (``f ≤ T``): aynı bölümleme, roller yer değiştirir. Eşitlik
  (f = T) her zaman ``f ≤ T`` tarafındadır.
* **Çıktı:** ikili maske, ön plan 255 / arka plan 0, uint8; girdinin kanal
  düzeninde (BGR → üç eşit kanal). Girdi dizisi değiştirilmez.
* **Pencere sınırı (yerel yöntemler):** laboratuvardaki ``refl`` kuralı —
  kenar pikseli tekrarlanmadan yansıtma (reflect-101: … 2 1 | 0 1 2 … ;
  1 piksellik boyutta tek değer tekrarlanır). Pencere görüntüden büyük olabilir.
* **Sayısal kesinlik:** kutu ortalaması ve yerel varyans tamsayı toplamlarından
  (integral görüntü) tam olarak hesaplanır; T(x, y) yüzeyi 1e-9'a yuvarlanır.
  Böylece düz (tekdüze) bölgelerde kayan nokta gürültüsü kararı değiştirmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np

from processing import morphology as _morph
from processing.common import check_image, luminance, match_layout, round_to_uint8
from processing.registry import AlgorithmParameterError, ProcessResult

FOREGROUND_MODES = ("bright", "dark")
ITERATIVE_INIT = ("mean", "custom")

# Parametre sınırları — laboratuvar kontrolleriyle aynı (catalog.py bu değerleri kullanır)
BLOCK_RANGE = (3, 51)            # adaptive b ve Niblack/Sauvola penceresi; tek sayı
C_RANGE = (-10, 30)              # adaptive C
K_RANGE = (0.0, 0.6)             # Niblack / Sauvola k
R_RANGE = (32, 255)              # Sauvola R
MOVING_N_RANGE = (5, 80)         # moving averages n
MOVING_C_RANGE = (0.2, 1.0)      # moving averages c
DELTA_T_RANGE = (0.1, 5.0)       # automatic global threshold ΔT
T0_RANGE = (1, 254)              # automatic global threshold T₀ (chosen value)
MAX_ITERATIONS = 100             # laboratuvardaki üst sınır


# --------------------------------------------------------------------------
# Rapor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdReport:
    """Eşikleme özeti: yöntem, eşik (global sayı veya T(x, y) istatistiği) ve ön plan oranı."""

    method: str
    threshold: str               # "T = 127" veya "T(x, y) min … max, mean …"
    foreground: str              # "bright" | "dark"
    foreground_px: int
    total_px: int
    extra: str = ""

    @property
    def foreground_ratio(self) -> float:
        return self.foreground_px / self.total_px

    def summary(self) -> str:
        rule = "f > T" if self.foreground == "bright" else "f ≤ T"
        return (f"{self.method}: {self.threshold}; foreground ({rule}) {self.foreground_px} px "
                f"= {100 * self.foreground_ratio:.1f} %{self.extra}")


# --------------------------------------------------------------------------
# Doğrulama ve ortak yardımcılar
# --------------------------------------------------------------------------


def _int(value, name: str, lo: int, hi: int, odd: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, float)) or \
            (isinstance(value, float) and not value.is_integer()):
        raise AlgorithmParameterError(f"{name} must be an integer (got {value!r}).")
    v = int(value)
    if not lo <= v <= hi:
        raise AlgorithmParameterError(f"{name} must be in {lo}…{hi} (got {v}).")
    if odd and v % 2 == 0:
        raise AlgorithmParameterError(f"{name} must be odd so the window has a centre pixel (got {v}).")
    return v


def _real(value, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
        raise AlgorithmParameterError(f"{name} must be a finite number (got {value!r}).")
    v = float(value)
    if not lo <= v <= hi:
        raise AlgorithmParameterError(f"{name} must be in {lo:g}…{hi:g} (got {v:g}).")
    return v


def _foreground(value) -> str:
    if value not in FOREGROUND_MODES:
        raise AlgorithmParameterError(f"Foreground must be 'bright' or 'dark' (got {value!r}).")
    return value


def gray_input(image: np.ndarray) -> np.ndarray:
    """Eşiklenecek yoğunluk f (uint8, H×W): gri değer veya yuvarlanmış BT.601 Y."""
    check_image(image)
    return round_to_uint8(luminance(image))


def _clean(t: np.ndarray) -> np.ndarray:
    """T(x, y) yüzeyindeki kayan nokta artığını temizler (düz bölgede T = tam değer)."""
    return np.round(t, 9)


def _decide(f: np.ndarray, t, foreground: str) -> np.ndarray:
    bright = f > t
    return bright if foreground == "bright" else ~bright


def _finish(image, mask, method, threshold, foreground, extra="") -> ProcessResult:
    out = match_layout(np.where(mask, 255, 0).astype(np.uint8), image)
    return ProcessResult(out, ThresholdReport(method, threshold, foreground, int(mask.sum()), int(mask.size), extra))


def _surface_text(t: np.ndarray) -> str:
    return f"T(x, y) {t.min():.2f} … {t.max():.2f}, mean {t.mean():.2f}"


def reflect_index(i: np.ndarray, n: int) -> np.ndarray:
    """Laboratuvardaki ``refl``: reflect-101 (kenar tekrarlanmaz); n = 1 → 0. Çok katlı yansıma da desteklenir."""
    i = np.asarray(i)
    if n == 1:
        return np.zeros_like(i)
    period = 2 * n - 2
    i = np.mod(i, period)
    return np.where(i < n, i, period - i)


def _pad_reflect(f: np.ndarray, r: int) -> np.ndarray:
    h, w = f.shape
    rows = reflect_index(np.arange(-r, h + r), h)
    cols = reflect_index(np.arange(-r, w + r), w)
    return f[np.ix_(rows, cols)]


def _box_sums(f: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """b×b penceredeki Σf ve Σf² (int64, tam) — reflect-101 sınırıyla, integral görüntüden."""
    r = b // 2
    p = _pad_reflect(f.astype(np.int64), r)
    h, w = f.shape

    def window_sum(a):
        ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
        ii[1:, 1:] = a.cumsum(0).cumsum(1)
        return ii[b:b + h, b:b + w] - ii[:h, b:b + w] - ii[b:b + h, :w] + ii[:h, :w]
    return window_sum(p), window_sum(p * p)


def local_mean_std(f: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """b×b penceredeki ortalama m ve standart sapma s (nüfus: s² = E[f²] − m²)."""
    s1, s2 = _box_sums(f, b)
    n = b * b
    m = s1 / n
    var = (n * s2 - s1 * s1) / (n * n)          # tamsayı pay: düz pencerede tam olarak 0
    return m, np.sqrt(np.maximum(var, 0.0))


def gaussian_sigma(b: int) -> float:
    """OpenCV / laboratuvar ``cvSigma``: σ = 0.3 · ((b − 1) · 0.5 − 1) + 0.8."""
    return 0.3 * ((b - 1) * 0.5 - 1) + 0.8


def gaussian_kernel_1d(b: int) -> np.ndarray:
    """Laboratuvar ``gaussK1``: exp(−i² / 2σ²), i = −r … r, toplamı 1'e normalize."""
    r = b // 2
    sigma = gaussian_sigma(b)
    k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma * sigma))
    return k / k.sum()


def local_gaussian_mean(f: np.ndarray, b: int) -> np.ndarray:
    """Ayrılabilir Gauss ağırlıklı ortalama (önce satır, sonra sütun), reflect-101 sınırı."""
    r = b // 2
    k = gaussian_kernel_1d(b)
    p = _pad_reflect(f.astype(np.float64), r)
    h, w = f.shape
    rows = sum(k[i] * p[:, i:i + w] for i in range(b))        # (h + 2r) × w
    return sum(k[i] * rows[i:i + h, :] for i in range(b))


# --------------------------------------------------------------------------
# Global eşikler
# --------------------------------------------------------------------------


def histogram(f: np.ndarray) -> np.ndarray:
    return np.bincount(f.ravel(), minlength=256).astype(np.int64)


def otsu_analysis(hist: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Laboratuvar ``otsu``: her k için σ²_B(k) = (m_G·ω₀ − m)² / (ω₀(1 − ω₀)); T* ilk en büyük.

    Döndürür: (T*, η = σ²_B(T*) / σ²_G, σ²_B dizisi). Tekdüze histogramda σ²_B ≡ 0 → T* = 0, η = 0.
    """
    n = float(hist.sum())
    p = [float(v) / n for v in hist]
    m_g = 0.0
    for i in range(256):
        m_g += i * p[i]
    sb = np.zeros(256)
    w = m = 0.0
    best, t = -1.0, 0
    for k in range(256):                     # laboratuvarla aynı sıra ve aynı eşitlik kuralı
        w += p[k]
        m += k * p[k]
        den = w * (1 - w)
        sb[k] = (m_g * w - m) ** 2 / den if den > 1e-12 else 0.0
        if sb[k] > best + 1e-9:
            best, t = sb[k], k
    sg = 0.0
    for i in range(256):
        sg += (i - m_g) ** 2 * p[i]
    return t, (best / sg if sg else 0.0), sb


@dataclass(frozen=True)
class Iteration:
    t: float          # T_k
    mu1: float        # f > T_k ortalaması (sınıf boşsa T_k)
    mu2: float        # f ≤ T_k ortalaması (sınıf boşsa T_k)
    t_next: float     # (μ₁ + μ₂) / 2


def iterative_threshold(hist: np.ndarray, t0: float, delta: float) -> list[Iteration]:
    """Laboratuvar ``iterT``: T_k+1 = (μ₁ + μ₂) / 2, |T_k+1 − T_k| < ΔT olunca durur (en fazla 100 adım).

    Son eşik ``its[-1].t_next``. Boş sınıfın ortalaması T_k alınır (laboratuvarla aynı).
    """
    levels = np.arange(256)
    its: list[Iteration] = []
    t = float(t0)
    for _ in range(MAX_ITERATIONS):
        above = levels > t
        n1, n2 = int(hist[above].sum()), int(hist[~above].sum())
        s1, s2 = int((levels * hist)[above].sum()), int((levels * hist)[~above].sum())
        mu1 = s1 / n1 if n1 else t
        mu2 = s2 / n2 if n2 else t
        tn = (mu1 + mu2) / 2
        its.append(Iteration(t, mu1, mu2, tn))
        if abs(tn - t) < delta:
            break
        t = tn
    return its


# --------------------------------------------------------------------------
# Yerel eşikler
# --------------------------------------------------------------------------


def moving_average_surface(f: np.ndarray, n: int) -> np.ndarray:
    """Laboratuvar ``movingAvg``: satırlar zikzak taranır (çift satır → sağa, tek satır → sola), tarama
    satırlar arasında kesintisiz sürer; m(k) = son n pikselin ortalaması (ilk n − 1 pikselde mevcut olanların)."""
    h, w = f.shape
    order = np.arange(h * w).reshape(h, w)
    order[1::2] = order[1::2, ::-1]
    order = order.ravel()                              # tarama sırası → piksel indisi
    z = f.ravel()[order].astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(z)))
    k = np.arange(1, z.size + 1)
    lo = np.maximum(k - n, 0)
    m_scan = (cs[k] - cs[lo]) / (k - lo)
    m = np.empty(h * w)
    m[order] = m_scan
    return m.reshape(h, w)


# --------------------------------------------------------------------------
# Kayıtlı işlevler (AYLAB parametre paneliyle aynı imzalar)
# --------------------------------------------------------------------------


def manual_threshold(image: np.ndarray, thresh: int = 127, foreground: str = "bright") -> ProcessResult:
    """Manual / fixed threshold: g = 255 ⇔ f > T (BRIGHT)."""
    t = _int(thresh, "Threshold T", 0, 255)
    fg = _foreground(foreground)
    f = gray_input(image)
    return _finish(image, _decide(f, t, fg), "Manual threshold", f"T = {t}", fg)


def automatic_threshold(image: np.ndarray, init: str = "mean", t0: int = 127, delta: float = 0.5,
                        foreground: str = "bright") -> ProcessResult:
    """Automatic global threshold (slayt 11): T₀ → böl → μ₁, μ₂ → T = (μ₁ + μ₂)/2, T sabitlenene dek."""
    if init not in ITERATIVE_INIT:
        raise AlgorithmParameterError(f"Initial T₀ must be 'mean' or 'custom' (got {init!r}).")
    t0v = _int(t0, "T₀", *T0_RANGE)
    d = _real(delta, "ΔT", *DELTA_T_RANGE)
    fg = _foreground(foreground)
    f = gray_input(image)
    start = float(f.mean()) if init == "mean" else float(t0v)
    its = iterative_threshold(histogram(f), start, d)
    tf = its[-1].t_next
    converged = abs(its[-1].t_next - its[-1].t) < d
    extra = (f"; T₀ = {start:.2f} ({'image mean' if init == 'mean' else 'chosen'}), {len(its)} iteration"
             f"{'s' if len(its) != 1 else ''}, μ₁ = {its[-1].mu1:.2f}, μ₂ = {its[-1].mu2:.2f}"
             + ("" if converged else f"; stopped at {MAX_ITERATIONS} iterations (not converged)"))
    return _finish(image, _decide(f, tf, fg), "Automatic global threshold", f"T = {tf:.2f}", fg, extra)


def otsu_threshold(image: np.ndarray, foreground: str = "bright") -> ProcessResult:
    """Otsu's method: sınıflar arası varyansı en büyük yapan T* (yalnızca histogramdan)."""
    fg = _foreground(foreground)
    f = gray_input(image)
    t, eta, _ = otsu_analysis(histogram(f))
    extra = f"; separability η = {eta:.3f}" + ("; uniform image: σ²_B ≡ 0, T* = 0" if eta == 0 else "")
    return _finish(image, _decide(f, t, fg), "Otsu's method", f"T* = {t}", fg, extra)


def _adaptive(image, block, c, foreground, gaussian: bool) -> ProcessResult:
    b = _int(block, "Block size b", *BLOCK_RANGE, odd=True)
    cv = _int(c, "C", *C_RANGE)
    fg = _foreground(foreground)
    f = gray_input(image)
    m = local_gaussian_mean(f, b) if gaussian else local_mean_std(f, b)[0]
    t = _clean(m - cv)
    name = "Adaptive Gaussian threshold" if gaussian else "Adaptive mean threshold"
    extra = f"; b = {b}" + (f" (σ = {gaussian_sigma(b):.2f})" if gaussian else "") + f", C = {cv}"
    return _finish(image, _decide(f, t, fg), name, _surface_text(t), fg, extra)


def adaptive_mean_threshold(image: np.ndarray, block: int = 15, c: int = 7, foreground: str = "bright") -> ProcessResult:
    """T(x, y) = b×b bloğun ortalaması − C (laboratuvar ``localMean``, kutu filtresi)."""
    return _adaptive(image, block, c, foreground, gaussian=False)


def adaptive_gaussian_threshold(image: np.ndarray, block: int = 15, c: int = 7, foreground: str = "bright") -> ProcessResult:
    """T(x, y) = b×b bloğun Gauss ağırlıklı ortalaması − C, σ = 0.3((b − 1)/2 − 1) + 0.8."""
    return _adaptive(image, block, c, foreground, gaussian=True)


def moving_average_threshold(image: np.ndarray, n: int = 20, c: float = 0.5, foreground: str = "bright") -> ProcessResult:
    """Moving averages (slayt 16): g = 255 ⇔ f > c · m, m zikzak taramada son n pikselin ortalaması."""
    nv = _int(n, "n", *MOVING_N_RANGE)
    cv = _real(c, "c", *MOVING_C_RANGE)
    fg = _foreground(foreground)
    f = gray_input(image)
    t = _clean(cv * moving_average_surface(f, nv))
    return _finish(image, _decide(f, t, fg), "Moving averages", _surface_text(t), fg,
                   f"; n = {nv}, c = {cv:g}, zigzag scan")


def _window_params(window, k):
    return _int(window, "Window", *BLOCK_RANGE, odd=True), _real(k, "k", *K_RANGE)


def niblack_threshold(image: np.ndarray, window: int = 15, k: float = 0.2, foreground: str = "bright") -> ProcessResult:
    """Niblack: T(x, y) = m − k · s (m, s: pencerenin ortalaması ve standart sapması)."""
    b, kv = _window_params(window, k)
    fg = _foreground(foreground)
    f = gray_input(image)
    m, s = local_mean_std(f, b)
    t = _clean(m - kv * s)
    return _finish(image, _decide(f, t, fg), "Niblack", _surface_text(t), fg, f"; window {b}×{b}, k = {kv:g}")


def sauvola_threshold(image: np.ndarray, window: int = 15, k: float = 0.2, r: int = 128,
                      foreground: str = "bright") -> ProcessResult:
    """Sauvola: T(x, y) = m · (1 + k · (s / R − 1))."""
    b, kv = _window_params(window, k)
    rv = _int(r, "R", *R_RANGE)
    fg = _foreground(foreground)
    f = gray_input(image)
    m, s = local_mean_std(f, b)
    t = _clean(m * (1 + kv * (s / rv - 1)))
    return _finish(image, _decide(f, t, fg), "Sauvola", _surface_text(t), fg, f"; window {b}×{b}, k = {kv:g}, R = {rv}")


# ==========================================================================
# Paket 2 — Kenar tabanlı bölütleme (slayt 18–28, laboratuvar "Edge-based")
# ==========================================================================
#
# Laboratuvarın ortak kuralları (JS ``conv``, ``refl``, ``dispFn``, ``dirRGBA``):
# * Çekirdekler KORELASYON olarak uygulanır (çevrilmez): R(x, y) = Σ w(s, t) · f(x + s, y + t).
# * Sınır: reflect-101 (yukarıdaki ``reflect_index``).
# * Eksenler: x sağa (sütun), y aşağı (satır). Slaytlar Gonzalez'in kuralını (x aşağı) kullanır;
#   bu yüzden slaytın ilk Prewitt/Sobel çekirdeği burada w_y'dir (laboratuvarla aynı).
# * Görüntüleme: signed → 128 + 127·v / max|v|;  minmax → 255·(v − min)/(max − min);
#   linmax → 255·v / max;  sonra en yakın tamsayı (yarım yukarı, JS Math.round).
# * Yön α = atan2(g_y, g_x); renk: ton = (α + π)/2π, doygunluk 0.85, açıklık 0.5·min(1, 2.2·M/max M).
#
# Laboratuvarda olmayan, standart uygulama tercihleri yardım metinlerinde ayrıca belirtilir.


GRADIENT_DISPLAYS = ("magnitude", "gx", "gy", "direction", "edges")
MAGNITUDE_MODES = ("l2", "abs")
SMOOTHING_MODES = ("none", "gauss5")
LINE_DIRECTIONS = ("horizontal", "p45", "vertical", "m45")
LAPLACIAN_KERNELS = ("4", "8")

# Slayt 20–23 / laboratuvar çekirdekleri (3×3, satır = y, sütun = x)
POINT_KERNEL = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
LINE_KERNELS = {
    "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
    "p45": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]),      # ╲ (Gonzalez +45°, x aşağı)
    "vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
    "m45": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),      # ╱
}
LAPLACIAN = {"4": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
             "8": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])}
GRADIENT_OPERATORS = {
    # ileri fark (slayt 19 f′ = f(x+1) − f(x), 2-B'de iki eksen) — 3×3 içine gömülü
    "difference": (np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]), np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])),
    # Roberts 2×2 (slayt 23), laboratuvardaki gibi 3×3 pencerenin merkezine gömülü
    "roberts": (np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]), np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])),
    "prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])),
    "sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])),
}
OPERATOR_NAMES = {"difference": "Gradient (forward difference)", "roberts": "Roberts", "prewitt": "Prewitt", "sobel": "Sobel"}

CANNY_DISPLAYS = ("edges", "nms", "double", "magnitude", "smoothed")
LINKING_ORIENTATIONS = ("both", "horizontal", "vertical", "angle")
LINKING_DISPLAYS = ("linked", "overlay", "candidates")
BOUNDARY_DISPLAYS = ("overlay", "boundaries")
HOUGH_DISPLAYS = ("lines", "accumulator", "edges")
POINT_DISPLAYS = ("detected", "abs", "signed")
LINE_DISPLAYS = ("detected", "positive", "signed")
LAPLACIAN_DISPLAYS = ("signed", "abs", "zero_crossings")
LOG_DISPLAYS = ("zero_crossings", "signed")
DISCONTINUITY_ORDERS = ("first", "second")
DISCONTINUITY_AXES = ("horizontal", "vertical")
DISCONTINUITY_DISPLAYS = ("signed", "abs")

ORANGE_BGR = (40, 150, 255)
HOUGH_LINE_BGR = (200, 190, 10)          # laboratuvar rgb(10, 190, 200)


@dataclass(frozen=True)
class EdgeReport:
    """Kenar işleminin özeti (durum çubuğu): yöntem + sayısal ayrıntılar."""

    method: str
    detail: str

    def summary(self) -> str:
        return f"{self.method}: {self.detail}"


def _choice(value, name: str, options) -> str:
    if value not in options:
        raise AlgorithmParameterError(f"{name} must be one of {', '.join(options)} (got {value!r}).")
    return value


def js_round(v):
    """JS Math.round: yarım yukarı (np.rint'in çifte yuvarlamasından farklı).

    floor(v) + [v − floor(v) ≥ 0.5] — floor(v + 0.5) DEĞİL: 0.49999999999999994 + 0.5 kayan noktada 1.0
    olur ve yanlışlıkla 1 verir (JS 0 verir)."""
    v = np.asarray(v, dtype=np.float64)
    fl = np.floor(v)
    return fl + (v - fl >= 0.5)


def q8(v) -> np.ndarray:
    """Laboratuvar ``q8``: [0, 255]'e kırp, yarım yukarı yuvarla."""
    return js_round(np.clip(v, 0, 255)).astype(np.uint8)


def correlate(f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Laboratuvar ``conv``: 3×3 (veya k×k) korelasyon, reflect-101 sınır; float64."""
    k = kernel.shape[0]
    r = k // 2
    p = _pad_reflect(f.astype(np.float64), r)
    h, w = f.shape
    out = np.zeros((h, w))
    for t in range(k):
        for s in range(k):
            if kernel[t, s]:
                out += kernel[t, s] * p[t:t + h, s:s + w]
    return out


def gaussian_kernel(sigma: float, k: int | None = None) -> np.ndarray:
    """Laboratuvar ``gaussK1``: k = 2⌈3σ⌉ + 1 (verilmezse), exp(−i²/2σ²), toplam 1."""
    k = k or 2 * int(np.ceil(3 * sigma)) + 1
    r = k // 2
    a = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma * sigma))
    return a / a.sum()


def gaussian_blur(f: np.ndarray, sigma: float, k: int | None = None) -> np.ndarray:
    """Laboratuvar ``gaussBlur`` (ayrılabilir: önce satır, sonra sütun; reflect-101) + ``q8``."""
    a = gaussian_kernel(sigma, k)
    r = a.size // 2
    h, w = f.shape
    p = _pad_reflect(f.astype(np.float64), r)
    rows = sum(a[i] * p[:, i:i + w] for i in range(a.size))
    return q8(sum(a[i] * rows[i:i + h, :] for i in range(a.size)))


def display_signed(v: np.ndarray) -> np.ndarray:
    m = np.abs(v).max() or 1.0
    return q8(128 + v * 127 / m)


def display_minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return q8((v - lo) * 255 / ((hi - lo) or 1.0))


def display_linmax(v: np.ndarray) -> np.ndarray:
    return q8(v * 255 / (float(v.max()) or 1.0))


def direction_bgr(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Laboratuvar ``dirRGBA`` + ``hsl2rgb``: ton = yön, parlaklık = büyüklük (BGR)."""
    m = np.hypot(gx, gy)
    mx = m.max() or 1.0
    h = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    light = 0.5 * np.minimum(1.0, m / mx * 2.2)
    a = 0.85 * np.minimum(light, 1 - light)

    def ch(n):
        k = (n + h * 12) % 12
        return js_round(255 * (light - a * np.maximum(-1, np.minimum(np.minimum(k - 3, 9 - k), 1))))
    return np.dstack([ch(4), ch(8), ch(0)]).astype(np.uint8)          # (B, G, R)


def _gray_out(img: np.ndarray, like: np.ndarray) -> np.ndarray:
    return match_layout(img, like)


def _color_out(bgr: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Renkli görüntüleme çıktısı: BGR; girdi BGRA ise alfa geri eklenir."""
    if like.ndim == 3 and like.shape[2] == 4 and bgr.shape[:2] == like.shape[:2]:
        return np.dstack([bgr, like[..., 3]])
    return bgr


def _binary(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def _frac(value, name="T / max", lo: float = 0.1) -> float:
    """Eşik oranı (T / max). Laboratuvar kontrolleri 0.1 … 1; kaynakta olmayan eşikler 0.05 … 1."""
    return _real(value, name, lo, 1.0)


# ------------------------------------------------------------------ 1) discontinuity (slayt 18–19)
def derivative_1d(f: np.ndarray, order: str, axis: str) -> np.ndarray:
    """Laboratuvar profili: f′(x) = f(x+1) − f(x), f″(x) = f(x+1) + f(x−1) − 2f(x).

    Satır (horizontal) veya sütun (vertical) boyunca. Tanımsız uçlar (f′: son sütun/satır,
    f″: ilk ve son) laboratuvardaki boş (null) değerler gibi 0 bırakılır.
    """
    g = f.astype(np.int64)
    if axis == "vertical":
        g = g.T
    d = np.zeros(g.shape, np.int64)
    if order == "first":
        d[:, :-1] = g[:, 1:] - g[:, :-1]
    else:
        d[:, 1:-1] = g[:, 2:] + g[:, :-2] - 2 * g[:, 1:-1]
    return d.T if axis == "vertical" else d


def discontinuity(image: np.ndarray, order: str = "first", axis: str = "horizontal",
                  display: str = "signed") -> ProcessResult:
    _choice(order, "Derivative", DISCONTINUITY_ORDERS)
    _choice(axis, "Direction", DISCONTINUITY_AXES)
    _choice(display, "Display", DISCONTINUITY_DISPLAYS)
    f = gray_input(image)
    d = derivative_1d(f, order, axis)
    out = display_signed(d) if display == "signed" else display_linmax(np.abs(d))
    name = "f′ = f(x+1) − f(x)" if order == "first" else "f″ = f(x+1) + f(x−1) − 2f(x)"
    detail = (f"{name} along {'rows (x)' if axis == 'horizontal' else 'columns (y)'}; range {int(d.min())} … "
              f"{int(d.max())}, non-zero {int((d != 0).sum())} px")
    return ProcessResult(_gray_out(out, image), EdgeReport("Discontinuity", detail))


# ------------------------------------------------------------------ 2) point detection (slayt 20)
def _detect(a: np.ndarray, t: float) -> tuple[np.ndarray, float]:
    """a ≥ T, T = t · max a. Düz yanıtta (max a = 0) hiçbir piksel işaretlenmez (laboratuvarda T = 0 olurdu)."""
    mx = float(a.max())
    thr = t * mx
    return (a >= thr) & (a > 0), thr


def point_detection(image: np.ndarray, t: float = 0.9, display: str = "detected") -> ProcessResult:
    tv = _frac(t, "T / max |R|")
    _choice(display, "Display", POINT_DISPLAYS)
    f = gray_input(image)
    r = correlate(f, POINT_KERNEL)
    a = np.abs(r)
    det, thr = _detect(a, tv)
    out = {"detected": lambda: _binary(det), "abs": lambda: display_minmax(a), "signed": lambda: display_signed(r)}[display]()
    detail = f"|R| ≥ T = {tv:g} · max |R| = {thr:.1f}; {int(det.sum())} point pixel(s)" + ("" if a.max() else "; flat image, R ≡ 0")
    return ProcessResult(_gray_out(out, image), EdgeReport("Point detection (8-neighbour Laplacian kernel)", detail))


# ------------------------------------------------------------------ 3) line detection (slayt 21)
def line_detection(image: np.ndarray, direction: str = "p45", t: float = 0.6, display: str = "detected") -> ProcessResult:
    _choice(direction, "Kernel", LINE_DIRECTIONS)
    tv = _frac(t, "T / max R")
    _choice(display, "Display", LINE_DISPLAYS)
    f = gray_input(image)
    r = correlate(f, LINE_KERNELS[direction])
    rp = np.maximum(r, 0)
    det, thr = _detect(rp, tv)
    out = {"detected": lambda: _binary(det), "positive": lambda: display_minmax(rp), "signed": lambda: display_signed(r)}[display]()
    name = {"horizontal": "horizontal", "p45": "+45°", "vertical": "vertical", "m45": "−45°"}[direction]
    detail = f"{name} kernel; R⁺ ≥ T = {tv:g} · max R⁺ = {thr:.1f}; {int(det.sum())} line pixel(s)"
    return ProcessResult(_gray_out(out, image), EdgeReport("Line detection", detail))


# ------------------------------------------------------------------ 4–7) gradient operators (slayt 22–24)
def gradient_components(f: np.ndarray, operator: str, smoothing: str = "none") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(girdi, g_x, g_y). ``gauss5``: laboratuvardaki 5×5 Gauss ön yumuşatma (σ = 1.1, sonra q8)."""
    src = gaussian_blur(f, gaussian_sigma(5), 5) if smoothing == "gauss5" else f
    kx, ky = GRADIENT_OPERATORS[operator]
    return src, correlate(src, kx), correlate(src, ky)


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray, mode: str = "l2") -> np.ndarray:
    return np.hypot(gx, gy) if mode == "l2" else np.abs(gx) + np.abs(gy)


def _gradient(image, operator, display, magnitude, smoothing, t) -> ProcessResult:
    _choice(display, "Display", GRADIENT_DISPLAYS)
    _choice(magnitude, "Magnitude", MAGNITUDE_MODES)
    _choice(smoothing, "Pre-smoothing", SMOOTHING_MODES)
    tv = _frac(t, "Edge threshold T / max M", 0.05)
    f = gray_input(image)
    _, gx, gy = gradient_components(f, operator, smoothing)
    m = gradient_magnitude(gx, gy, magnitude)
    edges, thr = _detect(m, tv)
    if display == "direction":
        out = _color_out(direction_bgr(gx, gy), image)
    else:
        out = _gray_out({"magnitude": lambda: display_minmax(m), "gx": lambda: display_signed(gx),
                         "gy": lambda: display_signed(gy), "edges": lambda: _binary(edges)}[display](), image)
    formula = "√(gx² + gy²)" if magnitude == "l2" else "|gx| + |gy|"
    detail = (f"M = {formula}, max {m.max():.1f}, mean {m.mean():.2f}; gx {gx.min():.0f} … {gx.max():.0f}, "
              f"gy {gy.min():.0f} … {gy.max():.0f}; edges M ≥ {tv:g} · max = {thr:.1f}: {int(edges.sum())} px"
              + ("; 5×5 Gaussian first" if smoothing == "gauss5" else ""))
    return ProcessResult(out, EdgeReport(OPERATOR_NAMES[operator], detail))


def gradient(image: np.ndarray, display: str = "magnitude", magnitude: str = "l2", smoothing: str = "none",
             t: float = 0.3) -> ProcessResult:
    """g_x = f(x+1, y) − f(x, y), g_y = f(x, y+1) − f(x, y); M ve α = atan2(g_y, g_x)."""
    return _gradient(image, "difference", display, magnitude, smoothing, t)


def roberts(image: np.ndarray, display: str = "magnitude", magnitude: str = "l2", smoothing: str = "none",
            t: float = 0.3) -> ProcessResult:
    return _gradient(image, "roberts", display, magnitude, smoothing, t)


def sobel(image: np.ndarray, display: str = "magnitude", magnitude: str = "l2", smoothing: str = "none",
          t: float = 0.3) -> ProcessResult:
    return _gradient(image, "sobel", display, magnitude, smoothing, t)


def prewitt(image: np.ndarray, display: str = "magnitude", magnitude: str = "l2", smoothing: str = "none",
            t: float = 0.3) -> ProcessResult:
    return _gradient(image, "prewitt", display, magnitude, smoothing, t)


# ------------------------------------------------------------------ 8–9) Laplacian, LoG (slayt 23–24)
ZC_PAIRS = ((-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1), (1, -1, -1, 1))    # (dx1, dy1, dx2, dy2)


def zero_crossings(lap: np.ndarray, thr: float) -> np.ndarray:
    """Laboratuvar ``zeroCross``: karşılıklı komşu çiftlerinden (yatay, dikey, iki köşegen) en az birinde
    u·v < 0 ve |u − v| > thr. Yalnızca iç pikseller (1 piksellik kenar 0)."""
    h, w = lap.shape
    out = np.zeros((h, w), bool)
    if h < 3 or w < 3:
        return out
    c = (slice(1, h - 1), slice(1, w - 1))
    for dx1, dy1, dx2, dy2 in ZC_PAIRS:
        u = lap[1 + dy1:h - 1 + dy1, 1 + dx1:w - 1 + dx1]
        v = lap[1 + dy2:h - 1 + dy2, 1 + dx2:w - 1 + dx2]
        out[c] |= (u * v < 0) & (np.abs(u - v) > thr)
    return out


def _laplacian_result(image, f, kernel, display, t, name, extra):
    lap = correlate(f, LAPLACIAN[kernel])
    thr = t * float(np.abs(lap).max())
    zc = zero_crossings(lap, thr)
    if display == "zero_crossings":
        out = _binary(zc)
    elif display == "abs":
        out = display_minmax(np.abs(lap))
    else:
        out = display_signed(lap)
    detail = (f"{kernel}-neighbour kernel{extra}; response {lap.min():.0f} … {lap.max():.0f}; zero crossings "
              f"(|u − v| > {t:g} · max |∇²| = {thr:.1f}): {int(zc.sum())} px")
    return ProcessResult(_gray_out(out, image), EdgeReport(name, detail))


def laplacian_edges(image: np.ndarray, kernel: str = "8", display: str = "signed", t: float = 0.1) -> ProcessResult:
    _choice(kernel, "Laplacian kernel", LAPLACIAN_KERNELS)
    _choice(display, "Display", LAPLACIAN_DISPLAYS)
    tv = _real(t, "Zero-crossing threshold", 0.0, 0.5)
    return _laplacian_result(image, gray_input(image), kernel, display, tv, "Laplacian", "")


def laplacian_of_gaussian(image: np.ndarray, sigma: float = 2.0, kernel: str = "8", display: str = "zero_crossings",
                          t: float = 0.1) -> ProcessResult:
    """Marr–Hildreth: G_σ ⋆ f (q8), sonra Laplacian; kenarlar = sıfır geçişleri."""
    sv = _real(sigma, "σ", 0.5, 5.0)
    _choice(kernel, "Laplacian kernel", LAPLACIAN_KERNELS)
    _choice(display, "Display", LOG_DISPLAYS)
    tv = _real(t, "Zero-crossing threshold", 0.0, 0.5)
    k = 2 * int(np.ceil(3 * sv)) + 1
    return _laplacian_result(image, gaussian_blur(gray_input(image), sv), kernel, display, tv,
                             "Laplacian of Gaussian", f", σ = {sv:g} ({k}×{k} Gaussian)")


# ------------------------------------------------------------------ 10) Canny (slayt 25–26)
NMS_NEIGHBOURS = ((-1, 0, 1, 0), (-1, -1, 1, 1), (0, -1, 0, 1), (1, -1, -1, 1))   # sektör 0…3: (dx1, dy1, dx2, dy2)
HYSTERESIS_MAX_STEPS = 3000                                                      # laboratuvardaki sınır


@dataclass(frozen=True)
class CannyStages:
    smoothed: np.ndarray       # uint8
    gx: np.ndarray
    gy: np.ndarray
    magnitude: np.ndarray      # 0–255'e ölçeklenmiş M
    sector: np.ndarray         # 0…3
    nms: np.ndarray
    state: np.ndarray          # 0 / 1 zayıf / 2 güçlü
    edges: np.ndarray          # bool
    steps: int                 # histerezis adımı


def canny_stages(f: np.ndarray, sigma: float, low: float, high: float) -> CannyStages:
    """Laboratuvar ``cannyAll``: (1) Gauss σ (q8), (2) Sobel g_x, g_y; M = √(g_x² + g_y²) 0–255'e ölçeklenir,
    (3) yön 4 sektöre yuvarlanır, iç piksellerde M ≥ iki komşu ise korunur (NMS), (4) M ≥ T₂ güçlü,
    T₁ ≤ M < T₂ zayıf, (5) histerezis: güçlülerden 8-komşulukla zayıflara yayılır (bağlama)."""
    b = gaussian_blur(f, sigma) if sigma > 0 else f
    gx, gy = correlate(b, GRADIENT_OPERATORS["sobel"][0]), correlate(b, GRADIENT_OPERATORS["sobel"][1])
    m = np.hypot(gx, gy)
    mn = m * 255 / (float(m.max()) or 1.0)
    ang = np.degrees(np.arctan2(gy, gx))
    sector = (js_round(((ang + 360) % 180) / 45) % 4).astype(np.int64)
    h, w = f.shape
    nms = np.zeros((h, w))
    if h >= 3 and w >= 3:
        inner = mn[1:-1, 1:-1]
        keep = np.zeros(inner.shape, bool)
        for s, (dx1, dy1, dx2, dy2) in enumerate(NMS_NEIGHBOURS):
            a = mn[1 + dy1:h - 1 + dy1, 1 + dx1:w - 1 + dx1]
            c = mn[1 + dy2:h - 1 + dy2, 1 + dx2:w - 1 + dx2]
            keep |= (sector[1:-1, 1:-1] == s) & (inner >= a) & (inner >= c)
        nms[1:-1, 1:-1] = np.where(keep, inner, 0)
    state = np.where(nms >= high, 2, np.where(nms >= low, 1, 0))
    edges = state == 2
    weak = state == 1
    steps = 0
    k = np.ones((3, 3), np.uint8)
    while steps < HYSTERESIS_MAX_STEPS:
        grown = (cv2.dilate(edges.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0) & weak & ~edges
        if not grown.any():
            break
        edges = edges | grown
        steps += 1
    return CannyStages(b, gx, gy, mn, sector, nms, state, edges, steps)


def _canny_params(sigma, low, high):
    sv = _real(sigma, "Gaussian σ", 0.0, 4.0)
    lo = _int(low, "T₁ (low)", 1, 150)
    hi = _int(high, "T₂ (high)", 2, 255)
    if lo >= hi:
        raise AlgorithmParameterError(f"T₁ (low) must be smaller than T₂ (high) (got {lo} ≥ {hi}).")
    return sv, lo, hi


def canny(image: np.ndarray, sigma: float = 1.4, low: int = 10, high: int = 45, display: str = "edges") -> ProcessResult:
    sv, lo, hi = _canny_params(sigma, low, high)
    _choice(display, "Display", CANNY_DISPLAYS)
    f = gray_input(image)
    st = canny_stages(f, sv, lo, hi)
    if display == "double":
        bgr = np.zeros(f.shape + (3,), np.uint8)
        bgr[st.state == 1] = ORANGE_BGR
        bgr[st.state == 2] = (255, 255, 255)
        out = _color_out(bgr, image)
    else:
        out = _gray_out({"edges": lambda: _binary(st.edges), "nms": lambda: q8(st.nms),
                         "magnitude": lambda: q8(st.magnitude), "smoothed": lambda: st.smoothed}[display](), image)
    n_strong, n_weak = int((st.state == 2).sum()), int((st.state == 1).sum())
    detail = (f"σ = {sv:g}, T₁ = {lo}, T₂ = {hi} (on M scaled to 0–255); strong {n_strong}, weak {n_weak}, "
              f"final edges {int(st.edges.sum())} px ({n_strong} + {int(st.edges.sum()) - n_strong} linked weak, "
              f"{st.steps} hysteresis step{'s' if st.steps != 1 else ''})")
    return ProcessResult(out, EdgeReport("Canny", detail))


# ------------------------------------------------------------------ 11) edge linking (slayt 27; Gonzalez §10.2 yerel işleme)
def fill_row_gaps(g: np.ndarray, k: int) -> np.ndarray:
    """Her satırda, iki 1 arasında kalan ve uzunluğu ≤ K olan 0 dizilerini 1 yapar (Gonzalez, yerel işleme adım 3)."""
    out = g.copy()
    for r in range(g.shape[0]):
        ones = np.flatnonzero(g[r])
        if ones.size < 2:
            continue
        gaps = np.diff(ones) - 1
        for start, gap in zip(ones[:-1], gaps):
            if 0 < gap <= k:
                out[r, start + 1:start + 1 + gap] = True
    return out


def scan_lines(shape: tuple[int, int], theta_deg: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """θ yönüne paralel sayısal doğrular ailesi — Gonzalez adım 4'teki "g'yi θ kadar döndür, satırları tara,
    geri döndür" işleminin yeniden örneklemesiz karşılığı.

    θ: ekranda kenar yönü (0° yatay, 90° dikey, saat yönünün tersine; y aşağı). Kenar vektörü (cos θ, −sin θ).
    Sığ (|cos θ| ≥ |sin θ|): doğru c = {(x, c + round(s·x))}, s = −tan θ; konum x.
    Dik: doğru c = {(c + round(t·y), y)}, t = −cot θ; konum y.
    Her piksel tam olarak bir doğruya düşer (id c = y − round(s·x) ya da x − round(t·y)); bir doğru üzerinde
    ardışık konumlar komşu piksellerdir ve iki görüntü pikseli arasındaki tüm doğru pikselleri görüntü içindedir.
    θ = 0° → satırlar, θ = 90° → sütunlar. Döndürür: (doğru kimliği, konum, sığ mı).
    """
    h, w = shape
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    yy, xx = np.mgrid[0:h, 0:w]
    if abs(c) >= abs(s):
        slope = -s / c
        r = js_round(slope * np.arange(w)).astype(np.int64)        # round(s·x), her sütun için
        return yy - r[None, :], xx, True
    slope = -c / s
    r = js_round(slope * np.arange(h)).astype(np.int64)            # round(t·y), her satır için
    return xx - r[:, None], yy, False


def fill_gaps_along(g: np.ndarray, theta_deg: float, k: int) -> np.ndarray:
    """θ yönündeki her sayısal doğru boyunca, iki aday piksel arasındaki uzunluğu ≤ K (doğru pikseli sayısı)
    0 dizilerini 1 yapar. Açık uçlar (görüntü kenarına kadar uzanan boşluklar) doldurulmaz."""
    out = g.copy()
    if not g.any():
        return out
    line, pos, shallow = scan_lines(g.shape, theta_deg)
    ys, xs = np.nonzero(g)
    li, po = line[ys, xs], pos[ys, xs]
    order = np.lexsort((po, li))
    li, po = li[order], po[order]
    same = li[1:] == li[:-1]
    gap = po[1:] - po[:-1] - 1
    sel = np.flatnonzero(same & (gap > 0) & (gap <= k))
    if not sel.size:
        return out
    h, w = g.shape
    th = np.radians(theta_deg)
    slope = (-np.sin(th) / np.cos(th)) if shallow else (-np.cos(th) / np.sin(th))
    for i in sel:
        p = np.arange(po[i] + 1, po[i + 1])
        q = li[i] + js_round(slope * p).astype(np.int64)
        if shallow:
            out[q, p] = True                                       # (y, x) = (c + round(s·x), x)
        else:
            out[p, q] = True                                       # (y, x) = (y, c + round(t·y))
    return out


def _angle_mask(alpha_deg: np.ndarray, target: float, ta: float) -> np.ndarray:
    """|α − A| ≤ T_A; kenar yönü işaretten bağımsız olduğu için fark 180° modunda alınır:
    d = |((α − A) + 90) mod 180 − 90| ∈ [0, 90] — örn. α = 179° ile A = 0° arasındaki fark 1°'dir."""
    d = np.abs(((alpha_deg - target) + 90) % 180 - 90)
    return d <= ta


def gradient_target(theta_deg: float) -> float:
    """Kenar yönü θ (ekranda, saat yönünün tersine) → kabul edilen gradyan yönü A = 90° − θ (α = atan2(g_y, g_x),
    y aşağı). θ = 0° (yatay kenar) → A = 90°; θ = 90° (dikey) → A = 0°."""
    return (90.0 - theta_deg) % 180.0


def edge_linking(image: np.ndarray, tm: float = 0.3, ta: int = 45, k: int = 25, orientation: str = "both",
                 angle: int = 45, display: str = "linked") -> ProcessResult:
    """Gonzalez yerel işleme ile kenar bağlama: Sobel M, α; g = 1 ⇔ M ≥ T_M (ve M > 0) ve |α − A| ≤ T_A
    (180° modunda); kenar yönü θ boyunca uzunluğu ≤ K boşluklar doldurulur (adım 3–4).
    ``horizontal``: θ = 0°, ``vertical``: θ = 90°, ``both``: ikisinin birleşimi, ``angle``: θ = ``angle``."""
    tmv = _frac(tm, "T_M / max M", 0.05)
    tav = _int(ta, "T_A", 0, 90)
    kv = _int(k, "Gap K", 1, 100)
    _choice(orientation, "Edge orientation", LINKING_ORIENTATIONS)
    av = _int(angle, "Edge angle θ", 0, 179)
    _choice(display, "Display", LINKING_DISPLAYS)
    f = gray_input(image)
    _, gx, gy = gradient_components(f, "sobel")
    m = gradient_magnitude(gx, gy)
    alpha = np.degrees(np.arctan2(gy, gx))
    t_m = tmv * float(m.max())
    strong = (m >= t_m) & (m > 0)
    thetas = {"both": (0, 90), "horizontal": (0,), "vertical": (90,), "angle": (av,)}[orientation]
    cand = np.zeros(f.shape, bool)
    linked = np.zeros(f.shape, bool)
    for theta in thetas:
        g = strong & _angle_mask(alpha, gradient_target(theta), tav)
        cand |= g
        linked |= fill_gaps_along(g, theta, kv)
    if display == "overlay":
        bgr = np.zeros(f.shape + (3,), np.uint8)
        bgr[linked & ~cand] = ORANGE_BGR
        bgr[cand] = (255, 255, 255)
        out = _color_out(bgr, image)
    else:
        out = _gray_out(_binary(linked if display == "linked" else cand), image)
    dirs = ", ".join(f"θ = {t}° (A = {gradient_target(t):g}°)" for t in thetas)
    detail = (f"Sobel M ≥ {tmv:g} · max = {t_m:.1f}, |α − A| ≤ {tav}°, {dirs}, gaps ≤ {kv} px; "
              f"candidates {int(cand.sum())} px, linked {int(linked.sum())} px (+{int((linked & ~cand).sum())} filled)")
    return ProcessResult(out, EdgeReport("Edge linking (local processing)", detail))


# ------------------------------------------------------------------ 12) boundary extraction (slayt 27; Moore izleme)
MOORE_DIRS = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))   # (dy, dx): W NW N NE E SE S SW — saat yönü


def moore_trace(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Gonzalez §11.1 sınır izleme (Moore): b₀ en üst-en sol nokta, c₀ batı komşusu; b'nin 8 komşusu c'den
    başlayarak saat yönünde taranır, ilk 1 → yeni b, ondan önceki 0 → yeni c. b = b₀ ve sıradaki nokta b₁
    olunca durur. Görüntü dışı 0. Döndürür: (satır, sütun) dizisi, b₀ ile başlar, b₀'a dönüşü içermez."""
    h, w = mask.shape

    def on(p):
        return 0 <= p[0] < h and 0 <= p[1] < w and bool(mask[p])

    def step(b, c):
        idx = MOORE_DIRS.index((c[0] - b[0], c[1] - b[1]))
        for j in range(8):
            d = (idx + j) % 8
            p = (b[0] + MOORE_DIRS[d][0], b[1] + MOORE_DIRS[d][1])
            if on(p):
                q = MOORE_DIRS[(d - 1) % 8]
                return p, (b[0] + q[0], b[1] + q[1])
        return None, None

    b0 = start
    b1, c1 = step(b0, (b0[0], b0[1] - 1))
    if b1 is None:
        return [b0]                                   # tek piksellik nesne
    path = [b0, b1]
    b, c = b1, c1
    while True:
        nb, nc = step(b, c)
        if b == b0 and nb == b1:
            break
        path.append(nb)
        b, c = nb, nc
    return path[:-1]                                  # son eleman b₀ (kapanış)


@dataclass(frozen=True)
class BoundaryReport(_morph.ComponentReport):
    """Bileşen tablosu (ComponentReport) + her nesnenin Moore sınır uzunluğu."""

    lengths: tuple[int, ...] = ()
    holes: int = 0

    def summary(self) -> str:
        total = sum(self.lengths)
        return (f"{self.count} boundar{'ies' if self.count != 1 else 'y'} (Moore tracing, 8-connected objects), "
                f"total length {total} steps; {self.holes} enclosed region{'s' if self.holes != 1 else ''} "
                f"(closed contours around holes)")


def boundary_extraction(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
                        display: str = "overlay") -> ProcessResult:
    _choice(display, "Display", BOUNDARY_DISPLAYS)
    a = _morph.to_binary(image, binarize, thresh, foreground)
    cc = _morph.label_components(a, 8)
    boundary = np.zeros(a.shape, bool)
    lengths = []
    flat = cc.labels.ravel()
    firsts = {}
    for i in np.flatnonzero(flat):                       # raster sırasında ilk piksel = en üst-en sol b₀
        firsts.setdefault(int(flat[i]), i)
        if len(firsts) == cc.count:
            break
    paths = []
    for comp in cc.components:
        start = divmod(firsts[comp.label], a.shape[1])
        path = moore_trace(cc.labels == comp.label, start)
        paths.append(path)
        lengths.append(len(path))
        rr, cc_ = zip(*path)
        boundary[list(rr), list(cc_)] = True
    holes = len(_morph.hole_labels(a, 4)[1])
    report = BoundaryReport(cc.labels, cc.components, 8, tuple(lengths), holes)
    if display == "boundaries":
        return ProcessResult(_gray_out(_binary(boundary), image), report)
    f = gray_input(image)
    bgr = np.repeat((f // 3)[:, :, None], 3, axis=2)
    palette = _morph.label_colors(cc.count)
    lab = np.where(boundary, cc.labels, 0)
    bgr[boundary] = palette[lab[boundary]]
    return ProcessResult(_color_out(bgr, image), report)


# ------------------------------------------------------------------ 13) Hough (slayt 27–28)
@dataclass(frozen=True)
class HoughLine:
    rho: int
    theta: int       # derece, −90 … 89
    votes: int


def hough_accumulator(points: np.ndarray, w: int, h: int) -> tuple[np.ndarray, int]:
    """Laboratuvar ``houghSetup``/``houghAdd``: θ = −90°…89° (1°), ρ = round(x cos θ + y sin θ) + D,
    D = ⌈√(W² + H²)⌉, 2D + 1 satır. ``points``: (x, y) dizisi."""
    d = int(np.ceil(np.hypot(w, h)))
    th = np.radians(np.arange(180) - 90)
    acc = np.zeros((2 * d + 1, 180), np.int64)
    if len(points):
        x, y = points[:, 0:1].astype(np.float64), points[:, 1:2].astype(np.float64)
        r = js_round(x * np.cos(th) + y * np.sin(th)).astype(np.int64) + d
        np.add.at(acc, (r.ravel(), np.tile(np.arange(180), len(points))), 1)
    return acc, d


def hough_peaks(acc: np.ndarray, d: int, frac: float, max_lines: int) -> tuple[list[HoughLine], float]:
    """Laboratuvar ``houghPeaks``: v ≥ max(2, frac · max); ±5 (ρ, θ) komşulukta en büyük (eşitlikte
    raster sırasında önce gelen kazanır, θ sarmalanmaz); oy sayısına göre azalan, ilk ``max_lines``."""
    mx = int(acc.max())
    thr = max(2.0, frac * mx)
    rows, cols = acc.shape
    p = np.full((rows + 10, cols + 10), -1, np.int64)
    p[5:-5, 5:-5] = acc
    ok = acc >= thr
    for dr in range(-5, 6):
        for dt in range(-5, 6):
            if not dr and not dt:
                continue
            u = p[5 + dr:5 + dr + rows, 5 + dt:5 + dt + cols]
            earlier = dr < 0 or (dr == 0 and dt < 0)
            ok &= ~((u > acc) | ((u == acc) & earlier))
    r_idx, t_idx = np.nonzero(ok)                      # raster sırası
    peaks = [HoughLine(int(r - d), int(t - 90), int(acc[r, t])) for r, t in zip(r_idx, t_idx)]
    peaks.sort(key=lambda pk: -pk.votes)               # kararlı: eşit oyda raster sırası korunur
    return peaks[:max_lines], thr


def _line_endpoints(rho, theta, w, h):
    th = np.radians(theta)
    c, s = np.cos(th), np.sin(th)
    if abs(s) > 0.5:
        return (0.0, rho / s), (float(w), (rho - w * c) / s)
    return (rho / c, 0.0), ((rho - h * s) / c, float(h))


def hough_lines(image: np.ndarray, low: int = 20, high: int = 50, frac: float = 0.5, max_lines: int = 24,
                display: str = "lines") -> ProcessResult:
    """Kenar noktaları laboratuvar Canny'si (σ = 1.4, T₁, T₂) ile bulunur; her nokta geçen her (ρ, θ)
    doğrusuna oy verir; tepeler doğrulardır."""
    _, lo, hi = _canny_params(1.4, low, high)
    fv = _frac(frac, "Peak threshold")
    nv = _int(max_lines, "Max lines", 1, 60)
    _choice(display, "Display", HOUGH_DISPLAYS)
    f = gray_input(image)
    h, w = f.shape
    edges = canny_stages(f, 1.4, lo, hi).edges
    ys, xs = np.nonzero(edges)
    acc, d = hough_accumulator(np.column_stack([xs, ys]), w, h)
    peaks, thr = hough_peaks(acc, d, fv, nv)
    if display == "accumulator":
        a = display_linmax(acc.astype(np.float64))     # boyut (2D + 1) × 180 — girdiden farklı; alfa eklenmez
        out = a if image.ndim == 2 else (a[:, :, None] if image.shape[2] == 1 else np.repeat(a[:, :, None], 3, axis=2))
    elif display == "edges":
        out = _gray_out(_binary(edges), image)
    else:
        bgr = np.repeat(f[:, :, None], 3, axis=2).copy()
        lw = max(1, int(round(w / 200)))
        for pk in peaks:
            (x0, y0), (x1, y1) = _line_endpoints(pk.rho, pk.theta, w, h)
            if all(np.isfinite(v) and abs(v) < 1e6 for v in (x0, y0, x1, y1)):
                cv2.line(bgr, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), HOUGH_LINE_BGR, lw)
        out = _color_out(bgr, image)
    top = ", ".join(f"(ρ = {pk.rho}, θ = {pk.theta}°, {pk.votes})" for pk in peaks[:3])
    detail = (f"{len(xs)} Canny edge points (σ = 1.4, T₁ = {lo}, T₂ = {hi}) vote; accumulator {acc.shape[0]}×180, "
              f"max {int(acc.max())}; {len(peaks)} line{'s' if len(peaks) != 1 else ''} with ≥ {thr:.1f} votes"
              + (f": {top}" + (" …" if len(peaks) > 3 else "") if peaks else ""))
    return ProcessResult(out, EdgeReport("Hough transform", detail))


# ====================================================================================================
# Paket 3: bölge tabanlı bölütleme (slayt 29–38) — region growing, splitting & merging, watershed,
# marker-controlled watershed. Algoritmalar laboratuvarın kendi kodundan (growAll, quadtree + birleştirme,
# regionalMinima + watershed, viewMWS) aktarılmıştır; slaytlar yalnızca adımları ve Gonzalez şekillerini
# (10.46, 10.48, 10.57, 10.61) verir. Laboratuvar görüntüleri Float32Array'de tuttuğu için yüzey
# hesaplarında ara sonuçlar float32'ye yuvarlanır (aynı değerler, aynı eşitlik kararları).
# Bağlantılı bileşenler (Connected Components) Morphological Processing'de zaten var; burada yeniden
# yazılmaz — tohum gruplarını ve iç işaretleyicileri etiketlemek için ``_morph.label_components`` kullanılır.
# ====================================================================================================
GROW_SEED_MODES = ("point", "brightest")
REGION_CONNECTIVITY = ("8", "4")
GROW_DISPLAYS = ("overlay", "binary", "labels", "difference")
GROW_T_RANGE = (1, 120)                  # laboratuvar kaydırıcısı
GROW_Q_RANGE = (0.05, 3.0)               # laboratuvar "Auto seeds: top %"
GROW_MAX_LAYERS = 5000                   # laboratuvardaki üst sınır (growAll: L < 5000)
SEED_COORD_MAX = 9999
SPLIT_PREDICATES = ("homog", "gonzalez")
SPLIT_MIN_BLOCKS = ("2", "4", "8", "16", "32")
SPLIT_DISPLAYS = ("merged", "split", "labels")
SPLIT_MAX_DEPTH = 12                     # AYLAB güvenlik sınırı (laboratuvarda yok): 2 px bloklarla 8192 px'e yeter
WATERSHED_SURFACES = ("gradient", "intensity", "inverted")
WATERSHED_DISPLAYS = ("overlay", "labels", "boundaries", "surface", "minima")
MARKER_SURFACES = ("gradient", "distance")
MARKER_DISPLAYS = ("overlay", "labels", "boundaries", "markers", "binary", "sure_bg", "distance", "surface")
TEAL_BGR = (180, 170, 10)                # laboratuvar TEAL rgb(10, 170, 180)
EDT_INF = 1e12                           # laboratuvar edt: ön plan başlangıç değeri
N8_OFFSETS = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))   # (dx, dy), laboratuvar N8
N4_OFFSETS = ((0, -1), (-1, 0), (1, 0), (0, 1))                                       # laboratuvar N4


@dataclass(frozen=True)
class RegionReport(_morph.ComponentReport):
    """Bölge tablosu (ComponentReport: etiket, alan, kutu, merkez) + yöntem özeti."""

    method: str = ""
    detail: str = ""

    def summary(self) -> str:
        return f"{self.method}: {self.count} region{'s' if self.count != 1 else ''} — {self.detail}"


def region_components(labels: np.ndarray) -> tuple[_morph.Component, ...]:
    """Etiket matrisinden (0 = arka plan, 1…n) bölge istatistikleri; bölgenin bağlantılı olması gerekmez."""
    n = int(labels.max()) if labels.size else 0
    if n <= 0:
        return ()
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs].astype(np.int64)
    area = np.bincount(lab, minlength=n + 1)
    big = np.iinfo(np.int64).max
    x0, y0 = np.full(n + 1, big), np.full(n + 1, big)
    x1, y1 = np.full(n + 1, -1), np.full(n + 1, -1)
    np.minimum.at(x0, lab, xs)
    np.minimum.at(y0, lab, ys)
    np.maximum.at(x1, lab, xs)
    np.maximum.at(y1, lab, ys)
    sx = np.bincount(lab, weights=xs, minlength=n + 1)
    sy = np.bincount(lab, weights=ys, minlength=n + 1)
    return tuple(_morph.Component(i, int(area[i]), int(x0[i]), int(y0[i]), int(x1[i] - x0[i] + 1),
                                  int(y1[i] - y0[i] + 1), float(sx[i] / area[i]), float(sy[i] / area[i]))
                 for i in range(1, n + 1) if area[i])


def boundary_of(labels: np.ndarray) -> np.ndarray:
    """Laboratuvar ``boundaryOf``: L > 0 ve sağ ya da alt komşu başka bir pozitif etiket → sınır pikseli."""
    lab = labels
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= (lab[:, :-1] > 0) & (lab[:, 1:] > 0) & (lab[:, 1:] != lab[:, :-1])
    b[:-1, :] |= (lab[:-1, :] > 0) & (lab[1:, :] > 0) & (lab[1:, :] != lab[:-1, :])
    return b


def label_overlay(labels: np.ndarray, base: np.ndarray, dim: float = 0.45, alpha: float = 0.8,
                  lines: np.ndarray | None = None) -> np.ndarray:
    """Laboratuvar ``labelRGBA``: taban q8(base)·dim; etiketli pikselde renk α ile karıştırılır; çizgiler beyaz.
    Uint8ClampedArray gibi her yazımda en yakına (eşitlikte çifte) yuvarlanır. Renkler AYLAB etiket paleti
    (bileşen tablosundaki renk örnekleriyle aynı); laboratuvarın kendi paleti başka tonlar kullanır."""
    g = np.clip(np.rint(q8(base).astype(np.float64) * dim), 0, 255)
    out = np.repeat(g[:, :, None], 3, axis=2)
    lab = labels.astype(np.int64)
    m = lab > 0
    if m.any():
        pal = _morph.label_colors(int(lab.max())).astype(np.float64)
        out[m] = np.clip(np.rint(out[m] * (1 - alpha) + pal[lab[m]] * alpha), 0, 255)
    if lines is not None:
        out[lines] = 255
    return out.astype(np.uint8)


def _labels_bgr(labels: np.ndarray) -> np.ndarray:
    return _morph.colorize_labels(labels.astype(np.int32))


def _connectivity(value) -> str:
    if isinstance(value, bool) or str(value) not in REGION_CONNECTIVITY:
        raise AlgorithmParameterError(f"Connectivity must be 4 or 8 (got {value!r}).")
    return str(value)


def _first_order_labels(keys: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Anahtarlara raster sırasında ilk görülüşlerine göre 1…n numarası verir; geçersiz pikseller 0."""
    flat, ok = keys.ravel(), valid.ravel()
    out = np.zeros(flat.size, np.int32)
    if ok.any():
        idx = np.flatnonzero(ok)
        uniq, first, inv = np.unique(flat[idx], return_index=True, return_inverse=True)
        rank = np.empty(uniq.size, np.int32)
        rank[np.argsort(first, kind="stable")] = np.arange(1, uniq.size + 1, dtype=np.int32)
        out[idx] = rank[inv.ravel()]
    return out.reshape(keys.shape)


# ------------------------------------------------------------------ 14) region growing (slayt 30–31)
def brightest_seeds(f: np.ndarray, q: float) -> tuple[np.ndarray, int]:
    """Laboratuvar 'Brightest pixels (Gonzalez)': sıralı değerlerde s[⌊N(1 − q/100)⌋] eşiği; f ≥ eşik olan
    pikseller tohumdur; 8-bağlantılı her tohum grubu bir bölge numarası alır (raster sırası).
    Döndürür: (tohum grubu etiketleri, eşik)."""
    s = np.sort(f.ravel())
    thr = int(s[int(np.floor(f.size * (1 - q / 100)))])
    return _morph.label_components(f >= thr, 8).labels, thr


def grow_regions(f: np.ndarray, seed_index: np.ndarray, seed_region: np.ndarray, t: int, connectivity: str = "8",
                 max_layers: int = GROW_MAX_LAYERS) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Laboratuvar ``growAll``: katman katman büyütme. Her adımda, önceki adımda eklenen her p için ziyaret
    edilmemiş komşu q, |f(q) − ref(p)| ≤ T ise bölgeye girer; p'nin bölgesini ve referans değerini (p'yi
    yetiştiren tohumun değeri) devralır. q'yu ilk kabul eden p kazanır (cephe sırası, sonra komşu sırası).
    Yeni piksel kalmayınca (bölge kararlı) ya da ``max_layers`` katmanda durur.
    Döndürür: (bölge etiketleri, eklenme katmanı (−1 = dışarıda), katman sayısı, sınıra takıldı mı)."""
    h, w = f.shape
    n = h * w
    ff = f.ravel().astype(np.int64)
    add_at = np.full(n, -1, np.int64)
    reg = np.zeros(n, np.int32)
    ref = np.zeros(n, np.int64)
    front = []
    for i, r in zip(seed_index.tolist(), seed_region.tolist()):
        if add_at[i] < 0:
            add_at[i] = 0
            reg[i] = r
            ref[i] = ff[i]
            front.append(i)
    front = np.array(front, np.int64)
    offs = np.array(N4_OFFSETS if connectivity == "4" else N8_OFFSETS)
    layer = 0
    capped = False
    while front.size:
        if layer >= max_layers:
            capped = True
            break
        x, y = front % w, front // w
        xx = x[:, None] + offs[None, :, 0]
        yy = y[:, None] + offs[None, :, 1]
        parent = np.broadcast_to(front[:, None], xx.shape)
        inside = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
        j = (yy * w + xx)[inside]
        p = parent[inside]
        free = add_at[j] < 0
        j, p = j[free], p[free]
        acc = np.abs(ff[j] - ref[p]) <= t
        j, p = j[acc], p[acc]
        if not j.size:
            break
        layer += 1
        _, first = np.unique(j, return_index=True)
        first.sort()
        new, par = j[first], p[first]
        add_at[new] = layer
        reg[new] = reg[par]
        ref[new] = ref[par]
        front = new
    return reg.reshape(h, w), add_at.reshape(h, w), layer, capped


def region_growing(image: np.ndarray, seeds: str = "point", seed_x: int = -1, seed_y: int = -1, q: float = 0.3,
                   t: int = 40, connectivity: str = "8", display: str = "overlay") -> ProcessResult:
    """Region growing (slayt 30–31; laboratuvar 'Region growing'). POINT: tek tohum (x, y); −1 = görüntü
    merkezi (⌊w/2⌋, ⌊h/2⌋). BRIGHTEST: en parlak q % pikseller tohum; her 8-bağlantılı grup ayrı bölge."""
    _choice(seeds, "Seeds", GROW_SEED_MODES)
    sx = _int(seed_x, "Seed x", -1, SEED_COORD_MAX)
    sy = _int(seed_y, "Seed y", -1, SEED_COORD_MAX)
    qv = _real(q, "Auto seeds: top %", *GROW_Q_RANGE)
    tv = _int(t, "T", *GROW_T_RANGE)
    conn = _connectivity(connectivity)
    _choice(display, "Display", GROW_DISPLAYS)
    f = gray_input(image)
    h, w = f.shape
    if seeds == "point":
        px = w // 2 if sx < 0 else sx
        py = h // 2 if sy < 0 else sy
        if px >= w or py >= h:
            raise AlgorithmParameterError(f"Seed ({px}, {py}) is outside the {w}×{h} image.")
        idx, regs = np.array([py * w + px]), np.array([1])
        seed_text = f"seed ({px}, {py}), f(seed) = {int(f[py, px])}"
    else:
        if display == "difference":
            raise AlgorithmParameterError("The |f − f(seed)| display needs a single seed (Seeds = Point).")
        groups, thr = brightest_seeds(f, qv)
        idx = np.flatnonzero(groups)
        regs = groups.ravel()[idx]
        seed_text = f"{idx.size} seed pixel{'s' if idx.size != 1 else ''} with f ≥ {thr} (top {qv:g} %)"
    labels, _, layers, capped = grow_regions(f, idx, regs, tv, conn)
    report = RegionReport(labels, region_components(labels), int(conn), "Region growing",
                          f"{seed_text}; q joins if |f(q) − f(seed)| ≤ {tv}, {conn}-neighbours; "
                          f"{int((labels > 0).sum())} px after {layers} layer{'s' if layers != 1 else ''}"
                          + (f" (stopped at the {GROW_MAX_LAYERS}-layer limit)" if capped else " (stable)"))
    if display == "binary":
        return ProcessResult(_gray_out(_binary(labels > 0), image), report)
    if display == "labels":
        return ProcessResult(_color_out(_labels_bgr(labels), image), report)
    if display == "difference":
        diff = np.abs(f.astype(np.int16) - int(f[py, px])).astype(np.uint8)
        return ProcessResult(_gray_out(diff, image), report)
    return ProcessResult(_color_out(label_overlay(labels, f), image), report)


# ------------------------------------------------------------------ 15) region splitting & merging (slayt 32–34)
@dataclass(frozen=True)
class QuadNode:
    x: int
    y: int
    w: int
    h: int
    depth: int
    mean: float
    std: float
    q: bool
    leaf: bool


def quadtree(f: np.ndarray, predicate, min_size: int, invert: bool = False,
             max_depth: int = SPLIT_MAX_DEPTH) -> list[QuadNode]:
    """Laboratuvar ``quadtree``: kök = tüm görüntü; kuyruk sırasıyla (genişlik öncelikli) her blok için
    m ve σ = √max(0, Σf²/n − m²); Q(R) = predicate(m, σ). Q = FALSE (``invert``: Q = TRUE) ve
    w, h ≥ 2·min ise blok dörde bölünür: w₁ = ⌊w/2⌋, h₁ = ⌊h/2⌋ → (w₁, h₁), (w − w₁, h₁), (w₁, h − h₁),
    (w − w₁, h − h₁); 2'nin kuvveti olmayan boyutlarda parçalar bir piksel farklı olabilir, hiçbir blok
    görüntü dışına taşmaz. ``max_depth`` AYLAB güvenlik sınırıdır (laboratuvarda yok)."""
    h, w = f.shape
    ii = np.zeros((h + 1, w + 1), np.int64)
    ii[1:, 1:] = f.astype(np.int64).cumsum(0).cumsum(1)
    i2 = np.zeros((h + 1, w + 1), np.int64)
    i2[1:, 1:] = (f.astype(np.int64) ** 2).cumsum(0).cumsum(1)

    def block_sum(a, x, y, bw, bh):
        return int(a[y + bh, x + bw] - a[y, x + bw] - a[y + bh, x] + a[y, x])

    nodes = []
    queue = [(0, 0, w, h, 0)]
    qi = 0
    while qi < len(queue):
        x, y, bw, bh, d = queue[qi]
        qi += 1
        c = bw * bh
        s, s2 = block_sum(ii, x, y, bw, bh), block_sum(i2, x, y, bw, bh)
        mean = s / c
        std = float(np.sqrt(max(0.0, s2 / c - mean * mean)))
        qv = bool(predicate(mean, std))
        split = (qv if invert else not qv) and bw >= 2 * min_size and bh >= 2 * min_size and d < max_depth
        nodes.append(QuadNode(x, y, bw, bh, d, mean, std, qv, not split))
        if split:
            w1, h1 = bw >> 1, bh >> 1
            for dx, dy, ww, hh in ((0, 0, w1, h1), (w1, 0, bw - w1, h1), (0, h1, w1, bh - h1),
                                   (w1, h1, bw - w1, bh - h1)):
                queue.append((x + dx, y + dy, ww, hh, d + 1))
    return nodes


def merge_leaves(leaves: list[QuadNode], shape: tuple[int, int], similar) -> tuple[np.ndarray, np.ndarray]:
    """Laboratuvar birleştirme: yatay ya da dikey bitişik iki yaprak ``similar(A, B)`` ise aynı bölge
    (birleşim–bul; zincirleme birleşmeler geçişlidir, karşılaştırma yaprak ortalamalarıyla yapılır).
    Döndürür: (piksel başına yaprak no, yaprak → kök yaprak)."""
    lid = np.zeros(shape, np.int64)
    for k, n in enumerate(leaves):
        lid[n.y:n.y + n.h, n.x:n.x + n.w] = k
    pairs = np.concatenate([np.stack([lid[:, :-1].ravel(), lid[:, 1:].ravel()], 1),
                            np.stack([lid[:-1, :].ravel(), lid[1:, :].ravel()], 1)])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if pairs.size:
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    par = list(range(len(leaves)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for a, b in pairs.tolist():
        if similar(leaves[a], leaves[b]):
            ra, rb = find(a), find(b)
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)
    return lid, np.array([find(k) for k in range(len(leaves))], np.int64)


def _min_block(value) -> int:
    if isinstance(value, bool) or str(value) not in SPLIT_MIN_BLOCKS:
        raise AlgorithmParameterError(f"Min block must be one of {', '.join(SPLIT_MIN_BLOCKS)} (got {value!r}).")
    return int(value)


def split_and_merge(image: np.ndarray, predicate: str = "homog", a: int = 8, b: int = 125, min_size: str = "4",
                    max_depth: int = SPLIT_MAX_DEPTH, merge_t: int = 12, display: str = "merged") -> ProcessResult:
    """Region splitting & merging (slayt 32–34; laboratuvar 'Region splitting & merging').
    HOMOG: Q(R) = σ_R ≤ a; Q = FALSE bloklar bölünür; bitişik yapraklar |m_A − m_B| ≤ merge_t ise birleşir.
    GONZALEZ (Fig. 10.48): Q(R) = σ_R > a ve 0 < m_R < b; DIPUM ``splitmerge`` gibi Q = TRUE bloklar en küçük
    boyuta kadar bölünmeye devam eder; Q = TRUE yapraklar birleştirilir (beyaz), gerisi arka plan."""
    _choice(predicate, "Predicate Q", SPLIT_PREDICATES)
    av = _int(a, "a (σ limit)", 1, 40)
    bv = _int(b, "b (mean limit)", 20, 255)
    mv = _min_block(min_size)
    dv = _int(max_depth, "Max depth", 0, SPLIT_MAX_DEPTH)
    tv = _int(merge_t, "Merge if |Δmean| ≤", 0, 60)
    _choice(display, "Display", SPLIT_DISPLAYS)
    f = gray_input(image)
    gz = predicate == "gonzalez"
    pred = (lambda m, s: s > av and 0 < m < bv) if gz else (lambda m, s: s <= av)
    nodes = quadtree(f, pred, mv, invert=gz, max_depth=dv)
    leaves = [n for n in nodes if n.leaf]
    similar = (lambda p, r: p.q and r.q) if gz else (lambda p, r: abs(p.mean - r.mean) <= tv)
    lid, root = merge_leaves(leaves, f.shape, similar)
    keep = np.array([n.q for n in leaves])[lid] if gz else np.ones(f.shape, bool)
    labels = _first_order_labels(root[lid], keep)
    depth = max(n.depth for n in nodes)
    rule = (f"Q = σ > {av} and 0 < m < {bv}, Q = TRUE leaves merged" if gz
            else f"Q = σ ≤ {av}, merge if |Δmean| ≤ {tv}")
    report = RegionReport(labels, region_components(labels), 4, "Split & merge",
                          f"{rule}; {len(nodes)} quadtree nodes, {len(leaves)} leaves (min {mv}×{mv}, "
                          f"depth {depth})")
    if display == "labels":
        return ProcessResult(_color_out(_labels_bgr(labels), image), report)
    means = np.zeros(f.shape)
    for n in leaves:
        means[n.y:n.y + n.h, n.x:n.x + n.w] = n.mean
    if display == "split":
        bgr = np.repeat(q8(means)[:, :, None], 3, axis=2)
        for n in leaves:                                     # laboratuvar partRGBA: yaprak çerçevesi teal
            bgr[n.y, n.x:n.x + n.w] = TEAL_BGR
            bgr[n.y + n.h - 1, n.x:n.x + n.w] = TEAL_BGR
            bgr[n.y:n.y + n.h, n.x] = TEAL_BGR
            bgr[n.y:n.y + n.h, n.x + n.w - 1] = TEAL_BGR
        return ProcessResult(_color_out(bgr, image), report)
    if gz:
        return ProcessResult(_gray_out(_binary(labels > 0), image), report)
    means32 = means.astype(np.float32).astype(np.float64)    # laboratuvar meanImg: Float32Array
    return ProcessResult(_color_out(label_overlay(labels, means32, 0.5, 0.55, boundary_of(labels)), image), report)


# ------------------------------------------------------------------ 16) watershed (slayt 35–36)
def gaussian_blur_f32(f: np.ndarray, sigma: float) -> np.ndarray:
    """Laboratuvar ``gaussBlur`` (yuvarlamasız): ayrılabilir Gauss, reflect-101, her geçiş float32 saklanır."""
    a = gaussian_kernel(sigma)
    r = a.size // 2
    h, w = f.shape
    p = _pad_reflect(f.astype(np.float32).astype(np.float64), r)
    rows = sum(a[i] * p[:, i:i + w] for i in range(a.size)).astype(np.float32).astype(np.float64)
    return sum(a[i] * rows[i:i + h, :] for i in range(a.size)).astype(np.float32)


def surface_gradient(f: np.ndarray, sigma: float) -> np.ndarray:
    """Laboratuvar ``surfaceGrad``: (σ > 0 ise) Gauss, Sobel g_x, g_y, M = √(g_x² + g_y²), round(M·255 / max M)."""
    b = gaussian_blur_f32(f, sigma) if sigma > 0 else f.astype(np.float32)
    gx = correlate(b, GRADIENT_OPERATORS["sobel"][0]).astype(np.float32)
    gy = correlate(b, GRADIENT_OPERATORS["sobel"][1]).astype(np.float32)
    m = np.hypot(gx.astype(np.float64), gy.astype(np.float64)).astype(np.float32).astype(np.float64)
    mx = float(m.max()) or 1.0
    return js_round(m * 255 / mx).astype(np.uint8)


def watershed_surface(f: np.ndarray, surface: str, sigma: float) -> np.ndarray:
    """Laboratuvar ``viewWS`` yüzeyi: gradient = surfaceGrad(σ); intensity = q8(Gauss_σ(f));
    inverted = q8(255 − Gauss_σ(f)). σ = 0 → yumuşatma yok."""
    if surface == "gradient":
        return surface_gradient(f, sigma)
    b = gaussian_blur_f32(f, sigma) if sigma > 0 else f.astype(np.float32)
    if surface == "inverted":
        b = (255.0 - b.astype(np.float64)).astype(np.float32)
    return q8(b.astype(np.float64))


def regional_minima(s: np.ndarray) -> np.ndarray:
    """Laboratuvar ``regionalMinima``: eşit değerli 8-bağlantılı düzlükler; hiçbir 8-komşusu daha alçak olmayan
    düzlük bir bölgesel minimumdur. Etiketler raster sırasında ilk pikseline göre 1…n; diğerleri 0."""
    h, w = s.shape
    v = s.astype(np.int64)
    pad = np.pad(v, 1, constant_values=np.iinfo(np.int64).max)
    lower = np.zeros(s.shape, bool)
    for dx, dy in N8_OFFSETS:
        lower |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] < v
    plateau = np.zeros(s.shape, np.int64)
    offset = 0
    for val in np.unique(v):
        n, lab = cv2.connectedComponents((v == val).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        m = lab > 0
        plateau[m] = lab[m] + offset
        offset += n - 1
    bad = np.zeros(offset + 1, bool)
    bad[plateau[lower]] = True
    return _first_order_labels(plateau, ~bad[plateau])


def flood(s: np.ndarray, markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Laboratuvar ``watershed`` (Meyer taşırma, 4-komşu, 256 kova): işaretli pikseller kendi yüksekliğinin
    kovasına girer; seviyeler 0 → 255 sırayla, kova içinde ilk giren ilk çıkar. Etiketsiz komşu (sol, sağ,
    üst, alt), çıkan pikselin etiketini alır ve max(L, s(q)) kovasına girer. Baraj pikseli yoktur: ulaşılan
    her piksel bir havzaya düşer; havza sınırları ``boundary_of`` ile çizilir.
    Döndürür: (etiketler, pikselin su seviyesi (−1 = ulaşılmadı))."""
    h, w = s.shape
    n = h * w
    sv = s.ravel().astype(np.int64).tolist()
    lab = markers.ravel().astype(np.int64).tolist()
    arr = [-1] * n
    buckets = [[] for _ in range(256)]
    for i in np.flatnonzero(markers.ravel() > 0).tolist():
        arr[i] = sv[i]
        buckets[sv[i]].append(i)
    for level in range(256):
        q = buckets[level]
        k = 0
        while k < len(q):
            p = q[k]
            k += 1
            x = p % w
            lp = lab[p]
            for j in (p - 1 if x > 0 else -1, p + 1 if x < w - 1 else -1, p - w, p + w):
                if j < 0 or j >= n or lab[j] != 0:
                    continue
                lab[j] = lp
                lv = sv[j] if sv[j] > level else level
                arr[j] = lv
                buckets[lv].append(j)
        buckets[level] = None
    return np.array(lab, np.int32).reshape(h, w), np.array(arr, np.int16).reshape(h, w)


def watershed(image: np.ndarray, surface: str = "gradient", sigma: float = 3.0,
              display: str = "overlay") -> ProcessResult:
    """Watershed (slayt 35–36; laboratuvar 'Watershed segmentation'): görüntü topografik yüzeydir; her bölgesel
    minimum bir havzadır; su yükselirken havzalar dolar, buluştukları yerler watershed çizgileridir.
    Gürültünün yarattığı her minimum da bölge olur (aşırı bölütleme); σ büyüdükçe azalır."""
    _choice(surface, "Surface", WATERSHED_SURFACES)
    sg = _real(sigma, "Smoothing σ", 0.0, 6.0)
    _choice(display, "Display", WATERSHED_DISPLAYS)
    f = gray_input(image)
    s = watershed_surface(f, surface, sg)
    minima = regional_minima(s)
    labels, _ = flood(s, minima)
    lines = boundary_of(labels)
    name = {"gradient": "gradient magnitude", "intensity": "intensity", "inverted": "inverted intensity"}[surface]
    report = RegionReport(labels, region_components(labels), 4, "Watershed",
                          f"surface = {name}, σ = {sg:g}; one basin per regional minimum ({int(minima.max())} "
                          f"minima), 4-neighbour flooding; {int(lines.sum())} watershed-line px")
    if display == "labels":
        return ProcessResult(_color_out(_labels_bgr(labels), image), report)
    if display == "boundaries":
        return ProcessResult(_gray_out(_binary(lines), image), report)
    if display == "surface":
        return ProcessResult(_gray_out(s, image), report)
    if display == "minima":
        return ProcessResult(_color_out(label_overlay(minima, s, 0.35, 1.0), image), report)
    return ProcessResult(_color_out(label_overlay(labels, f, 0.5, 0.45, lines), image), report)


# ------------------------------------------------------------------ 17) marker-controlled watershed (slayt 37–38)
def morph3(mask: np.ndarray, op: str, iterations: int) -> np.ndarray:
    """Laboratuvar ``morph``: 3×3 kare yapı elemanı; aşındırmada görüntü dışı 1, genişletmede 0 sayılır."""
    a = mask.astype(bool)
    h, w = a.shape
    erode = op == "erode"
    for _ in range(iterations):
        p = np.pad(a, 1, constant_values=erode)
        out = np.full(a.shape, erode)
        for dy in range(3):
            for dx in range(3):
                win = p[dy:dy + h, dx:dx + w]
                out = (out & win) if erode else (out | win)
        a = out
    return a


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Laboratuvar ``edt``: her ön plan pikselinin en yakın arka plan pikseline tam Öklid uzaklığı (float32;
    arka plan 0). Görüntü dışı arka plan sayılmaz; hiç arka plan yoksa her piksel √10¹² = 10⁶ (laboratuvar INF)."""
    a = mask.astype(bool)
    if a.all():
        return np.full(a.shape, np.sqrt(EDT_INF), np.float32)
    if not a.any():
        return np.zeros(a.shape, np.float32)
    h, w = a.shape
    # 1) sütunlar: en yakın arka plan pikseline dikey uzaklığın karesi (sütunda arka plan yoksa INF)
    rows = np.arange(h)[:, None]
    above = np.maximum.accumulate(np.where(~a, rows, -1), axis=0)
    below = np.minimum.accumulate(np.where(~a, rows, 2 * h)[::-1], axis=0)[::-1]
    g = np.minimum(np.where(above >= 0, rows - above, 2 * h), np.where(below < 2 * h, below - rows, 2 * h))
    sq = np.where(g < 2 * h, g.astype(np.float64) ** 2, EDT_INF)
    # 2) satırlar: D(x) = min_t t² + G(x ± t); t² ≥ max D olunca hiçbir piksel iyileşemez (tamsayılar, tam)
    d = sq.copy()
    t = 1
    while t < w and t * t < d.max():
        d[:, t:] = np.minimum(d[:, t:], sq[:, :-t] + t * t)
        d[:, :-t] = np.minimum(d[:, :-t], sq[:, t:] + t * t)
        t += 1
    return np.sqrt(d).astype(np.float32)


@dataclass(frozen=True)
class MarkerStages:
    threshold: int
    binary: np.ndarray
    opened: np.ndarray
    sure_bg: np.ndarray
    dist: np.ndarray
    internal: np.ndarray          # iç işaretleyiciler: 8-bağlantılı etiketler 1…n
    markers: np.ndarray           # 1 = dış (arka plan), 2…n+1 = iç, 0 = bilinmeyen
    surface: np.ndarray
    flooded: np.ndarray           # taşırma etiketleri (1 = arka plan)


def marker_stages(f: np.ndarray, foreground: str, open_iter: int, bg_iter: int, alpha: float,
                  surface: str) -> MarkerStages:
    """Laboratuvar ``viewMWS`` (OpenCV coins akışı): Otsu → açma (aşındır ×k, genişlet ×k) → kesin arka plan
    (genişlet ×m) → uzaklık dönüşümü → iç işaretleyici dist > α·max → dış işaretleyici = kesin arka planın
    dışı → yüzey → yalnızca işaretleyicilerden taşırma."""
    t = otsu_analysis(histogram(f))[0]
    binary = _decide(f, t, foreground)
    opened = morph3(morph3(binary, "erode", open_iter), "dilate", open_iter)
    sure_bg = morph3(opened, "dilate", bg_iter)
    dist = distance_transform(opened)
    dmax = float(dist.max()) or 1.0
    fg = dist.astype(np.float64) > alpha * dmax
    internal = _morph.label_components(fg, 8).labels
    markers = np.zeros(f.shape, np.int32)
    markers[~sure_bg] = 1
    markers[sure_bg & fg] = internal[sure_bg & fg] + 1
    if surface == "distance":
        s = js_round(255 - dist.astype(np.float64) / dmax * 255).astype(np.uint8)
    else:
        s = surface_gradient(f, 1.0)
    flooded, _ = flood(s, markers)
    return MarkerStages(t, binary, opened, sure_bg, dist, internal, markers, s, flooded)


def marker_controlled_watershed(image: np.ndarray, foreground: str = "bright", open_iter: int = 2, bg_iter: int = 3,
                                alpha: float = 0.5, surface: str = "gradient",
                                display: str = "overlay") -> ProcessResult:
    """Marker-controlled watershed (slayt 37–38): (1) gradyan, (2) iç işaretleyiciler, (3) dış işaretleyici,
    (4) watershed, (5) bölge etiketleri. İç işaretleyici başına bir bölge; arka plan etiketi 0."""
    _foreground(foreground)
    ok = _int(open_iter, "Opening iterations", 0, 5)
    bk = _int(bg_iter, "Sure-background dilations", 0, 10)
    av = _real(alpha, "Foreground: dist > α·max", 0.1, 0.9)
    _choice(surface, "Surface", MARKER_SURFACES)
    _choice(display, "Display", MARKER_DISPLAYS)
    f = gray_input(image)
    st = marker_stages(f, foreground, ok, bk, av, surface)
    labels = np.where(st.flooded > 1, st.flooded - 1, 0).astype(np.int32)
    lines = boundary_of(st.flooded)
    n_obj = int(st.internal.max())
    sname = "gradient (σ = 1)" if surface == "gradient" else "inverted distance"
    report = RegionReport(labels, region_components(labels), 4, "Marker-controlled watershed",
                          f"Otsu T = {st.threshold} ({foreground}), opening {ok}×, sure background +{bk}×, internal "
                          f"markers dist > {av:g}·max ({n_obj} object{'s' if n_obj != 1 else ''}), "
                          f"surface = {sname}; background = label 0")
    if display == "labels":
        return ProcessResult(_color_out(_labels_bgr(labels), image), report)
    if display == "boundaries":
        return ProcessResult(_gray_out(_binary(lines), image), report)
    if display == "binary":
        return ProcessResult(_gray_out(_binary(st.binary), image), report)
    if display == "sure_bg":
        return ProcessResult(_gray_out(_binary(st.sure_bg), image), report)
    if display == "distance":
        return ProcessResult(_gray_out(display_minmax(st.dist.astype(np.float64)), image), report)
    if display == "surface":
        return ProcessResult(_gray_out(st.surface, image), report)
    if display == "markers":                                 # laboratuvar (f): koyu = dış, gri = bilinmeyen
        unk = np.where(st.markers > 1, st.markers - 1, 0)
        bgr = label_overlay(unk, f, 0.3, 1.0)
        bgr[st.markers == 1] = (35, 35, 35)
        bgr[st.markers == 0] = (150, 150, 150)
        return ProcessResult(_color_out(bgr, image), report)
    return ProcessResult(_color_out(label_overlay(labels, f, 0.5, 0.55, lines), image), report)
