"""Morfolojik işlemler — Gonzalez & Woods / COMP 4360 Chapter 4.

Desteklenen işlem grupları:

* İkili temel işlemler — erosion, dilation, opening, closing, gradient, hit-or-miss.
* İkili morfolojik algoritmalar — boundary extraction, hole filling, connected
  components, convex hull, thinning, thickening, skeletonization, pruning.
* Gri ton morfoloji (düz SE, eşikleme yapılmaz) — erosion, dilation, opening,
  closing, morfolojik gradient; top-hat, black-hat ve aydınlatma düzeltmesi
  (dosyanın sonundaki bölümler; sınır kuralı orada ayrıca açıklanır).

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
* **Sınır (ikili işlemler):** görüntünün dışındaki pikseller *arka plandır* (dersin laboratuvar
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
from processing.registry import AlgorithmParameterError, MaskParam, ProcessResult

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


def boundary_mask(mask: np.ndarray, se: StructuringElement) -> np.ndarray:
    """β(A) = A − (A ⊖ B) (slayt 53–55): erosion'ın sildiği ön plan pikselleri.

    Küme farkı A ∩ (A ⊖ B)ᶜ olarak alınır; sonuç her zaman A'nın alt kümesidir
    (iç sınır). Gradyandan farkı: dış kuşak (A ⊕ B) − A dahil değildir.
    Görüntü dışı arka plan olduğundan, kenara değen nesnelerin görüntü
    kenarındaki pikselleri de sınırdır.
    """
    a = _validate_mask(mask)
    return a & ~binary_erosion(a, se)


# --------------------------------------------------------------------------
# Delik doldurma (hole filling) — slayt 56–59
# --------------------------------------------------------------------------
#
# Delik: ön planla çevrili, görüntü dışına (arka plana) bağlanmayan arka plan
# bölgesi. Bağlantı, yeniden yapılandırmada kullanılan SE'ye karşılık gelir:
# 3×3 artı → 4-bağlantılı arka plan (ders kitabı), 3×3 kare → 8-bağlantılı.

HOLE_METHODS = ("flood", "reconstruction")
HOLE_SEED_POLICIES = ("largest", "centre", "all")
HOLE_SE_SHAPES = {"cross": 4, "rect": 8}   # 3×3 SE → arka plan bağlantılılığı


def hole_labels(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, list[int]]:
    """Arka plan bileşen etiketleri ve bunlardan *delik* olanlar (görüntü kenarına değmeyenler)."""
    a = _validate_mask(mask)
    if isinstance(connectivity, bool) or connectivity not in (4, 8):
        raise AlgorithmParameterError(f"Connectivity must be 4 or 8 (got {connectivity!r}).")
    n, labels = cv2.connectedComponents((~a).astype(np.uint8), connectivity=connectivity)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    holes = [i for i in range(1, n) if i not in border]
    return labels, holes


def _validate_seeds(a: np.ndarray, seeds, connectivity: int) -> list[tuple[int, int]]:
    h, w = a.shape
    labels, holes = hole_labels(a, connectivity)
    hole_set = set(holes)
    out = []
    for seed in seeds:
        if not isinstance(seed, (tuple, list)) or len(seed) != 2:
            raise AlgorithmParameterError(f"A seed must be an (x, y) pair (got {seed!r}).")
        x, y = _validate_int(seed[0], "Seed x"), _validate_int(seed[1], "Seed y")
        if not (0 <= x < w and 0 <= y < h):
            raise AlgorithmParameterError(f"Seed ({x}, {y}) is outside the {w}×{h} image.")
        if a[y, x]:
            raise AlgorithmParameterError(f"Seed ({x}, {y}) lies on the foreground; a seed must be inside a hole.")
        if labels[y, x] not in hole_set:
            raise AlgorithmParameterError(
                f"Seed ({x}, {y}) is not inside a closed hole: its background region reaches the image "
                f"border, so X_k would flood the whole background.")
        out.append((x, y))
    return out


def fill_holes_reconstruction(mask: np.ndarray, seeds, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """X_k = (X_{k−1} ⊕ B) ∩ Aᶜ, X₀ = tohumlar; X_k = X_{k−1} olunca durur. Döndürür: (X_k ∪ A, k).

    B = 3×3 artı (connectivity=4, ders kitabı) veya 3×3 kare (connectivity=8).
    Yalnızca tohum içeren delikler dolar. Tohumlar (x, y) = (sütun, satır).
    Ön plan üzerindeki, görüntü dışındaki veya kenara açılan bir arka plan
    bölgesindeki tohum ``AlgorithmParameterError`` verir.
    """
    a = _validate_mask(mask)
    seeds = _validate_seeds(a, list(seeds), connectivity)
    se = structuring_element("cross" if connectivity == 4 else "rect", 3, 3)
    ac = ~a
    x = np.zeros_like(a)
    for cx, cy in seeds:
        x[cy, cx] = True
    k = 0
    while True:
        nxt = binary_dilation(x, se) & ac
        k += 1
        if np.array_equal(nxt, x):
            break
        x = nxt
    return x | a, k


def fill_holes_flood(mask: np.ndarray) -> np.ndarray:
    """OpenCV yaklaşımı (slayt 59): arka planı sınırdan 4-bağlantılı taşır, tersler, A ile OR'lar.

    Görüntü bir piksellik arka plan çerçevesiyle genişletilir ve taşma bu
    çerçevenin köşesinden başlatılır. Böylece kenara değen TÜM arka plan
    bölgeleri "dışarı" sayılır ve (0, 0) ön plan olsa bile doğru çalışır.
    (Slayttaki kod doğrudan (0, 0)'dan başlar; (0, 0) ön plansa ya da kenardaki
    arka plan ayrık parçalara bölünmüşse yanlış sonuç verir.)
    """
    a = _validate_mask(mask)
    padded = np.pad(np.where(a, 255, 0).astype(np.uint8), 1)
    flood = padded.copy()
    ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255, flags=4)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1] > 0
    return a | holes


def auto_seeds(mask: np.ndarray, policy: str = "largest", connectivity: int = 4) -> list[tuple[int, int]]:
    """GUI için tohum seçimi: her delikten, ağırlık merkezine en yakın delik pikseli.

    largest — en büyük delik;  centre — tohumu görüntü merkezine en yakın delik;
    all — her delik için bir tohum (sonuç flood fill ile aynıdır, connectivity=4).
    """
    if policy not in HOLE_SEED_POLICIES:
        raise AlgorithmParameterError(f"Seed policy must be one of {', '.join(HOLE_SEED_POLICIES)} (got {policy!r}).")
    labels, holes = hole_labels(mask, connectivity)
    if not holes:
        return []
    seeds, sizes = [], []
    for i in holes:
        ys, xs = np.nonzero(labels == i)
        j = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
        seeds.append((int(xs[j]), int(ys[j])))
        sizes.append(len(ys))
    if policy == "all":
        return seeds
    if policy == "largest":
        return [seeds[int(np.argmax(sizes))]]
    h, w = labels.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    return [min(seeds, key=lambda s: (s[1] - cy) ** 2 + (s[0] - cx) ** 2)]


# --------------------------------------------------------------------------
# Bağlantılı bileşenler (connected components) — slayt 60–62
# --------------------------------------------------------------------------
#
# Ön plan pikselleri, B'nin tanımladığı komşulukla birbirine ulaşabiliyorsa
# aynı bileşendir: 3×3 artı → 4-bağlantı, 3×3 kare → 8-bağlantı.
# Etiketler: arka plan 0, bileşenler 1…N; numaralar, bileşenin raster
# sırasındaki (yukarıdan aşağı, soldan sağa) İLK pikseline göre verilir —
# dersin yöntemindeki "henüz etiketlenmemiş ilk ön plan pikseli tohumdur"
# kuralıyla aynı sıra. Etiket matrisi int32'dir; renkli görüntü ondan türetilir.

CC_CONNECTIVITY = {"4": 4, "8": 8}
CC_DISPLAYS = ("labels", "boxes")


@dataclass(frozen=True)
class Component:
    label: int
    area: int          # piksel sayısı
    x: int             # sınırlayıcı kutu: sol üst köşe (sütun, satır), genişlik, yükseklik
    y: int
    width: int
    height: int
    cx: float          # ağırlık merkezi (piksel koordinatlarının ortalaması)
    cy: float


@dataclass(frozen=True)
class ComponentReport:
    """Etiket matrisi (int32, arka plan 0) ve bileşen istatistikleri."""

    labels: np.ndarray
    components: tuple[Component, ...]
    connectivity: int

    @property
    def count(self) -> int:
        """Bileşen sayısı (arka plan hariç)."""
        return len(self.components)

    def summary(self) -> str:
        se = "3×3 cross" if self.connectivity == 4 else "3×3 square"
        return f"{self.count} component{'s' if self.count != 1 else ''} · {self.connectivity}-connected (B = {se})"


def _validate_connectivity(connectivity) -> int:
    if isinstance(connectivity, bool):
        raise AlgorithmParameterError(f"Connectivity must be 4 or 8 (got {connectivity!r}).")
    key = str(connectivity)
    if key not in CC_CONNECTIVITY:
        raise AlgorithmParameterError(f"Connectivity must be 4 or 8 (got {connectivity!r}).")
    return CC_CONNECTIVITY[key]


def _raster_order_lut(labels: np.ndarray, n: int) -> np.ndarray:
    """Eski etiket → yeni etiket tablosu: bileşenler raster sıradaki ilk piksellerine göre 1…n−1."""
    flat = labels.ravel()
    first = np.full(n, flat.size, np.int64)
    np.minimum.at(first, flat, np.arange(flat.size))
    order = np.argsort(first[1:], kind="stable") + 1        # eski etiketler, yeni sırayla
    lut = np.zeros(n, np.int32)
    lut[order] = np.arange(1, n, dtype=np.int32)
    return lut


def label_components(mask: np.ndarray, connectivity=8) -> ComponentReport:
    """OpenCV ``connectedComponentsWithStats`` ile etiketleme ve istatistikler.

    OpenCV'nin iç numaralaması algoritmaya göre değişebildiği için etiketler
    raster sırasına göre yeniden numaralanır; istatistikler aynı tabloyla taşınır.
    """
    a = _validate_mask(mask)
    conn = _validate_connectivity(connectivity)
    n, raw, stats, centroids = cv2.connectedComponentsWithStats(a.astype(np.uint8), connectivity=conn,
                                                                ltype=cv2.CV_32S)
    lut = _raster_order_lut(raw, n)
    labels = lut[raw]
    comps = [None] * (n - 1)
    for old in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[old])
        comps[lut[old] - 1] = Component(int(lut[old]), area, x, y, w, h,
                                        float(centroids[old][0]), float(centroids[old][1]))
    return ComponentReport(labels, tuple(comps), conn)


def label_components_iterative(mask: np.ndarray, connectivity=8) -> np.ndarray:
    """Dersin yöntemi (laboratuvar ``ccGrow``): X_k = (X_{k−1} ⊕ B) ∩ A, X_k = X_{k−1} olunca dur.

    Tohum, henüz etiketlenmemiş ilk (raster sırası) ön plan pikselidir; her
    bileşene sıradaki etiket verilir. Genişletme yalnızca X'in sınırlayıcı
    kutusu + 1 piksellik pencerede yapılır (X her adımda en fazla 1 piksel
    büyüdüğü için sonuç tam görüntüdekiyle aynıdır). Referans/eğitim amaçlıdır;
    GUI ``label_components`` (OpenCV) kullanır.
    """
    a = _validate_mask(mask)
    conn = _validate_connectivity(connectivity)
    se = structuring_element("cross" if conn == 4 else "rect", 3, 3)
    h, w = a.shape
    labels = np.zeros((h, w), np.int32)
    remaining = a.copy()
    flat = remaining.ravel()          # remaining ile aynı bellek
    nxt, start = 1, 0
    while True:
        hits = np.flatnonzero(flat[start:])
        if hits.size == 0:
            break
        start += int(hits[0])
        r, c = divmod(start, w)
        r0, r1, c0, c1 = r, r + 1, c, c + 1
        x = np.zeros((1, 1), bool)
        x[0, 0] = True
        while True:
            R0, R1, C0, C1 = max(r0 - 1, 0), min(r1 + 1, h), max(c0 - 1, 0), min(c1 + 1, w)
            win = np.zeros((R1 - R0, C1 - C0), bool)
            win[r0 - R0:r1 - R0, c0 - C0:c1 - C0] = x
            grown = binary_dilation(win, se) & a[R0:R1, C0:C1]
            if np.array_equal(grown, win):
                break
            rows, cols = np.nonzero(grown)
            r0, r1, c0, c1 = R0 + rows.min(), R0 + rows.max() + 1, C0 + cols.min(), C0 + cols.max() + 1
            x = grown[r0 - R0:r1 - R0, c0 - C0:c1 - C0]
        sub = labels[r0:r1, c0:c1]
        sub[x] = nxt
        remaining[r0:r1, c0:c1] &= ~x
        nxt += 1
    return labels


def label_colors(n: int) -> np.ndarray:
    """(n+1)×3 BGR palet: 0 = siyah (arka plan); i ≥ 1 altın açı (137.5°) adımlı tonlar.

    Ardışık etiketler belirgin biçimde farklı tonlar alır; ton döngüsel olduğundan
    çok sayıda bileşende benzer renkler yeniden görülebilir (etiketler yine farklıdır).
    """
    palette = np.zeros((n + 1, 3), np.uint8)
    if n:
        i = np.arange(1, n + 1)
        hsv = np.zeros((n, 1, 3), np.uint8)
        hsv[:, 0, 0] = np.round((i * 137.50776) % 360 / 2).astype(np.uint8) % 180
        hsv[:, 0, 1] = np.where(i % 2, 200, 150).astype(np.uint8)
        hsv[:, 0, 2] = np.where(i % 3 == 0, 205, 255).astype(np.uint8)
        palette[1:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    return palette


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """int32 etiket matrisini BGR görüntüye çevirir; indeksleme int ile yapılır (8-bit sarma yok)."""
    labels = np.asarray(labels)
    n = int(labels.max()) if labels.size else 0
    return label_colors(n)[labels]


# --------------------------------------------------------------------------
# Convex hull, thinning, thickening — slayt 63–71 (Gonzalez Şekil 9.21, 9.23, 9.24)
# --------------------------------------------------------------------------

# Şekil 9.21 — convex hull için B¹…B⁴ (merkez arka plan olmalı; bir yan sütun/satır ön plan)
HULL_KERNELS = tuple(np.array(k) for k in (
    [[1, 0, 0], [1, -1, 0], [1, 0, 0]],
    [[1, 1, 1], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, -1, 1], [0, 0, 1]],
    [[0, 0, 0], [0, -1, 0], [1, 1, 1]],
))
# Şekil 9.23 — thinning için B¹…B⁸ (1 = ön plan, −1 = arka plan, 0 = önemsiz)
THIN_KERNELS = tuple(np.array(k) for k in (
    [[-1, -1, -1], [0, 1, 0], [1, 1, 1]],
    [[0, -1, -1], [1, 1, -1], [1, 1, 0]],
    [[1, 0, -1], [1, 1, -1], [1, 0, -1]],
    [[1, 1, 0], [1, 1, -1], [0, -1, -1]],
    [[1, 1, 1], [0, 1, 0], [-1, -1, -1]],
    [[0, 1, 1], [-1, 1, 1], [-1, -1, 0]],
    [[-1, 0, 1], [-1, 1, 1], [-1, 0, 1]],
    [[-1, -1, 0], [-1, 1, 1], [0, 1, 1]],
))
HULL_METHODS = ("contour", "morph")
HULL_DISPLAYS = ("filled", "overlay")
THIN_METHODS = ("hmt", "zhang_suen")
THICKEN_CLEANUP = ("remove", "keep")


@dataclass(frozen=True)
class IterationReport:
    """Yinelemeli işlemin özeti (durum çubuğunda gösterilir)."""

    operation: str
    cycles: int          # tamamlanan tam döngü sayısı (son, değişiklik olmayan kontrol dahil)
    converged: bool      # X artık değişmiyor mu (aksi halde döngü sınırında durdu)
    before: int          # ön plan piksel sayısı, önce / sonra
    after: int

    def summary(self) -> str:
        state = f"converged after {self.cycles} cycle{'s' if self.cycles != 1 else ''}" if self.converged \
            else f"stopped at the {self.cycles}-cycle limit (not converged)"
        return f"{self.operation}: {state}, foreground {self.before} → {self.after} px"


@dataclass(frozen=True)
class HullReport:
    objects: int
    object_area: int
    hull_area: int

    def summary(self) -> str:
        solidity = self.object_area / self.hull_area if self.hull_area else 0.0
        return (f"{self.objects} object{'s' if self.objects != 1 else ''}, convex hull {self.hull_area} px "
                f"(solidity {solidity:.2f})")


def _validate_cycles(max_cycles) -> int:
    n = _validate_int(max_cycles, "Max cycles")
    if n < 0:
        raise AlgorithmParameterError(f"Max cycles must be ≥ 0 (0 = until convergence) (got {n}).")
    return n


def _fill_convex_polygon(vertices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Merkezi kapalı dışbükey poligonun içinde/üstünde olan pikseller (tam sayı aritmetiği, kesin).

    ``fillPoly`` yerine yarı düzlem testi: kenar üzerindeki merkezler de dahildir;
    1 köşe → tek piksel, 2 köşe → doğru parçası üzerindeki pikseller.
    """
    v = np.asarray(vertices, np.int64).reshape(-1, 2)
    out = np.zeros(shape, bool)
    x0, y0 = v.min(axis=0)
    x1, y1 = v.max(axis=0)
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    nxt = np.roll(v, -1, axis=0)
    area2 = int(np.sum(v[:, 0] * nxt[:, 1] - nxt[:, 0] * v[:, 1]))     # 2 × işaretli alan (shoelace)
    if area2 != 0:
        sign = 1 if area2 > 0 else -1
        inside = np.ones(xs.shape, bool)
        for (ax, ay), (bx, by) in zip(v, nxt):
            inside &= sign * ((bx - ax) * (ys - ay) - (by - ay) * (xs - ax)) >= 0
    else:                                   # tek nokta ya da doğrusal köşeler → doğru parçası
        order = np.lexsort((v[:, 1], v[:, 0]))
        (ax, ay), (bx, by) = v[order[0]], v[order[-1]]
        inside = (((bx - ax) * (ys - ay) - (by - ay) * (xs - ax)) == 0) \
            & (xs >= min(ax, bx)) & (xs <= max(ax, bx)) & (ys >= min(ay, by)) & (ys <= max(ay, by))
    out[y0:y1 + 1, x0:x1 + 1] = inside
    return out


def convex_hull_mask(mask: np.ndarray, method: str = "contour") -> tuple[np.ndarray, list[np.ndarray], int]:
    """Her nesnenin dışbükey zarfı; döndürür: (zarfların birleşimi ∪ A, zarf köşe listeleri, nesne sayısı).

    contour — slayt 65: ``findContours(RETR_EXTERNAL)`` ile DIŞ konturlar (delikler zarfı
              etkilemez), ``convexHull`` ile köşeler; zarf, merkezi poligonda olan piksellerdir.
    morph   — Şekil 9.21: her nesne için X_kⁱ = (X_{k−1}ⁱ ⊛ Bⁱ) ∪ X_{k−1}ⁱ, i = 1…4, yakınsayana dek;
              C(A) = ∪ Dⁱ. Büyüme nesnenin sınırlayıcı kutusuyla sınırlıdır (ders kitabının önerisi).
              Sonuç, gerçek dışbükey zarfın 4 yönlü dijital yaklaşımıdır (köşeler kutuya doğru dolar).
    """
    a = _validate_mask(mask)
    if method not in HULL_METHODS:
        raise AlgorithmParameterError(f"Hull method must be 'contour' or 'morph' (got {method!r}).")
    out = a.copy()
    hulls: list[np.ndarray] = []
    if method == "contour":
        contours, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            hull = cv2.convexHull(c).reshape(-1, 2)
            hulls.append(hull)
            out |= _fill_convex_polygon(hull, a.shape)
        return out, hulls, len(contours)
    report = label_components(a, 8)
    for comp in report.components:
        ys, xs = slice(comp.y, comp.y + comp.height), slice(comp.x, comp.x + comp.width)
        obj = report.labels[ys, xs] == comp.label
        hull = np.zeros_like(obj)
        for k in HULL_KERNELS:
            x = obj.copy()
            for _ in range(obj.size + 1):
                nxt = x | hit_or_miss_mask(x, k)
                if np.array_equal(nxt, x):
                    break
                x = nxt
            hull |= x
        out[ys, xs] |= hull
    return out, hulls, report.count


def thin_mask(mask: np.ndarray, method: str = "hmt", max_cycles: int = 0) -> tuple[np.ndarray, IterationReport]:
    """Thinning, yakınsayana kadar (``max_cycles`` = 0) ya da en fazla ``max_cycles`` döngü.

    hmt        — Şekil 9.23: A ⊗ B = A − (A ⊛ B); bir döngü B¹…B⁸'i SIRAYLA uygular
                 ({A ⊗ B} = ((…(A ⊗ B¹) ⊗ B²) …) ⊗ B⁸); hiçbir piksel silinmeyince durur.
    zhang_suen — slayt 68: iki alt-adımlı paralel inceltme (Zhang & Suen, 1984).
    Her değişen döngü en az bir piksel sildiğinden yakınsama en fazla |A| + 1 döngüde garantidir;
    güvenlik sınırı olarak bu değer kullanılır.
    """
    a = _validate_mask(mask)
    if method not in THIN_METHODS:
        raise AlgorithmParameterError(f"Thinning method must be 'hmt' or 'zhang_suen' (got {method!r}).")
    limit = _validate_cycles(max_cycles) or int(a.sum()) + 1
    x, cycles, changed = a.copy(), 0, True
    while cycles < limit and changed:
        changed = False
        if method == "hmt":
            for k in THIN_KERNELS:
                hits = hit_or_miss_mask(x, k)
                if hits.any():
                    x &= ~hits
                    changed = True
        else:
            for step in (1, 2):
                delete = _zhang_suen_candidates(x, step)
                if delete.any():
                    x &= ~delete
                    changed = True
        cycles += 1
    name = "Thinning (hit-or-miss B¹…B⁸)" if method == "hmt" else "Thinning (Zhang–Suen)"
    return x, IterationReport(name, cycles, not changed, int(a.sum()), int(x.sum()))


def _zhang_suen_candidates(x: np.ndarray, step: int) -> np.ndarray:
    """Zhang–Suen alt-adımında silinecek pikseller. Komşular P2…P9: kuzeyden saat yönünde; dışarısı 0."""
    p = np.pad(x, 1).astype(np.uint8)
    h, w = x.shape
    n = [p[0:h, 1:w + 1], p[0:h, 2:w + 2], p[1:h + 1, 2:w + 2], p[2:h + 2, 2:w + 2],
         p[2:h + 2, 1:w + 1], p[2:h + 2, 0:w], p[1:h + 1, 0:w], p[0:h, 0:w]]          # P2 … P9
    b = sum(v.astype(np.int16) for v in n)
    a = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
    p2, p4, p6, p8 = n[0], n[2], n[4], n[6]
    if step == 1:
        c3, c4 = (p2 * p4 * p6) == 0, (p4 * p6 * p8) == 0
    else:
        c3, c4 = (p2 * p4 * p8) == 0, (p2 * p6 * p8) == 0
    return x & (b >= 2) & (b <= 6) & (a == 1) & c3 & c4


def thicken_mask(mask: np.ndarray, max_cycles: int = 0, cleanup: str = "remove") -> tuple[np.ndarray, IterationReport]:
    """Thickening, Şekil 9.24 / laboratuvar ``thicken``: arka planı incelt, tümleyenini al.

    T = (Aᶜ ⊗ {B})ᶜ — arka plan (B¹…B⁸ ile) inceltildikçe nesneler büyür; yakınsadığında
    nesneler, arka planın iskeletine kadar kalınlaşır (komşu nesneler birleşmez: aradaki
    iskelet arka plan kalır). ``cleanup="remove"``: T'nin A'ya değmeyen (8-bağlantılı)
    kopuk parçaları silinir — ders kitabının önerdiği son adım. ``max_cycles`` kalınlaşma
    miktarını sınırlar (0 = yakınsayana kadar; büyük görüntüde yavaş olabilir).

    Görüntü sınırı: projenin genel kuralı "görüntü dışı arka plandır". Aᶜ bu yüzden 1 piksellik,
    DEĞİŞMEYEN bir arka plan dolgusuyla (değer 1) genişletilir; yalnızca görüntü içindeki pikseller
    inceltilebilir. (Dışarısı 0 sayılırsa görüntü çerçevesi bir nesne gibi büyür ve kenara değen
    ayrı nesneleri bu şerit üzerinden birleştirebilir.) Böylece silinen her arka plan pikseli o anki
    bir nesne pikseline komşudur; ``remove`` yine uygulanır ama A boş değilse silecek parça kalmaz.
    """
    a = _validate_mask(mask)
    if cleanup not in THICKEN_CLEANUP:
        raise AlgorithmParameterError(f"Cleanup must be 'remove' or 'keep' (got {cleanup!r}).")
    c = np.pad(~a, 1, constant_values=True)            # Aᶜ + sabit arka plan dolgusu
    inner = np.zeros_like(c)
    inner[1:-1, 1:-1] = True
    limit = _validate_cycles(max_cycles) or int(c[inner].sum()) + 1
    cycles, changed = 0, True
    while cycles < limit and changed:                  # Şekil 9.23 inceltmesi: B¹…B⁸ sırayla
        changed = False
        for k in THIN_KERNELS:
            hits = hit_or_miss_mask(c, k) & inner      # dolgu pikselleri silinmez
            if hits.any():
                c &= ~hits
                changed = True
        cycles += 1
    t = ~c[1:-1, 1:-1]
    if cleanup == "remove":
        labels = label_components(t, 8).labels
        keep = np.unique(labels[a & (labels > 0)])
        t = np.isin(labels, keep) & (labels > 0)
    return t, IterationReport("Thickening (background thinning)", cycles, not changed, int(a.sum()), int(t.sum()))


# --------------------------------------------------------------------------
# Skeletonization ve pruning — slayt 72–81 (Gonzalez Şekil 9.26, 9.27)
# --------------------------------------------------------------------------

# Şekil 9.27(b) — uç nokta çekirdekleri B¹…B⁸ (laboratuvardaki ENDPT ile aynı; 1 / −1 / 0 = önemsiz).
# B¹…B⁴: tek komşu bir kenar komşusu (köşeleri önemsiz); B⁵…B⁸: tek komşu bir köşegen komşu.
PRUNE_KERNELS = tuple(np.array(k) for k in (
    [[0, -1, -1], [1, 1, -1], [0, -1, -1]],
    [[0, 1, 0], [-1, 1, -1], [-1, -1, -1]],
    [[-1, -1, 0], [-1, 1, 1], [-1, -1, 0]],
    [[-1, -1, -1], [-1, 1, -1], [0, 1, 0]],
    [[1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
    [[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]],
    [[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    [[-1, -1, -1], [-1, 1, -1], [1, -1, -1]],
))
SKELETON_DISPLAYS = ("skeleton", "levels", "layer", "reconstruction")
PRUNE_DISPLAYS = ("result", "overlay", "x1", "x2", "x3")


@dataclass(frozen=True)
class SkeletonLevel:
    k: int
    eroded: int          # |A ⊖ kB|
    opened: int          # |(A ⊖ kB) ∘ B|
    skeleton: int        # |S_k(A)|


@dataclass(frozen=True)
class SkeletonReport:
    """Morfolojik iskeletin seviye seviye özeti (durum çubuğunda gösterilir)."""

    se: str
    levels: tuple[SkeletonLevel, ...]   # k = 0…K (A boşsa boş)
    object_px: int
    skeleton_px: int
    components: int                     # S(A)'nın 8-bağlantılı bileşen sayısı
    reconstructed: bool                 # ⋃ (S_k ⊕ kB) == A
    shown: str = ""                     # gösterilen katman ("layer" gösteriminde)

    @property
    def K(self) -> int:
        return len(self.levels) - 1

    def summary(self) -> str:
        if not self.levels:
            return f"Skeleton ({self.se}): A is empty — S(A) = ∅"
        # önemli bilgiler önce (durum çubuğu uzun metni sondan kırpar); seviye listesi en sonda
        sk = ", ".join(str(lv.skeleton) for lv in self.levels)
        return (f"Skeleton ({self.se}): K = {self.K}, |S(A)| = {self.skeleton_px} of {self.object_px} px, "
                f"{self.components} part{'s' if self.components != 1 else ''}, "
                f"reconstruction {'= A ✓' if self.reconstructed else '≠ A ✗'}{self.shown}; |S_k| k=0…{self.K}: {sk}")


@dataclass(frozen=True)
class PruneReport:
    n: int
    thinning_cycles: int    # X1 için tamamlanan döngü (değişiklik kalmazsa n'den önce durur)
    a_px: int
    x1_px: int
    end_points: int         # |X2|
    x3_px: int
    x4_px: int

    def summary(self) -> str:
        return (f"Pruning n = {self.n}: X1 = A ⊗ {{B}} ({self.thinning_cycles} cycle"
                f"{'s' if self.thinning_cycles != 1 else ''}) {self.a_px} → {self.x1_px} px; "
                f"X2 = {self.end_points} end point{'s' if self.end_points != 1 else ''}; X3 {self.x3_px} px; "
                f"X4 = X1 ∪ X3 {self.x4_px} px; removed {self.a_px - self.x4_px} px")


def _require_eroding_se(se: StructuringElement) -> StructuringElement:
    """1×1 SE (yalnızca orijin) reddedilir: A ⊖ B = A olur, K tanımsızdır (erosion hiç bitmez)."""
    if se.mask.sum() == 1:
        raise AlgorithmParameterError("Skeleton SE must have more than one cell (a 1 × 1 SE never erodes A).")
    return se


def _skeleton_se(shape: str, width: int, height: int) -> StructuringElement:
    """İskelet SE'si: orijin her zaman merkezde (B orijini içerir → A ⊖ kB ⊆ A, ⋃ (S_k ⊕ kB) ⊆ A)."""
    return _require_eroding_se(structuring_element(shape, width, height))


def skeleton_levels(mask: np.ndarray, se: StructuringElement) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Her k = 0…K için (A ⊖ kB, (A ⊖ kB) ∘ B, S_k(A)); K = max{k | A ⊖ kB ≠ ∅}.

    S_k(A) = (A ⊖ kB) − (A ⊖ kB) ∘ B; A ⊖ kB, B ile k kez art arda erosion'dur. A boşsa liste boştur.
    Her erosion en az bir pikseli siler (B orijini içerir, A sonlu), bu yüzden döngü en fazla |A| adımdır.
    """
    a = _validate_mask(mask)
    _require_eroding_se(se)
    out = []
    e = a
    while e.any():
        o = binary_opening(e, se)
        out.append((e, o, e & ~o))
        e = binary_erosion(e, se)
    return out


def skeleton_reconstruct(layers, se: StructuringElement) -> np.ndarray:
    """A = ⋃_{k=0…K} (S_k(A) ⊕ kB) — S_k, B ile k kez art arda dilation."""
    if not layers:
        raise AlgorithmParameterError("Reconstruction needs at least one skeleton level.")
    out = np.zeros(layers[0][2].shape, bool)
    for k, (_, _, s) in enumerate(layers):
        out |= binary_dilation(s, se, k) if k else s
    return out


def prune_mask(mask: np.ndarray, n: int = 3) -> tuple[dict[str, np.ndarray], PruneReport]:
    """Pruning (Şekil 9.27 / laboratuvar ``prune``), dört adım:

    X1 = A ⊗ {B}              — uç nokta çekirdekleri B¹…B⁸ ile SIRALI inceltme, n döngü
                                (bir döngüde değişiklik olmazsa erken durur);
    X2 = ⋃_{k=1…8} (X1 ⊛ Bᵏ)  — X1'in uç noktaları;
    X3 = (X2 ⊕ H) ∩ A         — H = 3×3 kare, n kez (A ile koşullu dilation);
    X4 = X1 ∪ X3.
    """
    a = _validate_mask(mask)
    n = _validate_int(n, "Spur length n")
    if n < 1:
        raise AlgorithmParameterError(f"Spur length n must be ≥ 1 (got {n}).")
    x1, cycles = a.copy(), 0
    while cycles < n:
        changed = False
        for k in PRUNE_KERNELS:
            hits = hit_or_miss_mask(x1, k)
            if hits.any():
                x1 &= ~hits
                changed = True
        cycles += 1
        if not changed:
            break
    x2 = np.zeros_like(a)
    for k in PRUNE_KERNELS:
        x2 |= hit_or_miss_mask(x1, k)
    h = structuring_element("rect", 3, 3)
    x3 = x2.copy()
    for _ in range(n):
        x3 = binary_dilation(x3, h) & a
    x4 = x1 | x3
    report = PruneReport(n, cycles, int(a.sum()), int(x1.sum()), int(x2.sum()), int(x3.sum()), int(x4.sum()))
    return {"A": a, "X1": x1, "X2": x2, "X3": x3, "X4": x4}, report


BOX_MIN_SIZE = 3   # bu boyuttan küçük bileşenlere kutu/merkez çizilmez (rengi örtülmesin)


def draw_component_boxes(color: np.ndarray, report: ComponentReport) -> np.ndarray:
    """Sınırlayıcı kutuları ve ağırlık merkezlerini (küçük artı) beyaz çizer.

    Kutu bileşenin 1 piksel DIŞINA çizilir (bileşenin kendi piksellerini örtmez).
    En az ``BOX_MIN_SIZE`` × ``BOX_MIN_SIZE`` olmayan bileşenlere (tek pikseller,
    ince kırıntılar) çizim yapılmaz; tabloda yine listelenirler.
    """
    out = color.copy()
    for comp in report.components:
        if comp.width < BOX_MIN_SIZE or comp.height < BOX_MIN_SIZE:
            continue
        cv2.rectangle(out, (comp.x - 1, comp.y - 1), (comp.x + comp.width, comp.y + comp.height), (255, 255, 255), 1)
        cv2.drawMarker(out, (int(round(comp.cx)), int(round(comp.cy))), (255, 255, 255), cv2.MARKER_CROSS, 5, 1)
    return out


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
# Gri seviye erosion / dilation / opening / closing / gradient — slayt 82–85
# (Gonzalez Şekil 9.37, 9.39, 9.41; laboratuvar gErode, gDilate, gOpen, gClose, gGradient)
# --------------------------------------------------------------------------
#
# Yukarıdaki gray_erosion / gray_dilation (Top-hat ile ortak) aynen kullanılır: düz SE,
# erosion min f(z + s), dilation max f(z − s) (B̂), görüntü dışı YOK SAYILIR (laboratuvar
# gWindow). İkili morfolojideki "dışarısı arka plan" kuralı burada uygulanmaz.
# SE seçenekleri laboratuvarın SEPicker'ı ile aynıdır: rect, cross, ellipse (cv2 elipsi),
# hline (1 × n), vline (n × 1), custom (hücreler + orijin). Laboratuvar boyutları 3 ve 5.

GRAY_SE_SHAPES = ("rect", "cross", "ellipse", "hline", "vline", "custom")
GRAY_OPEN_DISPLAYS = ("opening", "erosion")
GRAY_CLOSE_DISPLAYS = ("closing", "dilation")
GRAY_GRADIENT_DISPLAYS = ("gradient", "dilation", "erosion")
DEFAULT_CUSTOM_SE = "010/011/000@1,1"   # asimetrik: merkez + üst + sağ (yansıması merkez + alt + sol)


def gray_structuring_element(shape: str = "rect", size: int = 3, custom: str = DEFAULT_CUSTOM_SE) -> StructuringElement:
    """Laboratuvarın makeSE / özel SE'si. ``size`` tek sayı (custom'da kullanılmaz)."""
    if shape not in GRAY_SE_SHAPES:
        raise AlgorithmParameterError(f"SE shape must be one of {', '.join(GRAY_SE_SHAPES)} (got {shape!r}).")
    if shape == "custom":
        parsed = MaskParam.decode(custom, 31)
        if parsed is None:
            raise AlgorithmParameterError(f"Custom SE must look like '010/011/000@1,1' with at least one cell (got {custom!r}).")
        grid, (r, c) = parsed
        return StructuringElement(np.array(grid, np.uint8), (c, r))
    n = validate_ksize(size, name="SE size")
    if shape == "hline":
        return StructuringElement(np.ones((1, n), np.uint8), (n // 2, 0))
    if shape == "vline":
        return StructuringElement(np.ones((n, 1), np.uint8), (0, n // 2))
    return structuring_element(shape, n, n)


def gray_gradient(f: np.ndarray, se: StructuringElement) -> np.ndarray:
    """g = (f ⊕ b) − (f ⊖ b), laboratuvardaki gSub gibi 0'da kırpılır (orijin B'de ise zaten ≥ 0)."""
    return np.clip(gray_dilation(f, se).astype(np.int16) - gray_erosion(f, se).astype(np.int16), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class GrayReport:
    """Gri seviye işlemin özeti: SE ve giriş/çıkış yoğunluk istatistikleri."""

    operation: str
    se: str
    before: tuple[float, int, int]      # (ortalama, min, maks)
    after: tuple[float, int, int]
    extra: str = ""

    def summary(self) -> str:
        (m0, a0, b0), (m1, a1, b1) = self.before, self.after
        return (f"{self.operation} ({self.se}): mean {m0:.1f} → {m1:.1f}, range {a0}–{b0} → {a1}–{b1}"
                f"{self.extra}")


def _stats(x: np.ndarray) -> tuple[float, int, int]:
    return float(x.mean()), int(x.min()), int(x.max())


def _se_text(shape: str, se: StructuringElement) -> str:
    h, w = se.mask.shape
    ax, ay = se.anchor
    return f"{shape} {w}×{h}" + ("" if shape != "custom" else f", origin r{ay} c{ax}")


def _gray_apply(image, shape, size, custom, name, fn, extra_fn=None) -> ProcessResult:
    se = gray_structuring_element(shape, size, custom)
    f = intensity_channel(image)
    out = fn(f, se)
    extra = extra_fn(f, se) if extra_fn else ""
    return ProcessResult(match_layout(out, image), GrayReport(name, _se_text(shape, se), _stats(f), _stats(out), extra))


def grayscale_erosion(image: np.ndarray, shape: str = "rect", size: int = 3,
                      custom: str = DEFAULT_CUSTOM_SE) -> ProcessResult:
    """[f ⊖ b](x, y) = min f(x + s, y + t), (s, t) ∈ b — eşikleme yok; renkli girdide parlaklık Y."""
    return _gray_apply(image, shape, size, custom, "Grayscale erosion", gray_erosion)


def grayscale_dilation(image: np.ndarray, shape: str = "rect", size: int = 3,
                       custom: str = DEFAULT_CUSTOM_SE) -> ProcessResult:
    """[f ⊕ b](x, y) = max f(x − s, y − t), (s, t) ∈ b (yansıtılmış SE)."""
    return _gray_apply(image, shape, size, custom, "Grayscale dilation", gray_dilation)


def grayscale_opening(image: np.ndarray, shape: str = "rect", size: int = 3, custom: str = DEFAULT_CUSTOM_SE,
                      display: str = "opening") -> ProcessResult:
    """f ∘ b = (f ⊖ b) ⊕ b. ``display="erosion"``: 1. aşama f ⊖ b."""
    if display not in GRAY_OPEN_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be 'opening' or 'erosion' (got {display!r}).")
    if display == "erosion":
        return _gray_apply(image, shape, size, custom, "Grayscale opening · stage 1 f ⊖ b", gray_erosion)
    return _gray_apply(image, shape, size, custom, "Grayscale opening", gray_opening)


def grayscale_closing(image: np.ndarray, shape: str = "rect", size: int = 3, custom: str = DEFAULT_CUSTOM_SE,
                      display: str = "closing") -> ProcessResult:
    """f • b = (f ⊕ b) ⊖ b. ``display="dilation"``: 1. aşama f ⊕ b."""
    if display not in GRAY_CLOSE_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be 'closing' or 'dilation' (got {display!r}).")
    if display == "dilation":
        return _gray_apply(image, shape, size, custom, "Grayscale closing · stage 1 f ⊕ b", gray_dilation)
    return _gray_apply(image, shape, size, custom, "Grayscale closing", gray_closing)


def grayscale_gradient(image: np.ndarray, shape: str = "rect", size: int = 3, custom: str = DEFAULT_CUSTOM_SE,
                       display: str = "gradient") -> ProcessResult:
    """g = (f ⊕ b) − (f ⊖ b). ``display``: gradient | dilation (f ⊕ b) | erosion (f ⊖ b)."""
    if display not in GRAY_GRADIENT_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be one of {', '.join(GRAY_GRADIENT_DISPLAYS)} (got {display!r}).")

    def parts(f, se):
        d, e = gray_dilation(f, se), gray_erosion(f, se)
        return f"; f ⊕ b mean {d.mean():.1f}, f ⊖ b mean {e.mean():.1f}"
    fn = {"gradient": gray_gradient, "dilation": gray_dilation, "erosion": gray_erosion}[display]
    name = {"gradient": "Grayscale gradient", "dilation": "Grayscale gradient · f ⊕ b",
            "erosion": "Grayscale gradient · f ⊖ b"}[display]
    return _gray_apply(image, shape, size, custom, name, fn, parts)


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


def boundary_extraction(image: np.ndarray, binarize: str = "otsu", thresh: int = 127,
                        foreground: str = "bright", shape: str = "rect", width: int = 3, height: int = 3,
                        anchor_x: int = -1, anchor_y: int = -1) -> np.ndarray:
    """Sınır çıkarma β(A) = A − (A ⊖ B); sınır pikselleri 255, diğerleri 0."""
    a, se = _prepare(image, binarize, thresh, foreground, shape, width, height, anchor_x, anchor_y)
    return _to_output(boundary_mask(a, se), image)


def hole_filling(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
                 method: str = "flood", seeds: str = "largest", shape: str = "cross") -> np.ndarray:
    """Delik doldurma; çıktı A ∪ (doldurulan delikler), 0/255.

    flood          — tüm kapalı delikler tek seferde (4-bağlantılı, slayt 59).
    reconstruction — X_k = (X_{k−1} ⊕ B) ∩ Aᶜ; yalnızca ``seeds`` politikasının
                     seçtiği delik(ler) dolar. ``shape``: 3×3 cross (4-bağlantılı) / rect (8).
    """
    if method not in HOLE_METHODS:
        raise AlgorithmParameterError(f"Method must be 'flood' or 'reconstruction' (got {method!r}).")
    if shape not in HOLE_SE_SHAPES:
        raise AlgorithmParameterError(f"Hole-filling SE must be 'cross' or 'rect' (got {shape!r}).")
    if seeds not in HOLE_SEED_POLICIES:
        raise AlgorithmParameterError(f"Seed policy must be one of {', '.join(HOLE_SEED_POLICIES)} (got {seeds!r}).")
    a = to_binary(image, binarize, thresh, foreground)
    if method == "flood":
        return _to_output(fill_holes_flood(a), image)
    conn = HOLE_SE_SHAPES[shape]
    filled, _ = fill_holes_reconstruction(a, auto_seeds(a, seeds, conn), conn)
    return _to_output(filled, image)


def connected_components(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
                         connectivity: str = "8", display: str = "labels") -> ProcessResult:
    """Bağlantılı bileşenleri etiketler; renkli etiket görüntüsü + ``ComponentReport`` döndürür.

    Çıktı her zaman renkli (BGR) etiket görüntüsüdür — gri girdide de, çünkü
    etiketler renkle ayırt edilir; BGRA girdide orijinal alfa eklenir.
    ``display="boxes"``: kutular ve ağırlık merkezleri de çizilir.
    """
    if display not in CC_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be 'labels' or 'boxes' (got {display!r}).")
    a = to_binary(image, binarize, thresh, foreground)
    report = label_components(a, connectivity)
    color = colorize_labels(report.labels)
    if display == "boxes":
        color = draw_component_boxes(color, report)
    _, alpha = split_alpha(image)
    return ProcessResult(merge_alpha(color, alpha), report)


HULL_COLORS = {  # BGR — overlay gösterimi
    "object": (200, 200, 200),     # A: açık gri
    "added": (120, 85, 20),        # zarfın eklediği alan (C(A) − A): koyu mavi-camgöbeği
    "contour": (90, 220, 90),      # nesnenin dış konturu: yeşil
    "hull": (0, 215, 255),         # zarfın sınırı: sarı
}


def convex_hull(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
                method: str = "contour", display: str = "filled") -> ProcessResult:
    """Her nesnenin dışbükey zarfı. ``filled``: C(A) ∪ A ikili (0/255);
    ``overlay``: renkli — nesne gri, zarfın eklediği alan mavi, dış kontur yeşil, zarf sınırı sarı."""
    if display not in HULL_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be 'filled' or 'overlay' (got {display!r}).")
    a = to_binary(image, binarize, thresh, foreground)
    hull, _, objects = convex_hull_mask(a, method)
    report = HullReport(objects, int(a.sum()), int(hull.sum()))
    if display == "filled":
        return ProcessResult(_to_output(hull, image), report)
    color = np.zeros(a.shape + (3,), np.uint8)
    color[hull & ~a] = HULL_COLORS["added"]
    color[a] = HULL_COLORS["object"]
    contours, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(color, contours, -1, HULL_COLORS["contour"], 1)
    color[boundary_mask(hull, structuring_element("cross", 3, 3))] = HULL_COLORS["hull"]
    _, alpha = split_alpha(image)
    return ProcessResult(merge_alpha(color, alpha), report)


def thinning(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
             method: str = "hmt", max_cycles: int = 0) -> ProcessResult:
    a = to_binary(image, binarize, thresh, foreground)
    x, report = thin_mask(a, method, max_cycles)
    return ProcessResult(_to_output(x, image), report)


def thickening(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
               max_cycles: int = 10, cleanup: str = "remove") -> ProcessResult:
    a = to_binary(image, binarize, thresh, foreground)
    t, report = thicken_mask(a, max_cycles, cleanup)
    return ProcessResult(_to_output(t, image), report)


SKELETON_COLORS = {  # BGR — "layer" gösterimi
    "object": (70, 70, 70),        # A: koyu gri
    "eroded": (150, 150, 150),     # (A ⊖ kB) ∘ B: açık gri (opening'in geri getirdiği kısım)
    "skeleton": (0, 215, 255),     # S_k(A): sarı
}
PRUNE_COLORS = {  # BGR — "overlay" gösterimi
    "removed": (60, 60, 230),      # A − X4: budanan pikseller, kırmızı
    "kept": (235, 235, 235),       # X1: inceltmeden sağ çıkan, beyaz
    "restored": (90, 220, 90),     # X4 − X1: X3 ile geri büyütülen uçlar, yeşil
    "end_point": (0, 215, 255),    # X2: X1'in uç noktaları, sarı
}


def _color_output(color: np.ndarray, image: np.ndarray) -> np.ndarray:
    _, alpha = split_alpha(image)
    return merge_alpha(color, alpha)


def skeletonization(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
                    shape: str = "rect", width: int = 3, height: int = 3, display: str = "skeleton",
                    level: int = 0) -> ProcessResult:
    """Morfolojik iskelet S(A) = ⋃ S_k(A) (Şekil 9.26).

    skeleton       — S(A), 0/255.
    levels         — renkli: A koyu gri, S_k pikselleri k'ya göre renk (k = 0 ilk palet rengi).
    layer          — renkli, seçilen ``level`` = k için: A koyu gri, (A ⊖ kB) ∘ B açık gri, S_k sarı
                     (k > K ise A ⊖ kB = ∅, yalnızca A görünür).
    reconstruction — ⋃ (S_k ⊕ kB), 0/255 (A ile aynı olmalı).
    """
    if display not in SKELETON_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be one of {', '.join(SKELETON_DISPLAYS)} (got {display!r}).")
    level = _validate_int(level, "Level k")
    if level < 0:
        raise AlgorithmParameterError(f"Level k must be ≥ 0 (got {level}).")
    se = _skeleton_se(shape, width, height)
    a = to_binary(image, binarize, thresh, foreground)
    layers = skeleton_levels(a, se)
    skel = np.zeros_like(a)
    for _, _, s in layers:
        skel |= s
    rec = skeleton_reconstruct(layers, se) if layers else np.zeros_like(a)
    shown = ""
    if display == "layer":
        shown = f"; showing k = {level}" + (f": |S_{level}| = {int(layers[level][2].sum())} px" if level < len(layers)
                                             else " (> K: A ⊖ kB = ∅)")
    report = SkeletonReport(
        f"{width}×{height} {shape}",
        tuple(SkeletonLevel(k, int(e.sum()), int(o.sum()), int(s.sum())) for k, (e, o, s) in enumerate(layers)),
        int(a.sum()), int(skel.sum()), label_components(skel, 8).count, bool(np.array_equal(rec, a)), shown)
    if display == "skeleton":
        return ProcessResult(_to_output(skel, image), report)
    if display == "reconstruction":
        return ProcessResult(_to_output(rec, image), report)
    color = np.zeros(a.shape + (3,), np.uint8)
    color[a] = SKELETON_COLORS["object"]
    if display == "levels":
        palette = label_colors(len(layers))
        for k, (_, _, s) in enumerate(layers):
            color[s] = palette[k + 1]
    elif level < len(layers):
        _, o, s = layers[level]
        color[o] = SKELETON_COLORS["eroded"]
        color[s] = SKELETON_COLORS["skeleton"]
    return ProcessResult(_color_output(color, image), report)


def pruning(image: np.ndarray, binarize: str = "otsu", thresh: int = 127, foreground: str = "bright",
            n: int = 3, display: str = "result") -> ProcessResult:
    """Pruning (Şekil 9.27). ``result``: X4; ``x1`` / ``x2`` / ``x3``: ara adımlar (0/255);
    ``overlay``: budanan kırmızı, X1 beyaz, geri büyütülen uçlar yeşil, X2 uç noktaları sarı."""
    if display not in PRUNE_DISPLAYS:
        raise AlgorithmParameterError(f"Display must be one of {', '.join(PRUNE_DISPLAYS)} (got {display!r}).")
    a = to_binary(image, binarize, thresh, foreground)
    steps, report = prune_mask(a, n)
    if display != "overlay":
        key = "X4" if display == "result" else display.upper()
        return ProcessResult(_to_output(steps[key], image), report)
    color = np.zeros(a.shape + (3,), np.uint8)
    color[a & ~steps["X4"]] = PRUNE_COLORS["removed"]
    color[steps["X1"]] = PRUNE_COLORS["kept"]
    color[steps["X4"] & ~steps["X1"]] = PRUNE_COLORS["restored"]
    color[steps["X2"]] = PRUNE_COLORS["end_point"]
    return ProcessResult(_color_output(color, image), report)


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
