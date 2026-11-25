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
from processing.registry import AlgorithmParameterError, ProcessResult

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
