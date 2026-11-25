"""Gri seviye morfoloji (Chapter 4, slayt 82–85; laboratuvar gErode / gDilate / gOpen / gClose / gGradient).

    [f ⊖ b](x, y) = min f(x + s, y + t)      [f ⊕ b](x, y) = max f(x − s, y − t),  (s, t) ∈ b
    f ∘ b = (f ⊖ b) ⊕ b      f • b = (f ⊕ b) ⊖ b      g = (f ⊕ b) − (f ⊖ b)

Bağımsız referanslar OpenCV kullanmaz:
* ``px_erode`` / ``px_dilate`` — tanımın piksel piksel saf Python hali (laboratuvarın gWindow'u:
  görüntü dışı yok sayılır; boş pencere erosion'da 255, dilation'da 0).
* ``ref_erode`` / ``ref_dilate`` — aynı tanımın NumPy ötelemeli (vektörel) hali; büyük görüntüler için.
  Önce küçük görüntülerde piksel referansıyla birebir doğrulanır.
SE maskeleri (rect, cross, hline, vline) burada elle kurulur; elips maskesi laboratuvarın ELLIPSE
tablosuyla (cv2 çıktısı) karşılaştırılır.
"""

import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from app.controls import MaskControl
from app.main_window import MainWindow
from processing import morphology as M
from processing.analysis import histograms
from processing.catalog import build_registry
from processing.image_io import save_image
from processing.registry import AlgorithmParameterError, MaskParam
from tests.test_morphology_pruning import FIG_927_A, ref_prune
from tests.test_morphology_skeletonization import assert_previous_15_operations_unchanged, ref_skeleton, se_mask, union_skeleton
from tests.test_morphology_convex_hull import u8

# --------------------------------------------------------------------------
# Bağımsız referans
# --------------------------------------------------------------------------


def members(mask, anchor):
    """(s, t) = (sütun, satır) ofsetleri; anchor = (x, y)."""
    ax, ay = anchor
    return [(c - ax, r - ay) for r, c in zip(*np.nonzero(mask))]


def px_erode(f, mask, anchor):
    h, w = f.shape
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            vals = [int(f[y + t, x + s]) for s, t in members(mask, anchor) if 0 <= y + t < h and 0 <= x + s < w]
            out[y, x] = min(vals) if vals else 255
    return out


def px_dilate(f, mask, anchor):
    h, w = f.shape
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            vals = [int(f[y - t, x - s]) for s, t in members(mask, anchor) if 0 <= y - t < h and 0 <= x - s < w]
            out[y, x] = max(vals) if vals else 0
    return out


def _shifted(f, dy, dx, fill):
    """S[y, x] = f[y + dy, x + dx]; dışarısı ``fill`` (int32)."""
    h, w = f.shape
    out = np.full((h, w), fill, np.int32)
    r0, r1 = max(0, -dy), min(h, h - dy)
    c0, c1 = max(0, -dx), min(w, w - dx)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = f[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
    return out


def ref_erode(f, mask, anchor):
    out = np.full(f.shape, 256, np.int32)
    for s, t in members(mask, anchor):
        out = np.minimum(out, _shifted(f, t, s, 256))          # f(x + s, y + t); dışarısı yok sayılır
    return np.minimum(out, 255).astype(np.uint8)


def ref_dilate(f, mask, anchor):
    out = np.full(f.shape, -1, np.int32)
    for s, t in members(mask, anchor):
        out = np.maximum(out, _shifted(f, -t, -s, -1))         # f(x − s, y − t)
    return np.maximum(out, 0).astype(np.uint8)


def ref_open(f, mask, anchor):
    return ref_dilate(ref_erode(f, mask, anchor), mask, anchor)


def ref_close(f, mask, anchor):
    return ref_erode(ref_dilate(f, mask, anchor), mask, anchor)


def ref_gradient(f, mask, anchor):
    return np.maximum(ref_dilate(f, mask, anchor).astype(int) - ref_erode(f, mask, anchor).astype(int), 0).astype(np.uint8)


def ref_luma(img):
    """BT.601 Y = 0.299 R + 0.587 G + 0.114 B (BGR sırası), en yakın tamsayı — testte bağımsız hesap."""
    a = np.asarray(img)
    if a.ndim == 2:
        return a.copy()
    if a.shape[2] == 1:
        return a[..., 0].copy()
    b, g, r = (a[..., k].astype(np.float64) for k in range(3))
    return np.clip(np.rint(0.299 * r + 0.587 * g + 0.114 * b), 0, 255).astype(np.uint8)


LAB_ELLIPSE = {  # laboratuvardaki ELLIPSE tablosu (cv2.getStructuringElement(MORPH_ELLIPSE))
    3: [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    5: [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]],
    7: [[0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0]],
}


def ref_se(shape, n=3, custom=None):
    """(mask, anchor) — laboratuvarın makeSE'si."""
    c = n // 2
    if shape == "rect":
        return np.ones((n, n), np.uint8), (c, c)
    if shape == "cross":
        m = np.zeros((n, n), np.uint8)
        m[c, :] = m[:, c] = 1
        return m, (c, c)
    if shape == "ellipse":
        return np.array(LAB_ELLIPSE[n], np.uint8), (c, c)
    if shape == "hline":
        return np.ones((1, n), np.uint8), (c, 0)
    if shape == "vline":
        return np.ones((n, 1), np.uint8), (0, c)
    grid, (r, col) = custom
    return np.array(grid, np.uint8), (col, r)


ASYM = "010/011/000@1,1"                  # merkez + üst + sağ
ASYM_L = "100/100/111@2,0"                # L, orijin sol alt köşede
OUTSIDE = "110/000/000@2,2"               # orijin hücrelerin dışında (üye değil)
SE_CASES = [("rect", 3, None), ("rect", 5, None), ("cross", 3, None), ("cross", 5, None), ("ellipse", 5, None),
            ("ellipse", 7, None), ("hline", 5, None), ("vline", 3, None), ("custom", 3, ASYM), ("custom", 3, ASYM_L),
            ("custom", 3, OUTSIDE)]
SE_IDS = [f"{s}{n}" if c is None else f"custom:{c}" for s, n, c in SE_CASES]


def build(shape, n, custom):
    se = M.gray_structuring_element(shape, n, custom or M.DEFAULT_CUSTOM_SE)
    mask, anchor = ref_se(shape, n, MaskParam.decode(custom) if custom else None)
    return se, mask, anchor


def rand(seed, shape=(17, 23)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


# --------------------------------------------------------------------------
# Referansın kendisi ve SE tanımları
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", SE_CASES, ids=SE_IDS)
def test_vector_reference_equals_pixel_definition(case):
    _, mask, anchor = build(*case)
    for shape in ((1, 1), (1, 6), (6, 1), (5, 7)):
        f = rand(sum(shape), shape)
        np.testing.assert_array_equal(ref_erode(f, mask, anchor), px_erode(f, mask, anchor))
        np.testing.assert_array_equal(ref_dilate(f, mask, anchor), px_dilate(f, mask, anchor))


@pytest.mark.parametrize("case", SE_CASES, ids=SE_IDS)
def test_structuring_elements_match_the_lab(case):
    se, mask, anchor = build(*case)
    np.testing.assert_array_equal(se.mask, mask)
    assert se.anchor == anchor


def test_lab_ellipse_3_is_the_cross():
    np.testing.assert_array_equal(M.gray_structuring_element("ellipse", 3).mask, M.gray_structuring_element("cross", 3).mask)


@pytest.mark.parametrize("kwargs", [dict(shape="disk"), dict(size=4), dict(size=0), dict(size=-3), dict(size=2.5),
                                    dict(shape="custom", custom="000/000/000@1,1"), dict(shape="custom", custom="01/1@0,0"),
                                    dict(shape="custom", custom="010/111/010@3,1"), dict(shape="custom", custom="0a0@0,0"),
                                    dict(shape="custom", custom="111")])
def test_invalid_structuring_elements(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.gray_structuring_element(**kwargs)


def test_mask_param_round_trip_and_validation():
    p = MaskParam("custom", "Custom", "010/011/000@1,1", max_size=7)
    assert p.validate("10/01@1,0") == "10/01@1,0"
    for bad in ("", "000@0,0", "01/011@0,0", "1@1,0", "11111111@0,0", None, 7, "11@0"):
        assert p.validate(bad) == p.default
    grid, origin = MaskParam.decode("100/110/000@2,2")
    assert MaskParam.encode(grid, origin) == "100/110/000@2,2" and origin == (2, 2)


# --------------------------------------------------------------------------
# Formüller: beş işlem × SE × görüntü boyutu
# --------------------------------------------------------------------------

SIZES = [(1, 1), (1, 9), (9, 1), (2, 2), (17, 23)]


@pytest.mark.parametrize("case", SE_CASES, ids=SE_IDS)
@pytest.mark.parametrize("size", SIZES, ids=[f"{h}x{w}" for h, w in SIZES])
def test_five_operations_match_reference(case, size):
    se, mask, anchor = build(*case)
    f = rand(size[0] * 31 + size[1], size)
    np.testing.assert_array_equal(M.gray_erosion(f, se), ref_erode(f, mask, anchor))
    np.testing.assert_array_equal(M.gray_dilation(f, se), ref_dilate(f, mask, anchor))
    np.testing.assert_array_equal(M.gray_opening(f, se), ref_open(f, mask, anchor))
    np.testing.assert_array_equal(M.gray_closing(f, se), ref_close(f, mask, anchor))
    np.testing.assert_array_equal(M.gray_gradient(f, se), ref_gradient(f, mask, anchor))


@pytest.mark.parametrize("case", SE_CASES[:2] + SE_CASES[8:9], ids=SE_IDS[:2] + SE_IDS[8:9])
def test_registered_operations_match_reference(case):
    shape, n, custom = case
    _, mask, anchor = build(*case)
    f = rand(5, (20, 26))
    kw = dict(shape=shape, size=n, custom=custom or M.DEFAULT_CUSTOM_SE)
    pairs = [(M.grayscale_erosion, {}, ref_erode), (M.grayscale_dilation, {}, ref_dilate),
             (M.grayscale_opening, {}, ref_open), (M.grayscale_opening, dict(display="erosion"), ref_erode),
             (M.grayscale_closing, {}, ref_close), (M.grayscale_closing, dict(display="dilation"), ref_dilate),
             (M.grayscale_gradient, {}, ref_gradient), (M.grayscale_gradient, dict(display="dilation"), ref_dilate),
             (M.grayscale_gradient, dict(display="erosion"), ref_erode)]
    for fn, extra, ref in pairs:
        np.testing.assert_array_equal(fn(f, **kw, **extra).image, ref(f, mask, anchor), err_msg=f"{fn.__name__} {extra}")


def test_opening_and_closing_order_of_stages():
    se, mask, anchor = build("rect", 3, None)
    f = rand(9, (19, 21))
    e, d = ref_erode(f, mask, anchor), ref_dilate(f, mask, anchor)
    np.testing.assert_array_equal(M.gray_opening(f, se), ref_dilate(e, mask, anchor))      # önce erosion
    np.testing.assert_array_equal(M.gray_closing(f, se), ref_erode(d, mask, anchor))       # önce dilation
    assert not np.array_equal(M.gray_opening(f, se), M.gray_closing(f, se))
    np.testing.assert_array_equal(M.grayscale_opening(f, display="erosion").image, e)       # 1. aşama ayrı
    np.testing.assert_array_equal(M.grayscale_closing(f, display="dilation").image, d)


def test_gradient_is_dilation_minus_erosion():
    for case in SE_CASES[:8]:
        se, mask, anchor = build(*case)
        f = rand(11, (15, 18))
        d, e, g = M.gray_dilation(f, se), M.gray_erosion(f, se), M.gray_gradient(f, se)
        np.testing.assert_array_equal(g.astype(int), d.astype(int) - e.astype(int))        # orijin B'de: kırpma yok


def test_gradient_with_origin_outside_the_se_is_clipped_like_the_lab():
    se, mask, anchor = build("custom", 3, OUTSIDE)
    f = rand(12, (15, 18))
    raw = M.gray_dilation(f, se).astype(int) - M.gray_erosion(f, se).astype(int)
    assert (raw < 0).any()                                                # orijin B'de değilse fark negatif olabilir
    np.testing.assert_array_equal(M.gray_gradient(f, se), np.maximum(raw, 0))   # laboratuvarın gSub'u: max(0, ·)


# --------------------------------------------------------------------------
# Yansıma ve SE konvansiyonları (asimetrik özel SE)
# --------------------------------------------------------------------------


def test_dilation_of_a_bright_point_stamps_b_and_erosion_of_a_dark_point_stamps_b_hat():
    se = M.gray_structuring_element("custom", custom=ASYM_L)            # L: (0,0), (1,0), (2,0), (2,1), (2,2); orijin (2,0)
    f = np.zeros((9, 9), np.uint8)
    f[4, 4] = 200
    d = M.gray_dilation(f, se)
    # max f(z − s) → z = p + s: B'nin kendisi, orijin p'de (sağa ve yukarı)
    assert {(int(r), int(c)) for r, c in zip(*np.nonzero(d))} == {(2, 4), (3, 4), (4, 4), (4, 5), (4, 6)}
    g = np.full((9, 9), 200, np.uint8)
    g[4, 4] = 10
    e = M.gray_erosion(g, se)
    # min f(z + s) → z = p − s: yansıtılmış B̂ (sola ve aşağı)
    assert {(int(r), int(c)) for r, c in zip(*np.nonzero(e == 10))} == {(6, 4), (5, 4), (4, 4), (4, 3), (4, 2)}


def test_dilation_uses_the_reflected_se_not_the_raw_kernel():
    se, mask, anchor = build("custom", 3, ASYM)
    f = rand(13, (12, 14))
    reflected_window = ref_dilate(f, mask, anchor)                        # max f(z − s)
    raw_window = np.zeros_like(f)
    for y in range(f.shape[0]):                                           # max f(z + s) (yansıtmasız, yanlış)
        for x in range(f.shape[1]):
            v = [f[y + t, x + s] for s, t in members(mask, anchor) if 0 <= y + t < f.shape[0] and 0 <= x + s < f.shape[1]]
            raw_window[y, x] = max(v)
    np.testing.assert_array_equal(M.gray_dilation(f, se), reflected_window)
    assert not np.array_equal(M.gray_dilation(f, se), raw_window)
    # simetrik SE'de iki tanım aynıdır
    se3, m3, a3 = build("rect", 3, None)
    np.testing.assert_array_equal(M.gray_dilation(f, se3), ref_dilate(f, m3, a3))


# --------------------------------------------------------------------------
# Cebirsel özellikler, sınır ve özel görüntüler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", SE_CASES[:10], ids=SE_IDS[:10])
@pytest.mark.parametrize("seed", range(3))
def test_order_properties_with_origin_in_b(case, seed):
    se, _, _ = build(*case)
    f = rand(100 + seed, (21, 19))
    e, d = M.gray_erosion(f, se), M.gray_dilation(f, se)
    o, c, g = M.gray_opening(f, se), M.gray_closing(f, se), M.gray_gradient(f, se)
    assert (e <= f).all() and (f <= d).all()                              # erosion ≤ f ≤ dilation
    assert (o <= f).all() and (f <= c).all()                              # opening anti-extensive, closing extensive
    assert (e <= o).all() and (c <= d).all()
    assert (g >= 0).all() and g.dtype == np.uint8
    np.testing.assert_array_equal(M.gray_opening(o, se), o)               # idempotent
    np.testing.assert_array_equal(M.gray_closing(c, se), c)


@pytest.mark.parametrize("seed", range(3))
def test_opening_closing_bounds_hold_even_with_origin_outside_b(seed):
    se = M.gray_structuring_element("custom", custom=OUTSIDE)
    f = rand(200 + seed, (16, 20))
    assert (M.gray_opening(f, se) <= f).all() and (M.gray_closing(f, se) >= f).all()


def test_outside_is_ignored_not_background():
    """İkili morfolojinin "dışarısı arka plan" kuralı gri seviyeye kopyalanmaz: kenara değen parlak
    yapı kenardan karartılmaz (laboratuvar gWindow: yalnızca görüntü içindeki değerler)."""
    f = np.full((8, 10), 180, np.uint8)
    f[3:5, 4:6] = 40
    se = M.gray_structuring_element("rect", 3)
    e = M.gray_erosion(f, se)
    assert e[0, 0] == 180 and e[0, :].min() == 180 and e[:, -1].min() == 180   # kenar 0'a çekilmez
    full = np.full((6, 7), 255, np.uint8)
    np.testing.assert_array_equal(M.gray_erosion(full, se), full)
    assert M.binary_erosion(full > 0, se)[0].sum() == 0                  # ikili kuralda kenar aşınır (karşılaştırma)
    dark = np.zeros((6, 7), np.uint8)
    np.testing.assert_array_equal(M.gray_dilation(dark, se), dark)


@pytest.mark.parametrize("value", [0, 1, 128, 254, 255])
def test_constant_images_are_unchanged(value):
    f = np.full((7, 9), value, np.uint8)
    for case in SE_CASES[:10]:
        se, _, _ = build(*case)
        for fn in (M.gray_erosion, M.gray_dilation, M.gray_opening, M.gray_closing):
            np.testing.assert_array_equal(fn(f, se), f)
        assert not M.gray_gradient(f, se).any()


def test_extreme_values_stay_in_range():
    f = np.zeros((9, 9), np.uint8)
    f[::2, ::2] = 255
    se = M.gray_structuring_element("rect", 3)
    assert M.gray_erosion(f, se).max() == 0 and M.gray_dilation(f, se).min() == 255
    g = M.gray_gradient(f, se)
    assert g.dtype == np.uint8 and set(np.unique(g)) == {255}


def test_single_pixel_and_tiny_images():
    one = np.array([[77]], np.uint8)
    for case in SE_CASES:
        se, _, _ = build(*case)
        e = M.gray_erosion(one, se)
        np.testing.assert_array_equal(e, [[255 if case[2] == OUTSIDE else 77]])   # boş pencere → 255
    two = np.array([[10, 200]], np.uint8)
    se = M.gray_structuring_element("hline", 3)
    np.testing.assert_array_equal(M.gray_erosion(two, se), [[10, 10]])
    np.testing.assert_array_equal(M.gray_dilation(two, se), [[200, 200]])


def test_large_square_matches_reference_and_separable_minmax():
    """Kare SE'de OpenCV'nin min/max'ı (ayrılabilir: önce satır, sonra sütun) bağımsız referansla aynı."""
    f = rand(21, (160, 190))
    for n in (3, 9, 15):
        se, mask, anchor = build("rect", n, None)
        rows = ref_erode(f, np.ones((1, n), np.uint8), (n // 2, 0))
        separable = ref_erode(rows, np.ones((n, 1), np.uint8), (0, n // 2))
        np.testing.assert_array_equal(M.gray_erosion(f, se), ref_erode(f, mask, anchor))
        np.testing.assert_array_equal(M.gray_erosion(f, se), separable)
        np.testing.assert_array_equal(M.gray_dilation(f, se), ref_dilate(f, mask, anchor))


def test_performance_is_vectorised():
    f = rand(22, (512, 512))
    se = M.gray_structuring_element("rect", 15)
    t = time.perf_counter()
    M.grayscale_gradient(f, shape="rect", size=15)
    assert time.perf_counter() - t < 1.0                                  # Python döngüsü yok (ms düzeyi)
    np.testing.assert_array_equal(M.gray_gradient(f, se), ref_gradient(f, se.mask, se.anchor))


# --------------------------------------------------------------------------
# Görüntü düzeyi: kanal düzeni, renk, alfa, değişmezlik, rapor
# --------------------------------------------------------------------------

FUNCS = [M.grayscale_erosion, M.grayscale_dilation, M.grayscale_opening, M.grayscale_closing, M.grayscale_gradient]


@pytest.mark.parametrize("fn", FUNCS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("shape", [(14, 18), (14, 18, 1), (14, 18, 3), (14, 18, 4)])
def test_layout_alpha_immutability(fn, shape):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    before = img.copy()
    res = fn(img, shape="cross", size=3)
    out = res.image
    np.testing.assert_array_equal(img, before)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.shares_memory(out, img)
    f = ref_luma(img)
    if len(shape) == 3 and shape[2] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])
    ch = out if out.ndim == 2 else out[..., 0]
    for k in range(1, 3 if len(shape) == 3 and shape[2] >= 3 else 1):
        np.testing.assert_array_equal(out[..., k], ch)                    # renkli girdide üç eşit kanal
    se, mask, anchor = build("cross", 3, None)
    ref = {"grayscale_erosion": ref_erode, "grayscale_dilation": ref_dilate, "grayscale_opening": ref_open,
           "grayscale_closing": ref_close, "grayscale_gradient": ref_gradient}[fn.__name__]
    np.testing.assert_array_equal(ch, ref(f, mask, anchor))


def test_colour_uses_luminance_and_is_not_binarised():
    rng = np.random.default_rng(4)
    bgr = rng.integers(0, 256, (16, 20, 3), dtype=np.uint8)
    out = M.grayscale_opening(bgr).image
    y = ref_luma(bgr)                                                     # BT.601 Y, yuvarlanmış (bağımsız)
    np.testing.assert_array_equal(out[..., 0], ref_open(y, np.ones((3, 3), np.uint8), (1, 1)))
    assert len(np.unique(out)) > 20                                       # eşikleme yok: çok sayıda gri düzey
    gray = rng.integers(0, 256, (16, 20), dtype=np.uint8)
    np.testing.assert_array_equal(M.grayscale_erosion(gray).image, ref_erode(gray, np.ones((3, 3), np.uint8), (1, 1)))


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint8), [[1, 2]]])
def test_invalid_images(bad):
    with pytest.raises(AlgorithmParameterError):
        M.grayscale_erosion(bad)


@pytest.mark.parametrize("fn,kwargs", [(M.grayscale_opening, dict(display="closing")),
                                       (M.grayscale_closing, dict(display="erosion")),
                                       (M.grayscale_gradient, dict(display="laplace")),
                                       (M.grayscale_dilation, dict(shape="diamond")),
                                       (M.grayscale_erosion, dict(size=6))])
def test_invalid_parameters(fn, kwargs):
    with pytest.raises(AlgorithmParameterError):
        fn(rand(1, (5, 5)), **kwargs)


def test_reports():
    f = rand(6, (12, 15))
    r = M.grayscale_erosion(f, shape="ellipse", size=5).report
    e = ref_erode(f, *ref_se("ellipse", 5))
    assert r.before == (float(f.mean()), int(f.min()), int(f.max())) and r.after == (float(e.mean()), int(e.min()), int(e.max()))
    assert "Grayscale erosion (ellipse 5×5)" in r.summary()
    g = M.grayscale_gradient(f).report.summary()
    assert "f ⊕ b mean" in g and "f ⊖ b mean" in g
    assert "origin r2 c0" in M.grayscale_dilation(f, shape="custom", custom=ASYM_L).report.summary()
    assert "stage 1" in M.grayscale_opening(f, display="erosion").report.summary()


# --------------------------------------------------------------------------
# Mevcut Top-Hat / Black-Hat incelemesi
# --------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", [(-1, -1), (0, 0), (4, 1)])
@pytest.mark.parametrize("shape", ["rect", "ellipse", "cross"])
def test_top_hat_black_hat_are_intensity_differences(shape, anchor):
    f = rand(30, (24, 27))
    se = M.structuring_element(shape, 7, 5, *anchor)
    mask, anc = se.mask, se.anchor
    th = M.top_hat(f, shape=shape, width=7, height=5, anchor_x=anchor[0], anchor_y=anchor[1])
    bh = M.black_hat(f, shape=shape, width=7, height=5, anchor_x=anchor[0], anchor_y=anchor[1])
    np.testing.assert_array_equal(th, f.astype(int) - ref_open(f, mask, anc).astype(int))   # T = f − f ∘ b
    np.testing.assert_array_equal(bh, ref_close(f, mask, anc).astype(int) - f.astype(int))  # B = f • b − f
    assert len(np.unique(th)) > 5 and len(np.unique(bh)) > 5               # eşikleme yok


def test_top_hat_consistent_with_new_opening_and_closing_operations():
    rng = np.random.default_rng(31)
    bgr = rng.integers(0, 256, (20, 22, 3), dtype=np.uint8)
    f = ref_luma(bgr)
    opening = M.grayscale_opening(bgr, shape="rect", size=5).image[..., 0]
    closing = M.grayscale_closing(bgr, shape="rect", size=5).image[..., 0]
    np.testing.assert_array_equal(M.top_hat(bgr, width=5, height=5)[..., 0], f - opening)
    np.testing.assert_array_equal(M.black_hat(bgr, width=5, height=5)[..., 0], closing - f)


def test_hats_on_interior_and_border_touching_structures():
    """İçteki küçük leke tam çıkarılır. Kenara/köşeye değen aynı boyuttaki leke ise ÇIKARILMAZ:
    görüntü dışı yok sayıldığı için (laboratuvar gWindow, OpenCV varsayılanı) kenarda SE penceresi
    kırpılır ve kırpılmış pencere lekenin içine sığar. Bu, sınır kuralının bilinen sonucudur; sonuç
    bağımsız referansla aynıdır."""
    f = np.full((14, 16), 60, np.uint8)
    f[6:9, 6:9] = 220                                                     # içte 3 × 3 parlak leke
    f[0:3, 0:3] = 220                                                     # köşede 3 × 3 parlak leke
    f[11:14, 12:16] = 5                                                   # kenarda koyu leke
    f[2:4, 11:13] = 5                                                     # içte koyu leke
    th = M.top_hat(f, width=5, height=5)
    bh = M.black_hat(f, width=5, height=5)
    mask = np.ones((5, 5), np.uint8)
    np.testing.assert_array_equal(th, f.astype(int) - ref_open(f, mask, (2, 2)).astype(int))
    np.testing.assert_array_equal(bh, ref_close(f, mask, (2, 2)).astype(int) - f.astype(int))
    assert (th[6:9, 6:9] == 160).all() and (bh[2:4, 11:13] == 55).all()   # iç lekeler tam
    assert not th[0:3, 0:3].any() and not bh[11:14, 12:16].any()          # kenar lekeleri: kırpılmış pencere sığar


# --------------------------------------------------------------------------
# Kayıt ve önceki 17 işlem
# --------------------------------------------------------------------------


def test_registry_entries():
    reg = build_registry()
    keys = [a.key for a in reg.in_category("morphology")]
    assert keys[-5:] == ["gray_erosion", "gray_dilation", "gray_opening", "gray_closing", "gray_gradient"] and len(keys) == 22
    titles = {k: reg.get(k).title for k in keys[-5:]}
    assert titles == {"gray_erosion": "Grayscale Erosion", "gray_dilation": "Grayscale Dilation",
                      "gray_opening": "Grayscale Opening", "gray_closing": "Grayscale Closing",
                      "gray_gradient": "Grayscale Morphological Gradient"}
    for k in keys[-5:]:
        algo = reg.get(k)
        assert algo.implemented and [p.key for p in algo.params][:3] == ["shape", "size", "custom"]
        d = algo.default_params()
        assert (d["shape"], d["size"]) == ("rect", 3)
        assert [v for v, _ in algo.params[0].options] == list(M.GRAY_SE_SHAPES)
    assert "min f(x + s, y + t)" in reg.get("gray_erosion").summary
    assert "max f(x − s, y − t)" in reg.get("gray_dilation").summary


def test_previous_17_operations_unchanged():
    assert_previous_15_operations_unchanged(98)
    reg = build_registry()
    a = np.kron(FIG_927_A, np.ones((1, 1), bool))
    np.testing.assert_array_equal(reg.get("pruning").run(u8(a)), u8(ref_prune(a, 3)[0]["X4"]))
    blob = np.zeros((15, 18), bool)
    blob[3:12, 2:15] = True
    np.testing.assert_array_equal(reg.get("skeletonization").run(u8(blob)),
                                  u8(union_skeleton(ref_skeleton(blob, se_mask("rect", 3, 3)))))


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    yy, xx = np.mgrid[0:60, 0:80]
    img = (40 + xx * 2 + 10 * np.sin(yy / 4)).astype(np.uint8)
    img[20:23, 30:33] = 250
    img[40:42, 50:52] = 5
    img = np.repeat(img[..., None], 3, axis=2)
    path = tmp_path / "Gri" / "rampa.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


@pytest.mark.parametrize("key,ref", [("gray_erosion", ref_erode), ("gray_dilation", ref_dilate), ("gray_opening", ref_open),
                                     ("gray_closing", ref_close), ("gray_gradient", ref_gradient)])
def test_ui_apply_each_operation(window, key, ref):
    w, img = window
    w.navigator.select_algorithm(key)
    w.parameters.controls["shape"].set_value("ellipse")
    w.parameters.controls["size"].set_value(5)
    w.parameters.apply_button.click()
    np.testing.assert_array_equal(w.current_image[..., 0], ref(img[..., 0], *ref_se("ellipse", 5)))
    assert w.telemetry.state.state() == "DONE" and "ellipse 5×5" in w.statusBar().currentMessage()
    assert not w.parameters.report_view.isVisible()
    for name, counts in histograms(w.current_image).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)
    np.testing.assert_array_equal(w.original_image, img)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)


def test_ui_custom_se_grid_editing(window):
    w, img = window
    w.navigator.select_algorithm("gray_dilation")
    ctl = w.parameters.controls["custom"]
    assert isinstance(ctl, MaskControl) and ctl.value() == M.DEFAULT_CUSTOM_SE
    cell = ctl.grid.CELL
    QTest.mouseClick(ctl.grid, Qt.MouseButton.LeftButton, pos=QPoint(cell // 2, cell // 2))           # (0,0) aç
    QTest.mouseClick(ctl.grid, Qt.MouseButton.RightButton, pos=QPoint(cell * 2 + cell // 2, cell // 2))  # orijin (0,2)
    assert ctl.value() == "110/011/000@0,2" and "origin 0,2 (not in B)" in ctl.origin_label.text()
    ctl.size_buttons[5].click()
    assert ctl.value() == "00000/01100/00110/00000/00000@2,2"
    ctl.set_value("100/110/000@1,1")
    w.parameters.controls["shape"].set_value("custom")
    w.parameters.apply_button.click()
    mask, anchor = ref_se("custom", 3, MaskParam.decode("100/110/000@1,1"))
    np.testing.assert_array_equal(w.current_image[..., 0], ref_dilate(img[..., 0], mask, anchor))
    assert "origin r1 c1" in w.statusBar().currentMessage()
    w.parameters.defaults_button.click()
    assert ctl.value() == M.DEFAULT_CUSTOM_SE


def test_ui_chain_opening_then_gradient_and_reset(window):
    w, img = window
    w.navigator.select_algorithm("gray_opening")
    w.parameters.apply_button.click()
    opened = w.current_image.copy()
    w.navigator.select_algorithm("gray_gradient")
    w.parameters.apply_button.click()
    np.testing.assert_array_equal(w.current_image, M.grayscale_gradient(opened).image)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, opened)
    w.reset_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
