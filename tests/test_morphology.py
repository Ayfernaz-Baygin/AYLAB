"""İkili morfoloji: bilinen küçük matrisler, küme tanımlarından bağımsız referans ve özellikler.

Referans uygulama (``ref_*``) OpenCV kullanmaz: doğrudan ders kitabı tanımlarını
öteleme (shift) ile kurar ve görüntü dışını arka plan kabul eder:

    A ⊖ B = ∩_{b∈B} A_{−b}      A ⊕ B = ∪_{b∈B} A_b
"""

import itertools

import cv2
import numpy as np
import pytest

from processing import morphology as M
from processing.catalog import build_registry
from processing.registry import AlgorithmParameterError

# --------------------------------------------------------------------------
# Bağımsız referans
# --------------------------------------------------------------------------


def shift(a, dy, dx, fill=False):
    """S[r, c] = a[r + dy, c + dx]; dışarısı ``fill``."""
    h, w = a.shape
    out = np.full((h, w), fill, bool)
    r0, r1 = max(0, -dy), min(h, h - dy)
    c0, c1 = max(0, -dx), min(w, w - dx)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = a[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
    return out


def offsets(mask, anchor):
    ax, ay = anchor
    return [(r - ay, c - ax) for r, c in zip(*np.nonzero(mask))]


def ref_erode(a, mask, anchor):
    out = np.ones(a.shape, bool)
    for dy, dx in offsets(mask, anchor):
        out &= shift(a, dy, dx)          # (B)_z ⊆ A
    return out


def ref_dilate(a, mask, anchor):
    out = np.zeros(a.shape, bool)
    for dy, dx in offsets(mask, anchor):
        out |= shift(a, -dy, -dx)        # z − b ∈ A
    return out


def on_big_canvas(fn, a, pad):
    """Sonsuz düzlemi taklit eder: geniş arka plan tuvalinde hesapla, kırp."""
    big = np.pad(a, pad)
    return fn(big)[pad:pad + a.shape[0], pad:pad + a.shape[1]]


def ref_hmt(a, kernel):
    k = np.asarray(kernel)
    anchor = (k.shape[1] // 2, k.shape[0] // 2)
    out = np.ones(a.shape, bool)
    for (dy, dx) in offsets(k == 1, anchor):
        out &= shift(a, dy, dx, fill=False)
    for (dy, dx) in offsets(k == -1, anchor):
        out &= ~shift(a, dy, dx, fill=False)   # dışarısı arka plan → −1 hücresi sağlanır
    return out


def B(rows):
    return np.array(rows, bool)


def random_binary(seed, shape=(23, 29), p=0.55):
    return np.random.default_rng(seed).random(shape) < p


SHAPES = ("rect", "ellipse", "cross")
SE_CASES = [(s, w, h) for s in SHAPES for (w, h) in ((1, 1), (3, 3), (5, 5), (7, 3), (3, 7), (1, 5), (9, 9))]

# --------------------------------------------------------------------------
# Yapı elemanı
# --------------------------------------------------------------------------


def test_rect_square_and_rectangle():
    se = M.structuring_element("rect", 3, 3)
    assert se.mask.tolist() == [[1] * 3] * 3 and se.anchor == (1, 1)
    se = M.structuring_element("rect", 5, 3)
    assert se.mask.shape == (3, 5) and se.mask.all() and se.anchor == (2, 1)


def test_cross_is_centre_row_and_column():
    assert M.structuring_element("cross", 5, 5).mask.tolist() == [
        [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
    assert M.structuring_element("cross", 3, 5).mask.tolist() == [
        [0, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0]]


def test_cross_shape_does_not_move_with_anchor():
    """OpenCV MORPH_CROSS artıyı anchor'a taşır; ders kitabında şekil ve orijin bağımsızdır."""
    se = M.structuring_element("cross", 5, 5, anchor_x=0, anchor_y=0)
    assert np.array_equal(se.mask, M.structuring_element("cross", 5, 5).mask)
    assert se.anchor == (0, 0)


def test_ellipse_matches_textbook_tables():
    # Dersin laboratuvar sayfasındaki tablolar (cv2.getStructuringElement(MORPH_ELLIPSE) çıktısı)
    assert M.structuring_element("ellipse", 3, 3).mask.tolist() == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    assert M.structuring_element("ellipse", 5, 5).mask.tolist() == [
        [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]]
    assert M.structuring_element("ellipse", 7, 7).mask.tolist() == [
        [0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0]]


@pytest.mark.parametrize("w,h", [(5, 1), (1, 5), (1, 1), (9, 1)])
def test_degenerate_ellipse_is_a_line(w, h):
    se = M.structuring_element("ellipse", w, h)
    assert se.mask.shape == (h, w) and se.mask.all()


@pytest.mark.parametrize("shape,w,h", SE_CASES)
def test_se_is_symmetric_and_centred_by_default(shape, w, h):
    se = M.structuring_element(shape, w, h)
    assert se.anchor == (w // 2, h // 2)
    assert np.array_equal(se.mask, se.mask[::-1, ::-1])
    assert se.mask[h // 2, w // 2] == 1  # orijin üyedir


def test_reflection_rotates_mask_and_origin():
    se = M.StructuringElement(np.array([[1, 1, 0], [0, 1, 0]], np.uint8), (0, 0))
    r = se.reflected()
    assert r.mask.tolist() == [[0, 1, 0], [0, 1, 1]] and r.anchor == (2, 1)
    assert sorted(r.offsets) == sorted((-dy, -dx) for dy, dx in se.offsets)
    assert se.offsets == [(0, 0), (0, 1), (1, 1)]


@pytest.mark.parametrize("kwargs", [
    dict(shape="disk"), dict(shape=None), dict(width=0), dict(width=2), dict(width=4), dict(height=-3),
    dict(width=2.5), dict(width="3"), dict(width=None), dict(width=True),
    dict(anchor_x=3), dict(anchor_x=-2), dict(anchor_y=5, height=5), dict(anchor_x=1.5),
    dict(anchor_x=True), dict(anchor_y="0"),
])
def test_invalid_structuring_element(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.structuring_element(**kwargs)


def test_anchor_accepts_corners_and_integral_floats():
    assert M.structuring_element("rect", 3, 5, 0, 4).anchor == (0, 4)
    assert M.structuring_element("rect", 3, 5, 2.0, 0.0).anchor == (2, 0)
    assert M.structuring_element("rect", 3, 5, np.int64(2), np.int32(1)).anchor == (2, 1)


# --------------------------------------------------------------------------
# İkili görüntüye dönüşüm
# --------------------------------------------------------------------------


def test_manual_threshold_is_strictly_greater_and_dark_is_complement():
    g = np.array([[0, 100, 127, 128, 255]], np.uint8)
    assert M.to_binary(g, "manual", 127).tolist() == [[False, False, False, True, True]]
    assert M.to_binary(g, "manual", 127, "dark").tolist() == [[True, True, True, False, False]]
    assert M.to_binary(g, "manual", 255).sum() == 0 and M.to_binary(g, "manual", 0).sum() == 4


def test_color_uses_bt601_luminance():
    # Saf mavi Y≈29, saf kırmızı Y≈76, saf yeşil Y≈150 (BGR düzeni)
    img = np.array([[[255, 0, 0], [0, 0, 255], [0, 255, 0]]], np.uint8)
    assert M.to_binary(img, "manual", 100).tolist() == [[False, False, True]]
    assert M.to_binary(img, "manual", 50).tolist() == [[False, True, True]]


def test_alpha_is_ignored_for_binarization():
    color = np.array([[[200, 200, 200], [10, 10, 10]]], np.uint8)
    for alpha in (0, 128, 255):
        bgra = np.dstack([color, np.full((1, 2), alpha, np.uint8)])
        assert M.to_binary(bgra, "manual", 127).tolist() == [[True, False]]


def numpy_otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    levels = np.arange(256)
    best, best_t = -1.0, 0
    for t in range(256):
        w0, w1 = hist[:t + 1].sum(), hist[t + 1:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:t + 1] * levels[:t + 1]).sum() / w0
        m1 = (hist[t + 1:] * levels[t + 1:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, t
    return best_t


@pytest.mark.parametrize("seed", range(5))
def test_otsu_matches_independent_between_class_variance(seed):
    rng = np.random.default_rng(seed)
    g = np.concatenate([rng.normal(60, 12, 600), rng.normal(180, 15, 400)])
    g = np.clip(np.rint(g), 0, 255).astype(np.uint8).reshape(25, 40)
    t = numpy_otsu(g)
    assert abs(M.otsu_threshold(g) - t) <= 1
    mask = M.to_binary(g, "otsu")
    # Otsu sonucu iki modu ayırır: parlak modun neredeyse tamamı ön plan
    assert 0.38 < mask.mean() < 0.42


def test_otsu_on_already_binary_image_keeps_it():
    a = random_binary(1, (10, 12))
    img = np.where(a, 255, 0).astype(np.uint8)
    assert np.array_equal(M.to_binary(img, "otsu"), a)
    assert np.array_equal(M.to_binary(img, "otsu", foreground="dark"), ~a)


@pytest.mark.parametrize("value,expected", [(0, False), (90, True), (255, True)])
def test_otsu_on_constant_image(value, expected):
    """Tek düzeyli görüntüde Otsu (OpenCV) T = 0 verir: ön plan = Y > 0."""
    assert (M.to_binary(np.full((4, 4), value, np.uint8), "otsu") == expected).all()


@pytest.mark.parametrize("kwargs", [dict(binarize="auto"), dict(foreground="white"), dict(binarize="manual", thresh=-1),
                                    dict(binarize="manual", thresh=256), dict(binarize="manual", thresh=1.5),
                                    dict(binarize="manual", thresh=True)])
def test_invalid_binarization(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.to_binary(np.zeros((3, 3), np.uint8), **kwargs)


# --------------------------------------------------------------------------
# Erosion — bilinen matrisler
# --------------------------------------------------------------------------

SQUARE5 = np.zeros((7, 7), bool)
SQUARE5[1:6, 1:6] = True


def test_erosion_square_by_square_keeps_core():
    out = M.binary_erosion(SQUARE5, M.structuring_element("rect", 3, 3))
    expected = np.zeros((7, 7), bool)
    expected[2:5, 2:5] = True
    assert np.array_equal(out, expected)


def test_erosion_square_by_cross():
    a = np.zeros((5, 5), bool)
    a[1:4, 1:4] = True
    out = M.binary_erosion(a, M.structuring_element("cross", 3, 3))
    assert out.astype(int).tolist() == [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]


def test_erosion_removes_thin_line_and_isolated_pixels():
    a = np.zeros((7, 9), bool)
    a[3, 1:8] = True       # 1 piksel kalınlığında çizgi
    a[0, 0] = True         # tekil piksel
    assert not M.binary_erosion(a, M.structuring_element("rect", 3, 3)).any()
    # Yatay çizgi SE'si (1×3) çizgiyi korur, yalnızca uçlarını kısaltır
    out = M.binary_erosion(a, M.structuring_element("rect", 3, 1))
    assert out[3].astype(int).tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0] and out.sum() == 5


def test_erosion_with_corner_anchor_shifts_result():
    """(B)_z ⊆ A, B = 3×3, orijin sol üst: z + {0,1,2}² ⊆ [1..5]² → z ∈ [1..3]²."""
    out = M.binary_erosion(SQUARE5, M.structuring_element("rect", 3, 3, 0, 0))
    expected = np.zeros((7, 7), bool)
    expected[1:4, 1:4] = True
    assert np.array_equal(out, expected)
    out = M.binary_erosion(SQUARE5, M.structuring_element("rect", 3, 3, 2, 2))
    expected = np.zeros((7, 7), bool)
    expected[3:6, 3:6] = True
    assert np.array_equal(out, expected)


def test_erosion_outside_image_is_background():
    full = np.ones((4, 5), bool)
    out = M.binary_erosion(full, M.structuring_element("rect", 3, 3))
    expected = np.zeros((4, 5), bool)
    expected[1:3, 1:4] = True
    assert np.array_equal(out, expected)


def test_erosion_is_anti_extensive_and_increasing_with_se():
    a = random_binary(3)
    small, big = M.structuring_element("rect", 3, 3), M.structuring_element("rect", 5, 5)
    e3, e5 = M.binary_erosion(a, small), M.binary_erosion(a, big)
    assert not (e3 & ~a).any() and not (e5 & ~e3).any()


# --------------------------------------------------------------------------
# Dilation — bilinen matrisler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_dilation_of_single_pixel_stamps_the_se(shape):
    a = np.zeros((9, 9), bool)
    a[4, 4] = True
    se = M.structuring_element(shape, 5, 5)
    out = M.binary_dilation(a, se)
    assert np.array_equal(out[2:7, 2:7], se.mask.astype(bool)) and out.sum() == se.mask.sum()


def test_dilation_uses_reflected_se_textbook_definition():
    """A ⊕ B = ∪ A_b. B = {(0,0),(0,1),(0,2)} (orijin solda) tek pikseli SAĞA uzatır.

    cv2.dilate aynı çekirdekle SOLA uzatır (A ⊕ B̂); modül ders kitabını izler.
    """
    a = np.zeros((5, 7), bool)
    a[2, 3] = True
    se = M.structuring_element("rect", 3, 1, anchor_x=0)
    assert M.binary_dilation(a, se)[2].astype(int).tolist() == [0, 0, 0, 1, 1, 1, 0]
    raw = cv2.dilate(a.astype(np.uint8), se.mask, anchor=se.anchor)
    assert raw[2].tolist() == [0, 1, 1, 1, 0, 0, 0]


def test_dilation_with_corner_anchor():
    a = np.zeros((7, 7), bool)
    a[2, 2] = True
    out = M.binary_dilation(a, M.structuring_element("rect", 3, 3, 0, 0))
    expected = np.zeros((7, 7), bool)
    expected[2:5, 2:5] = True
    assert np.array_equal(out, expected)


def test_dilation_bridges_a_one_pixel_gap():
    a = np.zeros((5, 9), bool)
    a[2, 1:4] = True
    a[2, 5:8] = True
    out = M.binary_dilation(a, M.structuring_element("rect", 3, 1))
    assert out[2, 1:8].all() and out[2, 0] and out[2, 8] and out.sum() == 9


def test_dilation_at_border_is_clipped_not_wrapped():
    a = np.zeros((5, 5), bool)
    a[0, 0] = True
    out = M.binary_dilation(a, M.structuring_element("rect", 3, 3))
    expected = np.zeros((5, 5), bool)
    expected[:2, :2] = True
    assert np.array_equal(out, expected)


def test_iterations_equal_larger_se():
    a = np.zeros((25, 25), bool)
    a[12, 12] = True
    a[5:9, 15:20] = True
    for n in (1, 2, 3):
        big = M.structuring_element("rect", 2 * n + 1, 2 * n + 1)
        assert np.array_equal(M.binary_dilation(a, M.structuring_element(), n), M.binary_dilation(a, big))
        sq = np.zeros((25, 25), bool)
        sq[3:22, 3:22] = True
        assert np.array_equal(M.binary_erosion(sq, M.structuring_element(), n), M.binary_erosion(sq, big))


def test_iterated_dilation_does_not_lose_paths_through_outside():
    """Off-centre SE ile tekrarlanan dilation sonsuz düzlemdekiyle aynı olmalı."""
    a = random_binary(11, (15, 15), 0.1)
    se = M.structuring_element("cross", 5, 3, anchor_x=4, anchor_y=0)

    def twice(x):
        return ref_dilate(ref_dilate(x, se.mask, se.anchor), se.mask, se.anchor)

    assert np.array_equal(M.binary_dilation(a, se, 2), on_big_canvas(twice, a, 20))


# --------------------------------------------------------------------------
# Bağımsız referansla rastgele karşılaştırma
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape,w,h", SE_CASES)
def test_erosion_dilation_match_set_definitions(shape, w, h):
    a = random_binary(w * 31 + h)
    for ax, ay in {(w // 2, h // 2), (0, 0), (w - 1, 0), (0, h - 1)}:
        se = M.structuring_element(shape, w, h, ax, ay)
        assert np.array_equal(M.binary_erosion(a, se), ref_erode(a, se.mask, se.anchor)), (ax, ay)
        assert np.array_equal(M.binary_dilation(a, se), ref_dilate(a, se.mask, se.anchor)), (ax, ay)


@pytest.mark.parametrize("shape,w,h", SE_CASES)
def test_opening_closing_match_set_definitions(shape, w, h):
    a = random_binary(w * 7 + h + 100)
    for anchor in {(w // 2, h // 2), (0, 0), (w - 1, h - 1)}:
        se = M.structuring_element(shape, w, h, *anchor)
        m, an = se.mask, se.anchor
        pad = 2 * max(w, h)
        ref_open = on_big_canvas(lambda x: ref_dilate(ref_erode(x, m, an), m, an), a, pad)
        ref_close = on_big_canvas(lambda x: ref_erode(ref_dilate(x, m, an), m, an), a, pad)
        assert np.array_equal(M.binary_opening(a, se), ref_open)
        assert np.array_equal(M.binary_closing(a, se), ref_close)


def test_opening_is_union_of_fitting_translates():
    """A ∘ B = ∪ { (B)_z | (B)_z ⊆ A } — kaba kuvvet referansı."""
    a = random_binary(5, (14, 16), 0.7)
    se = M.structuring_element("ellipse", 5, 3)
    expected = np.zeros_like(a)
    offs = se.offsets
    for z in itertools.product(range(-3, a.shape[0] + 3), range(-3, a.shape[1] + 3)):
        cells = [(z[0] + dy, z[1] + dx) for dy, dx in offs]
        if all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] and a[r, c] for r, c in cells):
            for r, c in cells:
                expected[r, c] = True
    assert np.array_equal(M.binary_opening(a, se), expected)


def test_duality_away_from_border():
    """(A ⊖ B)ᶜ = Aᶜ ⊕ B̂ (sınırdan SE erimi kadar uzakta, iki kuralın sınır tanımı farklı)."""
    a = random_binary(8, (30, 30))
    se = M.structuring_element("rect", 5, 3, anchor_x=1, anchor_y=0)
    lhs = ~M.binary_erosion(a, se)
    rhs = M.binary_dilation(~a, se.reflected())
    r = se.reach
    assert np.array_equal(lhs[r:-r, r:-r], rhs[r:-r, r:-r])


def test_against_opencv_with_matching_conventions():
    """Simetrik SE + merkez orijinde, sınır kuralı eşlendiğinde OpenCV ile birebir."""
    a = random_binary(21, (40, 50))
    u8 = a.astype(np.uint8)
    for shape in SHAPES:
        se = M.structuring_element(shape, 5, 5)
        cv_e = cv2.erode(u8, se.mask, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        cv_d = cv2.dilate(u8, se.mask, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        assert np.array_equal(M.binary_erosion(a, se), cv_e)
        assert np.array_equal(M.binary_dilation(a, se), cv_d)


# --------------------------------------------------------------------------
# Opening / Closing
# --------------------------------------------------------------------------


def test_opening_removes_speck_and_bridge_keeps_squares():
    a = np.zeros((9, 15), bool)
    a[2:7, 1:6] = True
    a[2:7, 9:14] = True
    a[4, 6:9] = True    # 1 piksel köprü
    a[0, 7] = True      # leke
    expected = np.zeros_like(a)
    expected[2:7, 1:6] = True
    expected[2:7, 9:14] = True
    assert np.array_equal(M.binary_opening(a, M.structuring_element("rect", 3, 3)), expected)


def test_closing_fills_hole_and_gap():
    a = np.zeros((9, 13), bool)
    a[2:7, 2:11] = True
    a[4, 4] = False         # tek piksel delik
    a[2:7, 7] = False       # 1 piksel genişliğinde yarık
    expected = np.zeros_like(a)
    expected[2:7, 2:11] = True
    assert np.array_equal(M.binary_closing(a, M.structuring_element("rect", 3, 3)), expected)


def test_operation_order_matters_and_is_respected():
    a = np.zeros((9, 15), bool)
    a[2:7, 1:6] = True
    a[2:7, 9:14] = True
    a[4, 6:9] = True
    se = M.structuring_element("rect", 3, 3)
    e_then_d = M.binary_dilation(M.binary_erosion(a, se), se)
    d_then_e = M.binary_erosion(M.binary_dilation(a, se), se)
    assert np.array_equal(M.binary_opening(a, se), e_then_d)
    assert np.array_equal(M.binary_closing(a, se), d_then_e)
    assert not np.array_equal(e_then_d, d_then_e)   # köprü: açmada kopar, kapamada kalır
    assert not e_then_d[4, 7] and d_then_e[4, 7]


@pytest.mark.parametrize("shape", SHAPES)
def test_opening_closing_properties(shape):
    a = random_binary(40, (30, 34))
    se = M.structuring_element(shape, 5, 5)
    o, c = M.binary_opening(a, se), M.binary_closing(a, se)
    assert not (o & ~a).any() and not (a & ~c).any()            # A ∘ B ⊆ A ⊆ A • B (sınırda da)
    assert np.array_equal(M.binary_opening(o, se), o)           # idempotent
    assert np.array_equal(M.binary_closing(c, se), c)


def test_opening_and_closing_do_not_depend_on_origin():
    a = random_binary(44, (26, 31))
    ref_o = M.binary_opening(a, M.structuring_element("ellipse", 5, 7))
    ref_c = M.binary_closing(a, M.structuring_element("ellipse", 5, 7))
    for ax, ay in [(0, 0), (4, 6), (1, 5), (4, 0)]:
        se = M.structuring_element("ellipse", 5, 7, ax, ay)
        assert np.array_equal(M.binary_opening(a, se), ref_o)
        assert np.array_equal(M.binary_closing(a, se), ref_c)


# --------------------------------------------------------------------------
# Morfolojik gradyan
# --------------------------------------------------------------------------


def test_gradient_of_square_is_band():
    out = M.binary_gradient(np.pad(SQUARE5, 1), M.structuring_element("rect", 3, 3))
    assert out.sum() == 7 * 7 - 3 * 3
    expected = np.zeros((9, 9), bool)
    expected[1:8, 1:8] = True
    expected[3:6, 3:6] = False
    assert np.array_equal(out, expected)


def test_gradient_equals_dilation_minus_erosion():
    a = random_binary(9)
    for shape in SHAPES:
        se = M.structuring_element(shape, 3, 5, 0, 1)
        assert np.array_equal(M.binary_gradient(a, se), M.binary_dilation(a, se) & ~M.binary_erosion(a, se))


def test_gradient_of_constant_interior_is_empty():
    a = np.zeros((6, 6), bool)
    assert not M.binary_gradient(a, M.structuring_element()).any()
    full = np.ones((6, 6), bool)  # dışarısı arka plan → görüntü kenarı nesne sınırıdır
    g = M.binary_gradient(full, M.structuring_element())
    assert g[0].all() and g[-1].all() and not g[2:4, 2:4].any()


# --------------------------------------------------------------------------
# Hit-or-miss
# --------------------------------------------------------------------------


def test_hmt_corner_finds_only_upper_right_corner():
    hits = M.hit_or_miss_mask(np.pad(SQUARE5, 1), M.hmt_kernel("corner"))
    assert list(zip(*np.nonzero(hits))) == [(2, 6)]


@pytest.mark.parametrize("rotation,expected", [("0", (2, 6)), ("90", (2, 2)), ("180", (6, 2)), ("270", (6, 6))])
def test_hmt_rotations_find_each_corner(rotation, expected):
    hits = M.hit_or_miss_mask(np.pad(SQUARE5, 1), M.hmt_kernel("corner", rotation))
    assert list(zip(*np.nonzero(hits))) == [expected]


def test_hmt_rotation_kernels():
    assert M.hmt_kernel("corner", "90").tolist() == [[-1, -1, -1], [-1, 1, 1], [-1, 1, 1]]
    assert M.hmt_kernel("edge", "180").tolist() == [[-1, 0, 0], [-1, 1, 1], [-1, 0, 0]]


def test_hmt_right_edge_dont_care_cells():
    """[0 0 −1; 1 1 −1; 0 0 −1]: köşelerde 0 (önemsiz) olduğu için köşe pikselleri de kenardır."""
    hits = M.hit_or_miss_mask(SQUARE5, M.hmt_kernel("edge"))
    assert sorted(zip(*np.nonzero(hits))) == [(r, 5) for r in range(1, 6)]


def test_hmt_isolated_hole_point_and_line_end():
    a = np.ones((5, 5), bool)
    a[2, 2] = False
    a[0, 4] = False
    assert list(zip(*np.nonzero(M.hit_or_miss_mask(a, M.hmt_kernel("hole"))))) == [(2, 2)]

    b = np.zeros((6, 6), bool)
    b[1, 1] = True           # tek nokta
    b[4, 1:5] = True         # doğuya bakan çizgi
    assert list(zip(*np.nonzero(M.hit_or_miss_mask(b, M.hmt_kernel("isolated"))))) == [(1, 1)]
    assert list(zip(*np.nonzero(M.hit_or_miss_mask(b, M.hmt_kernel("end_point"))))) == [(4, 4)]
    assert list(zip(*np.nonzero(M.hit_or_miss_mask(b, M.hmt_kernel("end_point", "180"))))) == [(4, 1)]


def test_hmt_opencv_slide39_kernel():
    a = np.zeros((6, 6), bool)
    a[1:4, 1:4] = True
    hits = M.hit_or_miss_mask(a, M.hmt_kernel("cv_example"))
    assert list(zip(*np.nonzero(hits))) == [(3, 3)]   # sağ alt köşe
    raw = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_HITMISS, M.hmt_kernel("cv_example").astype(np.int8))
    assert np.array_equal(hits, raw.astype(bool))


def test_hmt_outside_image_is_background():
    full = np.ones((4, 4), bool)
    assert list(zip(*np.nonzero(M.hit_or_miss_mask(full, M.hmt_kernel("corner"))))) == [(0, 3)]
    assert M.hit_or_miss_mask(np.ones((1, 1), bool), M.hmt_kernel("isolated")).tolist() == [[True]]


@pytest.mark.parametrize("pattern,rotation", list(itertools.product(M.HMT_PATTERNS, M.HMT_ROTATIONS)))
def test_hmt_matches_definition(pattern, rotation):
    a = random_binary(hash((pattern, rotation)) % 1000, (20, 24), 0.5)
    k = M.hmt_kernel(pattern, rotation)
    hits = M.hit_or_miss_mask(a, k)
    assert np.array_equal(hits, ref_hmt(a, k))
    # (A ⊖ B1) ∩ (Aᶜ ⊖ B2) — sınırdan uzakta modülün kendi erosion'ıyla da aynı
    b1 = M.StructuringElement((k == 1).astype(np.uint8), (1, 1))
    b2 = M.StructuringElement((k == -1).astype(np.uint8), (1, 1))
    via = M.binary_erosion(a, b1) & M.binary_erosion(~a, b2)
    assert np.array_equal(hits[1:-1, 1:-1], via[1:-1, 1:-1])


def test_hmt_custom_kernels_and_empty_parts():
    a = random_binary(2, (8, 8))
    assert np.array_equal(M.hit_or_miss_mask(a, np.zeros((3, 3), int)), np.ones((8, 8), bool))
    only_fg = np.zeros((3, 3), int)
    only_fg[1, 1] = 1
    assert np.array_equal(M.hit_or_miss_mask(a, only_fg), a)
    k5 = np.zeros((5, 5), int)
    k5[2, 2], k5[2, 4] = 1, -1
    assert np.array_equal(M.hit_or_miss_mask(a, k5), ref_hmt(a, k5))


@pytest.mark.parametrize("kernel", [np.zeros((2, 3)), np.zeros((3, 3, 1)), np.full((3, 3), 2), np.zeros(3)])
def test_hmt_invalid_kernels(kernel):
    with pytest.raises(AlgorithmParameterError):
        M.hit_or_miss_mask(np.zeros((4, 4), bool), kernel)


@pytest.mark.parametrize("kwargs", [dict(pattern="star"), dict(rotation="45"), dict(rotation=90), dict(display="both")])
def test_hmt_invalid_options(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.hit_or_miss(np.zeros((4, 4), np.uint8), **kwargs)


def test_hmt_overlay_display():
    img = np.where(np.pad(SQUARE5, 1), 255, 0).astype(np.uint8)
    out = M.hit_or_miss(img, pattern="corner", display="overlay")
    assert out[2, 6] == 255 and out[4, 4] == M.OVERLAY_SHAPE_LEVEL and out[0, 0] == 0
    assert set(np.unique(out)) == {0, M.OVERLAY_SHAPE_LEVEL, 255}
    assert set(np.unique(M.hit_or_miss(img))) == {0, 255}


# --------------------------------------------------------------------------
# Görüntü düzeyindeki işlevler
# --------------------------------------------------------------------------

OPS = {
    "erode": lambda im, **k: M.erode(im, **k),
    "dilate": lambda im, **k: M.dilate(im, **k),
    "opening": lambda im, **k: M.opening(im, **k),
    "closing": lambda im, **k: M.closing(im, **k),
    "gradient": lambda im, **k: M.gradient(im, **k),
    "hit_or_miss": lambda im, **k: M.hit_or_miss(im, **{x: v for x, v in k.items() if x in ("binarize", "thresh", "foreground")}),
}
MASK_FN = {
    "erode": lambda a, se: M.binary_erosion(a, se),
    "dilate": lambda a, se: M.binary_dilation(a, se),
    "opening": M.binary_opening,
    "closing": M.binary_closing,
    "gradient": M.binary_gradient,
    "hit_or_miss": lambda a, se: M.hit_or_miss_mask(a, M.hmt_kernel()),
}


def make_image(shape, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = rng.integers(0, 256, shape[:2])
    return img


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape", [(12, 15), (12, 15, 1), (12, 15, 3), (12, 15, 4)])
def test_layout_dtype_alpha_and_immutability(op, shape):
    img = make_image(shape, 3)
    before = img.copy()
    out = OPS[op](img, binarize="manual", thresh=120)
    np.testing.assert_array_equal(img, before)
    assert out.shape == img.shape and out.dtype == np.uint8 and out is not img
    assert not np.shares_memory(out, img)
    color = out if out.ndim == 2 else out[..., :min(3, out.shape[2])]
    expected = np.where(MASK_FN[op](M.to_binary(img, "manual", 120), M.structuring_element()), 255, 0)
    for ch in range(1 if color.ndim == 2 else color.shape[2]):
        plane = color if color.ndim == 2 else color[..., ch]
        np.testing.assert_array_equal(plane, expected)
    if len(shape) == 3 and shape[2] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])   # alfa aynen korunur


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (3, 3, 3), (2, 3, 4)])
def test_tiny_images_with_large_se(op, shape):
    img = make_image(shape, 1)
    out = OPS[op](img, binarize="manual", thresh=100, **({} if op == "hit_or_miss" else dict(width=7, height=7)))
    assert out.shape == img.shape and set(np.unique(out[..., :3] if out.ndim == 3 else out)) <= {0, 255}


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("value", [0, 128, 255])
def test_constant_images(op, value):
    img = np.full((6, 7, 3), value, np.uint8)
    with np.errstate(all="raise"):
        out = OPS[op](img, binarize="manual", thresh=127)
    if value < 128 and op != "hit_or_miss":
        assert not out.any()   # boş küme her işlemde boş kalır (dilation dahil)


@pytest.mark.parametrize("op", ["erode", "dilate", "opening", "closing", "gradient"])
def test_se_parameters_are_forwarded(op):
    img = np.where(random_binary(13, (20, 20)), 255, 0).astype(np.uint8)
    se = M.structuring_element("cross", 5, 3, 0, 2)
    out = OPS[op](img, shape="cross", width=5, height=3, anchor_x=0, anchor_y=2)
    np.testing.assert_array_equal(out, np.where(MASK_FN[op](img > 127, se), 255, 0))


def test_dark_foreground_on_text_like_image():
    img = np.full((9, 9), 230, np.uint8)
    img[2:7, 4] = 20  # açık zemin üzerinde koyu dikey çizgi
    out = M.dilate(img, foreground="dark", shape="rect", width=3, height=1)
    assert (out[2:7, 3:6] == 255).all() and out.sum() == 255 * 15


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint8),
                                 np.zeros((3, 3, 2), np.uint8), [[0, 1]]])
def test_invalid_images(bad):
    for op in OPS.values():
        with pytest.raises(AlgorithmParameterError):
            op(bad)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True, "2"])
def test_invalid_iterations(iterations):
    with pytest.raises(AlgorithmParameterError):
        M.erode(np.zeros((4, 4), np.uint8), iterations=iterations)


# --------------------------------------------------------------------------
# Katalog / registry
# --------------------------------------------------------------------------


def test_registry_morphology_entries():
    registry = build_registry()
    algos = registry.in_category("morphology")
    assert [a.key for a in algos] == ["erode", "dilate", "opening", "closing", "gradient", "hit_or_miss"]
    assert all(a.implemented and a.summary for a in algos)
    se_keys = ["binarize", "thresh", "foreground", "shape", "width", "height", "anchor_x", "anchor_y"]
    for a in algos[:5]:
        assert [p.key for p in a.params][:8] == se_keys
        assert all(p.help or p.key == "width" for p in a.params)
    assert [p.key for p in registry.get("erode").params][-1] == "iterations"
    assert [p.key for p in registry.get("hit_or_miss").params] == ["binarize", "thresh", "foreground",
                                                                   "pattern", "rotation", "display"]
    assert set(registry.get("hit_or_miss").params[3].values) == set(M.HMT_PATTERNS)


def test_registry_defaults_run_and_values_are_sanitized():
    registry = build_registry()
    img = make_image((10, 11, 3))
    for algo in registry.in_category("morphology"):
        out = algo.run(img)
        assert out.shape == img.shape
    algo = registry.get("dilate")
    p = algo.resolve_params({"width": 4, "anchor_x": -9, "thresh": 999, "shape": "disk"})
    assert p["width"] == 5 and p["anchor_x"] == -1 and p["thresh"] == 255 and p["shape"] == "rect"


def test_registry_reports_anchor_outside_se():
    """Panel aralığı (−1…30) SE boyutundan bağımsızdır; tutarsızlık işlev içinde yakalanır."""
    algo = build_registry().get("erode")
    with pytest.raises(AlgorithmParameterError, match="Anchor X"):
        algo.run(np.zeros((5, 5), np.uint8), {"width": 3, "anchor_x": 5})
