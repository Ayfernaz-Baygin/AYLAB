"""Gri ton top-hat / black-hat ve birleşik düzeltme (Chapter 4, slayt 43–47).

Bağımsız referans (``ref_*``) OpenCV kullanmaz: düz SE için tanımları öteleme ile
kurar; görüntü dışı yok sayılır (erosion'da +∞, dilation'da −∞ dolgu), boş
komşuluk [0, 255] kafesinde inf ∅ = 255, sup ∅ = 0 olur.

    [f ⊖ b](z) = min_{s∈b} f(z + s)      [f ⊕ b](z) = max_{s∈b} f(z − s)
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


def shift(a, dy, dx, fill):
    h, w = a.shape
    out = np.full((h, w), fill, float)
    r0, r1 = max(0, -dy), min(h, h - dy)
    c0, c1 = max(0, -dx), min(w, w - dx)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = a[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
    return out


def offsets(mask, anchor):
    ax, ay = anchor
    return [(r - ay, c - ax) for r, c in zip(*np.nonzero(mask))]


def ref_erode(f, se):
    out = np.full(f.shape, np.inf)
    for dy, dx in offsets(se.mask, se.anchor):
        out = np.minimum(out, shift(f.astype(float), dy, dx, np.inf))
    return np.where(np.isinf(out), 255, out)


def ref_dilate(f, se):
    out = np.full(f.shape, -np.inf)
    for dy, dx in offsets(se.mask, se.anchor):
        out = np.maximum(out, shift(f.astype(float), -dy, -dx, -np.inf))
    return np.where(np.isinf(out), 0, out)


def ref_open(f, se):
    return ref_dilate(ref_erode(f, se), se)


def ref_close(f, se):
    return ref_erode(ref_dilate(f, se), se)


def ref_top_hat(f, se):
    return f.astype(float) - ref_open(f, se)


def ref_black_hat(f, se):
    return ref_close(f, se) - f.astype(float)


def ref_luma(bgr):
    """BT.601, yuvarlanmış (B, G, R düzeni) — common.luminance'tan bağımsız yazıldı."""
    b, g, r = (bgr[..., i].astype(float) for i in range(3))
    return np.clip(np.floor(0.299 * r + 0.587 * g + 0.114 * b + 0.5), 0, 255)


def se(shape="rect", w=3, h=3, ax=-1, ay=-1):
    return M.structuring_element(shape, w, h, ax, ay)


def row(*values):
    return np.array([values], np.uint8)


def kw(s):
    return dict(shape=s[0], width=s[1], height=s[2], anchor_x=s[3], anchor_y=s[4])


RNG_IMAGES = [np.random.default_rng(s).integers(0, 256, (17, 21), dtype=np.uint8) for s in range(3)]
SE_CASES = [(shape, w, h, ax, ay) for shape in M.SE_SHAPES
            for (w, h, ax, ay) in ((3, 3, -1, -1), (5, 5, -1, -1), (7, 3, -1, -1), (1, 5, -1, -1),
                                   (5, 5, 0, 0), (5, 3, 4, 2), (3, 7, 2, 0))]

# --------------------------------------------------------------------------
# Elle hesaplanmış küçük diziler
# --------------------------------------------------------------------------


def test_top_hat_extracts_bright_spike():
    f = row(10, 10, 10, 200, 10, 10, 10)
    b = se("rect", 3, 1)
    assert M.gray_opening(f, b).tolist() == [[10] * 7]
    assert M.white_top_hat_values(f, b).tolist() == [[0, 0, 0, 190, 0, 0, 0]]
    assert M.top_hat(f, width=3, height=1).tolist() == [[0, 0, 0, 190, 0, 0, 0]]


def test_black_hat_extracts_dark_pit():
    f = row(200, 200, 200, 20, 200, 200, 200)
    b = se("rect", 3, 1)
    assert M.gray_closing(f, b).tolist() == [[200] * 7]
    assert M.black_hat_values(f, b).tolist() == [[0, 0, 0, 180, 0, 0, 0]]
    assert M.black_hat(f, width=3, height=1).tolist() == [[0, 0, 0, 180, 0, 0, 0]]


def test_top_hat_ignores_dark_details_and_black_hat_ignores_bright_details():
    f = row(100, 100, 140, 100, 100, 80, 100, 100)
    b = se("rect", 3, 1)
    assert M.white_top_hat_values(f, b).tolist() == [[0, 0, 40, 0, 0, 0, 0, 0]]
    assert M.black_hat_values(f, b).tolist() == [[0, 0, 0, 0, 0, 20, 0, 0]]


def test_correction_formula_without_clipping():
    f = row(100, 100, 140, 100, 100, 80, 100, 100)
    # 140 + 40 − 0 = 180,  80 + 0 − 20 = 60
    assert M.hat_correction(f, width=3, height=1).tolist() == [[100, 100, 180, 100, 100, 60, 100, 100]]


def test_correction_clips_both_ends_instead_of_wrapping():
    f = row(50, 50, 200, 50, 50, 20, 50, 50)
    # parlak: 200 + 150 = 350 → 255;  koyu: 20 − 30 = −10 → 0  (uint8'de 94 ve 246'ya taşardı)
    out = M.hat_correction(f, width=3, height=1)
    assert out.dtype == np.uint8 and out.tolist() == [[50, 50, 255, 50, 50, 0, 50, 50]]
    wrapped = (f + M.white_top_hat_values(f, se("rect", 3, 1)).astype(np.uint8)
               - M.black_hat_values(f, se("rect", 3, 1)).astype(np.uint8))
    assert wrapped[0, 2] == 94 and wrapped[0, 5] == 246   # taşma olsaydı yanlış sonuç


def test_detail_must_be_smaller_than_se():
    f = np.full((7, 7), 100, np.uint8)
    f[2:5, 2:5] = 150                                    # 3×3 parlak blok
    assert not M.white_top_hat_values(f, se("rect", 3, 3)).any()   # SE sığar → opening korur
    th = M.white_top_hat_values(f, se("rect", 5, 5))     # SE sığmaz → blok çıkarılır
    expected = np.zeros((7, 7), int)
    expected[2:5, 2:5] = 50
    assert np.array_equal(th, expected)


def test_top_hat_flattens_uneven_illumination():
    """Doğrusal aydınlatma rampası flat SE ile açıldığında değişmez; T_hat yalnızca küçük ayrıntıları bırakır."""
    h, w = 40, 60
    ramp = np.add.outer(np.linspace(40, 120, h), np.linspace(0, 60, w))
    f = np.rint(ramp)
    dots = [(10, 10), (30, 45), (20, 30), (5, 50)]
    for r, c in dots:
        f[r:r + 2, c:c + 2] += 70
    f = f.astype(np.uint8)
    th = M.white_top_hat_values(f, se("rect", 7, 7))
    for r, c in dots:
        assert th[r:r + 2, c:c + 2].min() >= 60
    reach = 3
    mask = np.zeros_like(th, bool)
    mask[reach:-reach, reach:-reach] = True     # iç bölge
    for r, c in dots:
        mask[r - 3:r + 5, c - 3:c + 5] = False
    assert th[mask].max() <= 2            # zemin ≈ 0 (yalnızca yuvarlama basamakları)
    assert f[mask].max() - f[mask].min() > 100   # oysa girdide zemin 100+ düzey değişiyor
    # Sınır: dışarısı yok sayıldığı için rampanın PARLAK uçtaki (sağ/alt) ilk `reach` pikseli
    # SE'den dar bir tepe gibi davranır; T_hat orada eğim × uzaklık kadar değer alır.
    assert th[-1, reach:-reach].max() > 2 and th[reach:-reach, 0].max() <= 2


@pytest.mark.parametrize("value", [0, 1, 128, 254, 255])
def test_constant_image(value):
    f = np.full((6, 9), value, np.uint8)
    with np.errstate(all="raise"):
        assert not M.top_hat(f).any() and not M.black_hat(f).any()
        np.testing.assert_array_equal(M.hat_correction(f), f)


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (2, 2), (3, 3)])
def test_tiny_images_with_large_se(shape):
    f = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    b = se("rect", 15, 15)
    th, bh = M.white_top_hat_values(f, b), M.black_hat_values(f, b)
    assert (th >= 0).all() and (bh >= 0).all()
    assert np.array_equal(th, ref_top_hat(f, b)) and np.array_equal(bh, ref_black_hat(f, b))
    out = M.hat_correction(f, width=15, height=15)
    assert out.shape == f.shape


def test_single_pixel_is_unchanged():
    f = np.array([[77]], np.uint8)
    assert M.top_hat(f).tolist() == [[0]] and M.black_hat(f).tolist() == [[0]]
    assert M.hat_correction(f).tolist() == [[77]]


# --------------------------------------------------------------------------
# Bağımsız referans ve özellikler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("s", SE_CASES, ids=lambda s: "-".join(map(str, s)))
def test_gray_operations_match_definitions(s):
    b = se(*s)
    for f in RNG_IMAGES:
        assert np.array_equal(M.gray_erosion(f, b), ref_erode(f, b))
        assert np.array_equal(M.gray_dilation(f, b), ref_dilate(f, b))
        assert np.array_equal(M.gray_opening(f, b), ref_open(f, b))
        assert np.array_equal(M.gray_closing(f, b), ref_close(f, b))
        assert np.array_equal(M.white_top_hat_values(f, b), ref_top_hat(f, b))
        assert np.array_equal(M.black_hat_values(f, b), ref_black_hat(f, b))


@pytest.mark.parametrize("s", SE_CASES, ids=lambda s: "-".join(map(str, s)))
def test_hats_are_non_negative_and_ordering_holds(s):
    b = se(*s)
    for f in RNG_IMAGES:
        o, c = M.gray_opening(f, b), M.gray_closing(f, b)
        assert (o <= f).all() and (f <= c).all()          # f ∘ b ≤ f ≤ f • b (sınırda da)
        assert (M.white_top_hat_values(f, b) >= 0).all()
        assert (M.black_hat_values(f, b) >= 0).all()
        assert np.array_equal(M.gray_opening(o, b), o)    # idempotent
        assert np.array_equal(M.gray_closing(c, b), c)


@pytest.mark.parametrize("s", SE_CASES, ids=lambda s: "-".join(map(str, s)))
def test_correction_matches_course_formula(s):
    b = se(*s)
    for f in RNG_IMAGES:
        expected = np.clip(f.astype(float) + ref_top_hat(f, b) - ref_black_hat(f, b), 0, 255)
        out = M.hat_correction(f, **kw(s))
        assert out.dtype == np.uint8 and np.array_equal(out, expected)
        # eşdeğer biçim: 3f − (f ∘ b) − (f • b)
        assert np.array_equal(out, np.clip(3 * f.astype(int) - ref_open(f, b) - ref_close(f, b), 0, 255))


def test_per_pixel_loop_reference():
    """Tamamen döngüyle yazılmış ikinci referans (öteleme yok)."""
    f = RNG_IMAGES[0][:9, :11]
    b = se("cross", 5, 3, 1, 2)
    offs = offsets(b.mask, b.anchor)
    h, w = f.shape

    def ero(g):
        return np.array([[min([g[r + dy, c + dx] for dy, dx in offs if 0 <= r + dy < h and 0 <= c + dx < w], default=255)
                          for c in range(w)] for r in range(h)])

    def dil(g):
        return np.array([[max([g[r - dy, c - dx] for dy, dx in offs if 0 <= r - dy < h and 0 <= c - dx < w], default=0)
                          for c in range(w)] for r in range(h)])

    assert np.array_equal(M.white_top_hat_values(f, b), f.astype(int) - dil(ero(f)))
    assert np.array_equal(M.black_hat_values(f, b), ero(dil(f)) - f.astype(int))


def test_matches_opencv_tophat_blackhat_for_symmetric_se():
    f = RNG_IMAGES[1]
    for shape in M.SE_SHAPES:
        b = se(shape, 7, 7)
        assert np.array_equal(M.white_top_hat_values(f, b), cv2.morphologyEx(f, cv2.MORPH_TOPHAT, b.mask))
        assert np.array_equal(M.black_hat_values(f, b), cv2.morphologyEx(f, cv2.MORPH_BLACKHAT, b.mask))


def test_dilation_uses_reflected_se():
    f = np.zeros((1, 7), np.uint8)
    f[0, 3] = 90
    b = se("rect", 3, 1, ax=0)       # B = {0, +1, +2}: f ⊕ b sağa uzanır
    assert M.gray_dilation(f, b).tolist() == [[0, 0, 0, 90, 90, 90, 0]]
    assert M.gray_erosion(255 - f, b).tolist() == [[255, 165, 165, 165, 255, 255, 255]]


def test_anchor_shifts_erosion_but_not_hats_in_the_interior():
    f = RNG_IMAGES[2]
    centred, corner = se("rect", 5, 5), se("rect", 5, 5, 0, 0)
    e0, e1 = M.gray_erosion(f, centred), M.gray_erosion(f, corner)
    assert np.array_equal(e1[:-4, :-4], e0[2:-2, 2:-2])          # orijin taşınınca sonuç ötelenir
    r = 2 * centred.reach
    for fn in (M.white_top_hat_values, M.black_hat_values):
        assert np.array_equal(fn(f, corner)[r:-r, r:-r], fn(f, centred)[r:-r, r:-r])


# --------------------------------------------------------------------------
# Renk, alfa, kanal düzeni, değişmezlik
# --------------------------------------------------------------------------


def color_image(shape=(16, 19, 3), seed=4):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_gray_content_in_bgr_equals_gray():
    g = RNG_IMAGES[0]
    bgr = np.dstack([g, g, g])
    for fn in (M.top_hat, M.black_hat, M.hat_correction):
        out = fn(bgr)
        assert out.shape == bgr.shape
        for c in range(3):
            np.testing.assert_array_equal(out[..., c], fn(g))


def test_color_hats_use_luminance():
    img = color_image()
    y = ref_luma(img).astype(np.uint8)
    b = se("rect", 15, 15)
    for fn, ref in ((M.top_hat, ref_top_hat), (M.black_hat, ref_black_hat)):
        out = fn(img)
        assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
        np.testing.assert_array_equal(out[..., 0], ref(y, b))


def test_color_correction_adds_same_delta_to_each_channel():
    img = color_image()
    y = ref_luma(img).astype(np.uint8)
    b = se("rect", 15, 15)
    delta = ref_top_hat(y, b) - ref_black_hat(y, b)
    out = M.hat_correction(img)
    expected = np.clip(img.astype(float) + delta[..., None], 0, 255)
    np.testing.assert_array_equal(out, expected)
    # Kırpılmayan piksellerde renklilik farkları (B − G, R − G) aynen korunur
    unclipped = ((img.astype(float) + delta[..., None] >= 0) & (img.astype(float) + delta[..., None] <= 255)).all(-1)
    d_in = img.astype(int)[..., [0, 2]] - img.astype(int)[..., [1]]
    d_out = out.astype(int)[..., [0, 2]] - out.astype(int)[..., [1]]
    assert unclipped.mean() > 0.3 and np.array_equal(d_in[unclipped], d_out[unclipped])


def test_color_correction_does_not_create_false_color_edges():
    """Eşit parlaklıklı kırmızı nokta / yeşil zemin: Y'de ayrıntı yok → görüntü değişmez.

    Kanal başına top-hat ise kırmızı kanalda noktayı 'parlak ayrıntı' sayıp renk kaydırırdı.
    """
    img = np.zeros((15, 15, 3), np.uint8)
    img[...] = (0, 128, 0)                     # yeşil, Y ≈ 75
    img[7, 7] = (0, 0, 251)                    # kırmızı, Y ≈ 75
    assert ref_luma(img).min() == ref_luma(img).max()
    np.testing.assert_array_equal(M.hat_correction(img, width=5, height=5), img)
    per_channel = cv2.morphologyEx(img[..., 2], cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
    assert per_channel[7, 7] == 251            # kanal başına yaklaşım burada yanlış ayrıntı bulurdu


@pytest.mark.parametrize("fn", [M.top_hat, M.black_hat, M.hat_correction])
@pytest.mark.parametrize("shape", [(12, 15), (12, 15, 1), (12, 15, 3), (12, 15, 4)])
def test_layout_alpha_and_immutability(fn, shape):
    img = np.random.default_rng(7).integers(0, 256, shape, dtype=np.uint8)
    before = img.copy()
    out = fn(img, width=5, height=5)
    np.testing.assert_array_equal(img, before)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.shares_memory(out, img)
    if len(shape) == 3 and shape[2] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])
        np.testing.assert_array_equal(out[..., :3], fn(np.ascontiguousarray(img[..., :3]), width=5, height=5))
    if len(shape) == 3 and shape[2] == 1:
        np.testing.assert_array_equal(out[..., 0], fn(img[..., 0], width=5, height=5))


def test_alpha_does_not_influence_values():
    color = color_image((10, 12, 3))
    outs = [M.hat_correction(np.dstack([color, np.full((10, 12), a, np.uint8)]), width=5, height=5)
            for a in (0, 90, 255)]
    for o in outs[1:]:
        np.testing.assert_array_equal(o[..., :3], outs[0][..., :3])


# --------------------------------------------------------------------------
# Geçersiz girdiler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [M.top_hat, M.black_hat, M.hat_correction])
@pytest.mark.parametrize("kwargs", [dict(width=4), dict(height=0), dict(width=2.5), dict(shape="disk"),
                                    dict(width=5, anchor_x=5), dict(anchor_y=-2), dict(anchor_x=True)])
def test_invalid_se_parameters(fn, kwargs):
    with pytest.raises(AlgorithmParameterError):
        fn(np.zeros((5, 5), np.uint8), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint8),
                                 np.zeros((3, 3, 2), np.uint8), [[0, 1]]])
def test_invalid_images(bad):
    for fn in (M.top_hat, M.black_hat, M.hat_correction):
        with pytest.raises(AlgorithmParameterError):
            fn(bad)


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float64), np.zeros((3, 3, 3), np.uint8), np.zeros((0, 0), np.uint8)])
def test_gray_helpers_reject_non_2d_uint8(bad):
    with pytest.raises(AlgorithmParameterError):
        M.gray_erosion(bad, se())


# --------------------------------------------------------------------------
# Katalog ve ikili işlemlerde regresyon olmadığı
# --------------------------------------------------------------------------


def test_registry_hat_entries():
    registry = build_registry()
    keys = [a.key for a in registry.in_category("morphology")]
    assert keys[-3:] == ["top_hat", "black_hat", "hat_correction"]
    assert [registry.get(k).title for k in keys[-3:]] == ["White Top-Hat", "Black-Hat", "Illumination / Contrast Correction"]
    assert "f − (f ∘ b)" in registry.get("top_hat").summary
    assert "(f • b) − f" in registry.get("black_hat").summary
    assert "img + top_hat − black_hat" in registry.get("hat_correction").summary
    binary_se = [(p.key, p.label, p.minimum, p.maximum) for p in registry.get("opening").params[3:]
                 if hasattr(p, "minimum")]
    for k in keys[-3:]:
        algo = registry.get(k)
        assert algo.implemented
        assert [p.key for p in algo.params] == ["shape", "width", "height", "anchor_x", "anchor_y"]
        assert not any(p.key in ("binarize", "thresh", "foreground") for p in algo.params)   # eşikleme yok
        assert algo.default_params() == dict(shape="rect", width=15, height=15, anchor_x=-1, anchor_y=-1)
        assert [(p.key, p.label, p.minimum, p.maximum) for p in algo.params if hasattr(p, "minimum")] == binary_se


def test_binary_operations_keep_their_defaults():
    registry = build_registry()
    for k in ("erode", "dilate", "opening", "closing", "gradient"):
        d = registry.get(k).default_params()
        assert d["width"] == 3 and d["height"] == 3 and d["binarize"] == "otsu"


def test_registry_runs_and_sanitizes():
    registry = build_registry()
    img = color_image((20, 22, 3))
    for k in ("top_hat", "black_hat", "hat_correction"):
        algo = registry.get(k)
        assert algo.run(img).shape == img.shape
        p = algo.resolve_params({"width": 40, "height": 2, "anchor_x": -7})
        assert p["width"] == 31 and p["height"] == 3 and p["anchor_x"] == -1
    with pytest.raises(AlgorithmParameterError, match="Anchor Y"):
        registry.get("black_hat").run(img, {"height": 3, "anchor_y": 9})


def test_hat_operations_do_not_threshold():
    """Gri düzey bilgisi korunur: çıktı yalnızca 0/255 değildir."""
    f = np.add.outer(np.arange(20), np.arange(20)).astype(np.uint8) * 6
    f[5, 5] += 37
    f[14, 9] -= 23
    vals = set(np.unique(M.hat_correction(f, width=3, height=3)))
    assert len(vals) > 10
    assert M.top_hat(f, width=3, height=3)[5, 5] > 0 and M.black_hat(f, width=3, height=3)[14, 9] > 0
