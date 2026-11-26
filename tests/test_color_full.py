"""Color Processing / Paket 3: full-color — color transformation, color slicing, tone corrections,
color balancing / white balance, histogram equalization in HSI (slayt 51–56, laboratuvar).

Referanslar laboratuvar formüllerinin piksel piksel, ``math`` ile yazılmış bağımsız sürümleridir (NumPy yok)."""

import inspect
import math

import numpy as np
import pytest

from processing import color_processing as C
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError

FULL_KEYS = ["color_transform", "color_slicing", "tone_correction", "color_balance", "hsi_equalization"]
FUNCS = dict(zip(FULL_KEYS, (C.color_transformation, C.color_slicing, C.tone_correction, C.color_balance,
                             C.hsi_equalization)))


# ================================================================ bağımsız referanslar (laboratuvar, math)
def c8(v):
    return 0 if v <= 0 else 255 if v >= 255 else math.floor(v + 0.5)


def rgb2hsi(r8, g8, b8):
    R, G, B = r8 / 255, g8 / 255, b8 / 255
    num = 0.5 * ((R - G) + (R - B))
    den = math.sqrt((R - G) * (R - G) + (R - B) * (G - B))
    if den < 1e-9:
        H = 0.0
    else:
        theta = (math.acos(max(-1.0, min(1.0, num / den))) * 180) / math.pi
        H = theta if B <= G else 360 - theta
    s = R + G + B
    S = 0.0 if s < 1e-9 else 1 - (3 / s) * min(R, G, B)
    return H, S, s / 3


def hsi2rgb(H, S, I):
    H = ((H % 360) + 360) % 360
    rad = lambda d: (d * math.pi) / 180
    f = lambda h: I * (1 + (S * math.cos(rad(h))) / math.cos(rad(60 - h)))
    if H < 120:
        B = I * (1 - S); R = f(H); G = 3 * I - (R + B)
    elif H < 240:
        R = I * (1 - S); G = f(H - 120); B = 3 * I - (R + G)
    else:
        G = I * (1 - S); B = f(H - 240); R = 3 * I - (G + B)
    return R, G, B


def pixels(rgb):
    h, w = rgb.shape[:2]
    return [(y, x, [int(v) for v in rgb[y, x]]) for y in range(h) for x in range(w)]


def ref_map(rgb, fn):
    out = np.zeros(rgb.shape, np.uint8)
    for y, x, p in pixels(rgb):
        out[y, x] = [c8(v) for v in fn(p)]
    return out


def ref_transform(rgb, model, k):
    def fn(p):
        n = [v / 255 for v in p]
        if model == "rgb":
            return [(k * v) * 255 for v in n]
        if model == "cmy":
            return [(1 - (k * (1 - v) + (1 - k))) * 255 for v in n]
        H, S, I = rgb2hsi(*p)
        return [v * 255 for v in hsi2rgb(H, S, k * I)]
    return ref_map(rgb, fn)


def ref_slice(rgb, shape, W, R0, a):
    def fn(p):
        if shape == "cube":
            keep = all(abs(v - a[j]) <= W / 2 for j, v in enumerate(p))
        else:
            keep = sum((v - a[j]) ** 2 for j, v in enumerate(p)) <= R0 * R0
        return p if keep else [128, 128, 128]
    return ref_map(rgb, fn)


def ref_scurve(a):
    s = lambda x: 1 / (1 + math.exp(-a * (x - 0.5)))
    lo, hi = s(0), s(1)
    return lambda x: (s(x) - lo) / (hi - lo)


def ref_gains(rgb, method, p=None):
    px = [q for _, _, q in pixels(rgb)]
    n = len(px)
    if method == "gray":
        mean = [sum(q[c] for q in px) / n for c in range(3)]
        g = (mean[0] + mean[1] + mean[2]) / 3
        return [g / m for m in mean]
    if method == "white":
        return [255 / max(1, max(q[c] for q in px)) for c in range(3)]
    g = (p[0] + p[1] + p[2]) / 3
    return [g / max(1, v) for v in p]


def ref_eq_lut(values):
    h = [0] * 256
    for v in values:
        h[v] += 1
    lut, c = [], 0
    for k in range(256):
        c += h[k]
        lut.append(c8(255 * c / len(values)))
    return lut


def ref_hsi_eq(rgb, sat):
    px = pixels(rgb)
    hsi = [rgb2hsi(*p) for _, _, p in px]
    iv = [c8(I * 255) for _, _, I in hsi]
    lut = ref_eq_lut(iv)
    out = np.zeros(rgb.shape, np.uint8)
    for (y, x, _), (H, S, _), v in zip(px, hsi, iv):
        out[y, x] = [c8(q * 255) for q in hsi2rgb(H, min(1, S * sat), lut[v] / 255)]
    return out


# ================================================================ görüntüler (sentetik, lisans sorunu yok)
def scene(h=24, w=32, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([np.clip(30 + 7 * xx, 0, 255), np.clip(15 + 9 * yy, 0, 255), np.clip(220 - 5 * xx - 2 * yy, 0, 255)], -1)
    base[::5, ::3] = rng.integers(0, 256, base[::5, ::3].shape)
    return base.astype(np.uint8)                                               # RGB


SPECIAL = np.array([[[0, 0, 0], [255, 255, 255], [128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                     [255, 255, 0], [0, 255, 255], [255, 0, 255], [1, 1, 1], [254, 254, 254], [190, 25, 45]]], np.uint8)


def bgr(rgb):
    return np.ascontiguousarray(rgb[..., ::-1])


def rgb_of(result):
    return result.image[..., ::-1]


# ================================================================ katalog
def test_full_color_operations_follow_slide_order():
    keys = [a.key for a in REGISTRY.in_category("color")]
    assert keys[5:10] == FULL_KEYS and keys[10:] == ["color_smooth_sharpen"]   # slayt 52–56, + Paket 4 (57–58)
    assert [REGISTRY.get(k).title for k in FULL_KEYS] == ["Color Transformation", "Color Slicing",
                                                          "Tone & Color Corrections", "Color Balancing",
                                                          "Histogram Equalization in HSI"]
    for key, func in FUNCS.items():
        algo = REGISTRY.get(key)
        assert algo.implemented and algo.func is func
        sig = inspect.signature(func)
        assert list(sig.parameters)[1:] == [p.key for p in algo.params]
        assert all(sig.parameters[p.key].default == p.default for p in algo.params)


def test_lab_defaults_and_ranges():
    p = {k: {q.key: q for q in REGISTRY.get(k).params} for k in FULL_KEYS}
    assert (p["color_transform"]["k"].default, p["color_transform"]["k"].minimum, p["color_transform"]["k"].maximum) == (0.7, 0.1, 1.0)
    assert (p["color_slicing"]["width"].default, p["color_slicing"]["radius"].default) == (110, 80)
    assert (p["color_slicing"]["width"].minimum, p["color_slicing"]["width"].maximum) == (10, 255)
    assert [p["color_slicing"][f"proto_{c}"].default for c in "rgb"] == [190, 25, 45]
    assert (p["tone_correction"]["strength"].default, p["tone_correction"]["gamma"].default) == (9.0, 1.0)
    assert (p["tone_correction"]["gamma"].minimum, p["tone_correction"]["gamma"].maximum) == (0.3, 3.0)
    assert p["color_balance"]["method"].default == "white"
    assert (p["color_balance"]["gamma_r"].minimum, p["color_balance"]["gamma_r"].maximum) == (0.4, 2.5)
    assert (p["hsi_equalization"]["method"].default, p["hsi_equalization"]["saturation"].minimum,
            p["hsi_equalization"]["saturation"].maximum) == ("hsi", 0.5, 1.6)


# ================================================================ slayt 52: color transformation
@pytest.mark.parametrize("model", ["rgb", "cmy", "hsi"])
@pytest.mark.parametrize("k", [0.1, 0.35, 0.7, 1.0])
def test_transformation_matches_loop_reference(model, k):
    img = np.concatenate([scene(), np.repeat(np.resize(SPECIAL, (1, 32, 3)), 2, 0)], 0)
    r = C.color_transformation(bgr(img), model, k)
    np.testing.assert_array_equal(rgb_of(r), ref_transform(img, model, k))


@pytest.mark.parametrize("model", C.TRANSFORM_MODELS)
def test_transformation_is_the_same_effect_in_every_model(model):
    img = np.concatenate([scene(), np.resize(SPECIAL, (1, 32, 3))], 0)
    for k in (0.2, 0.7, 0.95):
        out = rgb_of(C.color_transformation(bgr(img), model, k)).astype(int)
        exact = k * img.astype(float)
        assert np.abs(out - exact).max() <= 0.5 + 1e-6                          # yalnızca yuvarlama (± 0.5)


def test_transformation_known_colors_and_identity():
    rgb = rgb_of(C.color_transformation(bgr(SPECIAL), "rgb", 0.7))
    assert rgb[0, 0].tolist() == [0, 0, 0] and rgb[0, 1].tolist() == [179, 179, 179]   # 0.7·255 = 178.5 → 179
    assert rgb[0, 3].tolist() == [179, 0, 0] and rgb[0, 2].tolist() == [90, 90, 90]
    r = C.color_transformation(bgr(SPECIAL), "hsi", 0.7)
    out = rgb_of(r)
    # HSI yuvarlaması: beyazda R = B = 178.5 ama G = 3I − (R + B) = 178.49999999999994 → 178 (laboratuvar da aynı)
    assert out[0, 1].tolist() == [179, 178, 179] and out[0, 3].tolist() == [178, 0, 0]
    assert np.abs(out.astype(int) - rgb).max() == 1                                    # yalnızca ± 1 yuvarlama
    assert "HSI 1 gray level" in C.color_transformation(bgr(SPECIAL), "rgb", 0.7).report.summary()
    for model in C.TRANSFORM_MODELS:                                                    # k = 1 → değişmez
        np.testing.assert_array_equal(rgb_of(C.color_transformation(bgr(scene()), model, 1.0)), scene())
    assert "largest difference from the other models" in r.report.summary()
    assert [c.name for c in r.report.curves] == ["H, S", "I"]
    np.testing.assert_allclose(r.report.curves[1].values, 0.7 * np.arange(256) / 255)
    cmy = C.color_transformation(bgr(SPECIAL), "cmy", 0.4).report
    np.testing.assert_allclose(cmy.curves[0].values[[0, 255]], [0.6, 1.0])           # s = k·r + (1 − k)


# ================================================================ slayt 53: color slicing
@pytest.mark.parametrize("shape", C.SLICE_SHAPES)
@pytest.mark.parametrize("size", [10, 60, 110, 255])
@pytest.mark.parametrize("a", [(190, 25, 45), (128, 128, 128), (0, 0, 0)])
def test_slicing_matches_loop_reference(shape, size, a):
    img = np.concatenate([scene(), np.resize(SPECIAL, (1, 32, 3))], 0)
    r = C.color_slicing(bgr(img), shape, size, size, *a)
    np.testing.assert_array_equal(rgb_of(r), ref_slice(img, shape, size, size, a))


def test_slicing_boundaries_cube_and_sphere():
    a = (100, 100, 100)
    # küp W = 20: |v − a| ≤ 10 içeride, 11 dışarıda (her kanalda ayrı ayrı)
    img = np.array([[[110, 100, 100], [111, 100, 100], [90, 90, 90], [89, 100, 100], [100, 110, 111]]], np.uint8)
    keep = (rgb_of(C.color_slicing(bgr(img), "cube", 20, 80, *a)) != 128).any(-1) | (img == 128).all(-1)
    assert keep[0].tolist() == [True, False, True, False, False]
    # tek W: W/2 = 10.5 → |v − a| = 11 hâlâ dışarıda; W = 22 → W/2 = 11 içeride
    assert rgb_of(C.color_slicing(bgr(img), "cube", 21, 80, *a))[0, 1].tolist() == [128, 128, 128]
    assert rgb_of(C.color_slicing(bgr(img), "cube", 22, 80, *a))[0, 1].tolist() == [111, 100, 100]
    # küre R₀ = 10: (6, 8, 0) uzaklık tam 10 → içeride; (6, 8, 1) → dışarıda
    sph = np.array([[[106, 108, 100], [106, 108, 101], [100, 100, 100]]], np.uint8)
    out = rgb_of(C.color_slicing(bgr(sph), "sphere", 110, 10, *a))
    assert out[0, 0].tolist() == [106, 108, 100] and out[0, 1].tolist() == [128, 128, 128]
    assert out[0, 2].tolist() == [100, 100, 100]


def test_slicing_keeps_colors_report_and_bgr_order():
    img = SPECIAL
    r = C.color_slicing(bgr(img), "cube", 60, 80, 255, 0, 0)                   # yalnızca saf kırmızı
    out = r.image
    assert out[0, 3].tolist() == [0, 0, 255]                                    # BGR: kırmızı korunur
    assert (out[0, [0, 1, 4, 5]] == 128).all()
    s = r.report.summary()
    assert "a = RGB 255, 0, 0" in s and "kept 1 of 12 px" in s and r.report.swatch == (255, 0, 0)
    assert r.report.rows[0] == "R ∈ [225, 255]"
    sph = C.color_slicing(bgr(img), "sphere", 60, 30, 255, 0, 0).report
    assert "Σ(r_j − a_j)² ≤ 900" in sph.summary()


# ================================================================ slayt 54: tone corrections
@pytest.mark.parametrize("curve,a,g", [("s", 1.0, 1.0), ("s", 4.5, 1.0), ("s", 9.0, 1.0), ("s", 16.0, 1.0),
                                       ("gamma", 9.0, 0.3), ("gamma", 9.0, 0.45), ("gamma", 9.0, 2.0),
                                       ("gamma", 9.0, 3.0)])
def test_tone_matches_loop_reference(curve, a, g):
    img = np.concatenate([scene(), np.resize(SPECIAL, (1, 32, 3))], 0)
    f = ref_scurve(a) if curve == "s" else (lambda x: math.pow(x, g))
    r = C.tone_correction(bgr(img), curve, a, g)
    np.testing.assert_array_equal(rgb_of(r), ref_map(img, lambda p: [255 * f(v / 255) for v in p]))


def test_tone_same_curve_for_all_channels_and_endpoints():
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, 2)
    for curve in C.TONE_CURVES:
        out = rgb_of(C.tone_correction(bgr(ramp), curve, 9.0, 2.0))
        assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()   # gri gri kalır
        assert out[0, 0].tolist() == [0, 0, 0] and out[0, 255].tolist() == [255, 255, 255]
        assert (np.diff(out[0, :, 0].astype(int)) >= 0).all()                               # monoton
    s = rgb_of(C.tone_correction(bgr(ramp), "s", 9.0, 1.0))[0, :, 0]
    assert s[64] < 64 and s[192] > 192 and s[127] < 128 <= s[128]                           # orta tonları gerer
    assert (s.astype(int) + s[::-1] == 255).all()                                           # 0.5'e göre simetrik
    np.testing.assert_array_equal(rgb_of(C.tone_correction(bgr(scene()), "gamma", 9.0, 1.0)), scene())   # γ = 1


# ================================================================ slayt 55: color balancing / white balance
@pytest.mark.parametrize("method", ["gray", "white", "neutral"])
def test_balance_gains_match_loop_reference(method):
    img = C.c8(scene().astype(float) * np.array([1.12, 0.94, 0.66]))          # laboratuvar sıcak kayma
    p = (200, 150, 90)
    k = ref_gains(img, method, p)
    r = C.color_balance(bgr(img), method, *p)
    np.testing.assert_array_equal(rgb_of(r), ref_map(img, lambda q: [q[c] * k[c] for c in range(3)]))


def test_balance_white_patch_and_neutral_pixel_make_reference_neutral():
    img = np.array([[[200, 180, 120], [100, 90, 60], [20, 30, 40]]], np.uint8)
    out = rgb_of(C.color_balance(bgr(img), "white"))
    assert out[0, 0].tolist() == [255, 255, 255]                                # en parlak değerler → 255
    neu = rgb_of(C.color_balance(bgr(img), "neutral", 200, 180, 120))
    assert neu[0, 0].tolist() == [167, 167, 167]                                # tıklanan piksel gri olur
    gw = rgb_of(C.color_balance(bgr(scene()), "gray")).reshape(-1, 3).mean(0)
    assert np.ptp(gw) < 1.0                                                     # gri dünya: ortalamalar eşit


def test_balance_gains_and_curves():
    img = scene()
    r = C.color_balance(bgr(img), "gains", gain_r=1.5, gain_g=1.0, gain_b=0.5)
    np.testing.assert_array_equal(rgb_of(r), ref_map(img, lambda q: [q[0] * 1.5, q[1], q[2] * 0.5]))
    gs = (0.4, 1.3, 2.5)
    r = C.color_balance(bgr(img), "curves", gamma_r=gs[0], gamma_g=gs[1], gamma_b=gs[2])
    np.testing.assert_array_equal(rgb_of(r), ref_map(img, lambda q: [255 * math.pow(q[c] / 255, gs[c]) for c in range(3)]))
    assert [c.name for c in r.report.curves] == ["R", "G", "B"]
    np.testing.assert_array_equal(rgb_of(C.color_balance(bgr(img), "curves")), img)   # γ = 1 → değişmez


def test_balance_degenerate_images():
    black = np.zeros((3, 4, 3), np.uint8)
    for m in ("gray", "white", "neutral"):
        np.testing.assert_array_equal(C.color_balance(black, m).image, black)   # 0/0 yok, siyah siyah kalır
    no_blue = scene().copy()
    no_blue[..., 2] = 0
    r = C.color_balance(bgr(no_blue), "gray")
    assert (rgb_of(r)[..., 2] == 0).all() and "k_B = 1.000" in r.report.summary()


# ================================================================ slayt 56: histogram equalization in HSI
@pytest.mark.parametrize("sat", [0.5, 1.0, 1.3, 1.6])
def test_hsi_equalization_matches_loop_reference(sat):
    img = np.concatenate([C.c8(255 * (scene() / 255) ** 2.2), np.resize(SPECIAL, (1, 32, 3))], 0)   # koyu görüntü
    r = C.hsi_equalization(bgr(img), "hsi", sat)
    np.testing.assert_array_equal(rgb_of(r), ref_hsi_eq(img, sat))


def test_hsi_equalization_keeps_hue_and_saturation():
    img = C.c8(255 * (scene() / 255) ** 2.2)
    hsi, new, _ = C.equalize_hsi(img.astype(np.float64), 1.0)
    np.testing.assert_array_equal(new[..., 0], hsi[..., 0])                    # H aynen
    np.testing.assert_array_equal(new[..., 1], hsi[..., 1])                    # S aynen (sat = 1)
    assert not np.array_equal(new[..., 2], hsi[..., 2])                        # yalnızca I değişir
    _, boosted, _ = C.equalize_hsi(img.astype(np.float64), 1.4)
    np.testing.assert_array_equal(boosted[..., 0], hsi[..., 0])
    np.testing.assert_allclose(boosted[..., 1], np.minimum(1, hsi[..., 1] * 1.4))
    # 8-bit çıktıda ton kayması: gamut içindeki kromatik piksellerde yalnızca yuvarlama kadar
    out = rgb_of(C.hsi_equalization(bgr(img))).astype(np.float64)
    _, _, raw = C.equalize_hsi(img.astype(np.float64), 1.0)
    inside = (raw <= 255).all(-1) & (raw >= 0).all(-1)
    h0, h1 = C.rgb_to_model(img.astype(float), "hsi"), C.rgb_to_model(out, "hsi")
    chroma = inside & (h0[..., 1] > 0.2) & (h1[..., 2] > 0.2)
    dh = np.abs((h1[..., 0] - h0[..., 0] + 540) % 360 - 180)[chroma]
    assert chroma.sum() > 100 and dh.max() < 3.0


def test_hsi_equalization_spreads_intensity_and_compares_with_rgb():
    img = C.c8(255 * (np.random.default_rng(5).integers(0, 256, (48, 64, 3)) / 255) ** 2.2)   # koyu görüntü
    r = C.hsi_equalization(bgr(img))
    hsi, new, _ = C.equalize_hsi(img.astype(np.float64))
    assert np.median(hsi[..., 2]) < 0.32 and abs(np.median(new[..., 2]) - 0.5) < 0.02  # slayt 56: medyan → ≈ 0.5
    i1 = C.rgb_to_model(rgb_of(r).astype(float), "hsi")[..., 2]
    assert np.median(i1) > np.median(hsi[..., 2]) + 0.1    # 8-bit çıktı da aydınlanır (gamut kırpması I′yı biraz düşürür)
    assert "median I" in r.report.summary() and [c.name for c in r.report.curves] == ["H", "I", "S"]
    lut = C.equalization_lut(C.c8(C.rgb_to_model(img.astype(float), "hsi")[..., 2] * 255))
    np.testing.assert_array_equal(np.round(r.report.curves[1].values * 255).astype(int), lut)
    # RGB yöntemi: laboratuvar karşılaştırması, kanal başına ayrı eqLUT
    rr = C.hsi_equalization(bgr(img), "rgb")
    exp = np.stack([np.array(ref_eq_lut(img[..., c].ravel().tolist()))[img[..., c]] for c in range(3)], -1)
    np.testing.assert_array_equal(rgb_of(rr), exp)
    assert "not HSI" in rr.report.summary() and len(rr.report.curves) == 3
    assert not np.array_equal(rr.image, r.image)                                # RGB, HSI'nın yerine geçmez


def test_hsi_equalization_gray_stays_gray():
    g = np.tile(np.arange(40, 120, 2, dtype=np.uint8), (6, 1))
    out = C.hsi_equalization(g).image
    assert out.shape == (6, 40, 3) and (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()
    np.testing.assert_array_equal(out[..., 0], np.array(ref_eq_lut(g.ravel().tolist()))[g])


# ================================================================ ortak: düzen, boyut, alfa, sınırlar
CALLS = [("color_transform", {"model": "hsi"}), ("color_transform", {"model": "cmyk"}), ("color_slicing", {}),
         ("color_slicing", {"shape": "sphere"}), ("tone_correction", {}), ("tone_correction", {"curve": "gamma", "gamma": 0.45}),
         ("color_balance", {}), ("color_balance", {"method": "gray"}), ("color_balance", {"method": "curves", "gamma_r": 0.6}),
         ("hsi_equalization", {}), ("hsi_equalization", {"method": "rgb"})]


@pytest.mark.parametrize("key,params", CALLS, ids=[f"{k}-{i}" for i, (k, _) in enumerate(CALLS)])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (5, 7)])
def test_layouts_sizes_and_alpha(key, params, shape):
    rng = np.random.default_rng(sum(shape))
    color = rng.integers(0, 256, shape + (3,), dtype=np.uint8)
    algo = REGISTRY.get(key)
    out = algo.execute(color, params)
    assert out.image.shape == shape + (3,) and out.image.dtype == np.uint8 and out.changes_image
    alpha = rng.integers(0, 256, shape + (1,), dtype=np.uint8)
    res = algo.execute(np.concatenate([color, alpha], -1), params).image
    assert res.shape == shape + (4,)
    np.testing.assert_array_equal(res[..., 3], alpha[..., 0])                   # alfa aynen
    np.testing.assert_array_equal(res[..., :3], out.image)
    gray = color[..., 1]
    for g in (gray, gray[..., None]):                                           # gri girdi = R = G = B
        np.testing.assert_array_equal(algo.execute(g, params).image,
                                      algo.execute(np.repeat(gray[..., None], 3, -1), params).image)


def test_bgr_channel_order_is_respected():
    red = bgr(np.array([[[200, 30, 10]]], np.uint8))                           # BGR (10, 30, 200)
    assert C.color_balance(red, "gains", gain_r=1.2, gain_g=1.0, gain_b=0.0).image[0, 0].tolist() == [0, 30, 240]
    assert C.color_slicing(red, "cube", 20, 80, 200, 30, 10).image[0, 0].tolist() == [10, 30, 200]
    assert C.color_slicing(red, "cube", 20, 80, 10, 30, 200).image[0, 0].tolist() == [128, 128, 128]
    assert C.color_balance(red, "curves", gamma_r=2.0).image[0, 0].tolist() == [10, 30, c8(255 * (200 / 255) ** 2)]


@pytest.mark.parametrize("call", [
    lambda im: C.color_transformation(im, "lab"), lambda im: C.color_transformation(im, "rgb", 0.05),
    lambda im: C.color_transformation(im, "rgb", 1.5), lambda im: C.color_transformation(im, "rgb", float("nan")),
    lambda im: C.color_slicing(im, "box"), lambda im: C.color_slicing(im, "cube", 9),
    lambda im: C.color_slicing(im, "cube", 110, 256), lambda im: C.color_slicing(im, "cube", 110.5),
    lambda im: C.color_slicing(im, proto_r=256), lambda im: C.color_slicing(im, proto_g=-1),
    lambda im: C.color_slicing(im, proto_b=True), lambda im: C.tone_correction(im, "log"),
    lambda im: C.tone_correction(im, "s", 0.5), lambda im: C.tone_correction(im, "gamma", 9.0, 3.5),
    lambda im: C.color_balance(im, "auto"), lambda im: C.color_balance(im, gain_r=4.5),
    lambda im: C.color_balance(im, gamma_b=0.3), lambda im: C.color_balance(im, neutral_r=300),
    lambda im: C.hsi_equalization(im, "hsv"), lambda im: C.hsi_equalization(im, "hsi", 2.0),
    lambda im: C.hsi_equalization(im, "hsi", "1"),
])
def test_bad_parameters_are_rejected(call):
    with pytest.raises(AlgorithmParameterError):
        call(scene(4, 4))


@pytest.mark.parametrize("key", FULL_KEYS)
def test_extreme_values_through_the_registry(key):
    algo = REGISTRY.get(key)
    lo = {p.key: getattr(p, "minimum", p.default) for p in algo.params}
    hi = {p.key: getattr(p, "maximum", p.default) for p in algo.params}
    for img in (np.zeros((3, 3, 3), np.uint8), np.full((3, 3, 3), 255, np.uint8), SPECIAL):
        for params in (lo, hi, {}):
            out = algo.execute(img, params).image
            assert out.dtype == np.uint8 and out.shape == img.shape
