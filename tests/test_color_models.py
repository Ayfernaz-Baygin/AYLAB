"""Color Processing / Paket 1: renk modelleri, dönüşümler, kanal görüntüleri, renk histogramları.

Kaynak: Chapter 6 slayt 24–42 ve Chapter 6 laboratuvarı. Bağımsız referanslar: OpenCV cvtColor (float ve
8-bit), Python colorsys, HSI için atan2 biçimindeki eşdeğer formül, düz Python döngüleri.
"""

import colorsys
import inspect
import itertools
import math

import cv2
import numpy as np
import pytest

from processing import color_processing as C
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError

COLOR_KEYS = ["color_space", "channel", "color_histogram"]
FUNCS = {"color_space": C.color_space_conversion, "channel": C.channel_visualization,
         "color_histogram": C.color_histogram}
NAMED = {  # RGB
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128), "red": (255, 0, 0),
    "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255)}


def model(rgb, m):
    return C.rgb_to_model(np.array(rgb, float).reshape(1, 1, 3), m)[0, 0]


def bgr_img(rgb_list):
    """RGB üçlüleri → 1×N BGR uint8 görüntü."""
    return np.array([[c[::-1] for c in rgb_list]], np.uint8)


def cube_sample(n=40000, seed=0):
    corners = np.array(list(itertools.product([0, 1, 127, 128, 254, 255], repeat=3)))
    rnd = np.random.default_rng(seed).integers(0, 256, (n, 3))
    return np.vstack([corners, rnd]).astype(float).reshape(-1, 1, 3)


def circ(d, period):
    d = np.abs(d) % period
    return np.minimum(d, period - d)


# ================================================================ katalog
def test_color_operations_follow_source_order_and_are_all_implemented():
    keys = [a.key for a in REGISTRY.in_category("color")]
    assert keys == ["color_space", "channel", "color_histogram", "intensity_slicing", "pseudocolor",   # + Paket 2
                    "color_transform", "color_slicing", "tone_correction", "color_balance", "hsi_equalization",   # + Paket 3
                    "color_smooth_sharpen",   # + Paket 4
                    "color_hue_threshold", "color_kmeans", "color_skin", "color_edges"]   # + Paket 5
    assert len(keys) == 15 and all(REGISTRY.get(k).implemented for k in keys)  # yer tutucu kalmadı (hsv_adjust kaldırıldı)
    assert "hsv_adjust" not in REGISTRY
    assert [REGISTRY.get(k).title for k in COLOR_KEYS] == ["Color Space Conversion", "Channel Visualization",
                                                           "Color Histogram"]


@pytest.mark.parametrize("key", COLOR_KEYS)
def test_catalog_parameters_match_signatures(key):
    algo = REGISTRY.get(key)
    sig = inspect.signature(FUNCS[key])
    assert [p.key for p in algo.params] == list(sig.parameters)[1:]
    for p in algo.params:
        assert sig.parameters[p.key].default == p.default
    model_param = next(p for p in algo.params if p.key == "model")
    assert model_param.values == C.MODELS                                     # slayt sırası


def test_channel_specs_cover_every_model():
    assert [s.name for s in C.CHANNELS["rgb"]] == ["R", "G", "B"]
    assert [s.name for s in C.CHANNELS["cmyk"]] == ["C", "M", "Y", "K"]
    assert [s.name for s in C.CHANNELS["hsi"]] == ["H", "S", "I"]
    assert [s.name for s in C.CHANNELS["lab"]] == ["L*", "a*", "b*"]
    assert [s.name for s in C.CHANNELS["ycbcr"]] == ["Y", "Cb", "Cr"]
    assert set(C.CHANNELS) == set(C.MODELS)


# ================================================================ bilinen renkler
EXPECT = {
    "cmy": {"black": (1, 1, 1), "white": (0, 0, 0), "red": (0, 1, 1), "cyan": (1, 0, 0), "yellow": (0, 0, 1)},
    "cmyk": {"black": (0, 0, 0, 1), "white": (0, 0, 0, 0), "red": (0, 1, 1, 0), "blue": (1, 1, 0, 0),
             "gray": (0, 0, 0, 127 / 255)},
    "hsv": {"black": (0, 0, 0), "white": (0, 0, 1), "gray": (0, 0, 128 / 255), "red": (0, 1, 1), "green": (120, 1, 1),
            "blue": (240, 1, 1), "yellow": (60, 1, 1), "cyan": (180, 1, 1), "magenta": (300, 1, 1)},
    "hsl": {"black": (0, 0, 0), "white": (0, 0, 1), "gray": (0, 0, 128 / 255), "red": (0, 1, 0.5),
            "green": (120, 1, 0.5), "blue": (240, 1, 0.5), "cyan": (180, 1, 0.5)},
    "hsi": {"black": (0, 0, 0), "white": (0, 0, 1), "gray": (0, 0, 128 / 255), "red": (0, 1, 1 / 3),
            "green": (120, 1, 1 / 3), "blue": (240, 1, 1 / 3), "yellow": (60, 1, 2 / 3), "cyan": (180, 1, 2 / 3),
            "magenta": (300, 1, 2 / 3)},
    "ycbcr": {"black": (0, 128, 128), "white": (255, 128, 128), "gray": (128, 128, 128),
              "red": (76.245, 128 - 76.245 * 0.564, 128 + (255 - 76.245) * 0.713)},
}


@pytest.mark.parametrize("m,name", [(m, n) for m in EXPECT for n in EXPECT[m]])
def test_known_colors(m, name):
    np.testing.assert_allclose(model(NAMED[name], m), EXPECT[m][name], atol=1e-9)


@pytest.mark.parametrize("name,lab", [("white", (100, 0, 0)), ("black", (0, 0, 0)), ("red", (53.2408, 80.0925, 67.2032)),
                                      ("green", (87.7347, -86.1827, 83.1793)), ("blue", (32.2970, 79.1875, -107.8602)),
                                      ("gray", (53.585, 0, 0))])
def test_cielab_reference_values(name, lab):
    np.testing.assert_allclose(model(NAMED[name], "lab"), lab, atol=2e-3)


def test_rgb_model_is_identity_and_gray_has_no_hue():
    np.testing.assert_array_equal(model((12, 34, 56), "rgb"), (12, 34, 56))
    for m in ("hsv", "hsl", "hsi"):
        for v in (0, 1, 77, 128, 254, 255):
            h, s, _ = model((v, v, v), m)
            assert h == 0 and s == 0                                          # tanımsız ton → 0°


# ================================================================ bağımsız referanslar
def test_hsv_and_hsl_match_opencv_float_and_colorsys():
    rgb = cube_sample()
    f = (rgb[..., ::-1] / 255).astype(np.float32)
    for m, code, perm in (("hsv", cv2.COLOR_BGR2HSV, [0, 1, 2]), ("hsl", cv2.COLOR_BGR2HLS, [0, 2, 1])):
        mine, ref = C.rgb_to_model(rgb, m), cv2.cvtColor(f, code)[..., perm].astype(float)
        assert circ(mine[..., 0] - ref[..., 0], 360).max() < 5e-3
        assert np.abs(mine[..., 1:] - ref[..., 1:]).max() < 5e-5
    for r, g, b in rgb[::97, 0].tolist():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        np.testing.assert_allclose(model((r, g, b), "hsv"), (h * 360, s, v), atol=1e-9)
        h, light, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        mine = model((r, g, b), "hsl")
        assert circ(mine[0] - h * 360, 360) < 1e-9 and abs(mine[1] - s) < 1e-9 and abs(mine[2] - light) < 1e-9


def test_hsv_8bit_encoding_matches_opencv():
    u = np.random.default_rng(3).integers(0, 256, (120, 140, 3)).astype(np.uint8)
    rgb, _ = C.rgb_input(u)
    hsv = C.rgb_to_model(rgb, "hsv")
    ref = cv2.cvtColor(u, cv2.COLOR_BGR2HSV).astype(int)
    assert circ(ref[..., 0] - C.js_round(hsv[..., 0] / 2), 180).max() <= 1      # OpenCV H = H/2 (0–179)
    assert np.abs(ref[..., 1] - C.c8(hsv[..., 1] * 255)).max() <= 1
    assert np.abs(ref[..., 2] - C.c8(hsv[..., 2] * 255)).max() == 0


def test_hsi_matches_the_equivalent_atan2_formula():
    """H = atan2(√3 (G − B), 2R − G − B) Gonzalez'in cos⁻¹ biçimine eşdeğerdir (bağımsız türetme)."""
    rgb = cube_sample(20000, 4)
    r, g, b = (rgb[..., i] / 255 for i in range(3))
    h = np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    mine = C.rgb_to_model(rgb, "hsi")
    chroma = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 0
    assert circ(mine[..., 0] - h, 360)[chroma].max() < 1e-6
    total = r + g + b
    np.testing.assert_allclose(mine[..., 2], total / 3, atol=1e-12)
    s_ref = np.where(total > 0, 1 - 3 * np.minimum(np.minimum(r, g), b) / np.where(total > 0, total, 1), 0)
    np.testing.assert_allclose(mine[..., 1], s_ref, atol=1e-12)


def test_cielab_matches_opencv_within_its_constant_differences():
    u = np.random.default_rng(5).integers(0, 256, (100, 100, 3)).astype(np.uint8)
    rgb, _ = C.rgb_input(u)
    mine = C.rgb_to_model(rgb, "lab")
    ref = cv2.cvtColor(u.astype(np.float32) / 255, cv2.COLOR_BGR2Lab).astype(float)
    assert np.abs(mine - ref).max() < 0.5                                     # D65 / gama sabitleri biraz farklı
    ref8 = cv2.cvtColor(u, cv2.COLOR_BGR2Lab).astype(int)
    assert np.abs(C.encode8(mine, "lab").astype(int) - ref8).max() <= 2          # 8-bit: L·255/100, a + 128, b + 128


def test_ycbcr_matches_opencv_8bit_with_cr_cb_order():
    u = np.random.default_rng(6).integers(0, 256, (100, 100, 3)).astype(np.uint8)
    rgb, _ = C.rgb_input(u)
    mine = C.encode8(C.rgb_to_model(rgb, "ycbcr"), "ycbcr").astype(int)
    ref = cv2.cvtColor(u, cv2.COLOR_BGR2YCrCb).astype(int)                     # OpenCV sırası: Y, Cr, Cb
    assert np.abs(mine - ref[..., [0, 2, 1]]).max() <= 1


def test_cmyk_matches_loop_formula():
    rgb = cube_sample(3000, 7)
    mine = C.rgb_to_model(rgb, "cmyk")
    for (r, g, b), v in zip(rgb[:, 0].tolist(), mine[:, 0]):
        c, m, y = 1 - r / 255, 1 - g / 255, 1 - b / 255
        k = min(c, m, y)
        exp = (0, 0, 0, 1) if k == 1 else ((c - k) / (1 - k), (m - k) / (1 - k), (y - k) / (1 - k), k)
        np.testing.assert_allclose(v, exp, atol=1e-12)


# ================================================================ aralıklar, yönler
def test_channel_ranges_over_the_rgb_cube():
    rgb = cube_sample(60000, 8)
    for m in ("hsv", "hsl", "hsi"):
        v = C.rgb_to_model(rgb, m)
        assert v[..., 0].min() >= 0 and v[..., 0].max() < 360
        assert v[..., 1:].min() >= -1e-12 and v[..., 1:].max() <= 1 + 1e-12
    lab = C.rgb_to_model(rgb, "lab")
    assert lab[..., 0].min() >= -1e-9 and lab[..., 0].max() <= 100 + 1e-5      # beyaz: 100.0000039 (matris sabitleri)
    assert lab[..., 1:].min() > -128 and lab[..., 1:].max() < 128
    for m in ("cmy", "cmyk"):
        v = C.rgb_to_model(rgb, m)
        assert v.min() >= -1e-12 and v.max() <= 1 + 1e-12
    y = C.rgb_to_model(rgb, "ycbcr")
    assert y.min() >= 0 and y.max() <= 255.5


@pytest.mark.parametrize("h,rgb", [(0, (1, 0, 0)), (120, (0, 1, 0)), (240, (0, 0, 1)), (60, (0.5, 0.5, 0)),
                                   (180, (0, 0.5, 0.5)), (300, (0.5, 0, 0.5)), (359.999, (1, 0, 0))])
def test_hsi_to_rgb_uses_the_three_sectors(h, rgb):
    out = C.model_to_rgb(np.array([[[h, 1.0, 1 / 3]]], float), "hsi")[0, 0] / 255     # S = 1, I = 1/3 → R + G + B = 1
    np.testing.assert_allclose(out, rgb, atol=1e-4)


def test_bgr_channel_order_is_handled():
    red_bgr = np.array([[[0, 0, 255]]], np.uint8)
    blue_bgr = np.array([[[255, 0, 0]]], np.uint8)
    rgb, _ = C.rgb_input(red_bgr)
    assert rgb[0, 0].tolist() == [255, 0, 0]
    assert C.rgb_to_model(C.rgb_input(blue_bgr)[0], "hsv")[0, 0, 0] == 240
    r = C.channel_visualization(red_bgr, "rgb", "1")
    assert r.image.tolist() == [[[255, 255, 255]]]                            # R kanalı = 255 (gri gösterim)
    assert C.channel_visualization(red_bgr, "rgb", "3").image.tolist() == [[[0, 0, 0]]]
    assert C.channel_visualization(red_bgr, "rgb", "1", "tint").image.tolist() == [[[0, 0, 255]]]   # BGR'de kırmızı
    out = C.color_space_conversion(bgr_img([(10, 20, 200)]), "hsi", "roundtrip").image
    assert out.tolist() == [[[200, 20, 10]]]                                  # çıktı yine BGR
    np.testing.assert_array_equal(C.to_bgr8(np.array([[[1, 2, 3]]], float)), [[[3, 2, 1]]])


# ================================================================ gri / renkli girdiler, düzen, yuvarlama
@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 1)])
def test_gray_input_is_r_equals_g_equals_b(shape):
    g = np.random.default_rng(1).integers(0, 256, shape).astype(np.uint8)
    rgb, alpha = C.rgb_input(g)
    assert alpha is None and rgb.shape == (5, 7, 3)
    assert (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()
    hsv = C.rgb_to_model(rgb, "hsv")
    assert (hsv[..., :2] == 0).all()
    vis = C.channel_visualization(g, "hsv", "3")
    assert vis.image.shape == g.shape and np.array_equal(vis.image.reshape(5, 7), g.reshape(5, 7))
    rt = C.color_space_conversion(g, "lab", "roundtrip").image
    assert rt.shape == (5, 7, 3) and (rt == g.reshape(5, 7, 1)).all()          # gidiş-dönüş BGR, hata yok
    mont = C.color_space_conversion(g, "hsv").image
    assert mont.shape[:2] == (5, 3 * 7 + 2 * 3) and mont.ndim == g.ndim


def test_alpha_is_preserved_and_input_not_modified():
    rng = np.random.default_rng(2)
    bgra = np.dstack([rng.integers(0, 256, (6, 8, 3)), np.full((6, 8), 99)]).astype(np.uint8)
    before = bgra.copy()
    assert (C.channel_visualization(bgra, "lab", "2").image[..., 3] == 99).all()
    assert (C.channel_visualization(bgra, "cmy", "1", "tint").image[..., 3] == 99).all()
    assert (C.color_space_conversion(bgra, "hsv", "roundtrip").image[..., 3] == 99).all()
    assert C.color_space_conversion(bgra, "cmyk").image.shape == (6, 4 * 8 + 3 * 3, 3)   # pano: alfa yok
    np.testing.assert_array_equal(C.color_histogram(bgra).image, bgra)
    np.testing.assert_array_equal(bgra, before)


def test_rounding_is_half_up_and_clipped():
    assert C.c8(np.array([127.5, 0.49999999999999994, -3, 255.6, 254.5])).tolist() == [128, 0, 0, 255, 255]
    assert C.c8(np.array([1.5])).dtype == np.uint8
    v = C.rgb_to_model(np.full((1, 1, 3), 128.0), "hsv")
    assert C.encode8(v, "hsv")[0, 0].tolist() == [0, 0, 128]


@pytest.mark.parametrize("m", C.MODELS)
def test_encode_decode_and_display_rules(m):
    rgb = cube_sample(5000, 9)
    v = C.rgb_to_model(rgb, m)
    codes = C.encode8(v, m)
    assert codes.dtype == np.uint8 and codes.shape == v.shape
    back = C.decode8(codes, m)
    for k, s in enumerate(C.CHANNELS[m]):
        step = {"value": 0.5, "ink": 0.5 / 255, "hue": 0.5 * 360 / 255, "unit": 0.5 / 255,
                "lightness": 0.5 * 100 / 255, "opponent": 0.5}[s.kind]
        assert np.abs(back[..., k] - v[..., k]).max() <= step + 1e-9 + (0.5 if m == "ycbcr" else 0)


def test_display_rules_for_special_channels():
    px = bgr_img([NAMED["red"], NAMED["blue"], NAMED["white"]])
    h = C.channel_visualization(px, "hsi", "1").image[..., 0][0].tolist()
    assert h == [0, 170, 0]                                                   # 240/360·255 = 170
    a = C.channel_visualization(px, "lab", "2").image[..., 0][0].tolist()
    assert a[2] == 128 and a[0] == int(C.c8(np.array(80.0925 + 128)))          # a* + 128; nötr = 128
    k = C.channel_visualization(bgr_img([NAMED["black"], NAMED["white"]]), "cmyk", "4").image[..., 0][0].tolist()
    assert k == [0, 255]                                                      # baskı: siyah mürekkep = koyu
    tint = C.channel_visualization(bgr_img([NAMED["red"]]), "cmy", "1", "tint").image[0, 0].tolist()
    assert tint == [255, 255, 255]                                            # kırmızıda C = 0: beyaz kâğıt
    tint = C.channel_visualization(bgr_img([NAMED["cyan"]]), "cmy", "1", "tint").image[0, 0].tolist()
    assert tint == [255, 255, 0]                                              # C = 1: camgöbeği (BGR)


# ================================================================ gidiş-dönüş
ROUNDTRIP8_MAX = {"rgb": 0, "cmy": 0, "cmyk": 0, "hsv": 3, "hsl": 4, "hsi": 4, "lab": 22, "ycbcr": 1}


@pytest.mark.parametrize("m", C.MODELS)
def test_float_roundtrip_is_exact_after_rounding(m):
    rgb = cube_sample(60000, 10)
    back = C.model_to_rgb(C.rgb_to_model(rgb, m), m)
    assert np.abs(back - rgb).max() < (0.08 if m == "ycbcr" else 1e-3)       # YCbCr: laboratuvarın 3 basamaklı ters katsayıları
    np.testing.assert_array_equal(C.c8(back), rgb.astype(np.uint8))


@pytest.mark.parametrize("m", C.MODELS)
def test_8bit_roundtrip_error_bounds(m):
    u = cube_sample(60000, 11)[..., ::-1].astype(np.uint8)
    r = C.color_space_conversion(u, m, "roundtrip8")
    err = np.abs(r.image.astype(int) - u.astype(int))
    assert err.max() <= ROUNDTRIP8_MAX[m]
    assert err.mean() < 0.7
    assert f"max |Δ| R {int(err[..., 2].max())}" in r.report.summary()
    exact = C.color_space_conversion(u, m, "roundtrip")
    np.testing.assert_array_equal(exact.image, u)
    assert "0 of" in exact.report.summary()


def test_montage_layout():
    u = np.random.default_rng(12).integers(0, 256, (9, 11, 3)).astype(np.uint8)
    r = C.color_space_conversion(u, "hsv")
    assert r.image.shape == (9, 3 * 11 + 6, 3)
    codes = C.encode8(C.rgb_to_model(C.rgb_input(u)[0], "hsv"), "hsv")
    for k in range(3):
        np.testing.assert_array_equal(r.image[:, k * 14:k * 14 + 11, 0], codes[..., k])
    assert (r.image[:, 11:14] == 255).all()
    assert "H: H/360°·255" in r.report.summary()


# ================================================================ histogramlar
@pytest.mark.parametrize("m", C.MODELS)
@pytest.mark.parametrize("bins", C.HIST_BINS)
def test_histogram_totals_equal_pixel_count(m, bins):
    u = np.random.default_rng(13).integers(0, 256, (37, 23, 3)).astype(np.uint8)
    rep = C.color_histogram(u, m, bins).report
    assert rep.pixels == 37 * 23 and rep.bins == int(bins)
    assert [c.name for c in rep.channels] == [s.name for s in C.CHANNELS[m]]
    for ch, s in zip(rep.channels, C.CHANNELS[m]):
        assert ch.total == 37 * 23 and ch.counts.size == int(bins) and ch.counts.dtype == np.int64
        assert (ch.lo, ch.hi) == (s.lo, s.hi) and ch.lo <= ch.minimum <= ch.mean <= ch.maximum <= ch.hi
        assert ch.edges().size == int(bins) + 1


def test_rgb_histogram_equals_calc_hist_and_lab_16_bins():
    u = np.random.default_rng(14).integers(0, 256, (50, 60, 3)).astype(np.uint8)
    rep = C.color_histogram(u).report
    for ch, k in zip(rep.channels, (2, 1, 0)):
        ref = cv2.calcHist([u], [k], None, [256], [0, 256]).ravel().astype(int)
        np.testing.assert_array_equal(ch.counts, ref)
    rep16 = C.color_histogram(u, bins="16").report
    np.testing.assert_array_equal(rep16.channels[0].counts, np.bincount(u[..., 2].ravel() >> 4, minlength=16))


def test_histogram_bins_follow_real_ranges():
    px = bgr_img([NAMED["red"], NAMED["green"], NAMED["blue"], NAMED["white"]])
    h = C.color_histogram(px, "hsv", "16").report.channels
    assert np.flatnonzero(h[0].counts).tolist() == [0, 5, 10]                 # 0°, 120°, 240° (22.5° kutular)
    assert h[0].counts[0] == 2                                                # kırmızı + beyaz (H = 0)
    assert h[1].counts[-1] == 3 and h[1].counts[0] == 1                       # S = 1 son kutuda
    lab = C.color_histogram(px, "lab", "16").report.channels
    assert lab[0].counts[-1] == 1 and lab[1].lo == -128                       # L* = 100 son kutu; a* aralığı negatif
    y = C.color_histogram(np.full((3, 3), 100, np.uint8), "ycbcr").report.channels
    assert y[0].counts[100] == 9 and y[1].counts[128] == 9                    # 99.99999999999999 → 100 (1e-9 yuvarlama)


def test_hue_mask_follows_the_lab_rule():
    px = bgr_img([NAMED["red"], NAMED["gray"], (20, 5, 5), (200, 190, 190), NAMED["cyan"]])
    rep = C.color_histogram(px, "hsv", "16", True).report
    assert rep.channels[0].total == 2 and rep.channels[1].total == 5          # yalnızca H maskeli
    assert "2 chromatic px" in rep.summary()
    with pytest.raises(AlgorithmParameterError, match="hue mask"):
        C.color_histogram(px, "rgb", "16", True)


def test_histogram_of_gray_image():
    g = np.random.default_rng(15).integers(0, 256, (20, 30)).astype(np.uint8)
    rep = C.color_histogram(g).report
    for ch in rep.channels:
        np.testing.assert_array_equal(ch.counts, np.bincount(g.ravel(), minlength=256))
    hsv = C.color_histogram(g, "hsv").report
    assert hsv.channels[0].counts[0] == 600 and hsv.channels[1].counts[0] == 600


# ================================================================ parametre doğrulama
@pytest.mark.parametrize("fn,kwargs,match", [
    (C.color_space_conversion, dict(model="xyz"), "Color model"),
    (C.color_space_conversion, dict(model="HSV"), "Color model"),
    (C.color_space_conversion, dict(output="pack"), "Output"),
    (C.channel_visualization, dict(channel="0"), "Channel"),
    (C.channel_visualization, dict(channel=True), "Channel"),
    (C.channel_visualization, dict(model="hsv", channel="4"), "has 3 channels"),
    (C.channel_visualization, dict(model="hsv", style="tint"), "Tinted"),
    (C.channel_visualization, dict(style="color"), "Style"),
    (C.color_histogram, dict(bins="100"), "Bins"),
    (C.color_histogram, dict(bins=True), "Bins"),
    (C.color_histogram, dict(hue_mask="yes"), "Hue mask"),
    (C.color_histogram, dict(model=None), "Color model")])
def test_bad_parameters_are_rejected(fn, kwargs, match):
    with pytest.raises(AlgorithmParameterError, match=match):
        fn(np.zeros((4, 4, 3), np.uint8), **kwargs)


def test_integer_channel_and_bins_are_equivalent_to_their_text():
    u = np.random.default_rng(17).integers(0, 256, (5, 6, 3)).astype(np.uint8)
    np.testing.assert_array_equal(C.channel_visualization(u, "hsv", 2).image, C.channel_visualization(u, "hsv", "2").image)
    assert C.color_histogram(u, bins=64).report.bins == 64


def test_bad_images_are_rejected():
    for bad in (np.zeros((4, 4), np.float32), np.zeros((4, 4, 2), np.uint8), np.zeros((0, 3), np.uint8)):
        with pytest.raises(AlgorithmParameterError):
            C.color_space_conversion(bad)


@pytest.mark.parametrize("m", C.MODELS)
@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 9, 3), (9, 1, 3), (1, 1)])
def test_tiny_images(m, shape):
    u = np.random.default_rng(16).integers(0, 256, shape).astype(np.uint8)
    for out in ("channels", "roundtrip", "roundtrip8"):
        assert C.color_space_conversion(u, m, out).image.dtype == np.uint8
    for k in range(len(C.CHANNELS[m])):
        assert C.channel_visualization(u, m, str(k + 1)).image.shape == u.shape
    assert C.color_histogram(u, m).report.channels[0].total == math.prod(shape[:2])
