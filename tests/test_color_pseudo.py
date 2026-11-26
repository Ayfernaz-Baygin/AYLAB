"""Color Processing / Paket 2: pseudo-color — intensity slicing ve color coding (slayt 43–50, laboratuvar).

Referanslar: laboratuvardaki duraklar ve kurallar (bağımsız döngüsel yazım), OpenCV yalnızca yakınlık için.
"""

import inspect
import re

import cv2
import numpy as np
import pytest

from processing import color_processing as C
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError, ChoiceParam

RAMP = np.arange(256, dtype=np.uint8).reshape(16, 16)


def ref_ramp(stops, t):
    """Laboratuvar ``ramp`` (döngüyle): ilk t ≤ t₁ aralığında doğrusal ara değer."""
    t = min(1.0, max(0.0, t))
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        if t <= t1:
            u = (t - t0) / ((t1 - t0) or 1)
            return [v + (w - v) * u for v, w in zip(c0, c1)]
    return list(stops[-1][1])


def ref_c8(v):
    return 0 if v <= 0 else 255 if v >= 255 else int(np.floor(v + 0.5)) if v - np.floor(v) != 0.5 else int(np.floor(v) + 1)


# ================================================================ katalog
def test_pseudo_color_operations_follow_source_order():
    keys = [a.key for a in REGISTRY.in_category("color")]
    assert keys[3:5] == ["intensity_slicing", "pseudocolor"]
    assert REGISTRY.get("intensity_slicing").title == "Intensity Slicing"
    assert REGISTRY.get("pseudocolor").title == "Color Coding" and REGISTRY.get("pseudocolor").implemented
    cm = next(p for p in REGISTRY.get("pseudocolor").params if p.key == "cmap")
    assert cm.values == C.COLORMAPS == ("jet", "hot", "parula", "viridis", "hsv", "bone", "gray")


@pytest.mark.parametrize("key,fn", [("intensity_slicing", C.intensity_slicing), ("pseudocolor", C.color_coding)])
def test_catalog_matches_signature(key, fn):
    algo = REGISTRY.get(key)
    sig = inspect.signature(fn)
    assert [p.key for p in algo.params] == list(sig.parameters)[1:]
    assert all(sig.parameters[p.key].default == p.default for p in algo.params)


def test_channel_choice_depends_on_the_color_model():
    spec = next(p for p in REGISTRY.get("channel").params if p.key == "channel")
    assert spec.depends_on == "model"
    for m in C.MODELS:
        opts = spec.options_for(m)
        assert [label.split(" · ")[1] for _, label in opts] == [s.name for s in C.CHANNELS[m]]
        assert [v for v, _ in opts] == [str(k + 1) for k in range(len(C.CHANNELS[m]))]
    assert spec.options_for("unknown") == spec.options
    with pytest.raises(ValueError):
        ChoiceParam("c", "C", (("1", "a"),), "1", option_sets=(("m", (("9", "x"),)),))


# ================================================================ LUT'lar
@pytest.mark.parametrize("name", C.COLORMAPS)
def test_colormap_lut_matches_the_lab_definition(name):
    lut = C.colormap_lut(name)
    assert lut.shape == (256, 3) and lut.dtype == np.uint8
    for v in range(256):
        t = v / 255
        if name == "gray":
            exp = [t * 255] * 3
        elif name == "hsv":
            exp = list(C.model_to_rgb(np.array([[[t * 300, 1.0, 1.0]]]), "hsv")[0, 0])
        else:
            exp = ref_ramp(C.COLORMAP_STOPS[name], t)
        assert lut[v].tolist() == [ref_c8(x) for x in exp], (name, v)


def test_colormap_endpoints_and_stops():
    assert C.colormap_lut("jet")[0].tolist() == [0, 0, 143] and C.colormap_lut("jet")[255].tolist() == [128, 0, 0]
    assert C.colormap_lut("hot")[0].tolist() == [10, 0, 0] and C.colormap_lut("hot")[255].tolist() == [255, 255, 255]
    assert C.colormap_lut("parula")[255].tolist() == [249, 251, 14]
    assert C.colormap_lut("gray")[:, 0].tolist() == list(range(256))
    hsv = C.colormap_lut("hsv")
    assert hsv[0].tolist() == [255, 0, 0] and hsv[255].tolist() == [255, 0, 255]   # ton 0° → 300° (kırmızı → macenta)
    hot = C.colormap_lut("hot").astype(int)
    assert (np.diff(hot, axis=0) >= 0).all()                                  # Hot: her kanal tekdüze artar


def test_colormaps_are_close_to_opencv_where_the_lab_approximates_them():
    g = np.arange(256, dtype=np.uint8).reshape(1, 256)
    for name, code, limit in (("jet", cv2.COLORMAP_JET, 16), ("bone", cv2.COLORMAP_BONE, 3), ("viridis", cv2.COLORMAP_VIRIDIS, 20)):
        ref = cv2.applyColorMap(g, code)[0, :, ::-1].astype(int)
        assert np.abs(C.colormap_lut(name).astype(int) - ref).max() <= limit


# ================================================================ color coding
def test_color_coding_is_a_lut_lookup_in_bgr():
    for name in C.COLORMAPS:
        r = C.color_coding(RAMP, name)
        lut = C.colormap_lut(name)
        assert r.image.shape == (16, 16, 3) and r.image.dtype == np.uint8
        np.testing.assert_array_equal(r.image, lut[RAMP][..., ::-1])          # RGB tablo → BGR çıktı
        np.testing.assert_array_equal(r.report.lut, lut)
        assert r.report.slices == ()
    px = C.color_coding(np.array([[0, 255]], np.uint8), "jet").image
    assert px[0, 0].tolist() == [143, 0, 0] and px[0, 1].tolist() == [0, 0, 128]   # BGR: koyu mavi, koyu kırmızı


def test_color_coding_constant_and_extreme_images():
    for v in (0, 1, 127, 128, 254, 255):
        out = C.color_coding(np.full((4, 5), v, np.uint8), "parula").image
        assert (out == C.colormap_lut("parula")[v][::-1]).all()
    r = C.color_coding(np.full((3, 3), 200, np.uint8))
    assert "f = 200…200" in r.report.summary()


def test_color_input_is_converted_with_lab_to_gray():
    rgb = np.random.default_rng(1).integers(0, 256, (9, 11, 3)).astype(np.uint8)
    f, alpha, was_color = C.gray_levels(rgb)
    r, g, b = (rgb[..., k].astype(float) for k in (2, 1, 0))
    np.testing.assert_array_equal(f, C.c8(0.299 * r + 0.587 * g + 0.114 * b))
    assert was_color and alpha is None
    out = C.color_coding(rgb, "hot")
    np.testing.assert_array_equal(out.image, C.color_coding(f, "hot").image)
    assert "color input → Y" in out.report.summary()
    assert "color input" not in C.color_coding(f, "hot").report.summary()


@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 1), (1, 1), (1, 9)])
def test_gray_layouts_and_alpha(shape):
    g = np.random.default_rng(2).integers(0, 256, shape).astype(np.uint8)
    for fn in (C.color_coding, C.intensity_slicing):
        out = fn(g).image
        assert out.shape == shape[:2] + (3,) and out.dtype == np.uint8         # pseudo-color çıktısı her zaman BGR
    bgra = np.dstack([np.full((4, 4, 3), 90), np.full((4, 4), 33)]).astype(np.uint8)
    before = bgra.copy()
    for fn in (C.color_coding, C.intensity_slicing):
        out = fn(bgra).image
        assert out.shape == (4, 4, 4) and (out[..., 3] == 33).all()
    np.testing.assert_array_equal(bgra, before)


# ================================================================ intensity slicing
def test_thresholds_follow_the_lab():
    assert C.slice_thresholds(2, 128) == (128,)
    assert C.slice_thresholds(2, 1) == (1,)
    assert C.slice_thresholds(4, 7) == (64, 128, 192)                          # l_i yalnızca 2 renkte
    assert C.slice_thresholds(8, 7) == (32, 64, 96, 128, 160, 192, 224)


@pytest.mark.parametrize("plane", [1, 2, 64, 127, 128, 129, 254, 255])
def test_one_plane_boundaries(plane):
    """f < l_i → c₁, f ≥ l_i → c₂ (eşitlik üst dilimde)."""
    f = np.array([[0, plane - 1, plane, min(255, plane + 1), 255]], np.uint8)
    out = C.intensity_slicing(f, "2", plane).image[0, :, ::-1].tolist()
    c1, c2 = list(C.SLICE_COLORS[2][0]), list(C.SLICE_COLORS[2][1])
    assert out[0] == c1 and out[1] == c1                                       # 0 ve l_i − 1 < l_i
    assert out[2] == c2 and out[3] == c2 and out[4] == c2                     # l_i, l_i + 1, 255 ≥ l_i


@pytest.mark.parametrize("n", [4, 8])
def test_every_interval_boundary_for_four_and_eight_colors(n):
    th = C.slice_thresholds(n, 128)
    cols = C.SLICE_COLORS[n]
    lut = C.intensity_slicing(RAMP, str(n)).report.lut
    edges = (0,) + th + (256,)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        assert lut[lo].tolist() == list(cols[k])                              # aralığın ilk değeri
        assert lut[hi - 1].tolist() == list(cols[k])                          # aralığın son değeri
    for t in th:
        assert lut[t - 1].tolist() != lut[t].tolist()                         # her düzlemde renk değişir
    assert len({tuple(c) for c in lut.tolist()}) == n


def test_slicing_matches_loop_reference_and_counts():
    f = np.random.default_rng(3).integers(0, 256, (31, 29)).astype(np.uint8)
    for n in (2, 4, 8):
        for plane in (1, 90, 255):
            r = C.intensity_slicing(f, str(n), plane)
            th = C.slice_thresholds(n, plane)
            cols = C.SLICE_COLORS[n]
            exp = np.zeros(f.shape + (3,), np.uint8)
            for y in range(f.shape[0]):
                for x in range(f.shape[1]):
                    k = 0
                    while k < len(th) and f[y, x] >= th[k]:
                        k += 1
                    exp[y, x] = cols[k][::-1]
            np.testing.assert_array_equal(r.image, exp)
            assert len(r.report.slices) == n
            assert [s[0] for s in r.report.slices] == [0, *th] and r.report.slices[-1][1] == 256
            counts = [int(v) for v in re.findall(r"\) (\d+) px", r.report.detail)]
            assert len(counts) == n and sum(counts) == f.size
            assert counts == np.bincount(C.slice_index(f, th).ravel(), minlength=n).tolist()


def test_constant_and_extreme_images():
    for v in (0, 255):
        out = C.intensity_slicing(np.full((3, 4), v, np.uint8), "8").image
        assert (out == np.array(C.SLICE_COLORS[8][0 if v == 0 else 7][::-1])).all()
    out = C.intensity_slicing(np.full((3, 4), 128, np.uint8), "2", 128).image
    assert (out == np.array(C.SLICE_COLORS[2][1][::-1])).all()                 # f = l_i → c₂


def test_palette_from_a_color_map():
    r = C.intensity_slicing(RAMP, "4", palette="jet")
    jet = C.colormap_lut("jet")
    mids = [31, 95, 159, 223]                                                  # (lo + hi − 1) // 2
    assert [list(s[2]) for s in r.report.slices] == [jet[m].tolist() for m in mids]
    assert "Jet colors" in r.report.summary()


def test_slicing_and_lab_palette_colors():
    assert C.SLICE_COLORS[2] == ((30, 80, 220), (250, 200, 30))
    assert C.SLICE_COLORS[4] == (C.PAL8[0], C.PAL8[3], C.PAL8[4], C.PAL8[6])
    assert len(C.SLICE_COLORS[8]) == 8


@pytest.mark.parametrize("kwargs,match", [
    (dict(colors="3"), "Colors"), (dict(colors=True), "Colors"), (dict(plane=0), "plane"), (dict(plane=256), "plane"),
    (dict(plane=12.5), "plane"), (dict(plane=True), "plane"), (dict(palette="rainbow"), "Palette")])
def test_slicing_rejects_bad_parameters(kwargs, match):
    with pytest.raises(AlgorithmParameterError, match=match):
        C.intensity_slicing(RAMP, **kwargs)


@pytest.mark.parametrize("bad", ["inferno", "Jet", None, 3])
def test_color_coding_rejects_unknown_maps(bad):
    with pytest.raises(AlgorithmParameterError, match="Color map"):
        C.color_coding(RAMP, bad)


# ================================================================ Color Histogram: yalnızca ölçüm
def test_color_histogram_does_not_change_the_image():
    r = C.color_histogram(np.zeros((3, 3, 3), np.uint8))
    assert r.changes_image is False
    assert C.color_coding(RAMP).changes_image is True and C.intensity_slicing(RAMP).changes_image is True
