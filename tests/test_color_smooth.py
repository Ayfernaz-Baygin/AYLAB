"""Color Processing / Paket 4: renkli görüntü yumuşatma ve keskinleştirme (slayt 57–58, Şekil 6.38–6.39, laboratuvar).

Referanslar laboratuvar filt / boxFn / lapFn / process kodunun piksel piksel, ``math`` ile yazılmış bağımsız
sürümleridir (kenar tekrarı, Float32Array saklaması np.float32 ile)."""

import inspect
import math

import numpy as np
import pytest

from processing import color_processing as C
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError
from tests.test_color_full import SPECIAL, bgr, c8, hsi2rgb, rgb2hsi, rgb_of, scene

LAPL = (0, -1, 0, -1, 4, -1, 0, -1, 0)
f32 = lambda v: float(np.float32(v))


# ================================================================ bağımsız referans (laboratuvar, döngü)
def ref_filt(G, k, fn):
    h, w = len(G), len(G[0])
    r = k // 2
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            win = [G[min(h - 1, max(0, y + dy))][min(w - 1, max(0, x + dx))]     # replicate border
                   for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
            out[y][x] = f32(fn(win))
    return out


def box(win):
    s = 0
    for v in win:
        s = s + v
    return s / len(win)


def lap(win):
    s = 0
    for c, v in zip(LAPL, win):
        s = s + c * v
    return s


def ref_process(rgb, op, k, method):
    h, w = rgb.shape[:2]
    kk = k if op == "smooth" else 3
    fn = box if op == "smooth" else lap
    out = np.zeros((h, w, 3), np.uint8)
    if method == "rgb":
        res = [ref_filt([[int(rgb[y, x, c]) for x in range(w)] for y in range(h)], kk, fn) for c in range(3)]
        for y in range(h):
            for x in range(w):
                out[y, x] = [c8(res[c][y][x] if op == "smooth" else int(rgb[y, x, c]) + res[c][y][x]) for c in range(3)]
        return out
    hsi = [[rgb2hsi(*(int(v) for v in rgb[y, x])) for x in range(w)] for y in range(h)]
    ig = [[f32(hsi[y][x][2] * 255) for x in range(w)] for y in range(h)]
    f = ref_filt(ig, kk, fn)
    for y in range(h):
        for x in range(w):
            H, S, _ = hsi[y][x]
            i2 = max(0, min(1, (f[y][x] if op == "smooth" else ig[y][x] + f[y][x]) / 255))
            out[y, x] = [c8(v * 255) for v in hsi2rgb(H, S, i2)]
    return out


def ref_diff(a, b):
    return np.array([[c8(8 * max(abs(int(a[y, x, c]) - int(b[y, x, c])) for c in range(3)))
                      for x in range(a.shape[1])] for y in range(a.shape[0])], np.uint8)


def patches(h=30, w=42):
    """Laboratuvar 'Colour patches' karşılığı (sentetik): gri zemin, kırmızı / yeşil, mavi daire, mor, koyu."""
    yy, xx = np.mgrid[0:h, 0:w]
    im = np.full((h, w, 3), 110, np.uint8)
    im[3:27, 3:23] = (230, 60, 70)
    im[3:27, 23:40] = (50, 152, 60)
    im[np.hypot(xx - 13, yy - 15) < 7] = (90, 95, 255)
    im[9:21, 30:38] = (190, 60, 190)
    im[13:17, 33:36] = (40, 40, 40)
    return im


# ================================================================ katalog
def test_catalog_entry_follows_the_slides():
    keys = [a.key for a in REGISTRY.in_category("color")]
    assert keys[10] == "color_smooth_sharpen" and keys[9] == "hsi_equalization" and len(keys) == 15   # + Paket 5
    algo = REGISTRY.get("color_smooth_sharpen")
    assert algo.title == "Color Smoothing & Sharpening" and algo.func is C.color_smoothing_sharpening
    sig = inspect.signature(C.color_smoothing_sharpening)
    assert list(sig.parameters)[1:] == [p.key for p in algo.params] == ["operation", "kernel", "process", "display"]
    assert algo.default_params() == {"operation": "smooth", "kernel": "3", "process": "rgb", "display": "result"}
    opts = {p.key: [v for v, _ in p.options] for p in algo.params}
    assert opts == {"operation": ["smooth", "sharpen"], "kernel": ["3", "5"], "process": ["rgb", "hsi"],
                    "display": ["result", "difference"]}


# ================================================================ bağımsız referansla birebir
IMAGES = {"scene": lambda: scene(14, 18), "patches": lambda: patches(14, 20), "special": lambda: SPECIAL,
          "row": lambda: scene(1, 9), "col": lambda: scene(9, 1)[:, :1], "one": lambda: SPECIAL[:, 3:4]}


@pytest.mark.parametrize("img", IMAGES)
@pytest.mark.parametrize("op,k", [("smooth", "3"), ("smooth", "5"), ("sharpen", "3")])
@pytest.mark.parametrize("method", C.SMOOTH_PROCESSES)
def test_matches_loop_reference(img, op, k, method):
    rgb = np.ascontiguousarray(IMAGES[img]())
    out = rgb_of(C.color_smoothing_sharpening(bgr(rgb), op, k, method))
    np.testing.assert_array_equal(out, ref_process(rgb, op, int(k), method))


@pytest.mark.parametrize("op,k", [("smooth", "3"), ("smooth", "5"), ("sharpen", "3")])
def test_difference_times_eight_matches_reference(op, k):
    rgb = patches()
    d = C.color_smoothing_sharpening(bgr(rgb), op, k, "rgb", "difference").image
    exp = ref_diff(ref_process(rgb, op, int(k), "rgb"), ref_process(rgb, op, int(k), "hsi"))
    assert (d[..., 0] == d[..., 1]).all() and (d[..., 1] == d[..., 2]).all()         # gri gösterim
    np.testing.assert_array_equal(d[..., 0], exp)
    # fark yalnızca renk kenarlarında: düz yama içi (kırmızı, pencere tamamen içeride) 0, kenarlarda > 0
    assert d[5, 5].max() == 0 and d.max() > 0


# ================================================================ bilinen sonuçlar
def test_smoothing_known_values_and_kernel_size():
    g = np.zeros((5, 5, 3), np.uint8)
    g[2, 2] = (90, 180, 45)                                                            # tek parlak piksel
    s3 = rgb_of(C.color_smoothing_sharpening(bgr(g), "smooth", "3", "rgb"))
    s5 = rgb_of(C.color_smoothing_sharpening(bgr(g), "smooth", "5", "rgb"))
    assert s3[2, 2].tolist() == [10, 20, 5] and s3[1, 1].tolist() == [10, 20, 5]     # 90/9, 180/9, 45/9
    assert s3[0, 0].tolist() == [0, 0, 0]                                             # 3×3 pencere dışında
    assert s5[0, 0].tolist() == [4, 7, 2]                                             # 90/25 = 3.6, 7.2, 1.8
    assert s5[2, 2].tolist() == [4, 7, 2]
    const = np.full((6, 7, 3), (40, 120, 200), np.uint8)                             # sabit görüntü değişmez
    for op in C.SMOOTH_OPERATIONS:
        for m in C.SMOOTH_PROCESSES:
            np.testing.assert_array_equal(rgb_of(C.color_smoothing_sharpening(bgr(const), op, "5", m)), const)


def test_replicate_border():
    row = np.array([[[0, 0, 0], [90, 90, 90], [180, 180, 180]]], np.uint8)          # 1×3
    s = rgb_of(C.color_smoothing_sharpening(bgr(row), "smooth", "3", "rgb"))[0, :, 0]
    # tekrar: sol pencere 3 satır × (0, 0, 90) → 30; sağ (90, 180, 180) → 150
    assert s.tolist() == [30, 90, 150]
    sh = rgb_of(C.color_smoothing_sharpening(bgr(row), "sharpen", "3", "rgb"))[0, :, 0]
    # f + (4f − sol − sağ − üst − alt); kenarda dışarısı = kendisi
    assert sh.tolist() == [c8(0 + 4 * 0 - 0 - 90 - 0 - 0), c8(90 + 360 - 0 - 180 - 90 - 90), c8(180 + 720 - 90 - 180 - 180 - 180)]


def test_sharpening_laplacian_sign_and_clipping():
    g = np.full((5, 5, 3), 100, np.uint8)
    g[2, 2] = (150, 120, 101)
    out = rgb_of(C.color_smoothing_sharpening(bgr(g), "sharpen", "3", "rgb")).astype(int)
    # merkez: f − ∇²f = f + 4(f − 100) → 150 + 200 = 350 → 255; 120 + 80 = 200; 101 + 4 = 105
    assert out[2, 2].tolist() == [255, 200, 105]
    # 4-komşu: f + (4·100 − 3·100 − f_merkez) = 100 − (f_merkez − 100)
    assert out[1, 2].tolist() == [50, 80, 99] and out[2, 1].tolist() == [50, 80, 99]
    assert out[1, 1].tolist() == [100, 100, 100]                                      # köşegen komşu (4-komşulu kernel)
    dark = np.zeros((3, 3, 3), np.uint8)
    dark[1, 1] = 200
    o = rgb_of(C.color_smoothing_sharpening(bgr(dark), "sharpen", "3", "rgb"))
    assert o[0, 1].tolist() == [0, 0, 0] and o[1, 1].tolist() == [255, 255, 255]     # negatif → 0, taşma → 255


def test_sharpening_ignores_the_smoothing_kernel():
    rgb = patches(14, 20)
    for m in C.SMOOTH_PROCESSES:
        a = C.color_smoothing_sharpening(bgr(rgb), "sharpen", "3", m)
        b = C.color_smoothing_sharpening(bgr(rgb), "sharpen", "5", m)
        np.testing.assert_array_equal(a.image, b.image)
        assert "3×3 Laplacian" in b.report.summary()


# ================================================================ HSI: H ve S korunur
@pytest.mark.parametrize("op,k", [("smooth", 3), ("smooth", 5), ("sharpen", 3)])
def test_hsi_keeps_hue_and_saturation(op, k):
    rgb = np.concatenate([scene(20, 24), patches(20, 24)], 0).astype(np.float64)
    _, info = C.smooth_sharpen_rgb(rgb, op, k, "hsi")
    hsi = C.rgb_to_model(rgb, "hsi")
    np.testing.assert_array_equal(info[..., 0], hsi[..., 0])                         # H aynen
    np.testing.assert_array_equal(info[..., 1], hsi[..., 1])                         # S aynen
    assert (info[..., 3] >= 0).all() and (info[..., 3] <= 1).all()                   # I′ ∈ [0, 1]
    assert not np.array_equal(info[..., 3], hsi[..., 2])
    # 8-bit çıktıda: gamut içindeki kromatik piksellerde ton yalnızca yuvarlama kadar kayar
    out = rgb_of(C.color_smoothing_sharpening(bgr(rgb.astype(np.uint8)), op, str(k), "hsi")).astype(np.float64)
    h1 = C.rgb_to_model(out, "hsi")
    raw = C.model_to_rgb(np.stack([hsi[..., 0], hsi[..., 1], info[..., 3]], -1), "hsi")
    ok = (raw >= 0).all(-1) & (raw <= 255).all(-1) & (hsi[..., 1] > 0.2) & (info[..., 3] > 0.15)
    dh = np.abs((h1[..., 0] - hsi[..., 0] + 540) % 360 - 180)[ok]
    assert ok.sum() > 200 and dh.max() < 3.0


def test_rgb_smoothing_can_create_new_hues_hsi_cannot():
    # kırmızı | yeşil kenarı: kanal başına ortalama ara ton (sarımsı) üretir; HSI yalnızca I'yı değiştirir
    edge = np.zeros((5, 6, 3), np.uint8)
    edge[:, :3] = (230, 60, 70)
    edge[:, 3:] = (50, 152, 60)
    hsi_in = C.rgb_to_model(edge.astype(float), "hsi")[..., 0]
    by_rgb = C.rgb_to_model(rgb_of(C.color_smoothing_sharpening(bgr(edge), "smooth", "3", "rgb")).astype(float), "hsi")[..., 0]
    by_hsi = C.rgb_to_model(rgb_of(C.color_smoothing_sharpening(bgr(edge), "smooth", "3", "hsi")).astype(float), "hsi")[..., 0]
    assert abs(by_rgb[2, 2] - hsi_in[2, 2]) > 20                                     # yeni ton
    assert np.abs((by_hsi - hsi_in + 540) % 360 - 180).max() < 2                     # ton korunur
    d = C.color_smoothing_sharpening(bgr(edge), "smooth", "3", "rgb", "difference").image[..., 0]
    assert d[:, [0, 5]].max() == 0 and d[:, 2:4].min() > 0                           # fark yalnızca kenarda


def test_gray_images_give_the_same_result_both_ways():
    g = scene(12, 16)[..., 1]
    for op, k in (("smooth", "3"), ("smooth", "5"), ("sharpen", "3")):
        a = C.color_smoothing_sharpening(g, op, k, "rgb").image
        b = C.color_smoothing_sharpening(g, op, k, "hsi").image
        assert a.shape == (12, 16, 3) and np.abs(a.astype(int) - b).max() <= 1      # S = 0: aynı (± yuvarlama)


# ================================================================ düzen, tip, alfa, rapor, sınırlar
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (6, 9)])
@pytest.mark.parametrize("params", [{}, {"kernel": "5"}, {"process": "hsi"}, {"operation": "sharpen"},
                                    {"operation": "sharpen", "process": "hsi", "display": "difference"}])
def test_layouts_sizes_alpha_and_bgr(shape, params):
    algo = REGISTRY.get("color_smooth_sharpen")
    rng = np.random.default_rng(shape[0] * 10 + shape[1])
    color = rng.integers(0, 256, shape + (3,), dtype=np.uint8)
    out = algo.execute(color, params)
    assert out.image.shape == shape + (3,) and out.image.dtype == np.uint8 and out.changes_image
    alpha = rng.integers(0, 256, shape + (1,), dtype=np.uint8)
    res = algo.execute(np.concatenate([color, alpha], -1), params).image
    np.testing.assert_array_equal(res[..., 3], alpha[..., 0])
    np.testing.assert_array_equal(res[..., :3], out.image)
    gray = color[..., 0]
    np.testing.assert_array_equal(algo.execute(gray, params).image, algo.execute(np.repeat(gray[..., None], 3, -1), params).image)
    # BGR sırası: kanalları ters çevirmek sonucu ters çevirir (RGB yöntemi kanal başına)
    if params.get("process", "rgb") == "rgb" and params.get("display", "result") == "result":
        np.testing.assert_array_equal(algo.execute(color[..., ::-1].copy(), params).image, out.image[..., ::-1])


def test_bgr_order_single_channel():
    im = np.zeros((3, 3, 3), np.uint8)
    im[1, 1] = (0, 0, 90)                                                              # BGR: yalnızca kırmızı
    out = C.color_smoothing_sharpening(im, "smooth", "3", "rgb").image
    assert out[0, 0].tolist() == [0, 0, 10]


def test_report_rows_and_example_pixel():
    r = C.color_smoothing_sharpening(bgr(patches(14, 20)), "smooth", "5", "hsi")
    assert r.report.section == "SMOOTHING & SHARPENING" and not r.report.curves
    assert r.report.rows[0] == "kernel (1/25)·ones(5×5), K = 25" and "replicate" in r.report.rows[1]
    assert r.report.rows[-1].strip().startswith("HSI: H") and "kept" in r.report.rows[-1]
    s = C.color_smoothing_sharpening(bgr(patches(14, 20)), "sharpen", "3", "rgb").report
    assert s.rows[0].startswith("kernel [0 −1 0; −1 4 −1; 0 −1 0]") and "f − ∇²f" in s.rows[0]
    assert "px differ" in s.summary()


@pytest.mark.parametrize("kw", [{"operation": "blur"}, {"kernel": "7"}, {"kernel": "3×3"}, {"process": "hsv"},
                                {"display": "board"}, {"operation": "sharpen", "kernel": "9"}])
def test_bad_parameters(kw):
    with pytest.raises(AlgorithmParameterError):
        C.color_smoothing_sharpening(scene(4, 4), **kw)


def test_extremes_and_determinism():
    for img in (np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 255, np.uint8), SPECIAL):
        for op in C.SMOOTH_OPERATIONS:
            for m in C.SMOOTH_PROCESSES:
                a = C.color_smoothing_sharpening(img, op, "5", m).image
                np.testing.assert_array_equal(a, C.color_smoothing_sharpening(img.copy(), op, "5", m).image)
    w = C.color_smoothing_sharpening(np.full((4, 4, 3), 255, np.uint8), "sharpen", "3", "hsi").image
    assert (w == 255).all()                                                           # I = 1 kırpılır, beyaz kalır
