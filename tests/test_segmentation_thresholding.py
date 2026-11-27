"""Segmentation / Package 1 — Thresholding (Chapter 5 slayt 7–17, laboratuvar).

Referanslar işlem modülünden BAĞIMSIZDIR: piksel piksel döngüler, laboratuvarın JS
kodundaki tanımların (``refl``, ``localMean``, ``gaussK1``, ``movingAvg``, ``iterT``)
doğrudan karşılıkları ve Otsu için farklı bir formül (ω₀ω₁(μ₀ − μ₁)²) + OpenCV.
"""

import math

import cv2
import numpy as np
import pytest

from processing import segmentation as S
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError

KEYS = ["threshold", "iterative_threshold", "otsu", "adaptive_mean", "adaptive_gaussian",
        "moving_average", "niblack", "sauvola"]
FUNCS = {
    "threshold": S.manual_threshold, "iterative_threshold": S.automatic_threshold, "otsu": S.otsu_threshold,
    "adaptive_mean": S.adaptive_mean_threshold, "adaptive_gaussian": S.adaptive_gaussian_threshold,
    "moving_average": S.moving_average_threshold, "niblack": S.niblack_threshold, "sauvola": S.sauvola_threshold,
}


# ================================================================ bağımsız referanslar
def ref_luma(img):
    if img.ndim == 2:
        return img.astype(np.int64)
    if img.shape[2] == 1:
        return img[..., 0].astype(np.int64)
    b, g, r = (img[..., i].astype(np.float64) for i in range(3))
    return np.clip(np.floor(0.299 * r + 0.587 * g + 0.114 * b + 0.5), 0, 255).astype(np.int64)


def refl(i, n):
    """Laboratuvar: function refl(i,n){if(n===1)return 0;while(i<0||i>=n){if(i<0)i=-i;if(i>=n)i=2*n-2-i;}return i;}"""
    if n == 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * n - 2 - i
    return i


def window_values(f, y, x, b):
    h, w = f.shape
    r = b // 2
    return [[int(f[refl(y + t, h), refl(x + u, w)]) for u in range(-r, r + 1)] for t in range(-r, r + 1)]


def ref_local_mean(f, b):
    out = np.zeros(f.shape)
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            out[y, x] = math.fsum(sum(window_values(f, y, x, b), [])) / (b * b)
    return out


def ref_local_std(f, b):
    out = np.zeros(f.shape)
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            v = np.array(window_values(f, y, x, b), float).ravel()
            out[y, x] = float(np.sqrt(np.mean((v - v.mean()) ** 2)))     # iki geçişli, nüfus std
    return out


def ref_gauss_mean(f, b):
    r = b // 2
    sigma = 0.3 * ((b - 1) * 0.5 - 1) + 0.8
    g1 = [math.exp(-i * i / (2 * sigma * sigma)) for i in range(-r, r + 1)]
    s = sum(g1)
    g1 = [v / s for v in g1]
    out = np.zeros(f.shape)
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            v = window_values(f, y, x, b)
            out[y, x] = math.fsum(g1[t] * g1[u] * v[t][u] for t in range(b) for u in range(b))
    return out


def ref_moving_mean(f, n):
    """Laboratuvar movingAvg: zikzak, halka tampon, sum/cnt."""
    h, w = f.shape
    m = np.zeros(f.shape)
    buf, total = [], 0
    for y in range(h):
        for j in range(w):
            x = j if y % 2 == 0 else w - 1 - j
            z = int(f[y, x])
            buf.append(z)
            total += z
            if len(buf) > n:
                total -= buf.pop(0)
            m[y, x] = total / len(buf)
    return m


def ref_otsu(f):
    """σ²_B(k) = ω₀ω₁(μ₀ − μ₁)² doğrudan piksel kümelerinden; ilk en büyük k."""
    v = f.ravel().astype(float)
    best, t = -1.0, 0
    for k in range(256):
        c0, c1 = v[v <= k], v[v > k]
        if c0.size == 0 or c1.size == 0:
            sb = 0.0
        else:
            w0, w1 = c0.size / v.size, c1.size / v.size
            sb = w0 * w1 * (c0.mean() - c1.mean()) ** 2
        if sb > best + 1e-9:
            best, t = sb, k
    return t


def ref_iterative(f, t0, eps):
    v = f.ravel().astype(float)
    t = t0
    for i in range(100):
        a, b = v[v > t], v[v <= t]
        m1 = a.mean() if a.size else t
        m2 = b.mean() if b.size else t
        tn = (m1 + m2) / 2
        if abs(tn - t) < eps:
            return tn, i + 1
        t = tn
    return tn, 100


def u8(mask):
    return np.where(mask, 255, 0).astype(np.uint8)


def doc_scene(h=60, w=80, seed=1):
    """Eğik aydınlatmalı 'belge': koyu yazı benzeri çizgiler + gürültü (sentetik)."""
    rng = np.random.default_rng(seed)
    base = np.add.outer(np.linspace(90, 220, h), np.linspace(-30, 30, w))
    for r in range(8, h - 6, 12):
        base[r:r + 3, 6:w - 6:5] -= 70
    return np.clip(np.rint(base + rng.normal(0, 6, (h, w))), 0, 255).astype(np.uint8)


def rand_img(shape, seed):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


# ================================================================ katalog / kayıt
def test_segmentation_category_lists_thresholding_in_source_order():
    keys = [a.key for a in REGISTRY.in_category("segmentation")]
    assert keys[:8] == KEYS
    assert len(keys) == 25 and all(REGISTRY.get(k).implemented for k in keys)   # kmeans yer tutucusu kaldırıldı (Chapter 6 k-means: Color)
    assert all(k.startswith("edge_") for k in keys[8:21])                       # Paket 2: kenar tabanlı işlemler
    assert all(k.startswith("region_") for k in keys[21:])                      # Paket 3: bölge tabanlı işlemler
    assert all(REGISTRY.get(k).implemented for k in KEYS)
    assert [REGISTRY.get(k).title for k in KEYS] == [
        "Manual / Fixed Threshold", "Automatic Global Threshold", "Otsu's Method", "Adaptive Mean Threshold",
        "Adaptive Gaussian Threshold", "Moving Averages Threshold", "Niblack Threshold", "Sauvola Threshold"]


@pytest.mark.parametrize("key", KEYS)
def test_gui_parameters_match_function_signature(key):
    import inspect
    algo = REGISTRY.get(key)
    sig = inspect.signature(FUNCS[key])
    names = list(sig.parameters)[1:]
    assert [p.key for p in algo.params] == names
    assert all(sig.parameters[p.key].default == p.default for p in algo.params)
    assert algo.params[-1].key == "foreground"


def test_parameter_ranges_follow_lab_controls():
    get = lambda k, p: next(x for x in REGISTRY.get(k).params if x.key == p)
    for key, p in (("adaptive_mean", "block"), ("adaptive_gaussian", "block"), ("niblack", "window"), ("sauvola", "window")):
        spec = get(key, p)
        assert (spec.minimum, spec.maximum, spec.default, spec.odd_only) == (3, 51, 15, True)
    assert (get("adaptive_mean", "c").minimum, get("adaptive_mean", "c").maximum, get("adaptive_mean", "c").default) == (-10, 30, 7)
    assert (get("niblack", "k").minimum, get("niblack", "k").maximum, get("niblack", "k").default) == (0.0, 0.6, 0.2)
    assert (get("sauvola", "r").minimum, get("sauvola", "r").maximum, get("sauvola", "r").default) == (32, 255, 128)
    assert (get("moving_average", "n").default, get("moving_average", "c").default) == (20, 0.5)
    assert (get("iterative_threshold", "delta").default, get("threshold", "thresh").default) == (0.5, 127)


# ================================================================ ortak sözleşme
ALL_DEFAULT = [(k, {}) for k in KEYS]


@pytest.mark.parametrize("key,kw", ALL_DEFAULT, ids=KEYS)
@pytest.mark.parametrize("shape", [(37, 53), (37, 53, 1), (37, 53, 3), (37, 53, 4)], ids=["gray", "gray1", "bgr", "bgra"])
def test_output_is_binary_uint8_in_input_layout(key, kw, shape):
    img = rand_img(shape, 5)
    before = img.copy()
    res = FUNCS[key](img, **kw)
    out = res.image
    assert out.dtype == np.uint8 and out.shape == img.shape
    np.testing.assert_array_equal(img, before)                         # girdi değişmedi
    color = out[..., :3] if out.ndim == 3 and out.shape[2] == 4 else out
    assert set(np.unique(color)) <= {0, 255}
    if out.ndim == 3 and out.shape[2] >= 3:
        assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()
    if shape[-1] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])        # alfa aynen
    assert res.report.total_px == 37 * 53 and res.report.foreground_px == int((color.reshape(37, 53, -1)[..., 0] > 0).sum())


@pytest.mark.parametrize("key", KEYS)
def test_color_input_uses_rounded_bt601_luma(key):
    img = rand_img((24, 31, 3), 9)
    y = ref_luma(img).astype(np.uint8)
    np.testing.assert_array_equal(FUNCS[key](img).image[..., 0], FUNCS[key](y).image)


@pytest.mark.parametrize("key", KEYS)
def test_dark_foreground_is_exact_complement(key):
    img = doc_scene()
    bright = FUNCS[key](img, foreground="bright")
    dark = FUNCS[key](img, foreground="dark")
    np.testing.assert_array_equal(dark.image, 255 - bright.image)
    assert bright.report.foreground_px + dark.report.foreground_px == img.size
    assert "f > T" in bright.report.summary() and "f ≤ T" in dark.report.summary()


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (2, 2), (3, 70), (70, 3), (5, 5)])
def test_small_and_thin_images(key, shape):
    img = rand_img(shape, sum(shape))
    out = FUNCS[key](img).image
    assert out.shape == shape and set(np.unique(out)) <= {0, 255}


@pytest.mark.parametrize("key", KEYS)
def test_rejects_invalid_images(key):
    for bad in (np.zeros((4, 4), np.float32), np.zeros((0, 4), np.uint8), np.zeros((4, 4, 2), np.uint8), [[1, 2]]):
        with pytest.raises(AlgorithmParameterError):
            FUNCS[key](bad)


@pytest.mark.parametrize("key", KEYS)
def test_rejects_bad_foreground(key):
    with pytest.raises(AlgorithmParameterError, match="Foreground"):
        FUNCS[key](rand_img((5, 5), 1), foreground="invert")


# ================================================================ sabit (tekdüze) görüntüler
CONSTANT_EXPECTED = {
    # değer v için BRIGHT ön planda beklenen sonuç (True = 255)
    "threshold": lambda v: v > 127,
    "iterative_threshold": lambda v: False,         # T₀ = v, μ₁ boş → T = v, f > v yanlış
    "otsu": lambda v: v > 0,                         # σ²_B ≡ 0 → T* = 0
    "adaptive_mean": lambda v: True,                 # T = v − 7 < v
    "adaptive_gaussian": lambda v: True,
    "moving_average": lambda v: v > 0,               # f > 0.5 · v
    "niblack": lambda v: False,                      # s = 0 → T = m = v
    "sauvola": lambda v: v > 0,                      # T = 0.8 · v
}


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("value", [0, 1, 127, 128, 200, 255])
@pytest.mark.parametrize("shape", [(1, 1), (6, 9), (40, 33)])
def test_constant_images_are_decided_exactly(key, value, shape):
    img = np.full(shape, value, np.uint8)
    res = FUNCS[key](img)
    expected = CONSTANT_EXPECTED[key](value)
    assert (res.image == (255 if expected else 0)).all(), res.report.summary()
    dark = FUNCS[key](img, foreground="dark").image
    assert (dark == (0 if expected else 255)).all()


def test_constant_local_threshold_surface_is_exact():
    f = np.full((11, 13), 77, np.uint8)
    for b in (3, 15, 51):
        m, s = S.local_mean_std(f, b)
        assert (m == 77).all() and (s == 0).all()
        assert (np.round(S.local_gaussian_mean(f, b), 9) == 77).all()
    assert (S.moving_average_surface(f, 20) == 77).all()


# ================================================================ 1) manual / fixed
def test_manual_known_matrix():
    f = np.array([[0, 126, 127], [128, 200, 255]], np.uint8)
    np.testing.assert_array_equal(S.manual_threshold(f).image, [[0, 0, 0], [255, 255, 255]])
    np.testing.assert_array_equal(S.manual_threshold(f, thresh=0).image, [[0, 255, 255], [255, 255, 255]])
    np.testing.assert_array_equal(S.manual_threshold(f, thresh=255).image, np.zeros((2, 3)))
    np.testing.assert_array_equal(S.manual_threshold(f, 126, "dark").image, [[255, 255, 0], [0, 0, 0]])
    assert S.manual_threshold(f).report.summary().startswith("Manual threshold: T = 127; foreground (f > T) 3 px = 50.0 %")


@pytest.mark.parametrize("t", [0, 1, 64, 127, 128, 254, 255])
def test_manual_matches_opencv_threshold(t):
    img = rand_img((50, 60), t)
    _, ref = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)            # laboratuvardaki kod
    np.testing.assert_array_equal(S.manual_threshold(img, t).image, ref)
    _, inv = cv2.threshold(img, t, 255, cv2.THRESH_BINARY_INV)
    np.testing.assert_array_equal(S.manual_threshold(img, t, "dark").image, inv)


@pytest.mark.parametrize("bad", [-1, 256, 12.5, True, "10", None])
def test_manual_rejects_bad_threshold(bad):
    with pytest.raises(AlgorithmParameterError, match="Threshold"):
        S.manual_threshold(rand_img((4, 4), 0), thresh=bad)


def test_manual_accepts_integral_float():
    img = rand_img((8, 8), 3)
    np.testing.assert_array_equal(S.manual_threshold(img, 100.0).image, S.manual_threshold(img, 100).image)


# ================================================================ 2) automatic global (iterative)
def test_iterative_known_two_level_image():
    f = np.array([[10, 10, 10, 200, 200, 200]], np.uint8)
    r = S.automatic_threshold(f)                                        # T₀ = ortalama 105
    np.testing.assert_array_equal(r.image, [[0, 0, 0, 255, 255, 255]])
    assert "T = 105.00" in r.report.summary() and "1 iteration," in r.report.summary()
    its = S.iterative_threshold(S.histogram(f), 40.0, 0.5)              # T₀ = 40 → 105 → 105
    assert [(i.t, i.mu1, i.mu2, i.t_next) for i in its] == [(40.0, 200.0, 10.0, 105.0), (105.0, 200.0, 10.0, 105.0)]
    assert "2 iterations" in S.automatic_threshold(f, init="custom", t0=40).report.summary()


def test_iterative_empty_class_uses_current_t():
    its = S.iterative_threshold(S.histogram(np.full((3, 3), 50, np.uint8)), 200.0, 0.5)
    assert its[0].mu1 == 200.0 and its[0].mu2 == 50.0 and its[0].t_next == 125.0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("init,t0,eps", [("mean", 127, 0.5), ("custom", 40, 0.5), ("custom", 230, 0.1), ("mean", 127, 5.0)])
def test_iterative_matches_pixel_reference(seed, init, t0, eps):
    img = doc_scene(seed=seed) if seed % 2 else rand_img((40, 50), seed)
    start = float(img.mean()) if init == "mean" else float(t0)
    t_ref, n_ref = ref_iterative(img, start, eps)
    r = S.automatic_threshold(img, init=init, t0=t0, delta=eps)
    np.testing.assert_array_equal(r.image, u8(img > t_ref))
    assert f"T = {t_ref:.2f}" in r.report.summary() and f"{n_ref} iteration" in r.report.summary()


def test_iterative_converges_between_modes():
    rng = np.random.default_rng(2)
    f = np.clip(np.concatenate([rng.normal(60, 10, 3000), rng.normal(190, 12, 2000)]), 0, 255).astype(np.uint8).reshape(50, 100)
    its = S.iterative_threshold(S.histogram(f), float(f.mean()), 0.5)
    assert 100 < its[-1].t_next < 150 and len(its) < 10
    assert abs(its[-1].t_next - its[-1].t) < 0.5


@pytest.mark.parametrize("kw,msg", [({"delta": 0.0}, "ΔT"), ({"delta": 5.5}, "ΔT"), ({"delta": float("nan")}, "ΔT"),
                                    ({"t0": 0}, "T₀"), ({"t0": 255}, "T₀"), ({"init": "median"}, "Initial")])
def test_iterative_rejects_bad_parameters(kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        S.automatic_threshold(rand_img((4, 4), 0), **kw)


# ================================================================ 3) Otsu
def test_otsu_known_matrix_first_maximum():
    f = np.array([[10, 10, 200, 200]], np.uint8)
    r = S.otsu_threshold(f)
    assert "T* = 10" in r.report.summary() and "η = 1.000" in r.report.summary()
    np.testing.assert_array_equal(r.image, [[0, 0, 255, 255]])


@pytest.mark.parametrize("seed", range(12))
def test_otsu_matches_independent_formula_and_opencv(seed):
    img = doc_scene(seed=seed) if seed % 3 == 0 else rand_img((30, 41), seed) // (1 + seed % 4) * (1 + seed % 4)
    t = ref_otsu(img)
    t_cv, ref_cv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # laboratuvardaki kod
    assert t == int(t_cv)
    r = S.otsu_threshold(img)
    assert f"T* = {t};" in r.report.summary()
    np.testing.assert_array_equal(r.image, ref_cv)


def test_otsu_separability_eta():
    f = np.array([[0, 0, 100, 100, 100, 250]], np.uint8)
    t, eta, sb = S.otsu_analysis(S.histogram(f))
    v = f.ravel().astype(float)
    assert eta == pytest.approx(sb[t] / v.var()) and 0 < eta <= 1 and sb.argmax() == t


def test_otsu_uniform_reports_zero_variance():
    r = S.otsu_threshold(np.full((4, 4), 90, np.uint8))
    assert "T* = 0" in r.report.summary() and "uniform image" in r.report.summary()


# ================================================================ 4–5) adaptive mean / Gaussian
def test_adaptive_mean_known_matrix_uses_reflect101_border():
    f = np.zeros((3, 3), np.uint8)
    f[1, 1] = 90
    m, _ = S.local_mean_std(f, 3)
    np.testing.assert_allclose(m, [[40, 20, 40], [20, 10, 20], [40, 20, 40]])   # replicate kenar köşede 10 verirdi
    np.testing.assert_array_equal(S.adaptive_mean_threshold(f, 3, 0).image, [[0, 0, 0], [0, 255, 0], [0, 0, 0]])
    np.testing.assert_array_equal(S.adaptive_mean_threshold(f, 3, 21).image, [[0, 255, 0], [255, 255, 255], [0, 255, 0]])


def test_reflect_index_matches_lab_refl():
    for n in (1, 2, 3, 7):
        idx = np.arange(-60, 60)
        assert list(S.reflect_index(idx, n)) == [refl(int(i), n) for i in idx]


@pytest.mark.parametrize("b", [3, 5, 15, 51])
@pytest.mark.parametrize("c", [-10, 0, 7, 30])
def test_adaptive_mean_matches_pixel_reference(b, c):
    img = doc_scene(23, 29, seed=b + c + 10)
    ref = ref_local_mean(img, b)
    np.testing.assert_allclose(S.local_mean_std(img, b)[0], ref, atol=1e-9)
    np.testing.assert_array_equal(S.adaptive_mean_threshold(img, b, c).image, u8(img > np.round(ref - c, 9)))


@pytest.mark.parametrize("b", [3, 5, 7, 15, 31])
@pytest.mark.parametrize("c", [-5, 0, 7])
def test_adaptive_gaussian_matches_pixel_reference(b, c):
    img = doc_scene(21, 26, seed=3 * b + c + 10)
    ref = ref_gauss_mean(img, b)
    np.testing.assert_allclose(S.local_gaussian_mean(img, b), ref, atol=1e-9)
    np.testing.assert_array_equal(S.adaptive_gaussian_threshold(img, b, c).image, u8(img > np.round(ref - c, 9)))


def test_gaussian_kernel_follows_opencv_sigma_rule():
    for b in (3, 5, 15, 51):
        k = S.gaussian_kernel_1d(b)
        assert k.sum() == pytest.approx(1) and np.allclose(k, k[::-1]) and k.argmax() == b // 2
        assert S.gaussian_sigma(b) == pytest.approx(0.3 * ((b - 1) * 0.5 - 1) + 0.8)
        np.testing.assert_allclose(k, cv2.getGaussianKernel(b, S.gaussian_sigma(b)).ravel(), atol=1e-12)


def test_adaptive_close_to_opencv_inside_image():
    """cv2.adaptiveThreshold (laboratuvardaki örnek kod) iç bölgede aynı kararı verir; fark yalnızca
    kenar kuralında (replicate) ve 8-bit yuvarlamada (|f − T| < 1) olabilir."""
    img = doc_scene(80, 100, seed=4)
    for method, fn in ((cv2.ADAPTIVE_THRESH_MEAN_C, S.adaptive_mean_threshold),):
        cv = cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY, 15, 7)
        ours = fn(img, 15, 7).image
        inner = (slice(7, -7), slice(7, -7))
        m = S.local_mean_std(img, 15)[0]
        safe = np.abs(img.astype(float) - (m - 7)) >= 1
        np.testing.assert_array_equal(ours[inner][safe[inner]], cv[inner][safe[inner]])


def test_adaptive_follows_uneven_lighting_where_global_fails():
    img = doc_scene(60, 80, seed=0)
    text = np.zeros(img.shape, bool)
    for r in range(8, 54, 12):
        text[r:r + 3, 6:74:5] = True
    glob = S.otsu_threshold(img, foreground="dark").image > 0
    loc = S.adaptive_mean_threshold(img, 15, 15, foreground="dark").image > 0      # C ≈ 2.5σ (gürültü σ = 6)
    err = lambda m: (m != text).mean()
    assert err(loc) < 0.05 < err(glob)


@pytest.mark.parametrize("fn", [S.adaptive_mean_threshold, S.adaptive_gaussian_threshold])
@pytest.mark.parametrize("kw,msg", [({"block": 4}, "odd"), ({"block": 1}, "Block"), ({"block": 53}, "Block"),
                                    ({"block": 15.5}, "integer"), ({"c": -11}, "C"), ({"c": 31}, "C"),
                                    ({"c": 2.5}, "integer"), ({"block": True}, "integer")])
def test_adaptive_rejects_bad_parameters(fn, kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        fn(rand_img((9, 9), 0), **kw)


@pytest.mark.parametrize("fn,param", [(S.adaptive_mean_threshold, "block"), (S.adaptive_gaussian_threshold, "block"),
                                      (S.niblack_threshold, "window"), (S.sauvola_threshold, "window")])
def test_window_larger_than_image_uses_repeated_reflection(fn, param):
    img = rand_img((4, 6), 8)
    out = fn(img, **{param: 51}).image
    assert out.shape == img.shape and set(np.unique(out)) <= {0, 255}


def test_window_larger_than_image_matches_reference():
    img = rand_img((4, 6), 8)
    ref = ref_local_mean(img, 51)
    np.testing.assert_allclose(S.local_mean_std(img, 51)[0], ref, atol=1e-9)
    np.testing.assert_allclose(S.local_gaussian_mean(img, 51), ref_gauss_mean(img, 51), atol=1e-9)


# ================================================================ 6) moving averages
def test_moving_average_known_row():
    f = np.array([[100, 100, 0, 0]], np.uint8)
    np.testing.assert_allclose(S.moving_average_surface(f, 2), [[100, 100, 50, 0]])
    np.testing.assert_array_equal(S.moving_average_threshold(f, 5, 0.5).image, [[255, 255, 0, 0]])


def test_moving_average_zigzag_scan_continues_across_rows():
    f = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)        # tarama sırası 1 2 3 6 5 4
    m = S.moving_average_surface(f, 2)
    np.testing.assert_allclose(m, [[1, 1.5, 2.5], [4.5, 5.5, 4.5]])
    assert m[1, 2] == (3 + 6) / 2                          # 2. satırın ilk taranan pikseli önceki satırın sonuna bağlı


@pytest.mark.parametrize("n", [5, 20, 80])
@pytest.mark.parametrize("c", [0.2, 0.5, 0.83, 1.0])
def test_moving_average_matches_lab_reference(n, c):
    img = doc_scene(17, 23, seed=n)
    ref = ref_moving_mean(img, n)
    np.testing.assert_allclose(S.moving_average_surface(img, n), ref, atol=1e-12)
    np.testing.assert_array_equal(S.moving_average_threshold(img, n, c).image, u8(img > np.round(c * ref, 9)))


@pytest.mark.parametrize("kw,msg", [({"n": 4}, "n"), ({"n": 81}, "n"), ({"n": 20.5}, "integer"),
                                    ({"c": 0.1}, "c"), ({"c": 1.01}, "c"), ({"c": float("inf")}, "c"), ({"c": "0.5"}, "c")])
def test_moving_average_rejects_bad_parameters(kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        S.moving_average_threshold(rand_img((5, 5), 0), **kw)


# ================================================================ 7–8) Niblack / Sauvola
def test_niblack_sauvola_known_matrix():
    f = np.array([[0, 0, 0], [0, 90, 0], [0, 0, 0]], np.uint8)
    m, s = S.local_mean_std(f, 3)
    assert m[1, 1] == 10 and s[1, 1] == pytest.approx(math.sqrt(900 - 100))       # E[f²] = 8100/9
    n = S.niblack_threshold(f, 3, 0.2)
    np.testing.assert_array_equal(n.image[1, 1], 255)                                # 90 > 10 − 0.2·28.28
    sv = S.sauvola_threshold(f, 3, 0.5, 32)
    assert sv.image[1, 1] == 255 and sv.image[1, 0] == 0


@pytest.mark.parametrize("b", [3, 15, 51])
@pytest.mark.parametrize("k", [0.0, 0.2, 0.6])
def test_niblack_matches_pixel_reference(b, k):
    img = doc_scene(19, 24, seed=b)
    m, s = ref_local_mean(img, b), ref_local_std(img, b)
    ms, ss = S.local_mean_std(img, b)
    np.testing.assert_allclose(ms, m, atol=1e-9)
    np.testing.assert_allclose(ss, s, atol=1e-7)
    np.testing.assert_array_equal(S.niblack_threshold(img, b, k).image, u8(img > np.round(m - k * s, 9)))


@pytest.mark.parametrize("b", [3, 15])
@pytest.mark.parametrize("k,r", [(0.2, 128), (0.5, 32), (0.0, 255), (0.6, 64)])
def test_sauvola_matches_pixel_reference(b, k, r):
    img = doc_scene(19, 24, seed=b + r)
    m, s = ref_local_mean(img, b), ref_local_std(img, b)
    np.testing.assert_array_equal(S.sauvola_threshold(img, b, k, r).image, u8(img > np.round(m * (1 + k * (s / r - 1)), 9)))


def test_k_zero_reduces_to_local_mean():
    img = doc_scene(seed=7)
    a = S.adaptive_mean_threshold(img, 15, 0).image
    np.testing.assert_array_equal(S.niblack_threshold(img, 15, 0.0).image, a)
    np.testing.assert_array_equal(S.sauvola_threshold(img, 15, 0.0).image, a)


def test_sauvola_keeps_flat_noisy_background_cleaner_than_niblack():
    rng = np.random.default_rng(3)
    img = np.clip(np.rint(rng.normal(200, 4, (60, 80))), 0, 255).astype(np.uint8)
    img[20:40:4, 10:70] = 40
    nib = S.niblack_threshold(img, 15, 0.2, "dark").image > 0
    sau = S.sauvola_threshold(img, 15, 0.2, 128, "dark").image > 0
    bg = np.ones(img.shape, bool)
    bg[15:45, 5:75] = False
    assert sau[bg].mean() < 0.01 < nib[bg].mean()                   # slayt 17: Niblack zeminde lekelenir


@pytest.mark.parametrize("fn", [S.niblack_threshold, S.sauvola_threshold])
@pytest.mark.parametrize("kw,msg", [({"window": 2}, "Window"), ({"window": 14}, "odd"), ({"window": 53}, "Window"),
                                    ({"k": -0.01}, "k"), ({"k": 0.61}, "k"), ({"k": float("nan")}, "k")])
def test_niblack_sauvola_reject_bad_parameters(fn, kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        fn(rand_img((9, 9), 0), **kw)


@pytest.mark.parametrize("r", [31, 256, 100.5])
def test_sauvola_rejects_bad_r(r):
    with pytest.raises(AlgorithmParameterError, match="R"):
        S.sauvola_threshold(rand_img((9, 9), 0), r=r)


# ================================================================ rapor
REPORT_NAMES = {"threshold": "Manual threshold", "iterative_threshold": "Automatic global threshold",
                "otsu": "Otsu's method", "adaptive_mean": "Adaptive mean threshold",
                "adaptive_gaussian": "Adaptive Gaussian threshold", "moving_average": "Moving averages",
                "niblack": "Niblack", "sauvola": "Sauvola"}


@pytest.mark.parametrize("key,piece", [("threshold", "T = 127"), ("iterative_threshold", "iteration"), ("otsu", "η ="),
                                       ("adaptive_mean", "T(x, y)"), ("adaptive_gaussian", "σ = 2.60"),
                                       ("moving_average", "zigzag"), ("niblack", "k = 0.2"), ("sauvola", "R = 128")])
def test_report_summary_names_method_and_threshold(key, piece):
    r = FUNCS[key](doc_scene()).report
    assert isinstance(r, S.ThresholdReport) and piece in r.summary()
    assert r.summary().startswith(REPORT_NAMES[key] + ":")


def test_registry_execute_returns_report_and_clamps_gui_values():
    algo = REGISTRY.get("adaptive_mean")
    res = algo.execute(doc_scene(), {"block": 16, "c": 99})        # GUI doğrulaması: tek sayıya ve aralığa çekilir
    assert "b = 17" in res.report.summary() and "C = 30" in res.report.summary()


# ================================================================ eşitlik (f = T) durumları
@pytest.mark.parametrize("b", [3, 5, 7, 15, 31, 51])
@pytest.mark.parametrize("value", [1, 37, 77, 128, 200, 255])
def test_gaussian_tie_on_flat_image_is_not_flipped_by_rounding_noise(b, value):
    """C = 0, düz görüntü: T(x, y) = f tam olarak; f > T yanlış → BRIGHT'ta hepsi 0.
    (Gauss ağırlıklarının toplamı kayan noktada tam 1 değildir; yüzey 1e-9'a yuvarlanmazsa karar titrer.)"""
    img = np.full((7, 9), value, np.uint8)
    assert (S.adaptive_gaussian_threshold(img, b, 0).image == 0).all()
    assert (S.adaptive_gaussian_threshold(img, b, 0, "dark").image == 255).all()


def test_iterative_split_puts_pixels_equal_to_t_in_lower_class():
    f = np.array([[10, 10, 10, 200, 200, 200]], np.uint8)
    its = S.iterative_threshold(S.histogram(f), 10.0, 0.5)             # T₀ = 10 bir gri düzeye eşit
    assert (its[0].mu1, its[0].mu2, its[0].t_next) == (200.0, 10.0, 105.0)   # μ₂: f ≤ T (slayt 11, laboratuvar)
    assert ref_iterative(f, 10.0, 0.5)[0] == 105.0
    assert "T = 105.00" in S.automatic_threshold(f, "custom", 10).report.summary()
