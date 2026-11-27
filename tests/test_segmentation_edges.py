"""Segmentation / Package 2 — Edge-based segmentation (Chapter 5 slayt 18–28, laboratuvar).

Bağımsız referanslar: piksel döngüleri (laboratuvar ``conv``/``refl``/``zeroCross``/``cannyAll``/
``houghAdd``/``houghPeaks`` karşılıkları), slayt tablolarından elle yazılmış çekirdekler,
OpenCV ``filter2D`` (korelasyon, BORDER_REFLECT_101), ``findContours`` ve ``colorsys``.
"""

import colorsys
import inspect
import math

import cv2
import numpy as np
import pytest

from processing import segmentation as S
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError
from tests.test_segmentation_thresholding import ref_luma, refl

EDGE_KEYS = ["edge_discontinuity", "edge_point", "edge_line", "edge_gradient", "edge_roberts", "edge_sobel",
             "edge_prewitt", "edge_laplacian", "edge_log", "edge_canny", "edge_linking", "edge_boundary", "edge_hough"]
FUNCS = {"edge_discontinuity": S.discontinuity, "edge_point": S.point_detection, "edge_line": S.line_detection,
         "edge_gradient": S.gradient, "edge_roberts": S.roberts, "edge_sobel": S.sobel, "edge_prewitt": S.prewitt,
         "edge_laplacian": S.laplacian_edges, "edge_log": S.laplacian_of_gaussian, "edge_canny": S.canny,
         "edge_linking": S.edge_linking, "edge_boundary": S.boundary_extraction, "edge_hough": S.hough_lines}

# Slayt 20–23 tabloları (satır = y aşağı, sütun = x sağa), testte elle yazıldı
SLIDE_POINT = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
SLIDE_LINES = {"horizontal": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], "p45": [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
               "vertical": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], "m45": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]}
SLIDE_LAPLACIAN = {"4": [[0, 1, 0], [1, -4, 1], [0, 1, 0]], "8": [[1, 1, 1], [1, -8, 1], [1, 1, 1]]}
SLIDE_PREWITT = ([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
SLIDE_SOBEL = ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


# ================================================================ referanslar
def ref_conv(f, k):
    k = np.asarray(k)
    r = k.shape[0] // 2
    h, w = f.shape
    out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            out[y, x] = sum(int(k[t + r, s + r]) * int(f[refl(y + t, h), refl(x + s, w)])
                            for t in range(-r, r + 1) for s in range(-r, r + 1))
    return out


def ref_roberts(f):
    h, w = f.shape
    gx, gy = np.zeros((h, w)), np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            z = lambda dy, dx: int(f[refl(y + dy, h), refl(x + dx, w)])
            gx[y, x] = z(1, 1) - z(0, 0)          # Gonzalez z9 − z5
            gy[y, x] = z(1, 0) - z(0, 1)          # z8 − z6
    return gx, gy


def js_round(v):
    """JS Math.round (ES spesifikasyonu): x.5 yukarı; 0.49999999999999994 → 0."""
    fl = math.floor(v)
    return fl + (1 if v - fl >= 0.5 else 0)


def ref_blur(f, sigma, k=None):
    """2-B doğrudan toplam (ayrılabilir değil) + yarım yukarı yuvarlama."""
    k = k or 2 * math.ceil(3 * sigma) + 1
    r = k // 2
    g = [math.exp(-i * i / (2 * sigma * sigma)) for i in range(-r, r + 1)]
    s = sum(g)
    g = [v / s for v in g]
    h, w = f.shape
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            v = math.fsum(g[t + r] * g[u + r] * int(f[refl(y + t, h), refl(x + u, w)])
                          for t in range(-r, r + 1) for u in range(-r, r + 1))
            out[y, x] = js_round(min(255.0, max(0.0, v)))
    return out


def ref_zero_cross(L, thr):
    h, w = L.shape
    out = np.zeros((h, w), bool)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            for a, b, c, d in ((-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1), (1, -1, -1, 1)):
                u, v = L[y + b, x + a], L[y + d, x + c]
                if u * v < 0 and abs(u - v) > thr:
                    out[y, x] = True
                    break
    return out


def ref_canny(f, sigma, lo, hi):
    """Laboratuvar cannyAll, piksel döngüleri + kuyruklu BFS."""
    b = ref_blur(f, sigma) if sigma > 0 else f
    gx, gy = ref_conv(b, SLIDE_SOBEL[0]), ref_conv(b, SLIDE_SOBEL[1])
    h, w = f.shape
    m = np.hypot(gx, gy)
    mx = m.max() or 1
    mn = m * 255 / mx
    nbs = [(-1, 0, 1, 0), (-1, -1, 1, 1), (0, -1, 0, 1), (1, -1, -1, 1)]
    nms = np.zeros((h, w))
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = math.degrees(math.atan2(gy[y, x], gx[y, x]))
            sec = js_round(((a + 360) % 180) / 45) % 4
            q = nbs[sec]
            v = mn[y, x]
            if v >= mn[y + q[1], x + q[0]] and v >= mn[y + q[3], x + q[2]]:
                nms[y, x] = v
    st = np.where(nms >= hi, 2, np.where(nms >= lo, 1, 0))
    acc = np.full((h, w), -1)
    front = [(y, x) for y in range(h) for x in range(w) if st[y, x] == 2]
    for p in front:
        acc[p] = 0
    it = 0
    while front and it < 3000:
        nxt = []
        for y, x in front:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (dy or dx) and 0 <= yy < h and 0 <= xx < w and st[yy, xx] == 1 and acc[yy, xx] < 0:
                        acc[yy, xx] = it + 1
                        nxt.append((yy, xx))
        if nxt:
            it += 1
        front = nxt
    return acc >= 0, st, nms


def ref_hough(points, w, h):
    d = math.ceil(math.hypot(w, h))
    acc = np.zeros((2 * d + 1, 180), int)
    cs = [math.cos((t - 90) * math.pi / 180) for t in range(180)]
    sn = [math.sin((t - 90) * math.pi / 180) for t in range(180)]
    for x, y in points:
        for t in range(180):
            acc[js_round(x * cs[t] + y * sn[t]) + d, t] += 1
    return acc, d


def ref_peaks(acc, d, frac, max_n):
    thr = max(2, frac * acc.max())
    rows = acc.shape[0]
    out = []
    for r in range(rows):
        for t in range(180):
            v = acc[r, t]
            if v < thr:
                continue
            ok = True
            for dr in range(-5, 6):
                for dt in range(-5, 6):
                    if not dr and not dt:
                        continue
                    rr, tt = r + dr, t + dt
                    if 0 <= rr < rows and 0 <= tt < 180:
                        u = acc[rr, tt]
                        if u > v or (u == v and (dr < 0 or (dr == 0 and dt < 0))):
                            ok = False
            if ok:
                out.append((r - d, t - 90, int(v)))
    out.sort(key=lambda p: -p[2])
    return out[:max_n]


def ref_fill_rows(g, k):
    out = g.copy()
    for r in range(g.shape[0]):
        cols = [c for c in range(g.shape[1]) if g[r, c]]
        for a, b in zip(cols[:-1], cols[1:]):
            if 0 < b - a - 1 <= k:
                out[r, a + 1:b] = True
    return out


def rand_img(shape, seed):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def scene(h=40, w=56, seed=0, noise=4.0):
    """Sentetik sahne: dikdörtgen, daire, ince parlak çizgi, tek parlak nokta, eğim + gürültü."""
    rng = np.random.default_rng(seed)
    f = np.add.outer(np.linspace(40, 80, h), np.linspace(0, 20, w))
    f[8:20, 6:22] = 190
    cv2.circle(f, (40, 26), 8, 30, -1)
    f[h - 10, 4:30] = 230
    f[5, w - 11] = 255
    return np.clip(np.rint(f + rng.normal(0, noise, (h, w))), 0, 255).astype(np.uint8)


# ================================================================ katalog
def test_edge_operations_follow_slide_order_after_thresholding():
    keys = [a.key for a in REGISTRY.in_category("segmentation")]
    assert keys[8:21] == EDGE_KEYS and len(keys) == 25   # kmeans yer tutucusu kaldırıldı (Chapter 6 k-means: Color)
    assert all(REGISTRY.get(k).implemented for k in EDGE_KEYS)
    assert [REGISTRY.get(k).title for k in EDGE_KEYS] == [
        "Discontinuities & Derivatives", "Point Detection", "Line Detection", "Gradient (First-Order Derivative)",
        "Roberts Operator", "Sobel Operator", "Prewitt Operator", "Laplacian", "Laplacian of Gaussian (LoG)",
        "Canny Edge Detector", "Edge Linking", "Boundary Extraction · Contour Tracing", "Hough Transform"]


@pytest.mark.parametrize("key", EDGE_KEYS)
def test_gui_parameters_match_function_signature(key):
    algo = REGISTRY.get(key)
    sig = inspect.signature(FUNCS[key])
    assert [p.key for p in algo.params] == list(sig.parameters)[1:]
    assert all(sig.parameters[p.key].default == p.default for p in algo.params)


def test_lab_defaults():
    d = lambda k: REGISTRY.get(k).default_params()
    assert d("edge_point")["t"] == 0.9 and d("edge_line") == {"direction": "p45", "t": 0.6, "display": "detected"}
    assert (d("edge_log")["sigma"], d("edge_log")["kernel"], d("edge_log")["t"]) == (2.0, "8", 0.1)
    assert (d("edge_canny")["sigma"], d("edge_canny")["low"], d("edge_canny")["high"]) == (1.4, 10, 45)
    assert (d("edge_hough")["low"], d("edge_hough")["high"], d("edge_hough")["frac"], d("edge_hough")["max_lines"]) == (20, 50, 0.5, 24)
    assert (d("edge_linking")["tm"], d("edge_linking")["ta"], d("edge_linking")["k"]) == (0.3, 45, 25)   # Gonzalez örneği


def test_kernels_equal_slide_tables():
    np.testing.assert_array_equal(S.POINT_KERNEL, SLIDE_POINT)
    for k, v in SLIDE_LINES.items():
        np.testing.assert_array_equal(S.LINE_KERNELS[k], v)
    for k, v in SLIDE_LAPLACIAN.items():
        np.testing.assert_array_equal(S.LAPLACIAN[k], v)
    for name, (kx, ky) in (("prewitt", SLIDE_PREWITT), ("sobel", SLIDE_SOBEL)):
        np.testing.assert_array_equal(S.GRADIENT_OPERATORS[name][0], kx)
        np.testing.assert_array_equal(S.GRADIENT_OPERATORS[name][1], ky)
    rx, ry = S.GRADIENT_OPERATORS["roberts"]                      # slayt: [−1 0; 0 1] ve [0 −1; 1 0]
    np.testing.assert_array_equal(rx[1:, 1:], [[-1, 0], [0, 1]])
    np.testing.assert_array_equal(ry[1:, 1:], [[0, -1], [1, 0]])
    assert not rx[0].any() and not rx[:, 0].any() and not ry[0].any() and not ry[:, 0].any()


# ================================================================ ortak sözleşme
@pytest.mark.parametrize("key", EDGE_KEYS)
@pytest.mark.parametrize("shape", [(23, 31), (23, 31, 1), (23, 31, 3), (23, 31, 4)], ids=["gray", "gray1", "bgr", "bgra"])
def test_default_output_layout_and_input_untouched(key, shape):
    img = rand_img(shape, 3)
    before = img.copy()
    out = FUNCS[key](img).image
    np.testing.assert_array_equal(img, before)
    assert out.dtype == np.uint8 and out.shape[:2] == img.shape[:2]
    if out.ndim == 3 and img.ndim == 3 and img.shape[2] == 4 and out.shape[2] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])
    if key not in ("edge_boundary", "edge_hough"):                # gri çıktılı varsayılanlar: girdi düzeninde
        assert out.shape == img.shape


@pytest.mark.parametrize("key", EDGE_KEYS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (2, 2), (3, 3), (4, 60)])
def test_tiny_and_thin_images(key, shape):
    out = FUNCS[key](rand_img(shape, sum(shape))).image
    assert out.shape[:2] == shape and out.dtype == np.uint8


@pytest.mark.parametrize("key", EDGE_KEYS)
def test_color_uses_rounded_luma(key):
    img = rand_img((20, 26, 3), 5)
    y = ref_luma(img).astype(np.uint8)
    a, b = FUNCS[key](img), FUNCS[key](y)
    assert a.report.summary() == b.report.summary()
    ga = a.image[..., 0] if a.image.ndim == 3 and b.image.ndim == 2 else a.image
    np.testing.assert_array_equal(ga, b.image)


@pytest.mark.parametrize("key", EDGE_KEYS)
def test_rejects_invalid_images(key):
    for bad in (np.zeros((4, 4), np.float32), np.zeros((0, 4), np.uint8), np.zeros((4, 4, 2), np.uint8)):
        with pytest.raises(AlgorithmParameterError):
            FUNCS[key](bad)


@pytest.mark.parametrize("key", [k for k in EDGE_KEYS if "display" in inspect.signature(FUNCS[k]).parameters])
def test_rejects_unknown_display(key):
    with pytest.raises(AlgorithmParameterError, match="Display"):
        FUNCS[key](rand_img((6, 6), 0), display="nope")


# ================================================================ sabit görüntü: kenar yok
@pytest.mark.parametrize("value", [0, 1, 128, 255])
@pytest.mark.parametrize("shape", [(1, 1), (9, 12)])
def test_constant_image_has_no_edges(value, shape):
    f = np.full(shape, value, np.uint8)
    assert (S.discontinuity(f).image == 128).all() and (S.discontinuity(f, display="abs").image == 0).all()
    assert (S.point_detection(f).image == 0).all() and "flat image" in S.point_detection(f).report.summary()
    assert (S.line_detection(f).image == 0).all()
    for fn in (S.gradient, S.roberts, S.sobel, S.prewitt):
        assert (fn(f).image == 0).all() and (fn(f, display="edges").image == 0).all()
        assert (fn(f, display="gx").image == 128).all()
        assert (fn(f, display="direction").image == 0).all()            # M = 0 → siyah
    assert (S.laplacian_edges(f).image == 128).all() and (S.laplacian_edges(f, display="zero_crossings").image == 0).all()
    assert (S.laplacian_of_gaussian(f).image == 0).all()
    assert (S.canny(f).image == 0).all()
    assert (S.edge_linking(f).image == 0).all()
    h = S.hough_lines(f)
    assert "0 lines" in h.report.summary() and (h.image[..., 0] == f).all()
    b = S.boundary_extraction(f, display="boundaries")
    if value > 0:                                                   # Otsu T* = 0 → tüm görüntü tek nesne
        assert b.report.count == 1 and (b.image[0] == 255).all() and (b.image[:, 0] == 255).all()
    else:
        assert b.report.count == 0 and (b.image == 0).all()


# ================================================================ 1) discontinuity
def test_discontinuity_known_profile():
    f = np.array([[10, 10, 20, 30, 30, 200, 30]], np.uint8)           # rampa, adım, nokta
    np.testing.assert_array_equal(S.derivative_1d(f, "first", "horizontal"), [[0, 10, 10, 0, 170, -170, 0]])
    np.testing.assert_array_equal(S.derivative_1d(f, "second", "horizontal"), [[0, 10, 0, -10, 170, -340, 0]])
    np.testing.assert_array_equal(S.derivative_1d(f.T, "first", "vertical"), S.derivative_1d(f, "first", "horizontal").T)


@pytest.mark.parametrize("order", ["first", "second"])
@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_discontinuity_matches_loop_reference(order, axis):
    f = rand_img((11, 13), 2)
    h, w = f.shape
    ref = np.zeros((h, w), int)
    for y in range(h):
        for x in range(w):
            g = (lambda i: int(f[y, i])) if axis == "horizontal" else (lambda i: int(f[i, x]))
            i, n = (x, w) if axis == "horizontal" else (y, h)
            if order == "first" and i + 1 < n:
                ref[y, x] = g(i + 1) - g(i)
            if order == "second" and 0 < i < n - 1:
                ref[y, x] = g(i + 1) + g(i - 1) - 2 * g(i)
    np.testing.assert_array_equal(S.derivative_1d(f, order, axis), ref)
    m = np.abs(ref).max()
    np.testing.assert_array_equal(S.discontinuity(f, order, axis).image,
                                  np.floor(128 + ref * 127 / m + 0.5).astype(np.uint8))


def test_display_mappings_follow_lab_dispfn():
    v = np.array([[-50.0, 0.0, 25.0, 100.0]])
    np.testing.assert_array_equal(S.display_signed(v), [[js_round(128 - 50 * 1.27), 128, js_round(128 + 25 * 1.27), 255]])
    np.testing.assert_array_equal(S.display_minmax(v), [[0, js_round(50 * 255 / 150), js_round(75 * 255 / 150), 255]])
    np.testing.assert_array_equal(S.display_linmax(np.array([[0.0, 1.0, 2.0]])), [[0, 128, 255]])   # 127.5 → 128 (yarım yukarı)
    assert S.q8(np.array([0.5, 1.5, 2.5, -3, 300])).tolist() == [1, 2, 3, 0, 255]
    assert S.js_round(np.array([0.49999999999999994, -2.5, -0.5, 2.5, 1e15 + 0.5])).tolist() == [0, -2, 0, 3, 1e15 + 1]


# ================================================================ 2–3) point / line
def test_point_detection_known_matrix():
    f = np.full((7, 7), 50, np.uint8)
    f[3, 3] = 200
    r = S.point_detection(f)
    np.testing.assert_array_equal(r.image, np.where(np.arange(49).reshape(7, 7) == 24, 255, 0))
    assert "1 point pixel" in r.report.summary() and "1080.0" in r.report.summary()   # |R| = 8 · 150 → T = 0.9 · 1200


@pytest.mark.parametrize("t", [0.1, 0.5, 0.9, 1.0])
def test_point_detection_matches_reference(t):
    f = scene(seed=int(t * 10))
    a = np.abs(ref_conv(f, SLIDE_POINT))
    np.testing.assert_allclose(np.abs(S.correlate(f, S.POINT_KERNEL)), a)
    np.testing.assert_allclose(cv2.filter2D(f.astype(np.float64), -1, np.array(SLIDE_POINT, float),
                                            borderType=cv2.BORDER_REFLECT_101), ref_conv(f, SLIDE_POINT), atol=1e-9)
    np.testing.assert_array_equal(S.point_detection(f, t).image, np.where(a >= t * a.max(), 255, 0))
    assert S.point_detection(f, 0.9).image[5, f.shape[1] - 11] == 255   # sahnedeki tek nokta


@pytest.mark.parametrize("direction", ["horizontal", "p45", "vertical", "m45"])
def test_line_detection_matches_reference_and_responds_to_its_direction(direction):
    f = np.full((21, 21), 40, np.uint8)
    if direction == "horizontal":
        f[10, :] = 200
    elif direction == "vertical":
        f[:, 10] = 200
    elif direction == "p45":
        np.fill_diagonal(f, 200)                                     # ╲
    else:
        np.fill_diagonal(np.fliplr(f), 200)                          # ╱
    rp = np.maximum(ref_conv(f, SLIDE_LINES[direction]), 0)
    det = S.line_detection(f, direction, 0.6).image > 0
    np.testing.assert_array_equal(det, rp >= 0.6 * rp.max())
    assert det[f == 200].mean() > 0.8 and not det[f == 40].any()
    others = [d for d in SLIDE_LINES if d != direction]
    for o in others:                                                 # diğer çekirdeklerin yanıtı daha zayıf
        assert np.maximum(ref_conv(f, SLIDE_LINES[o]), 0).max() < rp.max()


def test_line_detection_negative_response_discarded():
    f = np.full((9, 9), 200, np.uint8)
    f[4, :] = 40                                                     # koyu çizgi: R < 0
    assert (S.line_detection(f, "horizontal").image[4] == 0).all()


# ================================================================ 4–7) gradient operators
@pytest.mark.parametrize("op,kx,ky", [("prewitt", *SLIDE_PREWITT), ("sobel", *SLIDE_SOBEL),
                                      ("difference", [[0, 0, 0], [0, -1, 1], [0, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]])])
def test_gradient_components_match_reference(op, kx, ky):
    f = scene(seed=3)
    _, gx, gy = S.gradient_components(f, op)
    np.testing.assert_allclose(gx, ref_conv(f, kx))
    np.testing.assert_allclose(gy, ref_conv(f, ky))
    if op == "sobel":                                                 # OpenCV (korelasyon, reflect-101) ile de aynı
        np.testing.assert_allclose(gx, cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101))
        np.testing.assert_allclose(gy, cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101))


def test_roberts_matches_cross_differences():
    f = scene(seed=4)
    _, gx, gy = S.gradient_components(f, "roberts")
    rx, ry = ref_roberts(f)
    np.testing.assert_allclose(gx, rx)
    np.testing.assert_allclose(gy, ry)


def test_forward_difference_known_matrix():
    f = np.array([[0, 10, 30], [5, 5, 5], [0, 0, 100]], np.uint8)
    _, gx, gy = S.gradient_components(f, "difference")
    np.testing.assert_array_equal(gx, [[10, 20, -20], [0, 0, 0], [0, 100, -100]])   # son sütun: reflect-101
    np.testing.assert_array_equal(gy, [[5, -5, -25], [-5, -5, 95], [5, 5, -95]])


@pytest.mark.parametrize("fn,op", [(S.gradient, "difference"), (S.roberts, "roberts"), (S.sobel, "sobel"), (S.prewitt, "prewitt")])
@pytest.mark.parametrize("magnitude", ["l2", "abs"])
@pytest.mark.parametrize("smoothing", ["none", "gauss5"])
def test_gradient_displays(fn, op, magnitude, smoothing):
    f = scene(seed=1)
    src = ref_blur(f, 0.3 * ((5 - 1) * 0.5 - 1) + 0.8, 5) if smoothing == "gauss5" else f
    kx, ky = S.GRADIENT_OPERATORS[op]
    gx, gy = ref_conv(src, kx), ref_conv(src, ky)
    m = np.hypot(gx, gy) if magnitude == "l2" else np.abs(gx) + np.abs(gy)
    kw = dict(magnitude=magnitude, smoothing=smoothing)
    np.testing.assert_array_equal(fn(f, display="magnitude", **kw).image, np.floor(m * 255 / m.max() + 0.5).astype(np.uint8))
    np.testing.assert_array_equal(fn(f, display="gx", **kw).image,
                                  np.floor(128 + gx * 127 / np.abs(gx).max() + 0.5).astype(np.uint8))
    np.testing.assert_array_equal(fn(f, display="edges", t=0.4, **kw).image, np.where(m >= 0.4 * m.max(), 255, 0))
    d = fn(f, display="direction", **kw).image
    assert d.shape == f.shape + (3,)


def test_direction_colour_matches_hsl_definition():
    gx = np.array([[10.0, 0.0, -10.0, 0.0, 3.0]])
    gy = np.array([[0.0, 10.0, 0.0, -10.0, 4.0]])
    out = S.direction_bgr(gx, gy)
    m = np.hypot(gx, gy)
    for i in range(gx.shape[1]):
        h = (math.atan2(gy[0, i], gx[0, i]) + math.pi) / (2 * math.pi)
        l = 0.5 * min(1, m[0, i] / m.max() * 2.2)
        r, g, b = colorsys.hls_to_rgb(h % 1.0, l, 0.85)
        assert np.abs(out[0, i].astype(int) - [round(b * 255), round(g * 255), round(r * 255)]).max() <= 1


def test_gradient_direction_points_across_the_edge():
    f = np.zeros((9, 9), np.uint8)
    f[:, 5:] = 200                                                   # dikey kenar: α = 0°
    _, gx, gy = S.gradient_components(f, "sobel")
    assert gx[4, 4] > 0 and gy[4, 4] == 0
    f = np.zeros((9, 9), np.uint8)
    f[5:, :] = 200                                                   # yatay kenar: α = 90° (y aşağı)
    _, gx, gy = S.gradient_components(f, "sobel")
    assert gy[4, 4] > 0 and gx[4, 4] == 0


@pytest.mark.parametrize("fn", [S.gradient, S.roberts, S.sobel, S.prewitt])
@pytest.mark.parametrize("kw,msg", [({"t": 0.04}, "threshold"), ({"t": 1.5}, "threshold"), ({"magnitude": "l1"}, "Magnitude"),
                                    ({"smoothing": "box"}, "smoothing"), ({"t": float("nan")}, "threshold")])
def test_gradient_rejects_bad_parameters(fn, kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        fn(rand_img((6, 6), 0), **kw)


# ================================================================ 8–9) Laplacian / LoG
@pytest.mark.parametrize("kernel", ["4", "8"])
@pytest.mark.parametrize("t", [0.0, 0.1, 0.3])
def test_laplacian_matches_reference(kernel, t):
    f = scene(seed=5)
    lap = ref_conv(f, SLIDE_LAPLACIAN[kernel])
    np.testing.assert_allclose(S.correlate(f, S.LAPLACIAN[kernel]), lap)
    zc = ref_zero_cross(lap, t * np.abs(lap).max())
    np.testing.assert_array_equal(S.laplacian_edges(f, kernel, "zero_crossings", t).image, np.where(zc, 255, 0))
    np.testing.assert_array_equal(S.laplacian_edges(f, kernel, "signed", t).image,
                                  np.floor(128 + lap * 127 / np.abs(lap).max() + 0.5).astype(np.uint8))


def test_laplacian_known_step_and_ramp():
    ramp = np.tile(np.arange(0, 70, 10, dtype=np.uint8), (5, 1))
    assert (S.correlate(ramp, S.LAPLACIAN["4"])[:, 1:-1] == 0).all()   # doğrusal rampada ∇² = 0
    step = np.zeros((5, 8), np.uint8)
    step[:, 4:] = 100
    lap = S.correlate(step, S.LAPLACIAN["4"])
    assert (lap[:, 3] == 100).all() and (lap[:, 4] == -100).all()      # çift yanıt
    # laboratuvar kuralı: piksel, KARŞILIKLI komşuları işaret değiştirince işaretlenir → ideal 1 px adımda yok
    assert not (S.laplacian_edges(step, "4", "zero_crossings", 0.1).image > 0).any()
    ramp3 = np.tile(np.array([0, 0, 0, 50, 100, 100, 100], np.uint8), (5, 1))
    zc = S.laplacian_edges(ramp3, "4", "zero_crossings", 0.1).image > 0
    assert zc[1:-1, 3].all() and zc.sum() == 3                         # rampanın tam ortası


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("kernel", ["4", "8"])
def test_log_matches_reference(sigma, kernel):
    f = scene(30, 36, seed=6, noise=10)
    b = ref_blur(f, sigma)
    np.testing.assert_array_equal(S.gaussian_blur(f, sigma), b)
    lap = ref_conv(b, SLIDE_LAPLACIAN[kernel])
    zc = ref_zero_cross(lap, 0.1 * np.abs(lap).max())
    np.testing.assert_array_equal(S.laplacian_of_gaussian(f, sigma, kernel).image, np.where(zc, 255, 0))


@pytest.mark.parametrize("noise", [6, 12])
def test_log_has_fewer_noise_crossings_than_laplacian(noise):
    rng = np.random.default_rng(7)
    g = np.full((64, 80), 100.0)
    g[16:48, 20:60] = 180
    g = np.clip(np.rint(g + rng.normal(0, noise, g.shape)), 0, 255).astype(np.uint8)
    bg = np.ones(g.shape, bool)
    bg[12:52, 16:64] = False                                          # nesne kenarlarından uzak düz zemin
    plain = (S.laplacian_edges(g, "8", "zero_crossings", 0.1).image > 0)[bg].mean()
    log = (S.laplacian_of_gaussian(g, 2.0, "8", "zero_crossings", 0.1).image > 0)[bg].mean()
    assert log < plain


def test_gaussian_kernel_follows_lab_size_rule():
    for sigma, k in ((0.5, 5), (1.0, 7), (1.4, 11), (2.0, 13), (5.0, 31)):
        g = S.gaussian_kernel(sigma)
        assert g.size == k and g.sum() == pytest.approx(1)


@pytest.mark.parametrize("kw,msg", [({"sigma": 0.4}, "σ"), ({"sigma": 5.1}, "σ"), ({"kernel": "6"}, "kernel"),
                                    ({"t": -0.1}, "Zero-crossing"), ({"t": 0.6}, "Zero-crossing")])
def test_log_rejects_bad_parameters(kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        S.laplacian_of_gaussian(rand_img((6, 6), 0), **kw)


# ================================================================ 10) Canny
@pytest.mark.parametrize("sigma,lo,hi", [(1.4, 10, 45), (0.0, 20, 60), (2.0, 5, 30), (1.0, 40, 41)])
def test_canny_matches_lab_reference(sigma, lo, hi):
    f = scene(30, 40, seed=8, noise=6)
    edges, st, nms = ref_canny(f, sigma, lo, hi)
    stg = S.canny_stages(f, sigma, lo, hi)
    np.testing.assert_allclose(stg.nms, nms, atol=1e-9)
    np.testing.assert_array_equal(stg.state, st)
    np.testing.assert_array_equal(stg.edges, edges)
    np.testing.assert_array_equal(S.canny(f, sigma, lo, hi).image, np.where(edges, 255, 0))


def test_canny_hysteresis_keeps_only_weak_pixels_connected_to_strong():
    f = np.zeros((30, 30), np.uint8)
    f[5:25, 10:] = 200                                               # güçlü kenar
    f[5:25, 3] = 20                                                  # zayıf (ölçekli M 10…45 arası), kopuk çizgi
    st = S.canny_stages(f, 0, 10, 45)
    assert (st.state[:, :7] == 1).any() and not (st.state[:, :7] == 2).any()
    assert not st.edges[:, :7].any()                                  # güçlüye bağlı değil → atılır
    assert st.edges[10:20, 9:11].any()


@pytest.mark.parametrize("sigma", [0.0, 1.4])
def test_canny_edge_width_across_a_step(sigma):
    """NMS ≥ (laboratuvar): simetrik adımda iki eşit tepe pikseli de korunur → en fazla 2 px genişlik."""
    f = np.zeros((40, 40), np.uint8)
    f[:, 20:] = 200
    e = S.canny(f, sigma).image > 0
    assert all(list(np.nonzero(e[r])[0]) == [19, 20] for r in range(3, 37))


def test_canny_displays():
    f = scene(seed=9)
    st = S.canny_stages(f, 1.4, 10, 45)
    np.testing.assert_array_equal(S.canny(f, display="nms").image, np.floor(np.clip(st.nms, 0, 255) + 0.5).astype(np.uint8))
    dbl = S.canny(f, display="double").image
    assert (dbl[st.state == 2] == 255).all() and (dbl[st.state == 1] == S.ORANGE_BGR).all() and (dbl[st.state == 0] == 0).all()
    np.testing.assert_array_equal(S.canny(f, display="smoothed").image, ref_blur(f, 1.4))


@pytest.mark.parametrize("kw,msg", [({"low": 45, "high": 45}, "smaller"), ({"low": 50, "high": 20}, "smaller"),
                                    ({"low": 0}, "T₁"), ({"high": 256}, "T₂"), ({"sigma": -0.1}, "σ"), ({"sigma": 4.1}, "σ"),
                                    ({"low": 10.5}, "integer")])
def test_canny_rejects_bad_parameters(kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        S.canny(rand_img((8, 8), 0), **kw)


# ================================================================ 11) edge linking
def test_fill_row_gaps_matches_reference():
    g = np.random.default_rng(1).random((15, 40)) < 0.2
    for k in (1, 3, 10):
        np.testing.assert_array_equal(S.fill_row_gaps(g, k), ref_fill_rows(g, k))
    row = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], bool)
    np.testing.assert_array_equal(S.fill_row_gaps(row, 3)[0], [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0])   # açık uç doldurulmaz


def test_edge_linking_bridges_gap_in_broken_edge():
    f = np.zeros((40, 60), np.uint8)
    f[20:, :] = 180
    f[18:23, 25:35] = 0                                              # kenarın üzerinde 10 px boşluk
    f[20:23, 25:35] = 0
    f[23:, 25:35] = 180
    r_small = S.edge_linking(f, k=5, orientation="horizontal").image > 0
    r_big = S.edge_linking(f, k=25, orientation="horizontal").image > 0
    cand = S.edge_linking(f, k=25, orientation="horizontal", display="candidates").image > 0
    row = 19
    assert r_big[row, 22:38].all() and not r_small[row, 26:34].all() and r_big.sum() > cand.sum()


def test_edge_linking_matches_reference():
    f = scene(seed=10)
    _, gx, gy = S.gradient_components(f, "sobel")
    m = np.hypot(gx, gy)
    a = np.degrees(np.arctan2(gy, gx))
    strong = m > 0.3 * m.max()
    hor = strong & (np.abs(((a - 90) + 90) % 180 - 90) <= 45)
    ver = strong & (np.abs((a + 90) % 180 - 90) <= 45)
    ref = ref_fill_rows(hor, 7) | ref_fill_rows(ver.T, 7).T
    np.testing.assert_array_equal(S.edge_linking(f, 0.3, 45, 7).image, np.where(ref, 255, 0))
    np.testing.assert_array_equal(S.edge_linking(f, 0.3, 45, 7, "horizontal").image, np.where(ref_fill_rows(hor, 7), 255, 0))


@pytest.mark.parametrize("kw,msg", [({"tm": 0.01}, "T_M"), ({"ta": 91}, "T_A"), ({"ta": -1}, "T_A"), ({"k": 0}, "Gap"),
                                    ({"k": 101}, "Gap"), ({"orientation": "diagonal"}, "orientation")])
def test_edge_linking_rejects_bad_parameters(kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        S.edge_linking(rand_img((6, 6), 0), **kw)


# ================================================================ 12) boundary extraction (Moore)
def test_moore_trace_known_shapes():
    sq = np.ones((3, 3), bool)
    assert S.moore_trace(sq, (0, 0)) == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    assert S.moore_trace(np.ones((1, 1), bool), (0, 0)) == [(0, 0)]
    assert S.moore_trace(np.ones((1, 3), bool), (0, 0)) == [(0, 0), (0, 1), (0, 2), (0, 1)]
    diag = np.eye(3, dtype=bool)
    assert S.moore_trace(diag, (0, 0)) == [(0, 0), (1, 1), (2, 2), (1, 1)]


def test_boundary_pixels_match_opencv_external_contours():
    f = np.zeros((60, 80), np.uint8)
    cv2.circle(f, (20, 20), 12, 255, -1)
    cv2.rectangle(f, (40, 10), (70, 30), 255, -1)
    cv2.rectangle(f, (48, 16), (62, 24), 0, -1)                      # delik: dış sınır izlenir
    cv2.fillPoly(f, [np.array([[10, 55], [35, 40], [30, 58]])], 255)
    f[50, 60] = 255
    f[45:47, 70:79] = 255
    r = S.boundary_extraction(f, display="boundaries")
    ours = r.image > 0
    contours, _ = cv2.findContours((f > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ref = np.zeros(f.shape, bool)
    for c in contours:
        ref[c[:, 0, 1], c[:, 0, 0]] = True
    np.testing.assert_array_equal(ours, ref)
    assert r.report.count == len(contours) == 5 and r.report.holes == 1
    assert isinstance(r.report, S._morph.ComponentReport) and len(r.report.lengths) == 5
    assert all(L >= 1 for L in r.report.lengths) and "5 boundaries" in r.report.summary()


def test_boundary_touching_image_border():
    f = np.zeros((10, 10), np.uint8)
    f[:4, :4] = 255
    b = S.boundary_extraction(f, display="boundaries").image > 0
    assert b[0, :4].all() and b[:4, 0].all() and b[3, :4].all() and not b[1:3, 1:3].any()


def test_boundary_overlay_colours_and_foreground():
    f = np.zeros((20, 20), np.uint8)
    f[5:15, 5:15] = 200
    o = S.boundary_extraction(f).image
    assert o.shape == (20, 20, 3) and (o[5, 5] != o[0, 0]).any() and (o[10, 10] == 200 // 3).all()
    d = S.boundary_extraction(f, foreground="dark", display="boundaries")
    assert d.report.count == 1 and d.image[0, 0] == 255 and d.image[10, 10] == 0


# ================================================================ 13) Hough
def test_hough_rho_rounding_edge_case():
    """(x, y) = (0, 1), θ = 30°: x cos θ + y sin θ = 0.49999999999999994 → ρ = 0 (JS Math.round), np.rint ile de 0;
    floor(v + 0.5) burada yanlışlıkla 1 verirdi."""
    acc, d = S.hough_accumulator(np.array([[0, 1]]), 5, 5)
    assert acc[d + 0, 120] == 1 and acc[d + 1, 120] == 0


def test_hough_five_point_example():
    pts = np.array([[0, 0], [0, 100], [100, 0], [100, 100], [50, 50]])   # laboratuvar TOY5
    acc, d = S.hough_accumulator(pts, 101, 101)
    racc, rd = ref_hough(pts, 101, 101)
    assert d == rd
    np.testing.assert_array_equal(acc, racc)
    assert acc[d, 45] == 3                                            # θ = −45°, ρ = 0: (0,0)–(50,50)–(100,100)
    peaks, thr = S.hough_peaks(acc, d, 0.5, 24)
    assert [(p.rho, p.theta, p.votes) for p in peaks] == ref_peaks(racc, rd, 0.5, 24)


@pytest.mark.parametrize("frac,max_n", [(0.5, 24), (0.3, 5), (0.9, 60)])
def test_hough_matches_reference_on_line_image(frac, max_n):
    f = np.full((50, 70), 30, np.uint8)
    cv2.line(f, (5, 10), (65, 10), 220, 1)                            # yatay: θ = −90°, ρ = −10
    cv2.line(f, (20, 3), (20, 46), 220, 1)                            # dikey: θ = 0°, ρ = 20
    cv2.line(f, (5, 45), (45, 5), 220, 1)                             # köşegen
    edges = S.canny_stages(f, 1.4, 20, 50).edges
    ys, xs = np.nonzero(edges)
    racc, rd = ref_hough(list(zip(xs.tolist(), ys.tolist())), 70, 50)
    acc, d = S.hough_accumulator(np.column_stack([xs, ys]), 70, 50)
    np.testing.assert_array_equal(acc, racc)
    peaks, _ = S.hough_peaks(acc, d, frac, max_n)
    assert [(p.rho, p.theta, p.votes) for p in peaks] == ref_peaks(racc, rd, frac, max_n)


def test_hough_detects_the_drawn_lines():
    f = np.full((60, 80), 30, np.uint8)
    f[20, 5:75] = 220
    f[5:55, 50] = 220
    r = S.hough_lines(f, frac=0.5)
    peaks, _ = S.hough_peaks(*S.hough_accumulator(np.column_stack(np.nonzero(S.canny_stages(f, 1.4, 20, 50).edges)[::-1]), 80, 60), 0.5, 24)
    thetas = {p.theta for p in peaks}
    assert {-90, 0} & thetas == {-90, 0} or ({-90, 89} & thetas and 0 in thetas)
    assert "lines with" in r.report.summary()
    acc = S.hough_lines(f, display="accumulator").image
    assert acc.shape == (2 * math.ceil(math.hypot(80, 60)) + 1, 180)
    lines = S.hough_lines(f).image
    assert lines.shape == (60, 80, 3) and (lines == S.HOUGH_LINE_BGR).all(axis=2).any()


@pytest.mark.parametrize("kw,msg", [({"low": 50, "high": 50}, "smaller"), ({"frac": 0.05}, "Peak"), ({"max_lines": 0}, "Max lines"),
                                    ({"max_lines": 61}, "Max lines")])
def test_hough_rejects_bad_parameters(kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        S.hough_lines(rand_img((8, 8), 0), **kw)


# ================================================================ point / line parametre doğrulaması
@pytest.mark.parametrize("fn,kw,msg", [(S.point_detection, {"t": 0.05}, "T / max"), (S.point_detection, {"t": 1.01}, "T / max"),
                                       (S.line_detection, {"direction": "diag"}, "Kernel"), (S.line_detection, {"t": 0}, "T / max"),
                                       (S.discontinuity, {"order": "third"}, "Derivative"), (S.discontinuity, {"axis": "z"}, "Direction"),
                                       (S.laplacian_edges, {"kernel": 4}, "kernel")])
def test_misc_rejects_bad_parameters(fn, kw, msg):
    with pytest.raises(AlgorithmParameterError, match=msg):
        fn(rand_img((6, 6), 0), **kw)


def test_reports_are_edge_reports_except_boundary():
    img = scene()
    for key in EDGE_KEYS:
        rep = FUNCS[key](img).report
        if key == "edge_boundary":
            assert isinstance(rep, S.BoundaryReport)
        else:
            assert isinstance(rep, S.EdgeReport) and rep.summary().startswith(rep.method + ": ")


# ================================================================ 11b) edge linking — keyfi yönler (Gonzalez adım 4)
LINK_ANGLES = [0, 30, 45, 60, 90, 135]


def broken_bar(theta, gap=8, h=120, w=160, thickness=5):
    """θ yönünde (ekranda, saat yönünün tersine) kalın parlak çubuk; ortasında θ boyunca ``gap`` px kesinti."""
    t = math.radians(theta)
    d = np.array([math.cos(t), -math.sin(t)])
    c = np.array([w / 2, h / 2])
    full = np.full((h, w), 40, np.uint8)
    p0, p1 = np.round(c - 70 * d).astype(int), np.round(c + 70 * d).astype(int)
    cv2.line(full, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 200, thickness)
    yy, xx = np.mgrid[0:h, 0:w]
    along = (xx - c[0]) * d[0] + (yy - c[1]) * d[1]
    f = full.copy()
    f[np.abs(along) <= gap / 2] = 40
    return f, full, np.abs(along) <= gap / 2


def ref_fill_along(g, theta, k):
    """Döngülerle: θ'ya paralel her sayısal doğruyu uç uca yürü, iki 1 arasındaki ≤ K uzunluklu 0'ları doldur."""
    h, w = g.shape
    t = math.radians(theta)
    cs, sn = math.cos(t), math.sin(t)
    out = g.copy()
    shallow = abs(cs) >= abs(sn)
    n_pos, n_other = (w, h) if shallow else (h, w)
    slope = -sn / cs if shallow else -cs / sn
    rnd = [js_round(slope * p) for p in range(n_pos)]
    span = max(abs(r) for r in rnd) + 1
    for c in range(-span, n_other + span):
        pts = [(p, c + rnd[p]) for p in range(n_pos) if 0 <= c + rnd[p] < n_other]
        val = [g[q, p] if shallow else g[p, q] for p, q in pts]
        ones = [i for i, v in enumerate(val) if v]
        for a, b in zip(ones[:-1], ones[1:]):
            if 0 < b - a - 1 <= k:
                for i in range(a + 1, b):
                    p, q = pts[i]
                    if shallow:
                        out[q, p] = True
                    else:
                        out[p, q] = True
    return out


def test_angle_difference_wraps_at_0_and_180():
    a = np.array([179.0, 1.0, -179.0, 90.0, 46.0, -134.0, 135.0])
    assert S._angle_mask(a, 0, 2).tolist() == [True, True, True, False, False, False, False]
    assert S._angle_mask(a, 45, 1).tolist() == [False, False, False, False, True, True, False]   # −134° ≡ 46°
    assert S._angle_mask(np.array([-45.0, 135.0, 134.0]), 135, 0).tolist() == [True, True, False]


def test_edge_direction_to_gradient_target():
    assert [S.gradient_target(t) for t in (0, 30, 45, 60, 90, 135, 179)] == [90, 60, 45, 30, 0, 135, 91]


@pytest.mark.parametrize("theta", [0, 10, 30, 45, 60, 89, 90, 91, 120, 135, 150, 179])
def test_scan_lines_partition_image_into_adjacent_digital_lines(theta):
    shape = (23, 31)
    line, pos, _ = S.scan_lines(shape, theta)
    seen = {}
    for y in range(shape[0]):
        for x in range(shape[1]):
            seen.setdefault(int(line[y, x]), []).append((int(pos[y, x]), y, x))
    for pts in seen.values():                                     # her doğruda konumlar tekil, ardışıklar 8-komşu
        pts.sort()
        assert len({p for p, _, _ in pts}) == len(pts)
        for (p1, y1, x1), (p2, y2, x2) in zip(pts[:-1], pts[1:]):
            assert p2 - p1 == 1 and max(abs(y2 - y1), abs(x2 - x1)) == 1


def test_scan_lines_reduce_to_rows_and_columns():
    line0, _, sh0 = S.scan_lines((7, 9), 0)
    line90, _, sh90 = S.scan_lines((7, 9), 90)
    assert sh0 and not sh90
    np.testing.assert_array_equal(line0, np.mgrid[0:7, 0:9][0])
    np.testing.assert_array_equal(line90, np.mgrid[0:7, 0:9][1])
    g = np.random.default_rng(3).random((15, 40)) < 0.2
    for k in (1, 4, 12):
        np.testing.assert_array_equal(S.fill_gaps_along(g, 0, k), S.fill_row_gaps(g, k))
        np.testing.assert_array_equal(S.fill_gaps_along(g, 90, k), S.fill_row_gaps(g.T, k).T)


@pytest.mark.parametrize("theta", LINK_ANGLES + [17, 100, 160])
@pytest.mark.parametrize("k", [1, 3, 9])
def test_fill_gaps_along_matches_loop_reference(theta, k):
    g = np.random.default_rng(theta + k).random((19, 27)) < 0.15
    np.testing.assert_array_equal(S.fill_gaps_along(g, theta, k), ref_fill_along(g, theta, k))


def test_fill_gaps_along_never_fills_open_ends_or_leaves_the_image():
    g = np.zeros((10, 10), bool)
    g[2, 2] = True                                                 # tek piksel: eşi yok, hiçbir şey doldurulmaz
    for theta in LINK_ANGLES:
        np.testing.assert_array_equal(S.fill_gaps_along(g, theta, 100), g)
    g = np.zeros((10, 10), bool)
    g[9, 0] = g[6, 3] = True                                        # 45° (╱) doğrusu üzerinde, 2 px boşluk
    out = S.fill_gaps_along(g, 45, 2)
    assert out[8, 1] and out[7, 2] and out.sum() == 4
    assert S.fill_gaps_along(g, 45, 1).sum() == 2                  # K = 1 < 2 → dolmaz
    g = np.zeros((10, 10), bool)
    g[0, 5] = g[9, 5] = True                                        # dikey doğrunun iki ucu: 8 px boşluk
    out = S.fill_gaps_along(g, 90, 8)
    assert out[:, 5].all() and out.sum() == 10                     # yalnızca aradaki pikseller; görüntü dışı yok


@pytest.mark.parametrize("theta", LINK_ANGLES)
def test_edge_linking_bridges_broken_line_at_any_angle(theta):
    from processing import morphology as M
    f, full, gap = broken_bar(theta, gap=8)
    kw = dict(ta=20, orientation="angle", angle=theta)
    ref = S.edge_linking(full, **kw, display="candidates").image > 0          # kesintisiz çubuğun aday kenarları
    cand = S.edge_linking(f, **kw, display="candidates").image > 0
    big = S.edge_linking(f, k=20, **kw).image > 0
    small = S.edge_linking(f, k=4, **kw).image > 0
    assert M.label_components(cand, 8).count >= 4                               # iki kenar × iki parça
    assert M.label_components(big, 8).count == 2                                # K ≥ boşluk: her kenar tek parça
    assert M.label_components(small, 8).count >= 4 and (small & gap).sum() <= 2    # K < boşluk: köprü yok
    assert (big & gap).sum() >= (ref & gap).sum() > 0                           # boşluk en az kesintisiz kadar dolu
    band = cv2.dilate((ref | cand).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert not (big & ~band).any()          # doldurulanlar kenar bandının en fazla 1 px dışında (sayısal doğru fazı)


@pytest.mark.parametrize("theta", [30, 45, 60, 135])
def test_horizontal_vertical_scan_alone_does_not_bridge_diagonals(theta):
    """Düzeltmeden önceki sınırlama: yalnızca θ = 0° ve 90° taranırsa köşegen çubuğun boşluğu kalır."""
    from processing import morphology as M
    f, _, _ = broken_bar(theta, gap=8)
    both = S.edge_linking(f, ta=20, k=20, orientation="both").image > 0
    angle = S.edge_linking(f, ta=20, k=20, orientation="angle", angle=theta).image > 0
    assert M.label_components(angle, 8).count == 2 != M.label_components(both, 8).count


@pytest.mark.parametrize("theta,named", [(0, "horizontal"), (90, "vertical")])
def test_angle_option_matches_named_orientations(theta, named):
    img = scene(seed=12)
    for k in (3, 25):
        np.testing.assert_array_equal(S.edge_linking(img, k=k, orientation="angle", angle=theta).image,
                                      S.edge_linking(img, k=k, orientation=named).image)


def test_edge_linking_magnitude_condition_is_inclusive():
    """M ≥ T_M: T_M = max M olduğunda en güçlü pikseller aday kalır (M > T_M ile hiçbiri kalmazdı)."""
    f = np.zeros((9, 12), np.uint8)
    f[:, 6:] = 100                                                  # dikey adım: Sobel M = (1 + 2 + 1) · 100
    r = S.edge_linking(f, tm=1.0, orientation="vertical", display="candidates")
    assert (r.image > 0).sum() > 0 and "M ≥ 1 · max = 400.0" in r.report.summary()


@pytest.mark.parametrize("bad", [-1, 180, 45.5, True])
def test_edge_linking_rejects_bad_angle(bad):
    with pytest.raises(AlgorithmParameterError, match="θ"):
        S.edge_linking(rand_img((6, 6), 0), orientation="angle", angle=bad)
