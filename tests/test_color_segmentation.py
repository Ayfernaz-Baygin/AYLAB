"""Color Processing / Paket 5: renkli bölütleme — hue thresholding (HSV), k-means (Lab), ten bölgeleri (YCbCr),
renkli kenar tespiti (slayt 59–63, laboratuvar).

Referanslar laboratuvar kodunun piksel piksel, ``math`` ile yazılmış bağımsız sürümleridir (rgb2hsvCV, rgb2ycrcb,
sobel / windowAt, edgeData, kmeans)."""

import inspect
import math

import numpy as np
import pytest

from processing import color_processing as C
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError
from tests.test_color_full import SPECIAL, bgr, c8, rgb_of, scene
from tests.test_color_smooth import patches

SEG_KEYS = ["color_hue_threshold", "color_kmeans", "color_skin", "color_edges"]
f32 = lambda v: float(np.float32(v))


# ================================================================ bağımsız referanslar
def hsv_cv(r8, g8, b8):
    R, G, B = r8 / 255, g8 / 255, b8 / 255
    V, mn = max(R, G, B), min(R, G, B)
    d = V - mn
    H = 0.0
    if d > 1e-9:
        H = 60 * ((G - B) / d) if V == R else 120 + 60 * ((B - R) / d) if V == G else 240 + 60 * ((R - G) / d)
        if H < 0:
            H += 360
    S = 0 if V < 1e-9 else d / V
    return math.floor(H / 2 + 0.5) % 180, c8(S * 255), c8(V * 255)


def ref_hue(rgb, lo, hi, s, v):
    def inside(h):
        return lo <= h <= hi if lo <= hi else (h >= lo or h <= hi)
    return np.array([[255 if (lambda o: inside(o[0]) and o[1] >= s and o[2] >= v)(hsv_cv(*(int(c) for c in rgb[y, x])))
                      else 0 for x in range(rgb.shape[1])] for y in range(rgb.shape[0])], np.uint8)


def ref_skin(rgb, crl, crh, cbl, cbh):
    out = np.zeros(rgb.shape[:2], np.uint8)
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            r, g, b = (int(c) for c in rgb[y, x])
            Y = 0.299 * r + 0.587 * g + 0.114 * b
            cr, cb = (r - Y) * 0.713 + 128, (b - Y) * 0.564 + 128
            out[y, x] = 255 if crl <= cr <= crh and cbl <= cb <= cbh else 0
    return out


SX = (-1, 0, 1, -2, 0, 2, -1, 0, 1)
SY = (-1, -2, -1, 0, 0, 0, 1, 2, 1)


def ref_sobel(G):
    h, w = len(G), len(G[0])
    gx = [[0.0] * w for _ in range(h)]
    gy = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            win = [G[min(h - 1, max(0, y + dy))][min(w - 1, max(0, x + dx))] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            a = b = 0
            for i, v in enumerate(win):
                a = a + SX[i] * v
                b = b + SY[i] * v
            gx[y][x], gy[y][x] = f32(a), f32(b)
    return gx, gy


def ref_edges(rgb, method):
    h, w = rgb.shape[:2]
    px = lambda y, x: [int(c) for c in rgb[y, x]]
    if method == "gray":
        chans = [[[c8(0.299 * px(y, x)[0] + 0.587 * px(y, x)[1] + 0.114 * px(y, x)[2]) for x in range(w)] for y in range(h)]]
    elif method == "v":
        chans = [[[max(px(y, x)) for x in range(w)] for y in range(h)]]
    else:
        chans = [[[px(y, x)[c] for x in range(w)] for y in range(h)] for c in range(3)]
    grads = [ref_sobel(G) for G in chans]
    mag = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            if method == "dizenzo":
                gxx = gyy = gxy = 0
                for gx, gy in grads:
                    gxx += gx[y][x] ** 2; gyy += gy[y][x] ** 2; gxy += gx[y][x] * gy[y][x]
                th = 0.5 * math.atan2(2 * gxy, gxx - gyy)
                m = math.sqrt(max(0, 0.5 * ((gxx + gyy) + (gxx - gyy) * math.cos(2 * th) + 2 * gxy * math.sin(2 * th))))
            elif method == "rgb":
                m = max(math.hypot(gx[y][x], gy[y][x]) for gx, gy in grads)
            else:
                m = math.hypot(grads[0][0][y][x], grads[0][1][y][x])
            mag[y, x] = f32(m)
    return mag


# ================================================================ katalog
def test_catalog_follows_slides_59_to_63():
    keys = [a.key for a in REGISTRY.in_category("color")]
    assert keys[11:] == SEG_KEYS and len(keys) == 15
    assert [REGISTRY.get(k).title for k in SEG_KEYS] == ["Hue Thresholding in HSV", "k-Means Clustering in Lab",
                                                         "Skin Regions in YCbCr", "Color Edge Detection"]
    funcs = (C.hue_thresholding, C.kmeans_lab, C.skin_regions, C.color_edge_detection)
    for key, func in zip(SEG_KEYS, funcs):
        algo = REGISTRY.get(key)
        assert algo.implemented and algo.func is func
        sig = inspect.signature(func)
        assert list(sig.parameters)[1:] == [p.key for p in algo.params]
        assert all(sig.parameters[p.key].default == p.default for p in algo.params)
    d = {k: REGISTRY.get(k).default_params() for k in SEG_KEYS}
    assert d["color_hue_threshold"] == {"target": "blue", "h_min": 90, "h_max": 130, "s_min": 50, "v_min": 50,
                                        "display": "segmented"}
    assert d["color_kmeans"] == {"k": 4, "features": "lab", "seed": 3}
    assert d["color_skin"] == {"cr_min": 133, "cr_max": 173, "cb_min": 77, "cb_max": 127, "display": "segmented"}
    assert d["color_edges"] == {"method": "gray", "t": 120, "display": "edges"}
    p = {q.key: q for q in REGISTRY.get("color_hue_threshold").params}
    assert (p["h_min"].maximum, p["s_min"].maximum) == (179, 255)                      # OpenCV ölçeği
    p = {q.key: q for q in REGISTRY.get("color_kmeans").params}
    assert (p["k"].minimum, p["k"].maximum) == (2, 6)
    p = {q.key: q for q in REGISTRY.get("color_edges").params}
    assert (p["t"].minimum, p["t"].maximum) == (10, 600)
    # Segmentation kategorisindeki eski 'kmeans' yer tutucusu kaldırıldı; k-means yalnızca Color'da (slayt 61)
    assert "kmeans" not in REGISTRY and REGISTRY.get("color_kmeans").category == "color"


# ================================================================ slayt 59–60: hue thresholding
def hue_wheel():
    """Tam doygun ton çemberi (H 0…358°) + düşük doygunluk ve koyu satırlar."""
    h = np.arange(0, 360, 2.0)
    full = C.c8(C.model_to_rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], -1)[None], "hsv"))
    pale = C.c8(C.model_to_rgb(np.stack([h, np.full_like(h, 0.15), np.ones_like(h)], -1)[None], "hsv"))
    dark = C.c8(C.model_to_rgb(np.stack([h, np.ones_like(h), np.full_like(h, 0.15)], -1)[None], "hsv"))
    return np.concatenate([full, pale, dark], 0)


@pytest.mark.parametrize("target", list(C.HUE_TARGETS))
def test_hue_presets_match_reference(target):
    img = np.concatenate([hue_wheel(), np.resize(scene(), (3, 180, 3))], 0)
    lo, hi, s, v = C.HUE_TARGETS[target]
    m = C.hue_thresholding(bgr(img), target, display="mask").image
    assert (m[..., 0] == m[..., 1]).all() and (m[..., 1] == m[..., 2]).all()
    np.testing.assert_array_equal(m[..., 0], ref_hue(img, lo, hi, s, v))


@pytest.mark.parametrize("lo,hi,s,v", [(0, 179, 0, 0), (100, 100, 0, 0), (175, 3, 0, 0), (90, 130, 255, 0), (0, 179, 0, 255)])
def test_hue_custom_ranges_and_bounds(lo, hi, s, v):
    img = np.concatenate([hue_wheel(), patches(6, 180)], 0)
    m = C.hue_thresholding(bgr(img), "custom", lo, hi, s, v, "mask").image[..., 0]
    np.testing.assert_array_equal(m, ref_hue(img, lo, hi, s, v))


def test_hue_scale_wrap_around_and_saturation_rule():
    wheel = hue_wheel()
    H = C.hsv_cv(wheel.astype(np.float64))[..., 0]
    assert H.min() == 0 and H.max() == 179                                           # OpenCV: derece / 2
    m = C.hue_thresholding(bgr(wheel), "red", display="mask").image[..., 0]
    red = m[0] == 255
    assert red[H[0] >= 170].all() and red[H[0] <= 8].all() and not red[(H[0] > 8) & (H[0] < 170)].any()   # iki aralık
    assert not m[1].any() and not m[2].any()                                          # soluk (S düşük), koyu (V düşük)
    gray = np.full((2, 3, 3), 128, np.uint8)                                          # gri: H = 0, S = 0
    assert not C.hue_thresholding(gray, "custom", 0, 179, 1, 0, "mask").image.any()
    assert C.hue_thresholding(gray, "custom", 0, 179, 0, 0, "mask").image.all()
    seg = rgb_of(C.hue_thresholding(bgr(SPECIAL), "blue"))
    assert seg[0, 5].tolist() == [0, 0, 255] and seg[0, 3].tolist() == [0, 0, 0]      # mavi kalır, kırmızı siyah
    r = C.hue_thresholding(bgr(SPECIAL), "red").report
    assert "wraps around 0" in r.summary() and "[170, 179] ∪ [0, 8]" in r.summary()


def test_hue_custom_ignored_for_presets():
    img = scene()
    a = C.hue_thresholding(bgr(img), "green", 0, 179, 0, 0).image
    b = C.hue_thresholding(bgr(img), "custom", 35, 85, 60, 40).image
    np.testing.assert_array_equal(a, b)


# ================================================================ slayt 62: skin regions
def skin_grid():
    """Cb–Cr düzlemini tarayan sentetik ızgara (sabit Y yok — R, G, B rampaları)."""
    yy, xx = np.mgrid[0:32, 0:40]
    return np.stack([np.clip(80 + 5 * xx, 0, 255), np.clip(40 + 4 * yy, 0, 255), np.clip(30 + 3 * xx + 2 * yy, 0, 255)], -1).astype(np.uint8)


@pytest.mark.parametrize("lims", [(133, 173, 77, 127), (140, 160, 90, 110), (0, 255, 0, 255), (200, 210, 0, 10)])
def test_skin_matches_reference(lims):
    img = np.concatenate([skin_grid(), np.resize(SPECIAL, (2, 40, 3))], 0)
    m = C.skin_regions(bgr(img), *lims, display="mask").image[..., 0]
    np.testing.assert_array_equal(m, ref_skin(img, *lims))


def test_skin_ignores_y_and_known_colors():
    tone = np.array([200, 150, 120], float)                                           # tipik ten rengi
    ycc = C.rgb_to_model(tone[None, None], "ycbcr")[0, 0]
    assert 133 <= ycc[2] <= 173 and 77 <= ycc[1] <= 127
    shades = np.array([[tone * f for f in (0.45, 0.7, 1.0, 1.2)]]).clip(0, 255).astype(np.uint8)
    m = C.skin_regions(bgr(shades), display="mask").image[..., 0]
    assert (m == 255).all()                                                           # parlaklık farklı, hepsi ten
    m2 = C.skin_regions(bgr(SPECIAL), display="mask").image[..., 0]
    assert not m2[0, [0, 1, 2, 4, 5]].any()                                           # siyah, beyaz, gri, yeşil, mavi
    r = C.skin_regions(bgr(shades)).report
    assert "Y not used" in r.rows[0] and "(not used)" in r.rows[-1]


# ================================================================ slayt 63: color edges
@pytest.mark.parametrize("method", C.EDGE_METHODS)
@pytest.mark.parametrize("img", ["patches", "scene", "special", "row", "col"])
def test_edges_match_reference(method, img):
    rgb = {"patches": patches(14, 20), "scene": scene(10, 14), "special": SPECIAL,
           "row": scene(1, 8), "col": scene(8, 1)[:, :1]}[img]
    rgb = np.ascontiguousarray(rgb)
    mag = ref_edges(rgb, method)
    for t in (10, 120, 600):
        e = C.color_edge_detection(bgr(rgb), method, t, "edges").image[..., 0]
        np.testing.assert_array_equal(e, np.where(mag > t, 255, 0))
    m = C.color_edge_detection(bgr(rgb), method, 120, "magnitude").image[..., 0]
    np.testing.assert_array_equal(m, np.array([[c8(v / 4) for v in row] for row in mag], np.uint8))


def test_isoluminant_edge_vanishes_in_grayscale():
    # kırmızı | yeşil, aynı luma (Y = 0.299R + 0.587G ≈ 120): gri gradyan 0, renk yöntemleri kenarı bulur
    iso = np.zeros((6, 8, 3), np.uint8)
    iso[:, :4] = (200, 103, 0)
    iso[:, 4:] = (0, 204, 6)
    y = 0.299 * iso[..., 0] + 0.587 * iso[..., 1] + 0.114 * iso[..., 2]
    assert np.ptp(C.c8(y)) == 0
    for m in C.EDGE_METHODS:
        e = C.color_edge_detection(bgr(iso), m, 100).image[..., 0]
        if m in ("gray", "v"):
            assert not e.any()                                                       # V = max: 200 | 204 → da kaçırır
        else:
            assert e[:, 3:5].all() and not e[:, [0, 7]].any()                         # renk yöntemleri bulur
    vmag = C.edge_gradient(iso.astype(np.float64), "v")[0]
    assert vmag[2, 3] == 16 and vmag[2, 0] == 0                                       # |∇V| = 4·(204 − 200)
    v = C.color_edge_detection(bgr(iso), "v", 10, "magnitude").report
    d = C.color_edge_detection(bgr(iso), "dizenzo", 10).report
    assert "max(R, G, B)" in v.rows[0] and "θ" in d.rows[-1] and "gxx" in d.rows[-1]


def test_di_zenzo_known_values():
    # tek kanal (R) dikey kenar: F = |∇R| (gri kanallar 0) ve RGB-max ile aynı
    im = np.zeros((5, 6, 3), np.uint8)
    im[:, 3:, 0] = 100
    dz = C.edge_gradient(im.astype(np.float64), "dizenzo")[0]
    mx = C.edge_gradient(im.astype(np.float64), "rgb")[0]
    np.testing.assert_allclose(dz, mx, atol=1e-3)
    assert mx[2, 2] == 400 and mx[2, 3] == 400 and mx[2, 0] == 0                       # Sobel: 4·100
    # kanallar aynı yönde: Di Zenzo = √Σ|∇c|² ≥ max
    im2 = np.zeros((5, 6, 3), np.uint8)
    im2[:, 3:] = (100, 100, 100)
    dz2 = C.edge_gradient(im2.astype(np.float64), "dizenzo")[0]
    assert abs(dz2[2, 2] - 400 * math.sqrt(3)) < 1e-2


# ================================================================ slayt 61: k-means
def ref_kmeans(X, K, seed, max_it=30):
    """Laboratuvar C.kmeans, satır satır."""
    n, dim = X.shape
    s = [seed & 0xFFFFFFFF]

    def rnd():
        s[0] = (s[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return s[0] / 4294967296

    def dist2(i, c):
        d = 0.0
        for j in range(dim):
            t = X[i, j] - c[j]
            d += t * t
        return d
    first = math.floor(rnd() * n)
    centers = [list(X[first])]
    while len(centers) < K:
        d = [min(dist2(i, c) for c in centers) for i in range(n)]
        tot = 0.0
        for v in d:
            tot += v
        r, pick = rnd() * tot, 0
        for i in range(n):
            r -= d[i]
            if r <= 0:
                pick = i
                break
        centers.append(list(X[pick]))
    labels = [-1] * n
    cur = [c[:] for c in centers]
    its = 0
    for _ in range(max_it):
        nl, changed = [], 0
        for i in range(n):
            b, bd = 0, math.inf
            for k, c in enumerate(cur):
                dd = dist2(i, c)
                if dd < bd:
                    bd, b = dd, k
            nl.append(b)
            changed += nl[i] != labels[i]
        sums, cnt = [[0.0] * dim for _ in range(K)], [0] * K
        for i in range(n):
            cnt[nl[i]] += 1
            for j in range(dim):
                sums[nl[i]][j] += X[i, j]
        nxt = [[v / cnt[k] for v in sums[k]] if cnt[k] else cur[k][:] for k in range(K)]
        its += 1
        moved = any(abs(nxt[k][j] - cur[k][j]) > 1e-6 for k in range(K) for j in range(dim))
        labels, cur = nl, nxt
        if not changed or not moved:
            break
    return labels, its


@pytest.mark.parametrize("features", C.KMEANS_FEATURES)
@pytest.mark.parametrize("K,seed", [(2, 1), (4, 3), (6, 11)])
def test_kmeans_matches_reference(features, K, seed):
    rgb = np.concatenate([patches(10, 16), scene(4, 16)], 0)
    X = C.kmeans_features(rgb.astype(np.float64), features)
    labels, its = ref_kmeans(X, K, seed)
    km = C.lab_kmeans(X, K, seed)
    assert km["labels"].tolist() == labels and len(km["its"]) == its
    out = rgb_of(C.kmeans_lab(bgr(rgb), K, features, seed)).reshape(-1, 3)
    flat = rgb.reshape(-1, 3).astype(float)
    for k in set(labels):
        idx = [i for i, l in enumerate(labels) if l == k]
        mean = [c8(sum(flat[i, c] for i in idx) / len(idx)) for c in range(3)]
        assert (out[idx] == mean).all()                                               # ortalama RGB ile boyanır


def test_kmeans_separates_patches_and_is_deterministic():
    rgb = patches()
    a = C.kmeans_lab(bgr(rgb), 5, "lab", 3)
    b = C.kmeans_lab(bgr(rgb.copy()), 5, "lab", 3)
    np.testing.assert_array_equal(a.image, b.image)                                   # aynı tohum → aynı sonuç
    colors = {tuple(c) for c in rgb_of(a).reshape(-1, 3).tolist()}
    assert len(colors) <= 5
    assert "converged" in a.report.rows[2] and len([r for r in a.report.rows if r.startswith("cluster")]) == 5
    flat = C.kmeans_lab(np.full((4, 4, 3), 90, np.uint8), 2, "rgb", 1).image          # tek renk: boş küme sorunsuz
    assert (flat == 90).all()


def test_kmeans_ab_ignores_lightness():
    # aynı ton, iki açıklık (koyu / açık kırmızı): a*b* uzayında bir arada, L*a*b*'de ayrı
    rgb = np.zeros((4, 8, 3), np.uint8)
    rgb[:, :4] = (120, 20, 25)
    rgb[:, 4:] = (240, 60, 70)
    rgb[0, 0] = (20, 120, 230)                                                        # farklı renkte tek piksel
    lab2 = rgb_of(C.kmeans_lab(bgr(rgb), 2, "lab", 2))
    ab2 = rgb_of(C.kmeans_lab(bgr(rgb), 2, "ab", 2))
    assert len({tuple(c) for c in ab2[:, 1:].reshape(-1, 3).tolist()}) == 1           # koyu + açık kırmızı birlikte
    assert len({tuple(c) for c in lab2[1:, :].reshape(-1, 3).tolist()}) == 2


# ================================================================ ortak: düzen, tip, alfa, sınırlar
CALLS = [("color_hue_threshold", {}), ("color_hue_threshold", {"target": "red", "display": "mask"}),
         ("color_kmeans", {"k": 2}), ("color_kmeans", {"features": "ab"}), ("color_skin", {}),
         ("color_skin", {"display": "mask"}), ("color_edges", {}), ("color_edges", {"method": "dizenzo", "display": "magnitude"})]


@pytest.mark.parametrize("key,params", CALLS, ids=[f"{k}-{i}" for i, (k, _) in enumerate(CALLS)])
@pytest.mark.parametrize("shape", [(1, 6), (6, 1), (2, 3), (7, 9)])
def test_layouts_sizes_alpha(key, params, shape):
    algo = REGISTRY.get(key)
    rng = np.random.default_rng(shape[0] + 7 * shape[1])
    color = rng.integers(0, 256, shape + (3,), dtype=np.uint8)
    out = algo.execute(color, params)
    assert out.image.shape == shape + (3,) and out.image.dtype == np.uint8 and out.changes_image
    alpha = rng.integers(0, 256, shape + (1,), dtype=np.uint8)
    res = algo.execute(np.concatenate([color, alpha], -1), params).image
    np.testing.assert_array_equal(res[..., 3], alpha[..., 0])
    np.testing.assert_array_equal(res[..., :3], out.image)
    gray = color[..., 0]
    np.testing.assert_array_equal(algo.execute(gray, params).image, algo.execute(np.repeat(gray[..., None], 3, -1), params).image)


def test_one_pixel_and_bgr_order():
    px = np.array([[[255, 0, 0]]], np.uint8)                                          # BGR: saf mavi
    assert C.hue_thresholding(px, "blue").image[0, 0].tolist() == [255, 0, 0]
    assert C.hue_thresholding(px[..., ::-1].copy(), "blue").image[0, 0].tolist() == [0, 0, 0]   # kırmızı → dışarıda
    assert C.color_edge_detection(px).image.tolist() == [[[0, 0, 0]]]
    with pytest.raises(AlgorithmParameterError):
        C.kmeans_lab(px, 2)                                                           # K > piksel sayısı


@pytest.mark.parametrize("call", [
    lambda im: C.hue_thresholding(im, "purple"), lambda im: C.hue_thresholding(im, "custom", 180),
    lambda im: C.hue_thresholding(im, "custom", 0, 179, 256), lambda im: C.hue_thresholding(im, display="overlay"),
    lambda im: C.kmeans_lab(im, 1), lambda im: C.kmeans_lab(im, 7), lambda im: C.kmeans_lab(im, 3, "hsv"),
    lambda im: C.kmeans_lab(im, 3, "lab", 0), lambda im: C.kmeans_lab(im, 3.5),
    lambda im: C.skin_regions(im, -1), lambda im: C.skin_regions(im, cb_max=300), lambda im: C.skin_regions(im, display="x"),
    lambda im: C.color_edge_detection(im, "canny"), lambda im: C.color_edge_detection(im, "gray", 5),
    lambda im: C.color_edge_detection(im, "gray", 601), lambda im: C.color_edge_detection(im, display="both"),
])
def test_bad_parameters(call):
    with pytest.raises(AlgorithmParameterError):
        call(scene(4, 4))
