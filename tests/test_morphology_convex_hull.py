"""Convex hull (Chapter 4, slayt 63–65; Gonzalez Şekil 9.21).

Bağımsız referanslar OpenCV kullanmaz:
* ``ref_hull_contour`` — her 8-bağlantılı nesnenin piksel merkezlerinden monoton zincir
  (Andrew) ile dışbükey zarf; merkezi kapalı poligonda olan pikseller (saf Python).
* ``ref_hull_morph``   — Şekil 9.21: nesne kutusunda X_k = (X_{k−1} ⊛ Bⁱ) ∪ X_{k−1}, öteleme tabanlı HMT ile.
Ayrıca ``assert_previous_operations_unchanged`` önceki 12 işlemin regresyon kontrolüdür
(thinning / thickening test dosyaları da kullanır).
"""

import cv2
import numpy as np
import pytest

from app.main_window import MainWindow
from processing import morphology as M
from processing.analysis import histograms
from processing.catalog import build_registry
from processing.image_io import save_image
from processing.registry import AlgorithmParameterError
from tests.test_morphology import random_binary, ref_dilate, ref_erode, ref_hmt
from tests.test_morphology_connected_components import ref_label
from tests.test_morphology_hat import ref_black_hat, ref_top_hat
from tests.test_morphology_hole_filling import ref_flood

# --------------------------------------------------------------------------
# Referanslar
# --------------------------------------------------------------------------


def monotone_chain(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def in_closed_polygon(px, py, poly):
    if len(poly) == 1:
        return (px, py) == poly[0]
    if len(poly) == 2:
        (ax, ay), (bx, by) = poly
        return ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) == 0 and \
            min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)
    signs = set()
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
        c = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if c:
            signs.add(c > 0)
    return len(signs) <= 1


def ref_hull_contour(a):
    labels = ref_label(a, 8)
    out = a.copy()
    for k in range(1, labels.max() + 1):
        ys, xs = np.nonzero(labels == k)
        poly = monotone_chain(list(zip(xs.tolist(), ys.tolist())))
        for y in range(ys.min(), ys.max() + 1):
            for x in range(xs.min(), xs.max() + 1):
                if in_closed_polygon(x, y, poly):
                    out[y, x] = True
    return out


def ref_hull_morph(a):
    labels = ref_label(a, 8)
    out = a.copy()
    for k in range(1, labels.max() + 1):
        ys, xs = np.nonzero(labels == k)
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        obj = labels[sl] == k
        hull = np.zeros_like(obj)
        for kern in M.HULL_KERNELS:
            x = obj.copy()
            while True:
                nxt = x | ref_hmt(x, kern)
                if np.array_equal(nxt, x):
                    break
                x = nxt
            hull |= x
        out[sl] |= hull
    return out


def grid(rows):
    return np.array([[ch == "#" for ch in row] for row in rows], bool)


def u8(mask):
    return np.where(mask, 255, 0).astype(np.uint8)


def assert_previous_operations_unchanged(seed=93):
    """Önceki 12 morfoloji işlemi, bağımsız referanslarla birebir (registry üzerinden)."""
    registry = build_registry()
    img = u8(random_binary(seed, (22, 26), 0.55))
    a, b = img > 127, M.structuring_element()
    pad = 10

    def big(fn):
        return fn(np.pad(a, pad))[pad:-pad, pad:-pad]

    expected = {
        "erode": ref_erode(a, b.mask, b.anchor),
        "dilate": ref_dilate(a, b.mask, b.anchor),
        "opening": big(lambda x: ref_dilate(ref_erode(x, b.mask, b.anchor), b.mask, b.anchor)),
        "closing": big(lambda x: ref_erode(ref_dilate(x, b.mask, b.anchor), b.mask, b.anchor)),
        "gradient": ref_dilate(a, b.mask, b.anchor) & ~ref_erode(a, b.mask, b.anchor),
        "hit_or_miss": ref_hmt(a, M.hmt_kernel("corner")),
        "boundary": a & ~ref_erode(a, b.mask, b.anchor),
        "hole_filling": ref_flood(a),
    }
    for key, exp in expected.items():
        out = registry.get(key).execute(img)
        assert out.report is None, key
        np.testing.assert_array_equal(out.image, u8(exp), err_msg=key)
    cc = registry.get("connected_components").execute(img)
    np.testing.assert_array_equal(cc.report.labels, ref_label(a, 8))
    gray = np.random.default_rng(seed).integers(0, 256, (22, 26), dtype=np.uint8)
    b15 = M.structuring_element("rect", 15, 15)
    np.testing.assert_array_equal(registry.get("top_hat").run(gray), ref_top_hat(gray, b15))
    np.testing.assert_array_equal(registry.get("black_hat").run(gray), ref_black_hat(gray, b15))
    np.testing.assert_array_equal(registry.get("hat_correction").run(gray),
                                  np.clip(gray + ref_top_hat(gray, b15) - ref_black_hat(gray, b15), 0, 255))


# --------------------------------------------------------------------------
# Elle kurulmuş durumlar
# --------------------------------------------------------------------------

L_SHAPE = grid([".........",
                ".##......",
                ".##......",
                ".##......",
                ".##......",
                ".##......",
                ".#######.",
                ".#######.",
                "........."])
L_HULL = grid([".........",
               ".##......",
               ".###.....",
               ".####....",
               ".#####...",
               ".######..",
               ".#######.",
               ".#######.",
               "........."])


@pytest.mark.parametrize("method", M.HULL_METHODS)
def test_l_shape_hull(method):
    hull, _, n = M.convex_hull_mask(L_SHAPE, method)
    assert n == 1
    np.testing.assert_array_equal(hull, L_HULL)


def test_contour_hull_vertices_are_object_pixels():
    _, hulls, _ = M.convex_hull_mask(L_SHAPE, "contour")
    [h] = hulls
    assert all(L_SHAPE[y, x] for x, y in h)
    assert {tuple(p) for p in h} == {(1, 1), (2, 1), (7, 6), (7, 7), (1, 7)}


def test_convex_object_is_unchanged():
    rect = np.zeros((8, 10), bool)
    rect[2:6, 3:8] = True
    for method in M.HULL_METHODS:
        np.testing.assert_array_equal(M.convex_hull_mask(rect, method)[0], rect)


def test_degenerate_hulls_point_and_lines():
    a = np.zeros((10, 12), bool)
    a[1, 1] = True                                   # tek piksel
    a[3, 2:9] = True                                 # yatay çizgi
    for i in range(5):
        a[5 + i, 1 + i] = True                       # köşegen çizgi (8-bağlantılı)
    hull, hulls, n = M.convex_hull_mask(a, "contour")
    assert n == 3
    np.testing.assert_array_equal(hull, a)           # zarf = nesnenin kendisi
    np.testing.assert_array_equal(hull, ref_hull_contour(a))


def test_ring_hole_does_not_change_hull_external_contours_only():
    ring = np.zeros((25, 25), np.uint8)
    cv2.circle(ring, (12, 12), 10, 1, -1)
    cv2.circle(ring, (12, 12), 5, 0, -1)
    a = ring > 0
    hull, hulls, n = M.convex_hull_mask(a, "contour")
    assert n == 1 and len(hulls) == 1                # iç kontur (delik) ayrı zarf üretmez
    assert hull[12, 12] and hull[8:17, 8:17].all()    # delik zarfın içinde
    np.testing.assert_array_equal(hull, ref_hull_contour(a))
    filled = ref_flood(a)
    assert not (filled & ~hull).any()                 # dolu disk ⊆ zarf


def test_object_nested_in_a_hole_is_inside_the_outer_hull():
    a = np.zeros((25, 25), bool)
    a[2:23, 2:23] = True
    a[6:19, 6:19] = False                             # büyük delik
    a[11:14, 11:14] = True                            # delikteki iç nesne (dış kontur değil)
    hull, hulls, n = M.convex_hull_mask(a, "contour")
    assert n == 1                                     # RETR_EXTERNAL: yalnızca en dıştaki
    assert hull[2:23, 2:23].all()
    np.testing.assert_array_equal(hull, ref_hull_contour(a))


def test_multiple_objects_get_separate_hulls():
    a = grid(["..............",
              ".#....#.......",
              ".#....#.......",
              ".######.......",
              "..............",
              "........#.....",
              ".......###....",
              "......#####...",
              ".......#.#....",
              ".............."])
    for method in M.HULL_METHODS:
        hull, _, n = M.convex_hull_mask(a, method)
        assert n == 2
        assert hull[1:3, 2:6].all()                   # U'nun içi doldu
        assert not hull[4].any()                      # nesneler arasındaki boşluk korunur
    np.testing.assert_array_equal(M.convex_hull_mask(a, "contour")[0], ref_hull_contour(a))
    np.testing.assert_array_equal(M.convex_hull_mask(a, "morph")[0], ref_hull_morph(a))


def test_morph_hull_is_limited_to_bounding_box():
    a = grid([".......",
              ".#.....",
              ".......",
              ".....#.",
              "......."])
    a[1:4, 1] = True
    a[3, 1:6] = True
    hull = M.convex_hull_mask(a, "morph")[0]
    ys, xs = np.nonzero(a)
    assert not hull[:ys.min()].any() and not hull[ys.max() + 1:].any()
    assert not hull[:, :xs.min()].any() and not hull[:, xs.max() + 1:].any()


def test_objects_touching_the_border():
    a = np.zeros((10, 10), bool)
    a[0, 0:6] = True
    a[0:6, 0] = True
    for method in M.HULL_METHODS:
        hull = M.convex_hull_mask(a, method)[0]
        assert hull[0:3, 0:3].all() and not hull[9, 9]
    np.testing.assert_array_equal(M.convex_hull_mask(a, "contour")[0], ref_hull_contour(a))


def test_empty_and_full():
    for method in M.HULL_METHODS:
        h, hulls, n = M.convex_hull_mask(np.zeros((5, 6), bool), method)
        assert n == 0 and not h.any()
        h, _, n = M.convex_hull_mask(np.ones((5, 6), bool), method)
        assert n == 1 and h.all()


# --------------------------------------------------------------------------
# Bağımsız referans ve özellikler
# --------------------------------------------------------------------------


def blobs(seed, shape=(28, 34)):
    """Rastgele ama nesne biçimli (yumuşatılmış) ikili görüntü."""
    rng = np.random.default_rng(seed)
    f = cv2.GaussianBlur(rng.random(shape), (0, 0), 1.6)
    return f > np.quantile(f, 0.6)


@pytest.mark.parametrize("seed", range(8))
def test_contour_method_matches_monotone_chain_reference(seed):
    a = blobs(seed)
    np.testing.assert_array_equal(M.convex_hull_mask(a, "contour")[0], ref_hull_contour(a))


@pytest.mark.parametrize("seed", range(6))
def test_morph_method_matches_figure_9_21_reference(seed):
    a = blobs(50 + seed, (22, 26))
    np.testing.assert_array_equal(M.convex_hull_mask(a, "morph")[0], ref_hull_morph(a))


@pytest.mark.parametrize("seed", range(6))
def test_hull_properties(seed):
    a = blobs(100 + seed)
    hull = M.convex_hull_mask(a, "contour")[0]
    assert not (a & ~hull).any()                                        # A ⊆ C(A)
    assert not (a & ~M.convex_hull_mask(a, "morph")[0]).any()
    # Nesne başına: zarf idempotent ve her satır/sütunda kesintisiz (dijital dışbükeylik).
    # (Farklı nesnelerin zarfları çakışabilir; birleşimleri dışbükey olmak zorunda değildir.)
    labels = ref_label(a, 8)
    for k in range(1, labels.max() + 1):
        h = M.convex_hull_mask(labels == k, "contour")[0]
        np.testing.assert_array_equal(M.convex_hull_mask(h, "contour")[0], h)
        for line in list(h) + list(h.T):
            idx = np.flatnonzero(line)
            if idx.size:
                assert idx[-1] - idx[0] + 1 == idx.size


def test_overlapping_hulls_of_different_objects_merge():
    """İki L biçimi iç içe geçince zarfları çakışır; sonuç zarfların birleşimidir (dışbükey değildir)."""
    a = np.zeros((14, 14), bool)
    a[1:12, 1] = True
    a[11, 1:9] = True                     # L: zarfı (1,1)–(1,11)–(8,11) üçgeni
    a[8, 3:11] = True                     # L'ye dokunmayan çubuk; L'nin zarfına girip dışına taşar
    hull, _, n = M.convex_hull_mask(a, "contour")
    assert n == 2
    labels = ref_label(a, 8)
    parts = [M.convex_hull_mask(labels == k, "contour")[0] for k in (1, 2)]
    np.testing.assert_array_equal(hull, parts[0] | parts[1])
    assert (parts[0] & parts[1]).any()


def test_morph_approximation_contains_exact_hull_on_diagonal_edges():
    """Şekil 9.21 yaklaşımı, köşegen kenarlarda gerçek zarftan fazlasını doldurabilir (kutuya doğru)."""
    tri = np.zeros((12, 12), bool)
    tri[2, 2:10] = True
    tri[2:10, 2] = True
    for i in range(8):
        tri[2 + i, 2 + (7 - i)] = True
    exact = M.convex_hull_mask(tri, "contour")[0]
    morph = M.convex_hull_mask(tri, "morph")[0]
    assert not (exact & ~morph).any()


# --------------------------------------------------------------------------
# Görüntü düzeyi
# --------------------------------------------------------------------------


def test_filled_display_and_report():
    res = M.convex_hull(u8(L_SHAPE), binarize="manual")
    np.testing.assert_array_equal(res.image, u8(L_HULL))
    assert res.report.objects == 1 and res.report.object_area == L_SHAPE.sum()
    assert res.report.hull_area == L_HULL.sum()
    assert res.report.summary() == f"1 object, convex hull {L_HULL.sum()} px (solidity {L_SHAPE.sum() / L_HULL.sum():.2f})"


def test_overlay_display_colors():
    res = M.convex_hull(u8(L_SHAPE), binarize="manual", display="overlay")
    img = res.image
    assert img.shape == L_SHAPE.shape + (3,)
    assert not img[~L_HULL].any()                                       # arka plan siyah
    added = L_HULL & ~L_SHAPE
    colors = {tuple(c) for c in img[added]}
    assert colors <= {M.HULL_COLORS["added"], M.HULL_COLORS["hull"]}
    assert tuple(img[3, 4]) == M.HULL_COLORS["hull"]                    # zarfın köşegen kenarı sarı
    assert tuple(img[3, 3]) == M.HULL_COLORS["added"]                   # zarfın eklediği iç alan
    assert tuple(img[6, 2]) in (M.HULL_COLORS["object"], M.HULL_COLORS["contour"], M.HULL_COLORS["hull"])
    assert any(tuple(c) == M.HULL_COLORS["contour"] for c in img[L_SHAPE])


def test_threshold_polarity_and_color_input():
    img = np.where(L_SHAPE, 30, 220).astype(np.uint8)                   # koyu nesne
    np.testing.assert_array_equal(M.convex_hull(img, foreground="dark").image, u8(L_HULL))
    bgr = np.zeros(L_SHAPE.shape + (3,), np.uint8)
    bgr[L_SHAPE] = (0, 255, 0)
    bgr[~L_SHAPE] = (255, 0, 0)
    out = M.convex_hull(bgr, binarize="manual", thresh=100).image
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], u8(L_HULL))


@pytest.mark.parametrize("display", M.HULL_DISPLAYS)
@pytest.mark.parametrize("shape", [(9, 9), (9, 9, 1), (9, 9, 3), (9, 9, 4)])
def test_layout_alpha_immutability(display, shape):
    base = u8(L_SHAPE)
    img = base if len(shape) == 2 else np.repeat(base[..., None], shape[2], axis=2).copy()
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = np.arange(81, dtype=np.uint8).reshape(9, 9)
    before = img.copy()
    out = M.convex_hull(img, binarize="manual", display=display).image
    np.testing.assert_array_equal(img, before)
    assert out.dtype == np.uint8 and not np.shares_memory(out, img)
    if display == "filled":
        assert out.shape == img.shape
    else:
        assert out.shape == (9, 9, 4 if (len(shape) == 3 and shape[2] == 4) else 3)
    if len(shape) == 3 and shape[2] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])


@pytest.mark.parametrize("kwargs", [dict(method="graham"), dict(display="lines"), dict(binarize="auto"),
                                    dict(foreground="white"), dict(binarize="manual", thresh=300)])
def test_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.convex_hull(np.zeros((5, 5), np.uint8), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint8), [[0, 1]]])
def test_invalid_images(bad):
    with pytest.raises(AlgorithmParameterError):
        M.convex_hull(bad)


def test_registry_entry_and_previous_operations():
    registry = build_registry()
    algo = registry.get("convex_hull")
    assert algo.title == "Convex Hull" and algo.implemented
    assert [p.key for p in algo.params] == ["binarize", "thresh", "foreground", "method", "display"]
    assert algo.default_params()["method"] == "contour" and algo.default_params()["display"] == "filled"
    assert "∪ Dⁱ" in algo.summary
    assert_previous_operations_unchanged()


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    img = np.repeat(np.where(np.kron(L_SHAPE, np.ones((4, 4), bool)), 220, 30).astype(np.uint8)[..., None], 3, axis=2)
    path = tmp_path / "Zarf" / "L_şekli.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


@pytest.mark.parametrize("method", M.HULL_METHODS)
def test_ui_apply(window, method):
    w, img = window
    w.navigator.select_algorithm("convex_hull")
    w.parameters.controls["method"].set_value(method)
    w.parameters.apply_button.click()
    a = img[..., 0] > 127
    exp = ref_hull_contour(a) if method == "contour" else ref_hull_morph(a)
    np.testing.assert_array_equal(w.current_image[..., 0], u8(exp))
    np.testing.assert_array_equal(w.original_image, img)
    assert w.telemetry.state.state() == "DONE"
    assert "convex hull" in w.statusBar().currentMessage() and "solidity" in w.statusBar().currentMessage()
    assert not w.parameters.report_view.isVisible()          # bileşen tablosu yalnızca CC için
    for name, counts in histograms(w.current_image).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
