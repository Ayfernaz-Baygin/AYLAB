"""Bağlantılı bileşenler (Chapter 4, slayt 60–62).

Bağımsız referanslar OpenCV ve modülün kendi kodunu kullanmaz:
* ``ref_label`` — saf Python BFS; tohum, etiketlenmemiş ilk (raster) ön plan pikseli
  (dersin X_k = (X_{k−1} ⊕ B) ∩ A yönteminin seçtiği sırayla aynı).
* ``ref_stats`` — alan, sınırlayıcı kutu ve ağırlık merkezi döngüyle hesaplanır.
Küçük matrislerde etiket ve istatistikler ayrıca elle yazılmıştır.
"""

from collections import deque

import numpy as np
import pytest

from app.main_window import MainWindow
from processing import morphology as M
from processing.analysis import histograms
from processing.catalog import build_registry
from processing.image_io import save_image
from processing.registry import Algorithm, AlgorithmParameterError, AlgorithmRegistry, Category, ProcessResult
from tests.test_morphology import random_binary, ref_dilate, ref_erode, ref_hmt
from tests.test_morphology_hat import ref_black_hat, ref_top_hat
from tests.test_morphology_hole_filling import ref_flood

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# --------------------------------------------------------------------------
# Referanslar
# --------------------------------------------------------------------------


def ref_label(a, connectivity):
    neigh = N4 if connectivity == 4 else N8
    h, w = a.shape
    labels = np.zeros((h, w), np.int64)
    nxt = 1
    for r in range(h):
        for c in range(w):
            if a[r, c] and labels[r, c] == 0:
                labels[r, c] = nxt
                q = deque([(r, c)])
                while q:
                    y, x = q.popleft()
                    for dy, dx in neigh:
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and a[yy, xx] and labels[yy, xx] == 0:
                            labels[yy, xx] = nxt
                            q.append((yy, xx))
                nxt += 1
    return labels


def ref_stats(labels):
    out = {}
    for r in range(labels.shape[0]):
        for c in range(labels.shape[1]):
            k = int(labels[r, c])
            if k:
                s = out.setdefault(k, dict(n=0, xs=0, ys=0, x0=c, x1=c, y0=r, y1=r))
                s["n"] += 1
                s["xs"] += c
                s["ys"] += r
                s["x0"], s["x1"] = min(s["x0"], c), max(s["x1"], c)
                s["y0"], s["y1"] = min(s["y0"], r), max(s["y1"], r)
    return [(k, s["n"], s["x0"], s["y0"], s["x1"] - s["x0"] + 1, s["y1"] - s["y0"] + 1, s["xs"] / s["n"], s["ys"] / s["n"])
            for k, s in sorted(out.items())]


def as_tuples(report):
    return [(c.label, c.area, c.x, c.y, c.width, c.height, c.cx, c.cy) for c in report.components]


def assert_stats_equal(report, expected):
    got = as_tuples(report)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g[:6] == e[:6]
        assert g[6] == pytest.approx(e[6]) and g[7] == pytest.approx(e[7])


def grid(rows):
    return np.array([[ch == "#" for ch in row] for row in rows], bool)


def u8(mask):
    return np.where(mask, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Elle hesaplanmış küçük matrisler
# --------------------------------------------------------------------------


def test_single_component():
    a = grid(["......",
              ".###..",
              ".#....",
              ".####.",
              "......"])
    r = M.label_components(a, 8)
    assert r.count == 1
    np.testing.assert_array_equal(r.labels, a.astype(int))
    # alan 8; kutu x 1..4, y 1..3; merkez: x (1+2+3+1+1+2+3+4)/8 = 17/8, y (1·3 + 2·1 + 3·4)/8 = 17/8
    assert_stats_equal(r, [(1, 8, 1, 1, 4, 3, 17 / 8, 17 / 8)])


def test_three_separate_components_exact_labels():
    a = grid(["##....#",
              "##....#",
              ".......",
              "..###..",
              "..###.."])
    expected = np.array([[1, 1, 0, 0, 0, 0, 2],
                         [1, 1, 0, 0, 0, 0, 2],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 3, 3, 3, 0, 0],
                         [0, 0, 3, 3, 3, 0, 0]])
    for conn in (4, 8):
        r = M.label_components(a, conn)
        np.testing.assert_array_equal(r.labels, expected)
        assert r.count == 3
        assert_stats_equal(r, [(1, 4, 0, 0, 2, 2, 0.5, 0.5), (2, 2, 6, 0, 1, 2, 6.0, 0.5),
                               (3, 6, 2, 3, 3, 2, 3.0, 3.5)])


def test_diagonal_contact_four_vs_eight():
    a = grid(["#...",
              ".#..",
              "..#.",
              "..##"])
    r8, r4 = M.label_components(a, 8), M.label_components(a, 4)
    assert r8.count == 1 and r4.count == 3
    np.testing.assert_array_equal(r4.labels, [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 3, 3]])
    np.testing.assert_array_equal(r8.labels, a.astype(int))
    assert_stats_equal(r4, [(1, 1, 0, 0, 1, 1, 0.0, 0.0), (2, 1, 1, 1, 1, 1, 1.0, 1.0),
                            (3, 3, 2, 2, 2, 2, 7 / 3, 8 / 3)])


def test_single_pixel_components_and_areas():
    a = np.zeros((9, 12), bool)
    a[1:8, 1:7] = True        # büyük: 7 × 6 = 42
    a[2, 9] = True            # tek piksel
    a[6:8, 9:11] = True       # küçük: 2 × 2
    r = M.label_components(a, 8)
    assert [c.area for c in r.components] == [42, 1, 4]
    big, dot, small = r.components
    assert (big.x, big.y, big.width, big.height, big.cx, big.cy) == (1, 1, 6, 7, 3.5, 4.0)
    assert (dot.x, dot.y, dot.width, dot.height, dot.cx, dot.cy) == (9, 2, 1, 1, 9.0, 2.0)
    assert (small.x, small.y, small.width, small.height, small.cx, small.cy) == (9, 6, 2, 2, 9.5, 6.5)
    assert sum(c.area for c in r.components) == a.sum()


def test_centroid_can_lie_outside_the_component():
    """U biçimli nesne: ağırlık merkezi nesnenin dışındaki boşluğa düşer (doğru tanım)."""
    a = grid(["#...#",
              "#...#",
              "#####"])
    [c] = M.label_components(a, 4).components
    assert c.cx == pytest.approx(2.0) and c.cy == pytest.approx((0 * 2 + 1 * 2 + 2 * 5) / 9)
    assert not a[round(c.cy), round(c.cx)]


def test_empty_foreground():
    r = M.label_components(np.zeros((5, 6), bool), 8)
    assert r.count == 0 and not r.labels.any() and r.components == ()
    res = M.connected_components(np.zeros((5, 6), np.uint8), binarize="manual")
    assert res.report.count == 0 and not res.image.any()


def test_full_foreground():
    for conn in (4, 8):
        r = M.label_components(np.ones((4, 7), bool), conn)
        assert r.count == 1 and (r.labels == 1).all()
        assert_stats_equal(r, [(1, 28, 0, 0, 7, 4, 3.0, 1.5)])


def test_components_touching_the_border():
    a = grid(["##..#",
              "#...#",
              ".....",
              "#..##"])
    r = M.label_components(a, 8)
    assert r.count == 4
    assert_stats_equal(r, ref_stats(ref_label(a, 8)))
    assert [(c.x, c.y) for c in r.components] == [(0, 0), (4, 0), (0, 3), (3, 3)]


def test_background_is_zero_and_labels_are_consecutive():
    a = random_binary(3, (30, 30), 0.4)
    r = M.label_components(a, 4)
    assert (r.labels[~a] == 0).all() and (r.labels[a] > 0).all()
    assert set(np.unique(r.labels)) == set(range(r.count + 1))
    assert r.labels.dtype == np.int32


# --------------------------------------------------------------------------
# Dersin iteratif yöntemi ve bağımsız referans
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("conn", [4, 8])
def test_opencv_matches_bfs_reference_and_course_iteration(seed, conn):
    a = random_binary(500 + seed, (21, 27), 0.45)
    expected = ref_label(a, conn)
    r = M.label_components(a, conn)
    np.testing.assert_array_equal(r.labels, expected)
    np.testing.assert_array_equal(M.label_components_iterative(a, conn), expected)
    assert_stats_equal(r, ref_stats(expected))


def test_course_iteration_step_by_step():
    """X_k = (X_{k−1} ⊕ B) ∩ A: artı B ile bir adımda yalnızca 4-komşular eklenir."""
    a = grid(["###",
              "###",
              "###"])
    x = np.zeros_like(a)
    x[0, 0] = True
    se = M.structuring_element("cross", 3, 3)
    x1 = M.binary_dilation(x, se) & a
    np.testing.assert_array_equal(x1, grid(["##.", "#..", "..."]))
    x1_sq = M.binary_dilation(x, M.structuring_element("rect", 3, 3)) & a
    np.testing.assert_array_equal(x1_sq, grid(["##.", "##.", "..."]))


def test_iterative_on_large_component_and_border():
    a = np.zeros((40, 60), bool)
    a[:, 0] = True
    a[39, :] = True
    a[5:30, 10:50] = True
    a[20, 1:10] = True
    np.testing.assert_array_equal(M.label_components_iterative(a, 4), ref_label(a, 4))


# --------------------------------------------------------------------------
# Görselleştirme: etiket matrisi ayrı, renkler ayırt edilebilir
# --------------------------------------------------------------------------


def test_more_than_255_components_keep_distinct_labels():
    a = np.zeros((44, 44), bool)
    a[::2, ::2] = True                         # 22 × 22 = 484 tek piksellik bileşen (4-bağlantı)
    res = M.connected_components(u8(a), binarize="manual", connectivity="4")
    r = res.report
    assert r.count == 484 and r.labels.max() == 484 and r.labels.dtype == np.int32
    assert len(np.unique(r.labels[a])) == 484      # 8-bit sarma olsaydı 256 ve 0 karışırdı
    assert (res.image[a].max(axis=1) > 0).all()    # hiçbir bileşen (örn. etiket 256) siyah değil
    assert not res.image[~a].any()


def test_palette():
    pal = M.label_colors(600)
    assert pal.shape == (601, 3) and pal.dtype == np.uint8
    assert not pal[0].any()                                        # arka plan siyah
    assert (pal[1:].max(axis=1) >= 150).all()                      # bileşenler belirgin
    diff = np.abs(pal[1:-1].astype(int) - pal[2:].astype(int)).sum(axis=1)
    assert diff.min() > 60                                         # ardışık etiketler farklı renk
    assert len({tuple(c) for c in pal[1:21]}) == 20


def test_colorized_image_matches_labels():
    a = random_binary(8, (20, 25), 0.5)
    res = M.connected_components(u8(a), binarize="manual")
    pal = M.label_colors(res.report.count)
    np.testing.assert_array_equal(res.image, pal[res.report.labels])


def test_boxes_display_draws_bbox_outside_and_centroid():
    a = np.zeros((20, 30), bool)
    a[3:10, 4:15] = True
    res = M.connected_components(u8(a), binarize="manual", display="boxes")
    [c] = res.report.components
    img = res.image
    # kutu nesnenin 1 piksel dışında: köşeler (x−1, y−1) ve (x+w, y+h)
    assert (img[c.y - 1, c.x - 1] == 255).all() and (img[c.y + c.height, c.x + c.width] == 255).all()
    assert (img[round(c.cy), round(c.cx)] == 255).all()                 # merkez işareti
    plain = M.connected_components(u8(a), binarize="manual").image
    np.testing.assert_array_equal(img[c.y, c.x], plain[c.y, c.x])        # nesne köşesi örtülmez
    np.testing.assert_array_equal(res.report.labels, M.connected_components(u8(a), binarize="manual").report.labels)


def test_boxes_skip_tiny_components():
    a = np.zeros((12, 12), bool)
    a[2, 2] = True                      # tek piksel
    a[6, 3:9] = True                    # 6 × 1 ince çizgi
    a[8:11, 8:11] = True                # 3 × 3 → kutu çizilir
    boxed = M.connected_components(u8(a), binarize="manual", display="boxes").image
    plain = M.connected_components(u8(a), binarize="manual").image
    np.testing.assert_array_equal(boxed[:7], plain[:7])                 # küçükler aynen, renkleri görünür
    assert (boxed[7, 7] == 255).all() and (boxed[9, 9] == 255).all()


# --------------------------------------------------------------------------
# Görüntü düzeyi: eşikleme, kutupluluk, renk, alfa, değişmezlik
# --------------------------------------------------------------------------

THREE = grid(["##....#",
              "##....#",
              ".......",
              "..###..",
              "..###.."])


def test_threshold_and_polarity():
    img = np.where(THREE, 200, 40).astype(np.uint8)
    for kwargs in (dict(binarize="otsu"), dict(binarize="manual", thresh=127)):
        assert M.connected_components(img, **kwargs).report.count == 3
    assert M.connected_components(img, binarize="manual", thresh=250).report.count == 0
    dark = M.connected_components(255 - img, foreground="dark").report
    assert dark.count == 3 and np.array_equal(dark.labels, ref_label(THREE, 8))
    bright_bg = M.connected_components(255 - img, foreground="bright").report     # zemin tek bileşen
    assert bright_bg.count == 1 and bright_bg.components[0].area == (~THREE).sum()


def test_bgr_luminance():
    img = np.zeros(THREE.shape + (3,), np.uint8)
    img[THREE] = (0, 255, 0)      # yeşil Y≈150
    img[~THREE] = (255, 0, 0)     # mavi Y≈29
    r = M.connected_components(img, binarize="manual", thresh=100).report
    np.testing.assert_array_equal(r.labels, ref_label(THREE, 8))


@pytest.mark.parametrize("shape", [(10, 12), (10, 12, 1), (10, 12, 3), (10, 12, 4)])
def test_layout_alpha_and_immutability(shape):
    rng = np.random.default_rng(1)
    mask = random_binary(6, shape[:2], 0.5)
    base = u8(mask)
    img = base if len(shape) == 2 else np.repeat(base[..., None], shape[2], axis=2).copy()
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = rng.integers(0, 256, shape[:2])
    before = img.copy()
    res = M.connected_components(img, binarize="manual")
    np.testing.assert_array_equal(img, before)
    out = res.image
    assert out.dtype == np.uint8 and not np.shares_memory(out, img)
    expected_channels = 4 if len(shape) == 3 and shape[2] == 4 else 3     # etiketler renkle gösterilir
    assert out.shape == shape[:2] + (expected_channels,)
    if expected_channels == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])
    np.testing.assert_array_equal(res.report.labels, ref_label(mask, 8))


@pytest.mark.parametrize("kwargs", [dict(connectivity="6"), dict(connectivity=True), dict(display="ids"),
                                    dict(binarize="auto"), dict(foreground="white"),
                                    dict(binarize="manual", thresh=999)])
def test_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.connected_components(np.zeros((5, 5), np.uint8), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint8),
                                 np.zeros((3, 3, 2), np.uint8), [[0, 1]]])
def test_invalid_images(bad):
    with pytest.raises(AlgorithmParameterError):
        M.connected_components(bad)


def test_label_functions_validate():
    for conn in (6, "x", None):
        with pytest.raises(AlgorithmParameterError):
            M.label_components(np.zeros((3, 3), bool), conn)
    with pytest.raises(AlgorithmParameterError):
        M.label_components_iterative(np.zeros((3, 3, 3), bool), 4)
    assert M.label_components(np.zeros((3, 3), bool), "4").connectivity == 4


# --------------------------------------------------------------------------
# Registry: rapor kanalı ve katalog
# --------------------------------------------------------------------------


def test_execute_returns_report_and_run_returns_image():
    algo = build_registry().get("connected_components")
    img = u8(THREE)
    outcome = algo.execute(img)
    assert isinstance(outcome, ProcessResult) and outcome.report.count == 3
    np.testing.assert_array_equal(algo.run(img), outcome.image)
    assert "3 components · 8-connected (B = 3×3 square)" == outcome.report.summary()
    assert algo.execute(img, {"connectivity": "4"}).report.summary().endswith("4-connected (B = 3×3 cross)")


def test_execute_wraps_plain_image_results():
    reg = AlgorithmRegistry()
    reg.add_category(Category("c", 1, "C", "", 500))
    reg.add(Algorithm("inv", "Invert", "c", ""))
    reg.bind("inv", lambda image: 255 - image)
    outcome = reg.get("inv").execute(np.zeros((2, 2), np.uint8))
    assert outcome.report is None and (outcome.image == 255).all()


def test_registry_entry():
    registry = build_registry()
    keys = [a.key for a in registry.in_category("morphology")]
    assert keys[-1] == "connected_components" and len(keys) == 12 and len(registry.categories()) == 7
    algo = registry.get("connected_components")
    assert algo.title == "Connected Components" and algo.implemented
    assert "X_k = (X_k−1 ⊕ B) ∩ A" in algo.summary and "background = 0" in algo.summary
    assert [p.key for p in algo.params] == ["binarize", "thresh", "foreground", "connectivity", "display"]
    assert algo.default_params() == dict(binarize="otsu", thresh=127, foreground="bright", connectivity="8",
                                         display="labels")
    conn = algo.params[3]
    assert dict(conn.options) == {"4": "4 · Cross", "8": "8 · Square"} and "3×3 cross" in conn.help


def test_previous_eleven_operations_unchanged():
    registry = build_registry()
    img = u8(random_binary(92, (22, 26), 0.55))
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
    gray = np.random.default_rng(6).integers(0, 256, (22, 26), dtype=np.uint8)
    b15 = M.structuring_element("rect", 15, 15)
    np.testing.assert_array_equal(registry.get("top_hat").run(gray), ref_top_hat(gray, b15))
    np.testing.assert_array_equal(registry.get("black_hat").run(gray), ref_black_hat(gray, b15))
    np.testing.assert_array_equal(registry.get("hat_correction").run(gray),
                                  np.clip(gray + ref_top_hat(gray, b15) - ref_black_hat(gray, b15), 0, 255))


# --------------------------------------------------------------------------
# Arayüz (offscreen)
# --------------------------------------------------------------------------

DIAG = grid(["............",
             ".##.........",
             ".##.........",
             "...##....#..",
             "...##.......",
             "........###.",
             "........###.",
             "............"])


@pytest.fixture
def loaded(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    img = np.repeat(np.where(np.kron(DIAG, np.ones((3, 3), bool)), 220, 30).astype(np.uint8)[..., None], 3, axis=2)
    path = tmp_path / "Bileşen" / "köşegen_temas.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


def _apply(w, **values):
    w.navigator.select_algorithm("connected_components")
    for k, v in values.items():
        w.parameters.controls[k].set_value(v)
    w.parameters.apply_button.click()


def test_ui_listed_and_controls(loaded):
    w, _ = loaded
    w.navigator.tiles["morphology"].click()
    assert list(w.navigator.operation_rows)[-1] == "connected_components"
    w.navigator.select_algorithm("connected_components")
    c = w.parameters.controls
    assert list(c) == ["binarize", "thresh", "foreground", "connectivity", "display"]
    assert set(c["connectivity"].buttons) == {"4", "8"} and c["connectivity"].value() == "8"
    assert not w.parameters.report_view.isVisible()


@pytest.mark.parametrize("conn,count", [("4", 4), ("8", 3)])
def test_ui_report_table_matches_reference(loaded, conn, count):
    """3×3 büyütülmüş köşegen temas: kareler 8-bağlantıda birleşir, 4-bağlantıda ayrılır."""
    w, img = loaded
    _apply(w, connectivity=conn, display="labels")
    view = w.parameters.report_view
    assert view.isVisible()
    a = img[..., 0] > 127
    expected = ref_stats(ref_label(a, int(conn)))
    assert len(expected) == count and view.report.count == count
    assert view.count_label.text().startswith(f"{count} component")
    rows = view.rows()
    assert len(rows) == count
    for row, (k, n, x, y, ww, hh, cx, cy) in zip(rows, expected):
        assert row == (str(k), str(n), f"{x},{y} {ww}×{hh}", f"{cx:.1f}, {cy:.1f}")
    header, fm = view.table.horizontalHeader(), view.table.fontMetrics()
    assert header.length() <= view.table.viewport().width()
    for col in range(4):                                        # her sütunun en uzun metni kırpılmadan sığar
        assert header.sectionSize(col) >= max(fm.horizontalAdvance(r[col]) for r in rows), col
    np.testing.assert_array_equal(w.current_image, M.label_colors(count)[ref_label(a, int(conn))])
    np.testing.assert_array_equal(w.original_image, img)
    assert w.telemetry.state.state() == "DONE"
    assert f"{count} component" in w.statusBar().currentMessage()
    for name, counts in histograms(w.current_image).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)


def test_ui_table_sorts_numerically(loaded):
    w, _ = loaded
    _apply(w, connectivity="4")
    view = w.parameters.report_view
    view.table.sortItems(1)
    areas = [int(r[1]) for r in view.rows()]
    assert areas == sorted(areas) and areas[0] == 9           # tek piksel × 3×3 büyütme


def test_ui_report_cleared_by_undo_reset_other_ops_and_loading(loaded, tmp_path):
    w, img = loaded
    view = w.parameters.report_view
    _apply(w)
    labelled = w.current_image.copy()
    assert view.isVisible()
    w.undo_action.trigger()
    assert not view.isVisible()
    np.testing.assert_array_equal(w.current_image, img)
    _apply(w)
    w.reset_action.trigger()
    assert not view.isVisible()
    np.testing.assert_array_equal(w.current_image, img)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, labelled)
    assert not view.isVisible()                                # eski rapor geri gelmez (bayat olabilir)
    _apply(w)
    w.navigator.select_algorithm("erode")
    w.parameters.apply_button.click()
    assert not view.isVisible()
    _apply(w)
    save_image(tmp_path / "baska.png", np.zeros((8, 8), np.uint8))
    w.load_image(tmp_path / "baska.png")
    assert not view.isVisible()
