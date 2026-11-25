"""Delik doldurma (Chapter 4, slayt 56–59).

Bağımsız referanslar OpenCV kullanmaz:
* ``ref_flood``  — saf Python BFS: kenardaki tüm arka plan piksellerinden 4-komşulukla
  yayılır; ulaşılamayan arka plan = delik; sonuç A ∪ delikler.
* ``ref_recon``  — X_k = (X_{k−1} ⊕ B) ∩ Aᶜ formülünün piksel döngüsüyle yazılmış hali.
Ayrıca küçük matrislerde beklenen sonuçlar elle yazılmıştır.
"""

from collections import deque

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
from tests.test_morphology_hat import ref_black_hat, ref_top_hat

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# --------------------------------------------------------------------------
# Referanslar
# --------------------------------------------------------------------------


def ref_flood(a, neigh=N4):
    h, w = a.shape
    outside = np.zeros_like(a)
    q = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not a[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in neigh:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not a[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    return a | (~a & ~outside)


def ref_recon(a, seeds, neigh=N4):
    """X_k = (X_{k−1} ⊕ B) ∩ Aᶜ; B = merkez + komşular (artı ya da kare)."""
    h, w = a.shape
    x = np.zeros_like(a)
    for cx, cy in seeds:
        x[cy, cx] = True
    k = 0
    while True:
        d = x.copy()
        for r, c in zip(*np.nonzero(x)):
            for dr, dc in neigh:
                if 0 <= r + dr < h and 0 <= c + dc < w:
                    d[r + dr, c + dc] = True
        nxt = d & ~a
        k += 1
        if np.array_equal(nxt, x):
            return x | a, k
        x = nxt


def grid(rows):
    return np.array([[ch == "#" for ch in row] for row in rows], bool)


def u8(mask):
    return np.where(mask, 255, 0).astype(np.uint8)


# Ders kitabındaki (Fig. 9.17 benzeri) nesne: tek kapalı delik
SINGLE = grid([".........",
               ".#####...",
               ".#...#...",
               ".#...###.",
               ".#.....#.",
               ".#######.",
               "........."])
SINGLE_FILLED = grid([".........",
                      ".#####...",
                      ".#####...",
                      ".#######.",
                      ".#######.",
                      ".#######.",
                      "........."])

# İki delikli nesne + ayrı bir halka (3 kapalı delik) + dışarı açık "C"
MULTI = grid(["...............",
              ".#######..#####",
              ".#..#..#..#...#",
              ".#..#..#..#.#.#",
              ".#######..#...#",
              "..........###.#",
              "...............",
              "..#####........",
              "..#...#........",
              "..#.#.#........",
              "..#####........",
              "..............."])


# --------------------------------------------------------------------------
# Elle hesaplanmış durumlar
# --------------------------------------------------------------------------


def test_single_closed_hole_both_methods():
    assert np.array_equal(M.fill_holes_flood(SINGLE), SINGLE_FILLED)
    filled, k = M.fill_holes_reconstruction(SINGLE, [(2, 2)])
    assert np.array_equal(filled, SINGLE_FILLED)
    assert k >= 2


def test_multiple_closed_holes_flood_fills_all():
    out = M.fill_holes_flood(MULTI)
    expected = MULTI.copy()
    expected[2:4, 2:4] = True      # sol nesnenin sol deliği
    expected[2:4, 5:7] = True      # sol nesnenin sağ deliği
    expected[8:10, 3:6] = True     # alttaki halka (içindeki adayla birlikte)
    expected[9, 4] = True
    assert np.array_equal(out, expected | MULTI)
    assert not out[2:5, 11:14].all()   # sağ üstteki "C" kenara açık → dolmaz
    assert np.array_equal(out, ref_flood(MULTI))


def test_open_gap_is_not_filled():
    c_shape = grid([".......",
                    ".#####.",
                    ".#....." ,
                    ".#.....",
                    ".#####.",
                    "......."])
    np.testing.assert_array_equal(M.fill_holes_flood(c_shape), c_shape)
    with pytest.raises(AlgorithmParameterError, match="not inside a closed hole"):
        M.fill_holes_reconstruction(c_shape, [(3, 2)])
    assert M.auto_seeds(c_shape, "all") == []


def test_hole_leaking_through_image_border_is_not_filled():
    a = grid(["#####",
              "#...#",
              "#...#",
              "##.##"])      # alt kenarda açıklık: delik görüntü dışına bağlı
    np.testing.assert_array_equal(M.fill_holes_flood(a), a)
    b = a.copy()
    b[3, 2] = True           # açıklık kapanınca delik dolar
    assert M.fill_holes_flood(b).all()


HOLE_LEFT = [(2, 2), (2, 3), (3, 2), (3, 3)]                   # (satır, sütun)
HOLE_RIGHT = [(2, 5), (2, 6), (3, 5), (3, 6)]
HOLE_RING = [(8, 3), (8, 4), (8, 5), (9, 3), (9, 5)]


@pytest.mark.parametrize("seed,hole", [((2, 2), HOLE_LEFT), ((6, 3), HOLE_RIGHT), ((5, 9), HOLE_RING)])
def test_seed_fills_only_the_selected_hole(seed, hole):
    out, _ = M.fill_holes_reconstruction(MULTI, [seed])
    expected = MULTI.copy()
    for r, c in hole:
        expected[r, c] = True
    np.testing.assert_array_equal(out, expected)              # yalnızca bu delik dolar
    np.testing.assert_array_equal(out, ref_recon(MULTI, [seed])[0])


def test_multiple_seeds_and_seeds_in_the_same_hole():
    two, _ = M.fill_holes_reconstruction(MULTI, [(2, 2), (3, 8)])
    assert (two & ~MULTI).sum() == 4 + 5
    same, _ = M.fill_holes_reconstruction(MULTI, [(2, 2), (3, 3), (2, 3)])
    one, _ = M.fill_holes_reconstruction(MULTI, [(2, 2)])
    np.testing.assert_array_equal(same, one)
    every, _ = M.fill_holes_reconstruction(MULTI, M.auto_seeds(MULTI, "all"))
    np.testing.assert_array_equal(every, M.fill_holes_flood(MULTI))


def test_no_seeds_returns_a():
    out, k = M.fill_holes_reconstruction(MULTI, [])
    np.testing.assert_array_equal(out, MULTI)
    assert k == 1


def test_iteration_count_matches_geodesic_length():
    """1 × n koridor delik, tohum bir uçta: her adımda bir piksel ilerler → k = n."""
    for n in (1, 3, 8):
        a = np.ones((3, n + 2), bool)
        a[1, 1:n + 1] = False
        filled, k = M.fill_holes_reconstruction(a, [(1, 1)])
        assert filled.all() and k == n
        assert ref_recon(a, [(1, 1)])[1] == n


def test_four_connectivity_vs_diagonal_leak():
    """Delik dışarıya yalnızca köşegenden bağlı: 4-bağlantıda kapalı, 8-bağlantıda açık."""
    a = grid(["......",
              ".###..",
              ".#.#..",
              ".##.#.",
              "..#...",
              "......"])
    # (2,2) deliğinin 4-komşuları ön plan; köşegen komşusu (3,3) dışarıya açık
    flood = M.fill_holes_flood(a)
    assert flood[2, 2] and np.array_equal(flood, ref_flood(a, N4))
    cross, _ = M.fill_holes_reconstruction(a, [(2, 2)], connectivity=4)
    assert np.array_equal(cross, flood)
    with pytest.raises(AlgorithmParameterError, match="not inside a closed hole"):
        M.fill_holes_reconstruction(a, [(2, 2)], connectivity=8)
    assert M.auto_seeds(a, "all", 8) == []
    np.testing.assert_array_equal(ref_flood(a, N8), a)       # 8-bağlantıda delik yok
    raw = M.hole_filling(u8(a), binarize="manual", method="reconstruction", seeds="all", shape="rect")
    np.testing.assert_array_equal(raw > 0, a)


def test_cross_se_grows_four_connected_only():
    """Artı B ile X yalnızca 4-komşulara büyür: bir adımda tohumdan artı biçimi oluşur."""
    a = np.ones((7, 7), bool)
    a[1:6, 1:6] = False
    se = M.structuring_element("cross", 3, 3)
    x = np.zeros_like(a)
    x[3, 3] = True
    step1 = M.binary_dilation(x, se) & ~a
    assert step1.astype(int).sum() == 5 and step1[2, 3] and not step1[2, 2]
    sq = M.binary_dilation(x, M.structuring_element("rect", 3, 3)) & ~a
    assert sq.sum() == 9


# --------------------------------------------------------------------------
# Rastgele görüntülerde bağımsız referans
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_flood_matches_bfs_reference(seed):
    a = random_binary(200 + seed, (19, 23), 0.55)
    np.testing.assert_array_equal(M.fill_holes_flood(a), ref_flood(a))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("conn,neigh", [(4, N4), (8, N8)])
def test_reconstruction_matches_formula_reference(seed, conn, neigh):
    a = random_binary(300 + seed, (17, 21), 0.6)
    seeds = M.auto_seeds(a, "all", conn)
    out, k = M.fill_holes_reconstruction(a, seeds, conn)
    ref, kr = ref_recon(a, seeds, neigh)
    np.testing.assert_array_equal(out, ref)
    assert k == kr
    np.testing.assert_array_equal(out, ref_flood(a, neigh))   # tüm delikler tohumlanınca flood ile aynı


@pytest.mark.parametrize("seed", range(5))
def test_properties(seed):
    a = random_binary(400 + seed, (25, 25), 0.6)
    f = M.fill_holes_flood(a)
    assert not (a & ~f).any()                              # A ⊆ sonuç
    np.testing.assert_array_equal(M.fill_holes_flood(f), f)  # idempotent
    assert M.auto_seeds(f, "all") == []                    # doldurulmuş görüntüde delik kalmaz
    labels, holes = M.hole_labels(a, 4)
    added = f & ~a
    for i in holes:                                        # her delik ya tamamen dolar
        assert added[labels == i].all()
    assert added.sum() == sum((labels == i).sum() for i in holes)


# --------------------------------------------------------------------------
# Slayt 59'daki OpenCV kodunun tuzakları
# --------------------------------------------------------------------------


def slide59(bw_u8):
    """Slayttaki kod: floodFill (0, 0)'dan, tersle, OR."""
    flood = bw_u8.copy()
    h, w = bw_u8.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    return cv2.bitwise_or(bw_u8, cv2.bitwise_not(flood)) > 0


def test_corner_foreground_is_handled():
    a = SINGLE.copy()
    a[0, 0] = True                                     # (0, 0) ön plan
    assert slide59(u8(a)).all()                        # slayt kodu: tüm görüntü "dolar" (yanlış)
    np.testing.assert_array_equal(M.fill_holes_flood(a), SINGLE_FILLED | a)


def test_background_split_along_border_is_not_treated_as_hole():
    a = np.zeros((7, 9), bool)
    a[:, 4] = True                                     # tam yükseklikte dikey çizgi: arka plan iki parça
    wrong = slide59(u8(a))
    assert wrong[:, 5:].all()                          # slayt kodu sağ yarıyı delik sanır
    np.testing.assert_array_equal(M.fill_holes_flood(a), a)


def test_frame_touching_every_border():
    a = np.ones((6, 6), bool)
    a[2:4, 2:4] = False
    assert M.fill_holes_flood(a).all()
    filled, _ = M.fill_holes_reconstruction(a, [(2, 2)])
    assert filled.all()


# --------------------------------------------------------------------------
# Tohum doğrulama ve otomatik tohumlar
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seeds,match", [([(20, 2)], "outside"), ([(-1, 2)], "outside"), ([(1, 1)], "foreground"),
                                        ([(0, 0)], "not inside a closed hole"), ([(2.5, 2)], "integer"),
                                        ([(True, 2)], "integer"), ([(2,)], "pair"), (["22"], "pair")])
def test_invalid_seeds(seeds, match):
    with pytest.raises(AlgorithmParameterError, match=match):
        M.fill_holes_reconstruction(SINGLE, seeds)


def test_integral_float_and_numpy_seed_coordinates():
    out, _ = M.fill_holes_reconstruction(SINGLE, [(2.0, np.int64(2))])
    np.testing.assert_array_equal(out, SINGLE_FILLED)


def test_invalid_connectivity():
    for conn in (6, 0, True, "4"):
        with pytest.raises(AlgorithmParameterError):
            M.hole_labels(SINGLE, conn)


def test_auto_seed_policies():
    labels, holes = M.hole_labels(MULTI, 4)
    sizes = {i: (labels == i).sum() for i in holes}
    largest = max(sizes, key=sizes.get)
    [s] = M.auto_seeds(MULTI, "largest")
    assert labels[s[1], s[0]] == largest and not MULTI[s[1], s[0]]
    seeds = M.auto_seeds(MULTI, "all")
    assert len(seeds) == len(holes) == 3
    assert {labels[y, x] for x, y in seeds} == set(holes)
    [c] = M.auto_seeds(MULTI, "centre")
    cy, cx = (MULTI.shape[0] - 1) / 2, (MULTI.shape[1] - 1) / 2
    assert min((y - cy) ** 2 + (x - cx) ** 2 for x, y in seeds) == (c[1] - cy) ** 2 + (c[0] - cx) ** 2
    with pytest.raises(AlgorithmParameterError):
        M.auto_seeds(MULTI, "random")


def test_seed_lies_inside_concave_hole():
    """Ağırlık merkezi deliğin dışında kalan (U biçimli) delikte bile tohum deliğin içindedir."""
    a = np.ones((9, 9), bool)
    a[1:8, 1:3] = a[1:8, 6:8] = a[6:8, 1:8] = False    # U biçimli delik
    [s] = M.auto_seeds(a, "largest")
    assert not a[s[1], s[0]]
    filled, _ = M.fill_holes_reconstruction(a, [s])
    assert filled.all()


# --------------------------------------------------------------------------
# Görüntü düzeyi: eşikleme, kutupluluk, renk, alfa, değişmezlik
# --------------------------------------------------------------------------


def test_gray_manual_and_otsu():
    img = np.where(MULTI, 200, 40).astype(np.uint8)
    for kwargs in (dict(binarize="manual", thresh=127), dict(binarize="otsu")):
        np.testing.assert_array_equal(M.hole_filling(img, **kwargs), u8(ref_flood(MULTI)))
    assert not M.hole_filling(img, binarize="manual", thresh=250).any()     # eşik üstünde ön plan yok


def test_dark_foreground_polarity():
    img = np.where(SINGLE, 20, 230).astype(np.uint8)        # koyu nesne, açık zemin
    np.testing.assert_array_equal(M.hole_filling(img, foreground="dark"), u8(SINGLE_FILLED))
    # BRIGHT seçilirse ön plan zemin olur; koyu halka kenara değmeyen bir arka plan bölgesi,
    # yani zeminin "deliği" olur ve doldurulur → sonuç tamamen ön plan
    np.testing.assert_array_equal(M.hole_filling(img, foreground="bright"), np.full(SINGLE.shape, 255))


def test_bgr_luminance_and_three_equal_channels():
    img = np.zeros(MULTI.shape + (3,), np.uint8)
    img[MULTI] = (0, 255, 0)              # yeşil Y≈150
    img[~MULTI] = (255, 0, 0)             # mavi Y≈29
    out = M.hole_filling(img, binarize="manual", thresh=100)
    assert out.shape == img.shape
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], u8(ref_flood(MULTI)))


@pytest.mark.parametrize("method", M.HOLE_METHODS)
@pytest.mark.parametrize("shape", [(15, 15), (15, 15, 1), (15, 15, 3), (15, 15, 4)])
def test_layout_alpha_immutability(method, shape):
    rng = np.random.default_rng(9)
    base = np.where(np.pad(MULTI[:11, :11], 2), 210, 30).astype(np.uint8)
    img = base if len(shape) == 2 else np.repeat(base[..., None], shape[2], axis=2).copy()
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = rng.integers(0, 256, shape[:2])
    before = img.copy()
    out = M.hole_filling(img, method=method, seeds="all")
    np.testing.assert_array_equal(img, before)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.shares_memory(out, img)
    expected = u8(ref_flood(np.pad(MULTI[:11, :11], 2)))
    planes = [out] if out.ndim == 2 else [out[..., c] for c in range(min(3, out.shape[2]))]
    for p in planes:
        np.testing.assert_array_equal(p, expected)
    if len(shape) == 3 and shape[2] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])


@pytest.mark.parametrize("method", M.HOLE_METHODS)
def test_empty_constant_and_tiny_images(method):
    for value in (0, 128, 255):
        img = np.full((6, 7), value, np.uint8)
        out = M.hole_filling(img, binarize="manual", method=method)
        np.testing.assert_array_equal(out, np.where(img > 127, 255, 0))   # boş kalır / tamamı ön plan kalır
    one = np.array([[0]], np.uint8)
    np.testing.assert_array_equal(M.hole_filling(one, binarize="manual", method=method), [[0]])
    ring3 = u8(grid(["###", "#.#", "###"]))
    np.testing.assert_array_equal(M.hole_filling(ring3, binarize="manual", method=method, seeds="all"),
                                  np.full((3, 3), 255))
    two = u8(grid(["#.", ".#"]))
    np.testing.assert_array_equal(M.hole_filling(two, binarize="manual", method=method), two)


@pytest.mark.parametrize("kwargs", [dict(method="watershed"), dict(seeds="first"), dict(shape="ellipse"),
                                    dict(binarize="auto"), dict(foreground="white"),
                                    dict(binarize="manual", thresh=-5)])
def test_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.hole_filling(np.zeros((5, 5), np.uint8), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint8),
                                 np.zeros((3, 3, 2), np.uint8), [[0, 1]]])
def test_invalid_images(bad):
    with pytest.raises(AlgorithmParameterError):
        M.hole_filling(bad)


def test_masks_must_be_2d():
    with pytest.raises(AlgorithmParameterError):
        M.fill_holes_flood(np.zeros((3, 3, 3), bool))
    with pytest.raises(AlgorithmParameterError):
        M.fill_holes_reconstruction(np.zeros((3, 3, 3), bool), [])


# --------------------------------------------------------------------------
# Katalog ve regresyon
# --------------------------------------------------------------------------


def test_registry_entry():
    registry = build_registry()
    keys = [a.key for a in registry.in_category("morphology")]
    assert keys[10] == "hole_filling" and len(keys) == 22 and len(registry.categories()) == 7
    algo = registry.get("hole_filling")
    assert algo.title == "Hole Filling" and algo.implemented
    assert "X_k = (X_k−1 ⊕ B) ∩ Aᶜ" in algo.summary and "X_k ∪ A" in algo.summary
    assert [p.key for p in algo.params] == ["binarize", "thresh", "foreground", "method", "seeds", "shape"]
    assert algo.default_params() == dict(binarize="otsu", thresh=127, foreground="bright", method="flood",
                                         seeds="largest", shape="cross")
    assert all(p.help for p in algo.params if p.key in ("method", "seeds", "shape"))


def test_registry_runs_and_sanitizes():
    algo = build_registry().get("hole_filling")
    img = u8(MULTI)
    np.testing.assert_array_equal(algo.run(img), u8(ref_flood(MULTI)))
    p = algo.resolve_params({"method": "magic", "seeds": "x", "shape": "disk"})
    assert p["method"] == "flood" and p["seeds"] == "largest" and p["shape"] == "cross"
    largest = algo.run(img, {"method": "reconstruction", "seeds": "largest"})
    assert (largest > 0).sum() == MULTI.sum() + 5          # en büyük delik: halkanın içi (5 piksel)


def test_previous_morphology_operations_unchanged():
    registry = build_registry()
    img = u8(random_binary(91, (22, 26), 0.55))
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
    }
    for key, exp in expected.items():
        np.testing.assert_array_equal(registry.get(key).run(img), u8(exp), err_msg=key)
    gray = np.random.default_rng(5).integers(0, 256, (22, 26), dtype=np.uint8)
    b15 = M.structuring_element("rect", 15, 15)
    np.testing.assert_array_equal(registry.get("top_hat").run(gray), ref_top_hat(gray, b15))
    np.testing.assert_array_equal(registry.get("black_hat").run(gray), ref_black_hat(gray, b15))
    np.testing.assert_array_equal(registry.get("hat_correction").run(gray),
                                  np.clip(gray + ref_top_hat(gray, b15) - ref_black_hat(gray, b15), 0, 255))


# --------------------------------------------------------------------------
# Arayüz (offscreen)
# --------------------------------------------------------------------------


@pytest.fixture
def loaded(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    img = np.repeat(np.where(np.pad(MULTI, 3), 220, 35).astype(np.uint8)[..., None], 3, axis=2)
    path = tmp_path / "Delik" / "delikli_şekiller.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


def test_ui_listed_and_controls(loaded):
    w, _ = loaded
    w.navigator.tiles["morphology"].click()
    assert list(w.navigator.operation_rows)[10] == "hole_filling"  # 11. işlem (Aşama 5 ile 12)
    w.navigator.select_algorithm("hole_filling")
    c = w.parameters.controls
    assert list(c) == ["binarize", "thresh", "foreground", "method", "seeds", "shape"]
    assert set(c["method"].buttons) == {"flood", "reconstruction"}
    assert set(c["seeds"].buttons) == {"largest", "centre", "all"}
    assert set(c["shape"].buttons) == {"cross", "rect"}


@pytest.mark.parametrize("values", [dict(method="flood"), dict(method="reconstruction", seeds="largest"),
                                    dict(method="reconstruction", seeds="all")])
def test_ui_apply_matches_reference(loaded, values):
    w, img = loaded
    w.navigator.select_algorithm("hole_filling")
    for k, v in values.items():
        w.parameters.controls[k].set_value(v)
    w.parameters.apply_button.click()
    out = w.current_image
    a = np.pad(MULTI, 3)
    if values.get("seeds") == "largest":
        exp, _ = ref_recon(a, M.auto_seeds(a, "largest"))
    else:
        exp = ref_flood(a)
    for ch in range(3):
        np.testing.assert_array_equal(out[..., ch], u8(exp))
    np.testing.assert_array_equal(w.original_image, img)
    assert w.telemetry.state.state() == "DONE"
    for name, counts in histograms(out).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)


def test_ui_undo_reset(loaded):
    w, img = loaded
    w.navigator.select_algorithm("hole_filling")
    w.parameters.controls["method"].set_value("reconstruction")
    w.parameters.apply_button.click()
    one = w.current_image.copy()
    w.parameters.controls["method"].set_value("flood")
    w.parameters.apply_button.click()
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, one)
    w.reset_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, one)
    np.testing.assert_array_equal(w.original_image, img)
