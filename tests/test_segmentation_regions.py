"""Segmentation / Paket 3: bölge tabanlı bölütleme (slayt 29–38, Chapter 5 laboratuvarı).

Her işlem için: bilinen sonuçlu küçük sentetik görüntüler, döngülerle yazılmış bağımsız referanslar
(laboratuvar sırasını izleyen kuyruk, özyinelemeli quadtree, heapq taşırma, kaba kuvvet EDT),
görüntü sınırları, sabit görüntüler, farklı boyutlar ve parametre sınırları.
"""

import heapq
import inspect
import math
from collections import deque

import cv2
import numpy as np
import pytest

from processing import morphology as M
from processing import segmentation as S
from processing.catalog import REGISTRY
from processing.registry import AlgorithmParameterError

REGION_KEYS = ["region_growing", "region_split_merge", "region_watershed", "region_marker_watershed"]
FUNCS = {"region_growing": S.region_growing, "region_split_merge": S.split_and_merge,
         "region_watershed": S.watershed, "region_marker_watershed": S.marker_controlled_watershed}


def noisy(f, sigma=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(np.rint(np.asarray(f, np.float64) + rng.normal(0, sigma, np.shape(f))), 0, 255).astype(np.uint8)


def discs(centres, shape=(80, 120), bg=40, fg=200):
    f = np.full(shape, float(bg))
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for cx, cy, r in centres:
        f[np.hypot(xx - cx, yy - cy) < r] = fg
    return f


TOUCHING = [(40, 40, 22), (78, 40, 20)]                 # iki değen disk (merkezler 38 px, yarıçaplar 22 + 20)


# ================================================================ bağımsız referanslar
def ref_grow(f, seeds, t, conn):
    """Laboratuvar growAll'ın düz Python karşılığı: katman katman kuyruk, ilk kabul eden kazanır."""
    h, w = f.shape
    nb = S.N4_OFFSETS if conn == "4" else S.N8_OFFSETS
    lab = np.zeros((h, w), np.int32)
    ref = {}
    seen = np.zeros((h, w), bool)
    front = []
    for (y, x), r in seeds:
        if not seen[y, x]:
            seen[y, x] = True
            lab[y, x] = r
            ref[(y, x)] = int(f[y, x])
            front.append((y, x))
    layers = 0
    while front:
        nxt = []
        for y, x in front:
            for dx, dy in nb:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and abs(int(f[yy, xx]) - ref[(y, x)]) <= t:
                    seen[yy, xx] = True
                    lab[yy, xx] = lab[y, x]
                    ref[(yy, xx)] = ref[(y, x)]
                    nxt.append((yy, xx))
        if not nxt:
            break
        layers += 1
        front = nxt
    return lab, layers


def ref_leaves(f, pred, mn, invert, depth=0, x=0, y=0, w=None, h=None, max_depth=S.SPLIT_MAX_DEPTH):
    """Özyinelemeli quadtree (laboratuvardaki genişlik öncelikli kuyruktan bağımsız)."""
    if w is None:
        h, w = f.shape
    blk = f[y:y + h, x:x + w].astype(np.int64)
    s, s2, c = int(blk.sum()), int((blk ** 2).sum()), w * h
    m = s / c
    q = pred(m, math.sqrt(max(0.0, s2 / c - m * m)))
    if (q if invert else not q) and w >= 2 * mn and h >= 2 * mn and depth < max_depth:
        w1, h1 = w // 2, h // 2
        out = []
        for dx, dy, ww, hh in ((0, 0, w1, h1), (w1, 0, w - w1, h1), (0, h1, w1, h - h1), (w1, h1, w - w1, h - h1)):
            out += ref_leaves(f, pred, mn, invert, depth + 1, x + dx, y + dy, ww, hh, max_depth)
        return out
    return [(x, y, w, h, m, q)]


def ref_partition_labels(leaves, shape, similar, keep):
    """Yaprak komşuluk grafında BFS; bölgeler raster sırasında ilk piksele göre numaralanır."""
    lid = np.zeros(shape, int)
    for k, (x, y, w, h, *_ ) in enumerate(leaves):
        lid[y:y + h, x:x + w] = k
    adj = {k: set() for k in range(len(leaves))}
    hh, ww = shape
    for y in range(hh):
        for x in range(ww):
            for yy, xx in ((y, x + 1), (y + 1, x)):
                if yy < hh and xx < ww and lid[yy, xx] != lid[y, x]:
                    a, b = lid[y, x], lid[yy, xx]
                    if similar(leaves[a], leaves[b]):
                        adj[a].add(b)
                        adj[b].add(a)
    comp = {}
    for k in range(len(leaves)):
        if k in comp:
            continue
        comp[k] = k
        dq = deque([k])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp[v] = k
                    dq.append(v)
    out = np.zeros(shape, np.int32)
    names = {}
    for y in range(hh):
        for x in range(ww):
            k = lid[y, x]
            if not keep(leaves[k]):
                continue
            out[y, x] = names.setdefault(comp[k], len(names) + 1)
    return out


def ref_minima(s):
    h, w = s.shape
    lab = np.zeros((h, w), np.int32)
    seen = np.zeros((h, w), bool)
    n = 0
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            v = s[y, x]
            comp, dq, is_min = [], deque([(y, x)]), True
            seen[y, x] = True
            while dq:
                cy, cx = dq.popleft()
                comp.append((cy, cx))
                for dx, dy in S.N8_OFFSETS:
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        if s[yy, xx] < v:
                            is_min = False
                        elif s[yy, xx] == v and not seen[yy, xx]:
                            seen[yy, xx] = True
                            dq.append((yy, xx))
            if is_min:
                n += 1
                for p in comp:
                    lab[p] = n
    return lab


def ref_flood(s, markers):
    """Öncelik kuyruğu (heapq): anahtar (seviye, giriş sırası) — kova + FIFO ile aynı sıra."""
    h, w = s.shape
    lab = markers.astype(np.int64).copy()
    heap, cnt = [], 0
    for y in range(h):
        for x in range(w):
            if lab[y, x] > 0:
                heapq.heappush(heap, (int(s[y, x]), cnt, y, x))
                cnt += 1
    while heap:
        lv, _, y, x = heapq.heappop(heap)
        for yy, xx in ((y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)):
            if 0 <= yy < h and 0 <= xx < w and lab[yy, xx] == 0:
                lab[yy, xx] = lab[y, x]
                heapq.heappush(heap, (max(lv, int(s[yy, xx])), cnt, yy, xx))
                cnt += 1
    return lab


def ref_edt(mask):
    h, w = mask.shape
    bg = np.argwhere(~mask)
    out = np.zeros((h, w))
    for y, x in np.argwhere(mask):
        out[y, x] = math.sqrt(min((y - by) ** 2 + (x - bx) ** 2 for by, bx in bg))
    return out.astype(np.float32)


def ref_morph3(a, op, k):
    h, w = a.shape
    for _ in range(k):
        b = np.zeros_like(a)
        for y in range(h):
            for x in range(w):
                vals = [a[y + dy, x + dx] if 0 <= y + dy < h and 0 <= x + dx < w else (op == "erode")
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                b[y, x] = all(vals) if op == "erode" else any(vals)
        a = b
    return a


# ================================================================ katalog
def test_region_operations_follow_edge_operations_in_source_order():
    keys = [a.key for a in REGISTRY.in_category("segmentation")]
    assert keys[21:25] == REGION_KEYS and keys[-1] == "kmeans"
    assert [REGISTRY.get(k).title for k in REGION_KEYS] == [
        "Region Growing", "Region Splitting & Merging", "Watershed", "Marker-Controlled Watershed"]
    assert all(REGISTRY.get(k).implemented for k in REGION_KEYS)
    assert "connected_components" in [a.key for a in REGISTRY.in_category("morphology")]   # yeniden yazılmadı
    assert not any("component" in k for k in keys)


@pytest.mark.parametrize("key", REGION_KEYS)
def test_catalog_parameters_match_function_signature_defaults(key):
    algo = REGISTRY.get(key)
    sig = inspect.signature(FUNCS[key])
    names = list(sig.parameters)[1:]
    assert [p.key for p in algo.params] == names
    for p in algo.params:
        assert sig.parameters[p.key].default == p.default, p.key


def test_catalog_choice_values_match_module_constants():
    def values(key, name):
        return next(p for p in REGISTRY.get(key).params if p.key == name).values
    assert values("region_growing", "seeds") == S.GROW_SEED_MODES
    assert values("region_growing", "display") == S.GROW_DISPLAYS
    assert values("region_split_merge", "min_size") == S.SPLIT_MIN_BLOCKS
    assert values("region_split_merge", "display") == S.SPLIT_DISPLAYS
    assert values("region_watershed", "surface") == S.WATERSHED_SURFACES
    assert values("region_watershed", "display") == S.WATERSHED_DISPLAYS
    assert values("region_marker_watershed", "surface") == S.MARKER_SURFACES
    assert values("region_marker_watershed", "display") == S.MARKER_DISPLAYS
    assert next(p for p in REGISTRY.get("region_marker_watershed").params if p.key == "surface").default == "gradient"


# ================================================================ ortak: rapor, boyutlar, kanal düzeni
@pytest.mark.parametrize("key", REGION_KEYS)
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (2, 2), (5, 7), (23, 37), (37, 23), (64, 64)])
def test_every_operation_handles_any_size_and_reports_regions(key, shape):
    f = noisy(np.random.default_rng(shape[0] * 100 + shape[1]).integers(0, 256, shape), 0)
    r = FUNCS[key](f)
    assert r.image.shape[:2] == shape and r.image.dtype == np.uint8
    assert isinstance(r.report, M.ComponentReport) and isinstance(r.report, S.RegionReport)
    lab = r.report.labels
    assert lab.shape == shape and lab.dtype == np.int32 and lab.min() >= 0
    assert r.report.count == len(np.unique(lab[lab > 0]))
    assert sum(c.area for c in r.report.components) == int((lab > 0).sum())
    assert r.report.method in r.report.summary()


@pytest.mark.parametrize("key", REGION_KEYS)
def test_input_array_is_not_modified_and_layouts_are_kept(key):
    g = noisy(discs(TOUCHING), 4, 1)
    bgr = np.dstack([g, g, g])
    bgra = np.dstack([g, g, g, np.full_like(g, 77)])
    before = bgra.copy()
    rg, rb, ra = FUNCS[key](g), FUNCS[key](bgr), FUNCS[key](bgra)
    np.testing.assert_array_equal(bgra, before)
    np.testing.assert_array_equal(rg.image, rb.image)                           # gri ↔ eşit kanallı BGR aynı sonuç
    assert ra.image.shape == bgra.shape and (ra.image[..., 3] == 77).all()     # alfa korunur
    np.testing.assert_array_equal(ra.image[..., :3], rb.image)


def test_region_components_match_connected_components_for_connected_regions():
    lab = M.label_components(np.random.default_rng(4).random((30, 40)) > 0.6, 8).labels
    mine = S.region_components(lab)
    theirs = M.label_components(lab > 0, 8).components
    assert mine == theirs
    assert S.region_components(np.zeros((3, 3), np.int32)) == ()


def test_boundary_of_marks_right_and_down_neighbours_with_other_positive_label():
    lab = np.array([[1, 1, 2], [1, 3, 2], [0, 3, 3]])
    np.testing.assert_array_equal(S.boundary_of(lab), [[0, 1, 0], [1, 1, 1], [0, 0, 0]])   # (0, 2): altı da 2
    assert not S.boundary_of(np.ones((4, 4), int)).any()
    assert not S.boundary_of(np.array([[1, 0, 2]])).any()                         # 0 (arka plan) sınır sayılmaz


def test_label_overlay_blends_like_the_lab():
    base = np.array([[100, 101], [200, 0]], np.uint8)
    lab = np.array([[0, 1], [2, 0]])
    out = S.label_overlay(lab, base, 0.5, 0.25, lines=np.array([[False, False], [False, True]]))
    pal = M.label_colors(2).astype(float)
    assert out[0, 0].tolist() == [50, 50, 50]
    assert out[0, 1].tolist() == np.rint(np.rint(50.5) * 0.75 + pal[1] * 0.25).astype(int).tolist()   # 50.5 → 50 (çift)
    assert out[1, 0].tolist() == np.rint(100 * 0.75 + pal[2] * 0.25).astype(int).tolist()
    assert out[1, 1].tolist() == [255, 255, 255]


# ================================================================ region growing
def test_growing_from_a_seed_fills_exactly_the_similar_connected_area():
    f = np.full((20, 30), 50, np.uint8)
    f[:, 15:] = 200
    f[5:8, 3:6] = 52                                         # benzer (|Δ| ≤ T) → bölgede
    r = S.region_growing(f, seed_x=2, seed_y=2, t=10)
    np.testing.assert_array_equal(r.report.labels, (np.mgrid[0:20, 0:30][1] < 15).astype(np.int32))
    assert r.report.count == 1 and "(stable)" in r.report.summary()
    r = S.region_growing(f, seed_x=29, seed_y=19, t=10)
    assert (r.report.labels == 1).sum() == 20 * 15


def test_threshold_controls_acceptance_inclusively():
    f = np.array([[100, 110, 121, 100]], np.uint8)
    lab = lambda t: S.region_growing(f, seed_x=0, seed_y=0, t=t).report.labels[0].tolist()
    assert lab(9) == [1, 0, 0, 0]
    assert lab(10) == [1, 1, 0, 0]                           # |110 − 100| = 10 ≤ T
    assert lab(21) == [1, 1, 1, 1]                           # ölçüt tohuma göre, komşuya göre değil
    assert lab(20) == [1, 1, 0, 0]                           # 121 − 100 = 21 > 20: 100 de artık ulaşılamaz


def test_four_versus_eight_connectivity_on_a_diagonal():
    f = np.full((7, 7), 0, np.uint8)
    np.fill_diagonal(f, 200)
    r8 = S.region_growing(f, seed_x=0, seed_y=0, t=5, connectivity="8").report.labels
    r4 = S.region_growing(f, seed_x=0, seed_y=0, t=5, connectivity="4").report.labels
    np.testing.assert_array_equal(r8, np.eye(7, dtype=np.int32))
    assert r4.sum() == 1


@pytest.mark.parametrize("conn,dist", [("8", lambda dy, dx: np.maximum(dy, dx)), ("4", lambda dy, dx: dy + dx)])
def test_constant_image_grows_everywhere_in_chessboard_or_city_block_layers(conn, dist):
    f = np.full((9, 13), 90, np.uint8)
    lab, add_at, layers, capped = S.grow_regions(f, np.array([4 * 13 + 6]), np.array([1]), 1, conn)
    yy, xx = np.mgrid[0:9, 0:13]
    np.testing.assert_array_equal(add_at, dist(np.abs(yy - 4), np.abs(xx - 6)))
    assert (lab == 1).all() and layers == add_at.max() and not capped


def test_layer_limit_stops_growth_and_is_reported():
    f = np.full((1, 30), 10, np.uint8)
    lab, add_at, layers, capped = S.grow_regions(f, np.array([0]), np.array([1]), 1, "8", max_layers=5)
    assert capped and layers == 5 and lab.sum() == 6 and add_at.max() == 5


def test_default_seed_is_image_centre_and_seed_must_be_inside():
    f = noisy(np.full((11, 16), 80), 2)
    a = S.region_growing(f).report.labels
    b = S.region_growing(f, seed_x=8, seed_y=5).report.labels
    np.testing.assert_array_equal(a, b)
    assert "seed (8, 5)" in S.region_growing(f).report.summary()
    for x, y in ((16, 0), (0, 11), (999, 999)):
        with pytest.raises(AlgorithmParameterError, match="outside"):
            S.region_growing(f, seed_x=x, seed_y=y)
    for x, y in ((0, 0), (15, 10), (-1, 10), (15, -1)):       # köşeler ve sınırlar
        assert S.region_growing(f, seed_x=x, seed_y=y).report.count == 1


def test_brightest_seeds_follow_the_lab_percentile_rule_and_give_one_region_per_group():
    f = np.full((30, 40), 60, np.uint8)
    f[5:7, 5:7] = 250
    f[20:22, 30:32] = 245
    f[5, 30] = 240
    groups, thr = S.brightest_seeds(f, 0.75)                  # ⌊1200 · 0.9925⌋ = 1191 → 9. en parlak değer
    assert thr == int(np.sort(f.ravel())[1191]) == 240
    assert groups.max() == 3 and (groups > 0).sum() == 9
    r = S.region_growing(f, seeds="brightest", q=0.75, t=30)
    assert r.report.count == 3
    np.testing.assert_array_equal(r.report.labels > 0, f >= 240)
    groups, thr = S.brightest_seeds(f, 3.0)                   # ⌊1200 · 0.97⌋ = 1164 → değer 60: her piksel tohum
    assert thr == 60 and groups.max() == 1


def test_brightest_seed_groups_are_eight_connected_as_in_the_lab():
    f = np.full((10, 10), 20, np.uint8)
    f[1, 1] = f[2, 2] = 250                                    # yalnızca çapraz komşu
    groups, thr = S.brightest_seeds(f, 1.5)                   # ⌊100 · 0.985⌋ = 98 → s[98] = 250: tam iki piksel
    assert thr == 250 and (groups > 0).sum() == 2 and groups.max() == 1   # 8-bağlantı: tek grup (ccLabel '8')
    assert S.region_growing(f, seeds="brightest", q=1.5, t=10, connectivity="4").report.count == 1


def test_seed_reference_value_is_inherited_from_the_seed_that_reached_the_pixel():
    """İki tohum: 100 ve 200. 150'lik bir köprü pikseli ikisine de |Δ| = 50 ≤ T uzaklıkta; onu önce ulaşan alır."""
    f = np.array([[100, 120, 150, 180, 200]], np.uint8)
    lab, _, _, _ = S.grow_regions(f, np.array([0, 4]), np.array([1, 2]), 50, "8")
    assert lab.tolist() == [[1, 1, 1, 2, 2]]                  # katman 1: 120 (1) ve 180 (2); katman 2: 150 → ilk cephe 1
    lab, _, _, _ = S.grow_regions(f, np.array([4, 0]), np.array([2, 1]), 50, "8")
    assert lab.tolist() == [[1, 1, 2, 2, 2]]                  # cephe sırası tersine: 150'yi 2 alır


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("conn", ["8", "4"])
def test_growing_matches_loop_reference(seed, conn):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, 25)), int(rng.integers(1, 25)))
    f = np.clip(np.rint(rng.normal(120, 35, shape)), 0, 255).astype(np.uint8)
    t = int(rng.integers(1, 60))
    if seed % 2:
        groups, _ = S.brightest_seeds(f, float(rng.choice([0.05, 1.0, 3.0])))
        idx = np.flatnonzero(groups)
        seeds = [(divmod(int(i), shape[1]), int(groups.ravel()[i])) for i in idx]
        lab, _, layers, _ = S.grow_regions(f, idx, groups.ravel()[idx], t, conn)
    else:
        y, x = int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1]))
        seeds = [((y, x), 1)]
        lab, _, layers, _ = S.grow_regions(f, np.array([y * shape[1] + x]), np.array([1]), t, conn)
    ref, ref_layers = ref_grow(f, seeds, t, conn)
    np.testing.assert_array_equal(lab, ref)
    assert layers == ref_layers


def test_growing_displays():
    f = noisy(discs(TOUCHING), 3, 2)
    r = {d: S.region_growing(f, seed_x=40, seed_y=40, t=40, display=d) for d in S.GROW_DISPLAYS}
    lab = r["overlay"].report.labels
    np.testing.assert_array_equal(r["binary"].image, np.where(lab > 0, 255, 0))
    np.testing.assert_array_equal(r["labels"].image, M.colorize_labels(lab))
    np.testing.assert_array_equal(r["difference"].image, np.abs(f.astype(int) - int(f[40, 40])))
    np.testing.assert_array_equal(r["overlay"].image, S.label_overlay(lab, f))
    assert r["overlay"].image.ndim == 3
    with pytest.raises(AlgorithmParameterError, match="single seed"):
        S.region_growing(f, seeds="brightest", display="difference")


@pytest.mark.parametrize("kwargs,match", [
    (dict(seeds="click"), "Seeds"), (dict(seed_x=-2), "Seed x"), (dict(seed_y=10000), "Seed y"),
    (dict(seed_x=1.5), "Seed x"), (dict(seed_x=True), "Seed x"), (dict(q=0.0), "top %"), (dict(q=3.5), "top %"),
    (dict(q=float("nan")), "top %"), (dict(t=0), "T"), (dict(t=121), "T"), (dict(t="40"), "T"),
    (dict(connectivity="6"), "Connectivity"), (dict(connectivity=True), "Connectivity"), (dict(display="mask"), "Display")])
def test_growing_rejects_bad_parameters(kwargs, match):
    with pytest.raises(AlgorithmParameterError, match=match):
        S.region_growing(np.zeros((5, 5), np.uint8), **kwargs)


def test_growing_accepts_limits_and_int_connectivity():
    f = noisy(np.full((6, 6), 100), 5)
    for kw in (dict(t=1), dict(t=120), dict(seeds="brightest", q=0.05), dict(seeds="brightest", q=3.0),
               dict(connectivity=4), dict(connectivity=8), dict(seed_x=5, seed_y=5)):
        assert S.region_growing(f, **kw).report.count >= 1


# ================================================================ splitting & merging
def test_constant_image_is_a_single_homogeneous_root():
    nodes = S.quadtree(np.full((64, 48), 77, np.uint8), lambda m, s: s <= 8, 4)
    assert len(nodes) == 1 and nodes[0].leaf and nodes[0].std == 0 and nodes[0].mean == 77
    r = S.split_and_merge(np.full((64, 48), 77, np.uint8))
    assert r.report.count == 1 and (r.report.labels == 1).all()


def test_four_constant_quadrants_split_once_and_merge_by_mean_difference():
    f = np.zeros((64, 64), np.uint8)
    f[:32, :32], f[:32, 32:], f[32:, :32], f[32:, 32:] = 40, 50, 120, 200
    nodes = S.quadtree(f, lambda m, s: s <= 8, 4)
    assert len(nodes) == 5 and [n.leaf for n in nodes] == [False, True, True, True, True]
    assert [(n.x, n.y, n.w, n.h, n.mean) for n in nodes[1:]] == [
        (0, 0, 32, 32, 40), (32, 0, 32, 32, 50), (0, 32, 32, 32, 120), (32, 32, 32, 32, 200)]
    lab = lambda t: S.split_and_merge(f, merge_t=t).report.labels
    assert lab(0).max() == 4
    assert lab(10).max() == 3 and lab(10)[0, 0] == lab(10)[0, 63]           # |40 − 50| = 10 ≤ 10 (kapsayıcı)
    assert lab(9).max() == 4
    assert lab(60).max() == 3                                               # 120 ↔ 200 = 80 > 60
    assert (S.split_and_merge(f, merge_t=60, a=40).report.labels == 1).sum() == 2048


def test_merging_is_transitive_through_chains_of_similar_leaves():
    f = np.zeros((32, 32), np.uint8)
    f[:16, :16], f[:16, 16:], f[16:, 16:], f[16:, :16] = 10, 20, 30, 40     # komşu farkları 10, uç farkı 30
    lab = S.split_and_merge(f, merge_t=10, min_size="16").report.labels
    assert lab.max() == 1                                                    # zincir: 10–20–30–40


@pytest.mark.parametrize("shape", [(37, 23), (23, 37), (50, 70), (33, 33), (7, 5), (1, 64), (64, 1)])
@pytest.mark.parametrize("mn", [2, 4, 8])
def test_leaves_tile_non_power_of_two_images_exactly(shape, mn):
    f = np.random.default_rng(sum(shape) + mn).integers(0, 256, shape).astype(np.uint8)
    nodes = S.quadtree(f, lambda m, s: s <= 3, mn)
    leaves = [n for n in nodes if n.leaf]
    cover = np.zeros(shape, int)
    for n in leaves:
        assert n.x >= 0 and n.y >= 0 and n.x + n.w <= shape[1] and n.y + n.h <= shape[0]
        cover[n.y:n.y + n.h, n.x:n.x + n.w] += 1
        assert (n.w >= mn and n.h >= mn) or n.depth == 0                    # min blok: kök dışında altına inilmez
    assert (cover == 1).all()                                                # örtüşme yok, boşluk yok
    for p in [n for n in nodes if not n.leaf]:
        kids = [n for n in nodes if n.depth == p.depth + 1 and p.x <= n.x < p.x + p.w and p.y <= n.y < p.y + p.h]
        assert len(kids) == 4 and {k.w for k in kids} <= {p.w // 2, p.w - p.w // 2}


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_max_depth_limits_the_quadtree(depth):
    f = np.random.default_rng(1).integers(0, 256, (64, 64)).astype(np.uint8)
    nodes = S.quadtree(f, lambda m, s: s <= 1, 2, max_depth=depth)
    assert max(n.depth for n in nodes) == depth and len([n for n in nodes if n.leaf]) == 4 ** depth
    r = S.split_and_merge(f, a=1, min_size="2", max_depth=depth, merge_t=0)
    assert f"depth {depth}" in r.report.summary()


def test_gonzalez_predicate_keeps_splitting_textured_blocks_and_marks_them_white():
    rng = np.random.default_rng(5)
    f = np.full((64, 64), 18.0)
    f[16:48, 16:48] = 95 + rng.normal(0, 24, (32, 32))                      # dokulu bölge: σ büyük, m < b
    f = np.clip(np.rint(f), 0, 255).astype(np.uint8)
    r = S.split_and_merge(f, predicate="gonzalez", a=10, b=125, min_size="8", display="merged")
    lab = r.report.labels
    assert lab[16:48, 16:48].all() and not lab[:8].any() and r.report.count == 1
    np.testing.assert_array_equal(r.image, np.where(lab > 0, 255, 0))
    nodes = S.quadtree(f, lambda m, s: s > 10 and 0 < m < 125, 8, invert=True)
    assert all(n.w == 8 for n in nodes if n.leaf and n.q)                  # Q = TRUE bloklar en küçük boyuta iner


@pytest.mark.parametrize("seed", range(10))
def test_split_and_merge_match_recursive_reference(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(4, 40)), int(rng.integers(4, 40)))
    f = np.clip(np.rint(rng.normal(100, 30, shape) + 40 * (np.mgrid[0:shape[0], 0:shape[1]][1] > shape[1] // 2)),
                0, 255).astype(np.uint8)
    gz = seed % 3 == 0
    a, b, mn, mt = int(rng.integers(1, 40)), int(rng.integers(20, 256)), int(rng.choice([2, 4, 8])), int(rng.integers(0, 60))
    pred = (lambda m, s: s > a and 0 < m < b) if gz else (lambda m, s: s <= a)
    mine = [(n.x, n.y, n.w, n.h) for n in S.quadtree(f, pred, mn, gz) if n.leaf]
    ref = ref_leaves(f, pred, mn, gz)
    assert sorted(mine) == sorted(r[:4] for r in ref)
    order = {box: k for k, box in enumerate(mine)}
    ref = sorted(ref, key=lambda r: order[r[:4]])                           # yaprak sırası etiketleri etkilemez
    similar = (lambda p, r: p[5] and r[5]) if gz else (lambda p, r: abs(p[4] - r[4]) <= mt)
    expected = ref_partition_labels(ref, shape, similar, (lambda p: p[5]) if gz else (lambda p: True))
    got = S.split_and_merge(f, "gonzalez" if gz else "homog", a, b, str(mn), merge_t=mt).report.labels
    np.testing.assert_array_equal(got, expected)


def test_split_displays():
    f = noisy(discs(TOUCHING, (48, 96)), 2, 3)
    r = {d: S.split_and_merge(f, display=d) for d in S.SPLIT_DISPLAYS}
    lab = r["merged"].report.labels
    np.testing.assert_array_equal(r["labels"].image, M.colorize_labels(lab))
    split = r["split"].image
    assert (split[0, 0] == S.TEAL_BGR).all() and (split[-1, -1] == S.TEAL_BGR).all()    # yaprak çerçeveleri
    assert r["merged"].image.ndim == 3
    assert (r["merged"].image[S.boundary_of(lab)] == 255).all()


@pytest.mark.parametrize("kwargs,match", [
    (dict(predicate="mean"), "Predicate"), (dict(a=0), "σ limit"), (dict(a=41), "σ limit"), (dict(b=19), "mean limit"),
    (dict(b=256), "mean limit"), (dict(min_size="3"), "Min block"), (dict(min_size=True), "Min block"),
    (dict(min_size=64), "Min block"), (dict(max_depth=-1), "Max depth"), (dict(max_depth=13), "Max depth"),
    (dict(merge_t=61), "Merge"), (dict(merge_t=-1), "Merge"), (dict(display="tree"), "Display")])
def test_split_rejects_bad_parameters(kwargs, match):
    with pytest.raises(AlgorithmParameterError, match=match):
        S.split_and_merge(np.zeros((8, 8), np.uint8), **kwargs)


def test_split_accepts_int_min_block_and_limits():
    f = noisy(np.full((40, 40), 90), 6)
    for kw in (dict(min_size=2), dict(min_size="32"), dict(a=1), dict(a=40), dict(b=20), dict(b=255),
               dict(max_depth=0), dict(merge_t=0), dict(merge_t=60)):
        assert S.split_and_merge(f, **kw).report.count >= 1


# ================================================================ watershed
def test_two_valleys_give_two_basins_separated_on_the_ridge():
    x = np.arange(41)
    row = np.minimum(np.abs(x - 10), np.abs(x - 30)) * 6                     # vadiler x = 10 ve 30, sırt x = 20
    s = np.tile(row, (5, 1)).astype(np.uint8)
    mk = S.regional_minima(s)
    assert mk.max() == 2 and (mk[:, 10] == 1).all() and (mk[:, 30] == 2).all()
    lab, level = S.flood(s, mk)
    assert (lab[:, :20] == 1).all() and (lab[:, 21:] == 2).all()             # sırt pikseli ilk ulaşan havzaya
    assert (level >= s).all() and level.max() == s.max()
    r = S.watershed(s, surface="intensity", sigma=0)
    np.testing.assert_array_equal(r.report.labels, lab)
    lines = S.boundary_of(lab)
    assert set(np.nonzero(lines)[1].tolist()) == {20} and lines[:, 20].all()   # 1 px çizgi sırtta (sağ komşu farklı)


def test_constant_image_is_one_basin():
    for surface in S.WATERSHED_SURFACES:
        r = S.watershed(np.full((12, 17), 90, np.uint8), surface=surface, sigma=1.5)
        assert r.report.count == 1 and (r.report.labels == 1).all()


def test_noise_over_segments_and_smoothing_reduces_the_regions():
    f = noisy(discs(TOUCHING), 8, 4)
    n = [S.watershed(f, sigma=s).report.count for s in (0, 1, 3, 6)]
    assert n[0] > 50 and n[0] > n[1] > n[2] >= n[3] >= 1
    assert S.watershed(f, sigma=0).report.count == int(S.regional_minima(S.watershed_surface(f, "gradient", 0)).max())


def test_surfaces():
    f = noisy(discs(TOUCHING), 3, 5)
    np.testing.assert_array_equal(S.watershed_surface(f, "intensity", 0), f)
    np.testing.assert_array_equal(S.watershed_surface(f, "inverted", 0), 255 - f)
    g = S.watershed_surface(f, "gradient", 0)
    assert g.max() == 255 and g.dtype == np.uint8
    gx = S.correlate(f.astype(np.float32), S.GRADIENT_OPERATORS["sobel"][0])
    gy = S.correlate(f.astype(np.float32), S.GRADIENT_OPERATORS["sobel"][1])
    m = np.hypot(gx, gy)
    assert np.abs(g - m * 255 / m.max()).max() <= 0.5 + 1e-6
    blur = S.gaussian_blur_f32(f, 2.0)
    assert blur.dtype == np.float32
    # Aynı Gauss (Canny'nin gaussian_blur'u): laboratuvar ara sonucu float32 saklar, Paket 2 float64 — q8 sonrası
    # yalnızca .5 sınırındaki nadir piksellerde 1 düzey fark olabilir (burada 9600 pikselden 1).
    d = np.abs(S.q8(blur.astype(np.float64)).astype(int) - S.gaussian_blur(f, 2.0))
    assert d.max() <= 1 and (d > 0).sum() <= 2
    # Laboratuvar q8img(gaussBlur(f, 2)) değeri (Node'da laboratuvar kodu ile ölçüldü): (24, 88) → 149;
    # float64 ara değerle 148 olurdu. Yüzey laboratuvarı izler.
    assert S.watershed_surface(f, "intensity", 2.0)[24, 88] == 149 and S.gaussian_blur(f, 2.0)[24, 88] == 148
    np.testing.assert_array_equal(S.watershed_surface(f, "inverted", 2.0),
                                  S.q8(255.0 - blur.astype(np.float64)))


@pytest.mark.parametrize("seed", range(10))
def test_minima_and_flooding_match_loop_references(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, 30)), int(rng.integers(1, 30)))
    s = rng.integers(0, int(rng.choice([3, 8, 256])), shape).astype(np.uint8)     # az düzey → geniş düzlükler
    mk = S.regional_minima(s)
    np.testing.assert_array_equal(mk, ref_minima(s))
    lab, _ = S.flood(s, mk)
    np.testing.assert_array_equal(lab, ref_flood(s, mk))
    assert (lab > 0).all()                                                    # her piksel bir havzada
    markers = np.zeros(shape, np.int32)
    pts = rng.integers(0, s.size, 3)
    markers.ravel()[pts] = [1, 2, 3]
    np.testing.assert_array_equal(S.flood(s, markers)[0], ref_flood(s, markers))


def test_flood_leaves_pixels_unreached_without_markers():
    lab, level = S.flood(np.zeros((3, 4), np.uint8), np.zeros((3, 4), np.int32))
    assert (lab == 0).all() and (level == -1).all()


def test_watershed_displays():
    f = noisy(discs(TOUCHING), 3, 6)
    r = {d: S.watershed(f, display=d) for d in S.WATERSHED_DISPLAYS}
    lab = r["overlay"].report.labels
    s = S.watershed_surface(f, "gradient", 3.0)
    np.testing.assert_array_equal(r["surface"].image, s)
    np.testing.assert_array_equal(r["boundaries"].image, np.where(S.boundary_of(lab), 255, 0))
    np.testing.assert_array_equal(r["labels"].image, M.colorize_labels(lab))
    np.testing.assert_array_equal(r["minima"].image, S.label_overlay(S.regional_minima(s), s, 0.35, 1.0))
    np.testing.assert_array_equal(r["overlay"].image, S.label_overlay(lab, f, 0.5, 0.45, S.boundary_of(lab)))


@pytest.mark.parametrize("kwargs,match", [
    (dict(surface="distance"), "Surface"), (dict(sigma=-0.5), "σ"), (dict(sigma=6.5), "σ"),
    (dict(sigma=float("inf")), "σ"), (dict(sigma="3"), "σ"), (dict(display="flood"), "Display")])
def test_watershed_rejects_bad_parameters(kwargs, match):
    with pytest.raises(AlgorithmParameterError, match=match):
        S.watershed(np.zeros((6, 6), np.uint8), **kwargs)


# ================================================================ marker-controlled watershed
def test_markers_separate_touching_objects_that_plain_watershed_over_segments():
    f = noisy(discs(TOUCHING), 6, 7)
    plain = S.watershed(f, sigma=1).report.count
    for surface in S.MARKER_SURFACES:
        r = S.marker_controlled_watershed(f, surface=surface)
        lab = r.report.labels
        assert r.report.count == 2 < plain
        assert lab[40, 40] != lab[40, 78] and lab[40, 40] > 0 and lab[40, 78] > 0
        assert lab[0, 0] == 0 and lab[-1, -1] == 0                              # arka plan etiketi 0
        st = S.marker_stages(f, "bright", 2, 3, 0.5, surface)
        assert ((lab > 0) >= st.opened).all() and not ((lab > 0) & ~st.sure_bg).any()   # açma ⊆ nesne ⊆ kesin arka plan
    truth = discs(TOUCHING) > 100
    grad = S.marker_controlled_watershed(f, surface="gradient").report.labels > 0
    dist = S.marker_controlled_watershed(f, surface="distance").report.labels > 0
    assert (grad ^ truth).sum() <= 0.01 * truth.sum()                         # gradyan: sınır nesne kenarında
    # ters uzaklık: bilinmeyen bantta yüzey sabit 255 → paylaşım taşırma sırasıyla, nesneler banda taşar (laboratuvar)
    assert (dist & ~truth).sum() > 0.1 * truth.sum() and not (truth & ~dist).any()


def test_dark_foreground_finds_dark_objects():
    f = noisy(255 - discs(TOUCHING), 4, 8)
    assert S.marker_controlled_watershed(f, foreground="dark").report.count == 2
    st = S.marker_stages(f, "dark", 2, 3, 0.5, "gradient")
    np.testing.assert_array_equal(st.binary, f <= st.threshold)


def test_marker_stages_follow_the_lab_pipeline():
    f = noisy(discs(TOUCHING + [(105, 66, 9)]), 5, 9)
    st = S.marker_stages(f, "bright", 2, 3, 0.5, "distance")
    t = S.otsu_analysis(S.histogram(f))[0]
    assert st.threshold == t
    np.testing.assert_array_equal(st.binary, f > t)
    np.testing.assert_array_equal(st.opened, ref_morph3(ref_morph3(f > t, "erode", 2), "dilate", 2))
    np.testing.assert_array_equal(st.sure_bg, ref_morph3(st.opened, "dilate", 3))
    np.testing.assert_array_equal(st.dist, ref_edt(st.opened))
    fg = st.dist > 0.5 * st.dist.max()
    np.testing.assert_array_equal(st.internal, M.label_components(fg, 8).labels)
    assert ((st.markers == 1) == ~st.sure_bg).all()
    np.testing.assert_array_equal(st.markers[fg], st.internal[fg] + 1)
    assert ((st.markers == 0) == (st.sure_bg & ~fg)).all()                       # bilinmeyen bölge
    np.testing.assert_array_equal(st.surface, S.js_round(255 - st.dist.astype(np.float64) / st.dist.max() * 255))
    np.testing.assert_array_equal(st.flooded, ref_flood(st.surface, st.markers))
    assert set(np.unique(st.flooded)) == set(range(1, int(st.internal.max()) + 2))   # her işaretleyiciden tam bir bölge


@pytest.mark.parametrize("seed", range(6))
def test_distance_transform_is_exact(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((int(rng.integers(1, 30)), int(rng.integers(1, 30)))) > rng.choice([0.3, 0.9, 0.99])
    if m.all():
        m.flat[0] = False
    np.testing.assert_array_equal(S.distance_transform(m), ref_edt(m))


def test_distance_transform_edge_cases_follow_the_lab():
    assert (S.distance_transform(np.ones((4, 5), bool)) == np.float32(1e6)).all()      # arka plan yok → √10¹²
    assert (S.distance_transform(np.zeros((4, 5), bool)) == 0).all()
    m = np.ones((1, 9), bool)
    m[0, 0] = False
    assert S.distance_transform(m).tolist() == [list(range(9))]


def test_morph3_matches_loop_reference_including_borders():
    a = np.random.default_rng(3).random((9, 11)) > 0.4
    for op in ("erode", "dilate"):
        for k in (0, 1, 2, 4):
            np.testing.assert_array_equal(S.morph3(a, op, k), ref_morph3(a, op, k))
    assert S.morph3(np.ones((3, 3), bool), "erode", 5).all()                   # görüntü dışı 1 → kenar aşınmaz


def test_no_objects_and_constant_images():
    for f in (np.full((20, 20), 0, np.uint8), np.full((20, 20), 255, np.uint8), np.full((20, 20), 90, np.uint8)):
        for surface in S.MARKER_SURFACES:
            r = S.marker_controlled_watershed(f, surface=surface)
            assert r.report.labels.shape == f.shape and r.report.count <= 1


def test_marker_parameters_change_the_markers():
    f = noisy(discs(TOUCHING), 4, 10)
    st = lambda **kw: S.marker_stages(f, "bright", kw.get("o", 2), kw.get("b", 3), kw.get("a", 0.5), "gradient")
    assert (st(b=0).sure_bg == st(b=0).opened).all()
    assert st(b=6).sure_bg.sum() > st(b=3).sure_bg.sum()
    assert (st(o=0).opened == st(o=0).binary).all()
    assert (st(a=0.9).internal > 0).sum() < (st(a=0.3).internal > 0).sum()


def test_marker_displays():
    f = noisy(discs(TOUCHING), 4, 11)
    r = {d: S.marker_controlled_watershed(f, display=d) for d in S.MARKER_DISPLAYS}
    st = S.marker_stages(f, "bright", 2, 3, 0.5, "gradient")
    lab = r["overlay"].report.labels
    np.testing.assert_array_equal(r["binary"].image, np.where(st.binary, 255, 0))
    np.testing.assert_array_equal(r["sure_bg"].image, np.where(st.sure_bg, 255, 0))
    np.testing.assert_array_equal(r["surface"].image, st.surface)
    np.testing.assert_array_equal(r["distance"].image, S.display_minmax(st.dist.astype(np.float64)))
    np.testing.assert_array_equal(r["boundaries"].image, np.where(S.boundary_of(st.flooded), 255, 0))
    np.testing.assert_array_equal(r["labels"].image, M.colorize_labels(lab))
    mk = r["markers"].image
    assert (mk[st.markers == 1] == 35).all() and (mk[st.markers == 0] == 150).all()
    np.testing.assert_array_equal(r["overlay"].image, S.label_overlay(lab, f, 0.5, 0.55, S.boundary_of(st.flooded)))


@pytest.mark.parametrize("kwargs,match", [
    (dict(foreground="light"), "Foreground"), (dict(open_iter=-1), "Opening"), (dict(open_iter=6), "Opening"),
    (dict(bg_iter=11), "background"), (dict(bg_iter=2.5), "background"), (dict(alpha=0.05), "α"),
    (dict(alpha=0.95), "α"), (dict(alpha=None), "α"), (dict(surface="intensity"), "Surface"),
    (dict(display="dams"), "Display")])
def test_marker_watershed_rejects_bad_parameters(kwargs, match):
    with pytest.raises(AlgorithmParameterError, match=match):
        S.marker_controlled_watershed(np.zeros((6, 6), np.uint8), **kwargs)
