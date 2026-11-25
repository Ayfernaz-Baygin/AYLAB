"""Thinning (Chapter 4, slayt 66–68; Gonzalez Şekil 9.23).

Bağımsız referanslar (modülün kodunu kullanmaz):
* ``ref_thin_hmt``  — B¹…B⁸'i sırayla uygulayan saf Python döngüsü: her piksel için 3×3 desen
  doğrudan kontrol edilir (dışarısı arka plan), eşleşenler silinir, değişiklik kalmayana kadar.
* ``ref_zhang_suen`` — Zhang & Suen (1984) makalesindeki koşulların piksel döngüsüyle yazılmışı.
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
from tests.test_morphology import random_binary
from tests.test_morphology_connected_components import ref_label
from tests.test_morphology_convex_hull import assert_previous_operations_unchanged

# --------------------------------------------------------------------------
# Referanslar
# --------------------------------------------------------------------------


def px(x, r, c):
    return bool(x[r, c]) if 0 <= r < x.shape[0] and 0 <= c < x.shape[1] else False


def matches(x, r, c, kern):
    for i in range(3):
        for j in range(3):
            k = kern[i][j]
            v = px(x, r + i - 1, c + j - 1)
            if (k == 1 and not v) or (k == -1 and v):
                return False
    return True


def ref_thin_hmt(a, max_cycles=None):
    x = a.copy()
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        changed = False
        for kern in M.THIN_KERNELS:
            hits = [(r, c) for r in range(x.shape[0]) for c in range(x.shape[1]) if matches(x, r, c, kern)]
            for r, c in hits:
                x[r, c] = False
            changed |= bool(hits)
        cycles += 1
        if not changed:
            break
    return x, cycles


def ref_zhang_suen(a):
    x = a.copy()
    while True:
        changed = False
        for step in (1, 2):
            dele = []
            for r in range(x.shape[0]):
                for c in range(x.shape[1]):
                    if not x[r, c]:
                        continue
                    p = [px(x, r - 1, c), px(x, r - 1, c + 1), px(x, r, c + 1), px(x, r + 1, c + 1),
                         px(x, r + 1, c), px(x, r + 1, c - 1), px(x, r, c - 1), px(x, r - 1, c - 1)]
                    b = sum(p)
                    a01 = sum(1 for i in range(8) if not p[i] and p[(i + 1) % 8])
                    p2, p4, p6, p8 = p[0], p[2], p[4], p[6]
                    if step == 1:
                        ok = not (p2 and p4 and p6) and not (p4 and p6 and p8)
                    else:
                        ok = not (p2 and p4 and p8) and not (p2 and p6 and p8)
                    if 2 <= b <= 6 and a01 == 1 and ok:
                        dele.append((r, c))
            for r, c in dele:
                x[r, c] = False
            changed |= bool(dele)
        if not changed:
            return x


def holes(a):
    return len(M.hole_labels(a, 4)[1])


def blocks_2x2(x):
    return int((x[:-1, :-1] & x[1:, :-1] & x[:-1, 1:] & x[1:, 1:]).sum())


def grid(rows):
    return np.array([[ch == "#" for ch in row] for row in rows], bool)


def u8(mask):
    return np.where(mask, 255, 0).astype(np.uint8)


def shapes():
    out = {}
    for w in (3, 4, 5, 7):
        a = np.zeros((w + 4, 24), bool)
        a[2:2 + w, 2:22] = True
        out[f"bar{w}"] = a
    r = np.zeros((27, 27), np.uint8)
    cv2.circle(r, (13, 13), 11, 1, -1)
    cv2.circle(r, (13, 13), 5, 0, -1)
    out["ring"] = r > 0
    p = np.zeros((44, 64), np.uint8)
    cv2.putText(p, "AB", (3, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 1, 5)
    out["letters"] = p > 0
    c = np.zeros((25, 25), bool)
    c[10:15, 2:23] = True
    c[2:23, 10:15] = True
    out["cross"] = c
    return out


SHAPES = shapes()

# --------------------------------------------------------------------------
# Elle hesaplanmış durumlar
# --------------------------------------------------------------------------


def test_bar_of_width_three_becomes_its_centre_line():
    a = np.zeros((7, 12), bool)
    a[2:5, 1:11] = True
    out, rep = M.thin_mask(a)
    # B¹…B⁸ SIRAYLA uygulandığı için sonuç simetrik değildir: uçlarda kısa çatallar (spur) kalır.
    expected = grid(["............",
                     "............",
                     ".#........#.",
                     ".##########.",
                     ".#..........",
                     "............",
                     "............"])
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out, ref_thin_hmt(a)[0])
    assert rep.converged and blocks_2x2(out) == 0


def test_single_pass_of_b1_removes_top_edge():
    """B¹ = [−1 −1 −1; 0 1 0; 1 1 1]: üstü arka plan, altı dolu pikseller — yani üst kenar."""
    a = np.zeros((6, 8), bool)
    a[1:5, 1:7] = True
    hits = M.hit_or_miss_mask(a, M.THIN_KERNELS[0])
    np.testing.assert_array_equal(np.nonzero(hits)[0], np.full(4, 1))
    np.testing.assert_array_equal(np.nonzero(hits)[1], np.arange(2, 6))


def test_one_pixel_lines_and_points_are_stable():
    a = np.zeros((9, 9), bool)
    a[4, 1:8] = True                       # yatay çizgi
    a[0, 0] = True                         # tek nokta
    for method in M.THIN_METHODS:
        out, rep = M.thin_mask(a, method)
        assert rep.converged and rep.cycles == 1
        if method == "hmt":
            np.testing.assert_array_equal(out, a)


def test_isolated_2x2_square():
    """B², B⁴, B⁶, B⁸ birer köşeyle eşleşir; sırayla uygulandığından yalnızca B² bir piksel siler
    (sonra kalan L üçlüsü sabittir). Zhang–Suen ise 2×2 kareyi tamamen siler (bilinen kusuru)."""
    a = np.zeros((6, 6), bool)
    a[2:4, 2:4] = True
    assert [int(M.hit_or_miss_mask(a, k).sum()) for k in M.THIN_KERNELS] == [0, 1, 0, 1, 0, 1, 0, 1]
    expected = a.copy()
    expected[2, 3] = False                       # B² sağ üst köşeyi siler
    np.testing.assert_array_equal(M.thin_mask(a, "hmt")[0], expected)
    assert not M.thin_mask(a, "zhang_suen")[0].any()


def test_empty_and_full_images():
    for method in M.THIN_METHODS:
        out, rep = M.thin_mask(np.zeros((5, 7), bool), method)
        assert not out.any() and rep.converged and rep.cycles == 1
        full, rep = M.thin_mask(np.ones((9, 15), bool), method)    # dışarısı arka plan → inceltilir
        assert rep.converged and 0 < full.sum() < 9 * 15
        assert ref_label(full, 8).max() == 1


# --------------------------------------------------------------------------
# Referanslar ve özellikler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_hmt_matches_sequential_reference(name):
    a = SHAPES[name]
    out, rep = M.thin_mask(a, "hmt")
    ref, cycles = ref_thin_hmt(a)
    np.testing.assert_array_equal(out, ref)
    assert rep.cycles == cycles and rep.converged


@pytest.mark.parametrize("seed", range(5))
def test_hmt_matches_reference_on_random_images(seed):
    a = random_binary(700 + seed, (18, 22), 0.6)
    np.testing.assert_array_equal(M.thin_mask(a, "hmt")[0], ref_thin_hmt(a)[0])


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_zhang_suen_matches_reference(name):
    a = SHAPES[name]
    np.testing.assert_array_equal(M.thin_mask(a, "zhang_suen")[0], ref_zhang_suen(a))


@pytest.mark.parametrize("method", M.THIN_METHODS)
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_topology_preserved_and_one_pixel_wide(method, name):
    a = SHAPES[name]
    out, _ = M.thin_mask(a, method)
    assert not (out & ~a).any()                                   # yalnızca siler
    assert ref_label(out, 8).max() == ref_label(a, 8).max()       # bileşen sayısı korunur
    assert holes(out) == holes(a)                                 # delik sayısı korunur
    assert blocks_2x2(out) == 0                                   # tek piksel kalınlık (2×2 blok yok)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_hmt_result_is_a_fixed_point(name):
    out, _ = M.thin_mask(SHAPES[name], "hmt")
    for kern in M.THIN_KERNELS:
        assert not M.hit_or_miss_mask(out, kern).any()            # hiçbir B ile eşleşme kalmadı
    again, rep = M.thin_mask(out, "hmt")
    np.testing.assert_array_equal(again, out)
    assert rep.cycles == 1 and rep.converged


def test_multiple_objects_are_thinned_independently():
    two = np.zeros((30, 60), bool)
    two[2:7, 2:30] = True
    two[12:28, 40:55] = True
    out, _ = M.thin_mask(two)
    assert ref_label(out, 8).max() == 2
    assert out[:10].any() and out[10:].any()
    np.testing.assert_array_equal(out, ref_thin_hmt(two)[0])


def test_max_cycles_limit_and_convergence_flag():
    a = SHAPES["bar7"]
    one, rep1 = M.thin_mask(a, "hmt", 1)
    ref1, _ = ref_thin_hmt(a, 1)
    np.testing.assert_array_equal(one, ref1)
    assert rep1.cycles == 1 and not rep1.converged
    full, rep = M.thin_mask(a, "hmt", 0)
    assert rep.converged and rep.cycles > 2
    sizes = [M.thin_mask(a, "hmt", n)[0].sum() for n in range(1, rep.cycles + 1)]
    assert all(x >= y for x, y in zip(sizes, sizes[1:])) and sizes[-1] == full.sum()
    assert "not converged" in rep1.summary() and "converged after" in rep.summary()


@pytest.mark.parametrize("value", [-1, 1.5, True, "3", None])
def test_invalid_max_cycles(value):
    with pytest.raises(AlgorithmParameterError):
        M.thin_mask(SHAPES["bar3"], "hmt", value)


# --------------------------------------------------------------------------
# Görüntü düzeyi
# --------------------------------------------------------------------------


def test_threshold_polarity_and_color():
    a = SHAPES["letters"]
    expected = u8(ref_thin_hmt(a)[0])
    np.testing.assert_array_equal(M.thinning(u8(a)).image, expected)
    np.testing.assert_array_equal(M.thinning(255 - u8(a), foreground="dark").image, expected)
    bgr = np.zeros(a.shape + (3,), np.uint8)
    bgr[a] = (0, 255, 0)
    bgr[~a] = (255, 0, 0)
    out = M.thinning(bgr, binarize="manual", thresh=100).image
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], expected)


@pytest.mark.parametrize("shape", [(9, 24), (9, 24, 1), (9, 24, 3), (9, 24, 4)])
def test_layout_alpha_immutability_and_report(shape):
    base = u8(SHAPES["bar5"])
    img = base if len(shape) == 2 else np.repeat(base[..., None], shape[2], axis=2).copy()
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = 77
    before = img.copy()
    res = M.thinning(img, binarize="manual")
    np.testing.assert_array_equal(img, before)
    assert res.image.shape == img.shape and res.image.dtype == np.uint8 and not np.shares_memory(res.image, img)
    if len(shape) == 3 and shape[2] == 4:
        assert (res.image[..., 3] == 77).all()
    assert res.report.converged and res.report.before == SHAPES["bar5"].sum()


@pytest.mark.parametrize("kwargs", [dict(method="guo_hall"), dict(max_cycles=-2), dict(binarize="auto"),
                                    dict(foreground="white")])
def test_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.thinning(np.zeros((5, 5), np.uint8), **kwargs)


def test_registry_entry_and_previous_operations():
    algo = build_registry().get("thinning")
    assert algo.title == "Thinning" and algo.implemented
    assert [p.key for p in algo.params] == ["binarize", "thresh", "foreground", "method", "max_cycles"]
    assert algo.default_params()["method"] == "hmt" and algo.default_params()["max_cycles"] == 0
    assert "A ⊗ B = A − (A ⊛ B)" in algo.summary
    assert_previous_operations_unchanged(94)


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    img = np.repeat(np.where(SHAPES["letters"], 220, 30).astype(np.uint8)[..., None], 3, axis=2)
    path = tmp_path / "İnceltme" / "harfler.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


@pytest.mark.parametrize("method", M.THIN_METHODS)
def test_ui_apply(window, method):
    w, img = window
    w.navigator.select_algorithm("thinning")
    w.parameters.controls["method"].set_value(method)
    w.parameters.controls["max_cycles"].set_value(0)
    w.parameters.apply_button.click()
    a = img[..., 0] > 127
    exp = ref_thin_hmt(a)[0] if method == "hmt" else ref_zhang_suen(a)
    np.testing.assert_array_equal(w.current_image[..., 0], u8(exp))
    np.testing.assert_array_equal(w.original_image, img)
    assert w.telemetry.state.state() == "DONE" and "converged after" in w.statusBar().currentMessage()
    assert not w.parameters.report_view.isVisible()
    for name, counts in histograms(w.current_image).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)
    w.reset_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
