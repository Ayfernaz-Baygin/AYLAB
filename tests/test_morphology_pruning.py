"""Pruning (Chapter 4, slayt 78–81; Gonzalez Şekil 9.27, laboratuvar ``prune``).

    X1 = A ⊗ {B}   (uç nokta çekirdekleri B¹…B⁸, n döngü)     X2 = ⋃ (X1 ⊛ Bᵏ)
    X3 = (X2 ⊕ H) ∩ A   (n kez, H = 3×3)                         X4 = X1 ∪ X3

Bağımsız referans saf Python'dur (OpenCV yok): piksel piksel hit-or-miss (görüntü dışı arka plan),
sıralı inceltme ve 3×3 koşullu dilation. Şekil 9.27 matrisleri slayt 79'daki şeklin (a), (c),
(d), (e), (f) panellerinden hücre hücre okunmuştur.
"""

import numpy as np
import pytest

from app.main_window import MainWindow
from processing import morphology as M
from processing.analysis import histograms
from processing.catalog import build_registry
from processing.image_io import save_image
from processing.registry import AlgorithmParameterError
from tests.test_morphology import random_binary
from tests.test_morphology_convex_hull import grid, u8
from tests.test_morphology_skeletonization import assert_previous_15_operations_unchanged
from tests.test_morphology_thinning import ref_thin_hmt

# --------------------------------------------------------------------------
# Bağımsız referans
# --------------------------------------------------------------------------


def px(x, r, c):
    return bool(x[r, c]) if 0 <= r < x.shape[0] and 0 <= c < x.shape[1] else False


def ref_hmt(x, kern):
    out = np.zeros_like(x)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            ok = True
            for i in range(3):
                for j in range(3):
                    k, v = kern[i][j], px(x, r + i - 1, c + j - 1)
                    if (k == 1 and not v) or (k == -1 and v):
                        ok = False
            out[r, c] = ok
    return out


def ref_dilate3(x):
    out = np.zeros_like(x)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            out[r, c] = any(px(x, r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1))
    return out


def ref_prune(a, n):
    x1, cycles = a.copy(), 0
    while cycles < n:
        changed = False
        for k in FIG_927_B:
            hits = ref_hmt(x1, k)
            x1 &= ~hits
            changed |= bool(hits.any())
        cycles += 1
        if not changed:
            break
    x2 = np.zeros_like(a)
    for k in FIG_927_B:
        x2 |= ref_hmt(x1, k)
    x3 = x2.copy()
    for _ in range(n):
        x3 = ref_dilate3(x3) & a
    return {"A": a, "X1": x1, "X2": x2, "X3": x3, "X4": x1 | x3}, cycles


# Şekil 9.27(b) — gölgeli = 1, beyaz = −1, × = 0 (önemsiz)
FIG_927_B = [
    [[0, -1, -1], [1, 1, -1], [0, -1, -1]],
    [[0, 1, 0], [-1, 1, -1], [-1, -1, -1]],
    [[-1, -1, 0], [-1, 1, 1], [-1, -1, 0]],
    [[-1, -1, -1], [-1, 1, -1], [0, 1, 0]],
    [[1, -1, -1], [-1, 1, -1], [-1, -1, -1]],
    [[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]],
    [[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    [[-1, -1, -1], [-1, 1, -1], [1, -1, -1]],
]

FIG_927_A = grid(["................", "......###.......", ".....#...#......", ".#..#...........", ".#.#......#.....",
                  ".##.......##....", "..#......#.#....", "...#.....#.#....", "..#......#.#....", "..#.....#..#..#.",
                  "...#####....##..", "................"])
FIG_927_C = grid(["................", "................", ".....#..........", "....#...........", "...#............",
                  "..#.......##....", "..#......#.#....", "...#.....#.#....", "..#......#.#....", "..#.....#.......",
                  "...#####........", "................"])
FIG_927_E = grid(["................", "......###.......", ".....#..........", "....#...........", "...#............",
                  "..#.......##....", "...........#....", "...........#....", "...........#....", "...........#....",
                  "............##..", "................"])
FIG_927_F = grid(["................", "......###.......", ".....#..........", "....#...........", "...#............",
                  "..#.......##....", "..#......#.#....", "...#.....#.#....", "..#......#.#....", "..#.....#..#....",
                  "...#####....##..", "................"])


def line_with_spur(spur_len, length=22, height=10):
    a = np.zeros((height, length + 2), bool)
    a[height - 3, 1:length + 1] = True
    col = 1 + length // 2
    a[height - 3 - spur_len:height - 3, col] = True
    return a


LAB_SPURS = grid(["....................", "........#...........", "........#...........", ".##################.",
                  "..........#...#.....", "...........#..#.....", "..............#.....", "..............#.....",
                  "..............#.....", "...................."])

# --------------------------------------------------------------------------
# Çekirdekler ve uç nokta tespiti
# --------------------------------------------------------------------------


def test_kernels_are_figure_9_27b():
    assert len(M.PRUNE_KERNELS) == 8
    for k, ref in zip(M.PRUNE_KERNELS, FIG_927_B):
        np.testing.assert_array_equal(k, np.array(ref))
    # B¹…B⁴ ve B⁵…B⁸ kendi içinde 90° döndürmelerdir (dört yön)
    rot = lambda k: {np.rot90(np.array(k), i).tobytes() for i in range(4)}
    assert rot(FIG_927_B[0]) == {np.array(k).tobytes() for k in FIG_927_B[:4]}
    assert rot(FIG_927_B[4]) == {np.array(k).tobytes() for k in FIG_927_B[4:]}


@pytest.mark.parametrize("seed", range(4))
def test_existing_end_point_pattern_is_a_special_case_of_B1(seed):
    """Hit-or-Miss işlemindeki ``end_point`` deseni = B¹'in × hücreleri −1 yapılmış hali → eşleşmeleri ⊆ B¹."""
    a = random_binary(300 + seed, (20, 24), 0.25)
    strict = M.hit_or_miss_mask(a, M.hmt_kernel("end_point"))
    b1 = M.hit_or_miss_mask(a, M.PRUNE_KERNELS[0])
    assert not (strict & ~b1).any()
    np.testing.assert_array_equal(M.hmt_kernel("end_point"), np.where(M.PRUNE_KERNELS[0] == 0, -1, M.PRUNE_KERNELS[0]))


@pytest.mark.parametrize("seed", range(5))
def test_end_point_detection_matches_reference(seed):
    a = random_binary(400 + seed, (18, 22), 0.3)
    got = np.zeros_like(a)
    for k in M.PRUNE_KERNELS:
        got |= M.hit_or_miss_mask(a, k)
    exp = np.zeros_like(a)
    for k in FIG_927_B:
        exp |= ref_hmt(a, k)
    np.testing.assert_array_equal(got, exp)
    # tek 8-komşulu her ön plan pikseli bir uç noktadır
    for r, c in zip(*np.nonzero(a)):
        nb = sum(px(a, r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1)) - 1
        if nb == 1:
            assert got[r, c]
        if nb == 0:
            assert not got[r, c]                                          # yalnız piksel uç nokta değildir


# --------------------------------------------------------------------------
# Şekil 9.27 (slayt 79), n = 3
# --------------------------------------------------------------------------


def test_figure_9_27_every_step():
    steps, rep = M.prune_mask(FIG_927_A, 3)
    np.testing.assert_array_equal(steps["X1"], FIG_927_C)                # (c) üç döngü inceltme
    assert sorted(zip(*np.nonzero(steps["X2"]))) == [(2, 5), (8, 11)]    # (d) uç noktalar
    np.testing.assert_array_equal(steps["X3"], FIG_927_E)                # (e) A ile koşullu dilation
    np.testing.assert_array_equal(steps["X4"], FIG_927_F)                # (f) budanmış görüntü
    assert (rep.n, rep.thinning_cycles, rep.end_points) == (3, 3, 2)
    assert rep.a_px - rep.x4_px == int((FIG_927_A & ~FIG_927_F).sum()) == 6


# --------------------------------------------------------------------------
# Bağımsız referans ve n karşılaştırması
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", range(4))
def test_all_steps_match_reference(n, seed):
    a = ref_thin_hmt(random_binary(500 + seed, (16, 20), 0.45))[0]       # ince (iskelet benzeri) girdi
    steps, rep = M.prune_mask(a, n)
    ref, cycles = ref_prune(a, n)
    for key in ("X1", "X2", "X3", "X4"):
        np.testing.assert_array_equal(steps[key], ref[key], err_msg=key)
    assert rep.thinning_cycles == cycles
    assert not (steps["X4"] & ~a).any() and not (steps["X1"] & ~steps["X4"]).any()   # X1 ⊆ X4 ⊆ A


def test_spur_shorter_than_n_is_removed_main_branch_kept():
    a = line_with_spur(2)
    main = a.copy()
    main[:a.shape[0] - 3] = False
    np.testing.assert_array_equal(M.prune_mask(a, 1)[0]["X4"], a)        # n = 1 < 2: spur kalır
    np.testing.assert_array_equal(M.prune_mask(a, 3)[0]["X4"], main)     # n = 3: spur gider, ana dal tam
    x5 = M.prune_mask(a, 5)[0]["X4"]                                     # n = 5: ana dal tam, spur'un dibi geri büyür
    assert not (main & ~x5).any() and (x5 & ~main).sum() == 1


def test_spur_longer_than_n_is_kept():
    a = line_with_spur(5)
    for n in (1, 3):
        np.testing.assert_array_equal(M.prune_mask(a, n)[0]["X4"], a)
    x5 = M.prune_mask(a, 5)[0]["X4"]                                     # n = 5 ≥ 5: spur budanır,
    main = a.copy()
    main[:a.shape[0] - 3] = False
    assert not (main & ~x5).any() and (x5 & ~main).sum() == 1             # dibindeki 1 piksel geri büyür


def test_spur_next_to_a_real_end_point_grows_back():
    """Bilinen sınırlama: X3, gerçek dalın uç noktasından A içinde n adım büyür; bu bölgede kalan
    kısa bir spur da geri büyür ve budanmaz."""
    a = np.zeros((7, 16), bool)
    a[5, 0:12] = True
    a[3:5, 9] = True                                                     # gerçek uca 2 piksel uzaklıkta 2'lik spur
    steps, _ = M.prune_mask(a, 3)
    assert not (steps["X1"] & ~a).any() and not steps["X1"][3:5, 9].any()
    np.testing.assert_array_equal(steps["X4"], a)


def test_short_real_branches_are_removed_too():
    """n'den kısa GERÇEK dallar da silinir (laboratuvardaki 'spurs' örneği): sol üstteki dikey dal ve
    köşegen dal (2 piksel) n = 2 ve n = 3'te silinir; 5 piksellik dikey dal kalır."""
    removed = {(1, 8), (2, 8), (4, 10), (5, 11)}
    for n in (2, 3):
        x4 = M.prune_mask(LAB_SPURS, n)[0]["X4"]
        assert set(zip(*np.nonzero(LAB_SPURS & ~x4))) == removed
        assert x4[3, 1:19].all() and x4[4:9, 14].all()
    seg = np.zeros((5, 10), bool)
    seg[2, 2:7] = True                                                   # 5 piksellik tek parça
    x4 = M.prune_mask(seg, 2)[0]["X4"]
    assert x4.sum() == 1 and x4[2, 4]                                     # yalnız merkez pikseli kalır


def test_larger_n_is_not_monotonic():
    """n büyütmek her zaman daha çok budamaz: n = 5'te ana dallar 5 piksel kısalır ve 5 adım geri
    büyürken kısa dalları da geri getirir (laboratuvarın 'spurs' örneği)."""
    removed = [int((LAB_SPURS & ~M.prune_mask(LAB_SPURS, n)[0]["X4"]).sum()) for n in (1, 2, 3, 5)]
    assert removed == [0, 4, 4, 0]


@pytest.mark.parametrize("n", [1, 3, 5])
def test_sets_without_spurs_are_unchanged(n):
    ring = np.zeros((9, 9), bool)
    ring[1, 1:8] = ring[7, 1:8] = True
    ring[1:8, 1] = ring[1:8, 7] = True
    np.testing.assert_array_equal(M.prune_mask(ring, n)[0]["X4"], ring)  # uç nokta yok → X1 = A
    line = np.zeros((5, 24), bool)
    line[2, 1:23] = True
    np.testing.assert_array_equal(M.prune_mask(line, n)[0]["X4"], line)  # uçlar n kısalıp geri büyür


def test_isolated_pixel_and_empty():
    one = np.zeros((5, 5), bool)
    one[2, 2] = True
    steps, rep = M.prune_mask(one, 3)
    np.testing.assert_array_equal(steps["X4"], one)                      # hiçbir Bᵏ yalnız pikselle eşleşmez
    assert rep.thinning_cycles == 1 and rep.end_points == 0
    steps, rep = M.prune_mask(np.zeros((4, 6), bool), 2)
    assert not any(v.any() for v in steps.values()) and rep.x4_px == 0
    assert "removed 0 px" in rep.summary()


@pytest.mark.parametrize("n", [0, -1, 2.5, True, "3"])
def test_invalid_n(n):
    with pytest.raises(AlgorithmParameterError):
        M.prune_mask(FIG_927_A, n)


@pytest.mark.parametrize("kwargs", [dict(display="tree"), dict(n=0), dict(foreground="white")])
def test_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.pruning(u8(FIG_927_A), **kwargs)


# --------------------------------------------------------------------------
# Görüntü düzeyi
# --------------------------------------------------------------------------


def big927():
    return np.kron(FIG_927_A, np.ones((2, 2), bool))


@pytest.mark.parametrize("display,key", [("result", "X4"), ("x1", "X1"), ("x2", "X2"), ("x3", "X3")])
def test_binary_displays(display, key):
    steps, _ = ref_prune(FIG_927_A, 3)
    res = M.pruning(u8(FIG_927_A), display=display)
    np.testing.assert_array_equal(res.image, u8(steps[key]))


def test_overlay_colours_and_report():
    steps, _ = ref_prune(FIG_927_A, 3)
    res = M.pruning(u8(FIG_927_A), display="overlay")
    out, c = res.image, M.PRUNE_COLORS
    assert out.shape == FIG_927_A.shape + (3,)
    assert (out[FIG_927_A & ~steps["X4"]] == c["removed"]).all()
    assert (out[steps["X1"] & ~steps["X2"]] == c["kept"]).all()
    assert (out[steps["X4"] & ~steps["X1"]] == c["restored"]).all()
    assert (out[steps["X2"]] == c["end_point"]).all() and not out[~FIG_927_A].any()
    text = res.report.summary()
    for part in ("n = 3", "3 cycles", "2 end points", "removed 6 px", "X4 = X1 ∪ X3 28 px"):
        assert part in text


def test_threshold_polarity_and_color_input():
    exp = u8(ref_prune(FIG_927_A, 3)[0]["X4"])
    np.testing.assert_array_equal(M.pruning(255 - u8(FIG_927_A), foreground="dark").image, exp)
    bgr = np.zeros(FIG_927_A.shape + (3,), np.uint8)
    bgr[FIG_927_A] = (250, 30, 30)
    out = M.pruning(bgr, binarize="manual", thresh=20).image
    for ch in range(3):
        np.testing.assert_array_equal(out[..., ch], exp)


@pytest.mark.parametrize("display", M.PRUNE_DISPLAYS)
@pytest.mark.parametrize("shape", [(24, 32), (24, 32, 1), (24, 32, 3), (24, 32, 4)])
def test_layout_alpha_immutability(display, shape):
    base = u8(big927())
    img = base if len(shape) == 2 else np.repeat(base[..., None], shape[2], axis=2).copy()
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = 77
    before = img.copy()
    out = M.pruning(img, display=display).image
    np.testing.assert_array_equal(img, before)
    assert out.dtype == np.uint8 and not np.shares_memory(out, img)
    if len(shape) == 3 and shape[2] == 4:
        assert out.shape == img.shape and (out[..., 3] == 77).all()
    elif display == "overlay":
        assert out.shape == shape[:2] + (3,)
    else:
        assert out.shape == img.shape


def test_registry_entry_and_previous_operations():
    algo = build_registry().get("pruning")
    assert algo.title == "Pruning" and algo.implemented
    assert [p.key for p in algo.params] == ["binarize", "thresh", "foreground", "n", "display"]
    assert algo.default_params()["n"] == 3 and algo.default_params()["display"] == "result"
    assert "X4 = X1 ∪ X3" in algo.summary and "real branches" in algo.summary
    n_help = next(p for p in algo.params if p.key == "n").help
    assert "REAL" in n_help and "grow back" in n_help
    assert_previous_15_operations_unchanged(97)


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    img = np.repeat(u8(FIG_927_A)[..., None], 3, axis=2)
    path = tmp_path / "Budama" / "şekil_9_27.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


def test_ui_apply_steps_and_undo(window):
    w, img = window
    w.navigator.select_algorithm("pruning")
    w.parameters.apply_button.click()
    np.testing.assert_array_equal(w.current_image[..., 0], u8(FIG_927_F))
    assert w.telemetry.state.state() == "DONE" and "2 end points" in w.statusBar().currentMessage()
    assert not w.parameters.report_view.isVisible()
    for name, counts in histograms(w.current_image).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
    w.parameters.controls["display"].set_value("x1")
    w.parameters.apply_button.click()
    np.testing.assert_array_equal(w.current_image[..., 0], u8(FIG_927_C))
    np.testing.assert_array_equal(w.original_image, img)


def test_ui_chain_thinning_then_pruning(window):
    w, img = window
    w.navigator.select_algorithm("thinning")
    w.parameters.apply_button.click()
    thin = w.current_image.copy()
    w.navigator.select_algorithm("pruning")
    w.parameters.controls["n"].set_value(2)
    w.parameters.apply_button.click()
    np.testing.assert_array_equal(w.current_image, M.pruning(thin, n=2).image)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, thin)
    w.reset_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
