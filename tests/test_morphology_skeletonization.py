"""Skeletonization (Chapter 4, slayt 72–77; Gonzalez Şekil 9.26, laboratuvar ``skeleton``).

    S_k(A) = (A ⊖ kB) − (A ⊖ kB) ∘ B,   K = max{k | A ⊖ kB ≠ ∅},   S(A) = ⋃ S_k(A)
    A = ⋃ (S_k(A) ⊕ kB)

Bağımsız referans OpenCV kullanmaz: erosion/dilation, ``tests.test_morphology`` içindeki
öteleme (shift) tanımlarıyla kurulur; görüntü dışı arka plandır. SE maskeleri rect/cross için
burada elle kurulur (elips maskesi SE'nin tanımı olduğundan ``structuring_element``'ten alınır).
Şekil 9.26 beklentileri ders slaytındaki (slayt 75) şekilden hücre hücre okunmuştur.
"""

import numpy as np
import pytest

from app.main_window import MainWindow
from processing import morphology as M
from processing.analysis import histograms
from processing.catalog import build_registry
from processing.image_io import save_image
from processing.registry import AlgorithmParameterError
from tests.test_morphology import random_binary, ref_dilate, ref_erode
from tests.test_morphology_connected_components import ref_label
from tests.test_morphology_convex_hull import assert_previous_operations_unchanged, grid, ref_hull_contour, u8
from tests.test_morphology_thickening import ref_thicken
from tests.test_morphology_thinning import ref_thin_hmt

# --------------------------------------------------------------------------
# Bağımsız referans
# --------------------------------------------------------------------------


def se_mask(shape, w, h):
    if shape == "rect":
        return np.ones((h, w), bool)
    if shape == "cross":
        m = np.zeros((h, w), bool)
        m[h // 2, :] = True
        m[:, w // 2] = True
        return m
    return M.structuring_element("ellipse", w, h).mask.astype(bool)


def ref_skeleton(a, mask):
    """[(A ⊖ kB, (A ⊖ kB) ∘ B, S_k)] k = 0…K — doğrudan tanımlardan; merkez orijin."""
    anchor = (mask.shape[1] // 2, mask.shape[0] // 2)
    out, e = [], a.copy()
    while e.any():
        o = ref_dilate(ref_erode(e, mask, anchor), mask, anchor)
        out.append((e, o, e & ~o))
        e = ref_erode(e, mask, anchor)
    return out


def ref_reconstruct(levels, mask):
    anchor = (mask.shape[1] // 2, mask.shape[0] // 2)
    out = np.zeros(levels[0][0].shape, bool)
    for k, (_, _, s) in enumerate(levels):
        x = s.copy()
        for _ in range(k):
            x = ref_dilate(x, mask, anchor)
        out |= x
    return out


def union_skeleton(levels):
    s = np.zeros(levels[0][0].shape, bool)
    for _, _, sk in levels:
        s |= sk
    return s


SE_LIST = [("rect", 3, 3), ("cross", 3, 3), ("ellipse", 5, 5), ("rect", 3, 5), ("rect", 3, 1), ("cross", 5, 5)]


def assert_previous_15_operations_unchanged(seed):
    """Önceki 15 morfoloji işlemi: 12'si ``assert_previous_operations_unchanged`` ile, Convex Hull /
    Thinning / Thickening varsayılan parametrelerle bağımsız referanslara karşı."""
    assert_previous_operations_unchanged(seed)
    registry = build_registry()
    a = random_binary(seed, (18, 22), 0.5)
    img = u8(a)
    np.testing.assert_array_equal(registry.get("convex_hull").run(img), u8(ref_hull_contour(a)))
    np.testing.assert_array_equal(registry.get("thinning").run(img), u8(ref_thin_hmt(a)[0]))
    np.testing.assert_array_equal(registry.get("thickening").run(img), u8(ref_thicken(a, 10)[0]))


# --------------------------------------------------------------------------
# Şekil 9.26 (slayt 75) — B = 3×3 kare
# --------------------------------------------------------------------------

FIG_926 = grid(["#....", ".##..", ".##..", ".###.", ".###.", "#####", "#####", "#####", "#####", "#####"])


def cells(*rc):
    m = np.zeros(FIG_926.shape, bool)
    for r, c in rc:
        m[r, c] = True
    return m


def test_figure_9_26_every_column():
    se = M.structuring_element("rect", 3, 3)
    levels = M.skeleton_levels(FIG_926, se)
    assert len(levels) - 1 == 2                                           # K = 2
    e1 = cells((4, 2), (5, 2), *[(r, c) for r in (6, 7, 8) for c in (1, 2, 3)])
    e2 = cells((7, 2))
    o0 = FIG_926 & ~cells((0, 0), (1, 1), (1, 2), (2, 1), (2, 2))
    expected = [  # (A ⊖ kB, (A ⊖ kB) ∘ B, S_k) — şeklin 1.–3. sütunları
        (FIG_926, o0, cells((0, 0), (1, 1), (1, 2), (2, 1), (2, 2))),
        (e1, cells(*[(r, c) for r in (6, 7, 8) for c in (1, 2, 3)]), cells((4, 2), (5, 2))),
        (e2, np.zeros_like(FIG_926), e2),
    ]
    for k, ((e, o, s), (ee, oo, ss)) in enumerate(zip(levels, expected)):
        np.testing.assert_array_equal(e, ee, err_msg=f"A ⊖ {k}B")
        np.testing.assert_array_equal(o, oo, err_msg=f"(A ⊖ {k}B) ∘ B")
        np.testing.assert_array_equal(s, ss, err_msg=f"S_{k}")
    # 4. sütun: S(A); 5.–6. sütunlar: S_k ⊕ kB ve birleşimleri (son satırda A)
    np.testing.assert_array_equal(union_skeleton(levels),
                                  cells((0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (4, 2), (5, 2), (7, 2)))
    np.testing.assert_array_equal(M.binary_dilation(levels[1][2], se, 1),
                                  cells(*[(r, c) for r in range(3, 7) for c in (1, 2, 3)]))
    np.testing.assert_array_equal(M.binary_dilation(levels[2][2], se, 2),
                                  cells(*[(r, c) for r in range(5, 10) for c in range(5)]))
    np.testing.assert_array_equal(M.skeleton_reconstruct(levels, se), FIG_926)


# --------------------------------------------------------------------------
# Formül: her S_k ayrı ayrı, K ve yeniden yapılandırma
# --------------------------------------------------------------------------


@pytest.mark.parametrize("se_case", SE_LIST)
@pytest.mark.parametrize("seed", range(4))
def test_each_level_matches_definition(se_case, seed):
    shape, w, h = se_case
    a = random_binary(1000 + seed, (19, 23), 0.75)
    se = M.structuring_element(shape, w, h)
    levels, ref = M.skeleton_levels(a, se), ref_skeleton(a, se_mask(shape, w, h))
    assert len(levels) == len(ref)
    for k, ((e, o, s), (re, ro, rs)) in enumerate(zip(levels, ref)):
        np.testing.assert_array_equal(e, re, err_msg=f"A ⊖ {k}B")
        np.testing.assert_array_equal(o, ro, err_msg=f"(A ⊖ {k}B) ∘ B")
        np.testing.assert_array_equal(s, rs, err_msg=f"S_{k}")
        assert not (s & o).any() and np.array_equal(s | o, e)            # S_k ⊎ opening = A ⊖ kB


@pytest.mark.parametrize("se_case", SE_LIST)
def test_K_is_last_non_empty_erosion(se_case):
    shape, w, h = se_case
    se = M.structuring_element(shape, w, h)
    a = random_binary(77, (21, 25), 0.85)
    levels = M.skeleton_levels(a, se)
    K = len(levels) - 1
    assert M.binary_erosion(a, se, K).any() if K else a.any()
    assert not M.binary_erosion(a, se, K + 1).any()


@pytest.mark.parametrize("se_case", SE_LIST)
@pytest.mark.parametrize("seed", range(5))
def test_reconstruction_is_exactly_A(se_case, seed):
    shape, w, h = se_case
    a = random_binary(2000 + seed, (20, 26), 0.7)
    se = M.structuring_element(shape, w, h)
    levels = M.skeleton_levels(a, se)
    rec = M.skeleton_reconstruct(levels, se)
    np.testing.assert_array_equal(rec, a)
    np.testing.assert_array_equal(rec, ref_reconstruct(ref_skeleton(a, se_mask(shape, w, h)), se_mask(shape, w, h)))
    for k, (_, _, s) in enumerate(levels):                               # her katman A'nın içinde kalır
        assert not ((M.binary_dilation(s, se, k) if k else s) & ~a).any()


# --------------------------------------------------------------------------
# Özel şekiller ve kenar durumları
# --------------------------------------------------------------------------


def test_empty_image():
    se = M.structuring_element()
    assert M.skeleton_levels(np.zeros((6, 7), bool), se) == []
    res = M.skeletonization(np.zeros((6, 7), np.uint8), binarize="manual")
    assert not res.image.any() and res.report.levels == () and "empty" in res.report.summary()
    with pytest.raises(AlgorithmParameterError):
        M.skeleton_reconstruct([], se)
    assert not M.skeletonization(np.zeros((6, 7), np.uint8), binarize="manual", display="reconstruction").image.any()


@pytest.mark.parametrize("a", [
    grid([".....", "..#..", "....."]),                                    # tek piksel
    grid(["........", ".######.", "........"]),                           # 1 piksel kalınlıkta çizgi
    grid(["........", ".######.", ".######.", "........"]),               # 2 piksel: 3×3 sığmaz
    grid(["#.....", ".#....", "..#...", "...#..", "....#."]),            # köşegen çizgi
], ids=["pixel", "line", "line2", "diagonal"])
def test_K_zero_shapes_are_their_own_skeleton(a):
    levels = M.skeleton_levels(a, M.structuring_element())
    assert len(levels) == 1                                                # K = 0
    np.testing.assert_array_equal(levels[0][2], a)                         # S_0 = A − A ∘ B = A
    np.testing.assert_array_equal(M.skeleton_reconstruct(levels, M.structuring_element()), a)


def test_rectangle_odd_height_gives_centre_line():
    a = np.zeros((11, 15), bool)
    a[2:9, 2:13] = True                                                    # 7 × 11
    levels = M.skeleton_levels(a, M.structuring_element())
    assert len(levels) - 1 == 3
    assert [int(s.sum()) for _, _, s in levels] == [0, 0, 0, 5]            # yalnızca S_3 boş değil
    expected = np.zeros_like(a)
    expected[5, 5:10] = True                                               # orta satır, 5 piksel
    np.testing.assert_array_equal(union_skeleton(levels), expected)


def test_rectangle_even_height_skeleton_is_two_pixels_wide():
    """Morfolojik iskeletin 1 piksel kalınlıkta olması gerekmez: 8 × 12 dikdörtgende S_3 bir 2 × 6 bloktur."""
    a = np.zeros((12, 16), bool)
    a[2:10, 2:14] = True
    s = union_skeleton(M.skeleton_levels(a, M.structuring_element()))
    expected = np.zeros_like(a)
    expected[5:7, 5:11] = True
    np.testing.assert_array_equal(s, expected)


def y_shape(size=41, r=2.6):
    """Slayt 74'teki Y: üç kalın kol (kendi ürettiğimiz)."""
    yy, xx = np.mgrid[0:size, 0:size]
    a = np.zeros((size, size), bool)
    c = size // 2
    for (y0, x0), (y1, x1) in (((4, 6), (c, c)), ((4, size - 7), (c, c)), ((c, c), (size - 4, c))):
        for t in np.linspace(0, 1, 200):
            y, x = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
            a |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    return a


@pytest.mark.parametrize("shape", ["rect", "cross"])
def test_y_shape(shape):
    a = y_shape()
    se = M.structuring_element(shape, 3, 3)
    levels = M.skeleton_levels(a, se)
    ref = ref_skeleton(a, se_mask(shape, 3, 3))
    s = union_skeleton(levels)
    np.testing.assert_array_equal(s, union_skeleton(ref))
    np.testing.assert_array_equal(M.skeleton_reconstruct(levels, se), a)
    assert not (s & ~a).any() and 0 < s.sum() < a.sum()                 # S(A) ⊊ A (kesin oran varsayılmaz)
    # ders materyali (slayt 77 / laboratuvar): morfolojik iskelet bağlantısız olabilir — bu Y'de de öyle
    assert ref_label(s, 8).max() > 1


def test_rect_and_cross_give_different_skeletons():
    a = y_shape()
    s_rect = union_skeleton(M.skeleton_levels(a, M.structuring_element("rect", 3, 3)))
    s_cross = union_skeleton(M.skeleton_levels(a, M.structuring_element("cross", 3, 3)))
    assert not np.array_equal(s_rect, s_cross)


def test_multiple_separate_objects_have_independent_skeletons():
    se = M.structuring_element()
    a1, a2 = np.zeros((20, 30), bool), np.zeros((20, 30), bool)
    a1[2:9, 2:13] = True
    a2[11:18, 16:28] = True
    a2[12:14, 14:16] = True
    both = M.skeleton_levels(a1 | a2, se)
    np.testing.assert_array_equal(union_skeleton(both),
                                  union_skeleton(M.skeleton_levels(a1, se)) | union_skeleton(M.skeleton_levels(a2, se)))
    np.testing.assert_array_equal(M.skeleton_reconstruct(both, se), a1 | a2)


def test_object_touching_the_border():
    """Görüntü dışı arka plandır: kenara değen nesne de kenardan aşınır; yeniden yapılandırma yine tam."""
    a = np.zeros((9, 12), bool)
    a[0:7, 0:9] = True
    se = M.structuring_element()
    levels = M.skeleton_levels(a, se)
    ref = ref_skeleton(a, se_mask("rect", 3, 3))
    assert len(levels) == len(ref) == 4
    np.testing.assert_array_equal(union_skeleton(levels), union_skeleton(ref))
    np.testing.assert_array_equal(M.skeleton_reconstruct(levels, se), a)


def test_full_image():
    a = np.ones((7, 9), bool)
    levels = M.skeleton_levels(a, M.structuring_element())
    assert len(levels) - 1 == 3
    np.testing.assert_array_equal(M.skeleton_reconstruct(levels, M.structuring_element()), a)


@pytest.mark.parametrize("kwargs", [dict(width=1, height=1), dict(shape="ellipse", width=1, height=1),
                                    dict(width=4), dict(shape="disk"), dict(display="medial"), dict(level=-1),
                                    dict(level=1.5), dict(level=True), dict(binarize="manual", thresh=256)])
def test_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.skeletonization(u8(FIG_926), **kwargs)


def test_single_cell_se_rejected_in_mask_api():
    with pytest.raises(AlgorithmParameterError):
        M.skeleton_levels(FIG_926, M.structuring_element("rect", 1, 1))


# --------------------------------------------------------------------------
# Görüntü düzeyi: gösterimler, rapor, düzen
# --------------------------------------------------------------------------


def big_fig():
    return np.kron(FIG_926, np.ones((3, 3), bool))


def test_displays_and_report():
    a = big_fig()
    img = u8(a)
    se = M.structuring_element()
    levels = ref_skeleton(a, se_mask("rect", 3, 3))
    s = union_skeleton(levels)
    res = M.skeletonization(img)
    np.testing.assert_array_equal(res.image, u8(s))
    np.testing.assert_array_equal(M.skeletonization(img, display="reconstruction").image, img)
    rep = res.report
    assert rep.K == len(levels) - 1 and rep.reconstructed and rep.object_px == a.sum() and rep.skeleton_px == s.sum()
    assert [(lv.k, lv.eroded, lv.opened, lv.skeleton) for lv in rep.levels] == \
        [(k, int(e.sum()), int(o.sum()), int(sk.sum())) for k, (e, o, sk) in enumerate(levels)]
    assert rep.components == ref_label(s, 8).max()
    text = rep.summary()
    assert f"K = {rep.K}" in text and "reconstruction = A" in text and "3×3 rect" in text
    assert ", ".join(str(int(sk.sum())) for _, _, sk in levels) in text


def test_levels_display_colours_each_level():
    a = big_fig()
    levels = ref_skeleton(a, se_mask("rect", 3, 3))
    out = M.skeletonization(u8(a), display="levels").image
    palette = M.label_colors(len(levels))
    assert sum(1 for _, _, s in levels if s.any()) >= 2                    # en az iki seviye katkı verir
    for k, (_, _, s) in enumerate(levels):                                 # boş S_k da olabilir
        assert (out[s] == palette[k + 1]).all()
    rest = a & ~union_skeleton(levels)
    assert (out[rest] == M.SKELETON_COLORS["object"]).all() and not out[~a].any()


@pytest.mark.parametrize("k", [0, 1, 2, 5])
def test_layer_display(k):
    a = big_fig()
    levels = ref_skeleton(a, se_mask("rect", 3, 3))
    res = M.skeletonization(u8(a), display="layer", level=k)
    out = res.image
    if k < len(levels):
        _, o, s = levels[k]
        assert (out[s] == M.SKELETON_COLORS["skeleton"]).all()
        assert (out[o] == M.SKELETON_COLORS["eroded"]).all()
        assert (out[a & ~o & ~s] == M.SKELETON_COLORS["object"]).all()
        assert f"showing k = {k}" in res.report.summary() and "> K" not in res.report.summary()
    else:
        assert (out[a] == M.SKELETON_COLORS["object"]).all()
        assert "> K" in res.report.summary()
    assert not out[~a].any()


def test_threshold_polarity_and_color_input():
    a = big_fig()
    exp = u8(union_skeleton(ref_skeleton(a, se_mask("rect", 3, 3))))
    np.testing.assert_array_equal(M.skeletonization(255 - u8(a), foreground="dark").image, exp)
    bgr = np.zeros(a.shape + (3,), np.uint8)
    bgr[a] = (40, 200, 90)
    out = M.skeletonization(bgr, binarize="manual", thresh=60).image
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], exp)


@pytest.mark.parametrize("display", M.SKELETON_DISPLAYS)
@pytest.mark.parametrize("shape", [(30, 15), (30, 15, 1), (30, 15, 3), (30, 15, 4)])
def test_layout_alpha_immutability(display, shape):
    base = u8(big_fig())
    img = base if len(shape) == 2 else np.repeat(base[..., None], shape[2], axis=2).copy()
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = np.arange(15, dtype=np.uint8)[None, :] * 9
    before = img.copy()
    out = M.skeletonization(img, display=display, level=1).image
    np.testing.assert_array_equal(img, before)
    assert out.dtype == np.uint8 and not np.shares_memory(out, img)
    colour = display in ("levels", "layer")
    if len(shape) == 3 and shape[2] == 4:
        assert out.shape == img.shape
        np.testing.assert_array_equal(out[..., 3], img[..., 3])
    elif colour:
        assert out.shape == shape[:2] + (3,)
    else:
        assert out.shape == img.shape


def test_registry_entry_and_previous_operations():
    algo = build_registry().get("skeletonization")
    assert algo.title == "Skeletonization" and algo.implemented
    assert [p.key for p in algo.params] == ["binarize", "thresh", "foreground", "shape", "width", "height",
                                            "display", "level"]
    d = algo.default_params()
    assert (d["shape"], d["width"], d["height"], d["display"], d["level"]) == ("rect", 3, 3, "skeleton", 0)
    assert "S_k(A) = (A ⊖ kB) − (A ⊖ kB) ∘ B" in algo.summary and "disconnected" in algo.summary
    assert_previous_15_operations_unchanged(96)


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    img = np.repeat(u8(np.kron(FIG_926, np.ones((4, 4), bool)))[..., None], 3, axis=2)
    path = tmp_path / "İskelet" / "şekil_9_26.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


def test_ui_apply_layers_undo_and_reset(window):
    w, img = window
    a = img[..., 0] > 127
    levels = ref_skeleton(a, se_mask("rect", 3, 3))
    w.navigator.select_algorithm("skeletonization")
    w.parameters.apply_button.click()
    np.testing.assert_array_equal(w.current_image[..., 0], u8(union_skeleton(levels)))
    assert w.telemetry.state.state() == "DONE" and f"K = {len(levels) - 1}" in w.statusBar().currentMessage()
    assert not w.parameters.report_view.isVisible()
    for name, counts in histograms(w.current_image).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
    w.parameters.controls["display"].set_value("reconstruction")
    w.parameters.apply_button.click()
    np.testing.assert_array_equal(w.current_image, img)                  # ⋃ (S_k ⊕ kB) = A
    w.reset_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
    np.testing.assert_array_equal(w.original_image, img)
