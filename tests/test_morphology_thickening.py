"""Thickening (Chapter 4, slayt 69–71; Gonzalez Şekil 9.24, laboratuvar ``thicken``).

Bağımsız referans: Aᶜ, saf Python sıralı B¹…B⁸ inceltmesiyle inceltilir (görüntü dışı arka
plandır, yani Aᶜ için 1 ve değişmez), tümleyeni alınır ve BFS ile A'ya değmeyen 8-bağlantılı
parçalar silinir. ``ref_thin_hmt`` (dışarısı 0) yalnızca eski "çerçeve" kuralını göstermek için kullanılır.
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
from tests.test_morphology_connected_components import ref_label
from tests.test_morphology_convex_hull import assert_previous_operations_unchanged
from tests.test_morphology_thinning import ref_thin_hmt


def ref_thin_background(c, max_cycles=None):
    """Sıralı B¹…B⁸ inceltmesi; görüntü dışındaki pikseller arka plandır (C için 1) ve değişmez."""
    x = c.copy()

    def px(r, col):
        return bool(x[r, col]) if 0 <= r < x.shape[0] and 0 <= col < x.shape[1] else True

    def matches(r, col, kern):
        for i in range(3):
            for j in range(3):
                k, v = kern[i][j], px(r + i - 1, col + j - 1)
                if (k == 1 and not v) or (k == -1 and v):
                    return False
        return True

    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        changed = False
        for kern in M.THIN_KERNELS:
            hits = [(r, col) for r in range(x.shape[0]) for col in range(x.shape[1]) if matches(r, col, kern)]
            for r, col in hits:
                x[r, col] = False
            changed |= bool(hits)
        cycles += 1
        if not changed:
            break
    return x, cycles


def ref_thicken(a, max_cycles=None, cleanup="remove"):
    background, cycles = ref_thin_background(~a, max_cycles)
    t = ~background
    if cleanup == "remove":
        labels = ref_label(t, 8)
        keep = set(np.unique(labels[a & (labels > 0)]).tolist())
        t = np.isin(labels, list(keep)) & (labels > 0)
    return t, cycles


def u8(mask):
    return np.where(mask, 255, 0).astype(np.uint8)


def two_objects():
    a = np.zeros((20, 30), bool)
    a[5:10, 3:8] = True
    a[8:14, 20:26] = True
    return a


# --------------------------------------------------------------------------
# Tanım ve bağımsız referans
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cycles", [1, 2, 3, 0])
@pytest.mark.parametrize("cleanup", M.THICKEN_CLEANUP)
def test_matches_background_thinning_reference(cycles, cleanup):
    a = two_objects()
    out, rep = M.thicken_mask(a, cycles, cleanup)
    ref, ref_cycles = ref_thicken(a, cycles or None, cleanup)
    np.testing.assert_array_equal(out, ref)
    assert rep.cycles == ref_cycles


@pytest.mark.parametrize("seed", range(4))
def test_matches_reference_on_random_images(seed):
    a = random_binary(800 + seed, (16, 20), 0.25)
    for cycles in (2, 0):
        np.testing.assert_array_equal(M.thicken_mask(a, cycles)[0], ref_thicken(a, cycles or None)[0])


def test_is_not_thinning_and_only_adds_pixels():
    a = two_objects()
    thick, _ = M.thicken_mask(a, 0)
    thin, _ = M.thin_mask(a)
    assert not (a & ~thick).any()                         # A ⊆ thickening(A)
    assert thick.sum() > a.sum() > thin.sum()
    assert not np.array_equal(thick, thin)


def test_objects_grow_but_never_merge():
    a = two_objects()
    out, rep = M.thicken_mask(a, 0)
    assert rep.converged
    assert ref_label(out, 8).max() == 2                   # hâlâ iki ayrı nesne
    labels = ref_label(out, 8)
    assert labels[7, 5] != labels[10, 22]
    # nesnelerin arasında üstten alta uzanan tek piksellik arka plan ayrımı kalır (arka planın iskeleti)
    between = out[:, 8:20]
    assert (~between).any(axis=1).all()
    assert (~out).sum() == out.shape[0] + 2                # ince, 8-bağlantılı ayırıcı çizgi


def test_growth_is_monotonic_in_cycles_and_bounded_by_dilation():
    a = two_objects()
    sizes = [M.thicken_mask(a, n)[0].sum() for n in (1, 2, 3, 4, 0)]
    assert all(x <= y for x, y in zip(sizes, sizes[1:]))
    one, _ = M.thicken_mask(a, 1)
    dil = M.binary_dilation(a, M.structuring_element("rect", 3, 3))
    assert not (one & ~dil).any()                         # bir döngü en fazla 1 piksel büyütür
    assert (one & ~a).any()


@pytest.mark.parametrize("seed", range(6))
def test_no_stray_parts_so_cleanup_finds_nothing(seed):
    """Dışarısı sabit arka plan olduğundan silinen her arka plan pikseli bir nesneye komşudur:
    T'nin her parçası A'ya değer, ``keep`` ve ``remove`` aynı sonucu verir (çerçeve bandı oluşmaz)."""
    a = random_binary(900 + seed, (14, 18), 0.2)
    for cycles in (1, 3, 0):
        keep, _ = M.thicken_mask(a, cycles, "keep")
        remove, _ = M.thicken_mask(a, cycles, "remove")
        np.testing.assert_array_equal(keep, remove)
        np.testing.assert_array_equal(keep, ref_thicken(a, cycles or None, "keep")[0])
        labels = ref_label(keep, 8)
        assert all((a & (labels == k)).any() for k in range(1, labels.max() + 1))


# --------------------------------------------------------------------------
# Görüntü sınırı (kenara/köşeye değen nesneler)
# --------------------------------------------------------------------------


def assert_each_object_kept_separate(a, out):
    """T ⊇ A ve T'nin her 8-bileşeni A'nın tam olarak bir 8-bileşenini içerir."""
    assert not (a & ~out).any()
    la, lo = ref_label(a, 8), ref_label(out, 8)
    assert lo.max() == la.max()
    for k in range(1, lo.max() + 1):
        assert len(set(la[(lo == k) & a].tolist())) == 1


def frame_rule_thicken(a):
    """Eski kural (laboratuvar): görüntü dışı Aᶜ için de 0 → görüntü çerçevesi nesne gibi büyür."""
    t = ~ref_thin_hmt(~a)[0]
    labels = ref_label(t, 8)
    return np.isin(labels, np.unique(labels[a & (labels > 0)])) & (labels > 0)


def test_single_edge_object_grows_along_and_away_from_the_edge():
    a = np.zeros((12, 16), bool)
    a[0:4, 5:9] = True                                     # üst kenara değen tek nesne
    for cycles in (1, 2, 0):
        out, rep = M.thicken_mask(a, cycles)
        np.testing.assert_array_equal(out, ref_thicken(a, cycles or None)[0])
        assert_each_object_kept_separate(a, out)
    one = M.thicken_mask(a, 1)[0]
    assert one[0, 4] and one[1, 9] and one[4, 6]           # kenar boyunca ve aşağı doğru büyür
    assert not one[0, 9] and M.thicken_mask(a, 2)[0][0, 9]  # sıralı B'ler: köşe bir döngü gecikir
    assert not one[0, :3].any() and not one[:, 12:].any()  # kenardan bağımsız "çerçeve" büyümesi yok
    full, rep = M.thicken_mask(a, 0)
    assert rep.converged and full.all()                    # tek nesne: arka planın iskeleti boştur


def test_two_separate_edge_objects_stay_separate():
    """Eski kuralda birleşen örnek: alt kenarda iki ayrı nesne (çerçeve bandı ikisine de değiyor)."""
    a = np.zeros((15, 14), bool)
    a[14, 6:9] = True
    a[14, 13] = True                                       # sağ alt köşedeki tek piksel
    assert ref_label(frame_rule_thicken(a), 8).max() == 1  # eski kural: tek parçaya birleşir
    out, rep = M.thicken_mask(a, 0)
    assert rep.converged
    np.testing.assert_array_equal(out, ref_thicken(a)[0])
    assert_each_object_kept_separate(a, out)
    assert (~out[14]).any()                                # kenarda aralarında arka plan kalır


def test_corner_objects_are_thickened_and_kept_apart():
    a = np.zeros((14, 18), bool)
    a[0:3, 0:3] = True                                     # sol üst köşe
    a[12:14, 15:18] = True                                 # sağ alt köşe
    a[0, 17] = True                                        # sağ üst köşe (tek piksel)
    a[13, 0] = True                                        # sol alt köşe (tek piksel)
    for cycles in (1, 3, 0):
        out, _ = M.thicken_mask(a, cycles)
        np.testing.assert_array_equal(out, ref_thicken(a, cycles or None)[0])
        assert_each_object_kept_separate(a, out)
    one = M.thicken_mask(a, 1)[0]
    assert one[3, 1] and one[1, 3]                         # köşe nesnesi iki yöne de büyür


def test_interior_objects_unchanged_by_the_boundary_rule():
    """Kenardan uzak nesneler için (çerçeve bandı nesnelere ulaşmadan) eski kuralla aynı sonuç."""
    a = np.zeros((30, 34), bool)
    a[12:16, 10:14] = True
    a[13:18, 19:23] = True
    for cycles in (1, 2, 3):
        out = M.thicken_mask(a, cycles)[0]
        old = ~ref_thin_hmt(~a, cycles)[0]
        labels = ref_label(old, 8)
        old = np.isin(labels, np.unique(labels[a & (labels > 0)])) & (labels > 0)
        np.testing.assert_array_equal(out, old)
    assert_each_object_kept_separate(a, M.thicken_mask(a, 0)[0])


@pytest.mark.parametrize("seed", range(40))
def test_random_border_objects_never_merge(seed):
    rng = np.random.default_rng(seed)
    h, w = rng.integers(8, 15, 2)
    a = np.zeros((h, w), bool)
    for _ in range(rng.integers(2, 5)):
        bh, bw = rng.integers(1, 4, 2)
        y, x = rng.integers(0, h - bh + 1), rng.integers(0, w - bw + 1)
        side = rng.integers(0, 4)
        y = 0 if side == 0 else h - bh if side == 2 else y
        x = 0 if side == 1 else w - bw if side == 3 else x
        a[y:y + bh, x:x + bw] = True
    out, rep = M.thicken_mask(a, 0)
    assert rep.converged
    assert_each_object_kept_separate(a, out)


def test_holes_shrink_but_are_not_closed():
    """Arka plan inceltmesi arka planın topolojisini korur: delik küçülür ama kapanmaz
    (2×2 delikten B² ile bir piksel nesneye katılır, kalan 3 piksel delik olarak kalır)."""
    a = np.zeros((12, 12), bool)
    a[2:10, 2:10] = True
    a[5:7, 5:7] = False
    out, _ = M.thicken_mask(a, 0)
    assert out[5:7, 5:7].sum() == 1
    assert len(M.hole_labels(out, 4)[1]) == len(M.hole_labels(a, 4)[1]) == 1
    np.testing.assert_array_equal(out, ref_thicken(a)[0])


def test_empty_and_full():
    out, rep = M.thicken_mask(np.zeros((8, 10), bool), 3)
    assert not out.any()                                   # A boşsa büyüyen her şey kopuktur
    full, _ = M.thicken_mask(np.ones((8, 10), bool), 0)
    assert full.all()


def test_invalid_arguments():
    for kwargs in (dict(cleanup="trim"), dict(max_cycles=-1), dict(max_cycles=2.5), dict(max_cycles=True)):
        with pytest.raises(AlgorithmParameterError):
            M.thicken_mask(two_objects(), **kwargs)


# --------------------------------------------------------------------------
# Görüntü düzeyi
# --------------------------------------------------------------------------


def test_threshold_polarity_and_color():
    a = two_objects()
    expected = u8(ref_thicken(a, 2)[0])
    np.testing.assert_array_equal(M.thickening(u8(a), max_cycles=2).image, expected)
    np.testing.assert_array_equal(M.thickening(255 - u8(a), foreground="dark", max_cycles=2).image, expected)
    bgr = np.zeros(a.shape + (3,), np.uint8)
    bgr[a] = (0, 255, 0)
    bgr[~a] = (255, 0, 0)
    out = M.thickening(bgr, binarize="manual", thresh=100, max_cycles=2).image
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], expected)


@pytest.mark.parametrize("shape", [(20, 30), (20, 30, 1), (20, 30, 3), (20, 30, 4)])
def test_layout_alpha_immutability_and_report(shape):
    base = u8(two_objects())
    img = base if len(shape) == 2 else np.repeat(base[..., None], shape[2], axis=2).copy()
    if len(shape) == 3 and shape[2] == 4:
        img[..., 3] = 123
    before = img.copy()
    res = M.thickening(img, binarize="manual", max_cycles=2)
    np.testing.assert_array_equal(img, before)
    assert res.image.shape == img.shape and res.image.dtype == np.uint8 and not np.shares_memory(res.image, img)
    if len(shape) == 3 and shape[2] == 4:
        assert (res.image[..., 3] == 123).all()
    assert not res.report.converged and res.report.cycles == 2
    assert res.report.after > res.report.before
    assert "Thickening" in res.report.summary() and "not converged" in res.report.summary()


def test_registry_entry_and_previous_operations():
    algo = build_registry().get("thickening")
    assert algo.title == "Thickening" and algo.implemented
    assert [p.key for p in algo.params] == ["binarize", "thresh", "foreground", "max_cycles", "cleanup"]
    assert algo.default_params()["max_cycles"] == 10 and algo.default_params()["cleanup"] == "remove"
    assert "Fig. 9.24" in algo.summary
    assert_previous_operations_unchanged(95)


# --------------------------------------------------------------------------
# Arayüz
# --------------------------------------------------------------------------


@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow()
    w.resize(1400, 860)
    w.show()
    img = np.repeat(u8(two_objects())[..., None], 3, axis=2)
    path = tmp_path / "Kalınlaştırma" / "iki_nesne.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


def test_ui_apply_and_undo(window):
    w, img = window
    w.navigator.select_algorithm("thickening")
    w.parameters.controls["max_cycles"].set_value(0)
    w.parameters.apply_button.click()
    a = img[..., 0] > 127
    np.testing.assert_array_equal(w.current_image[..., 0], u8(ref_thicken(a)[0]))
    np.testing.assert_array_equal(w.original_image, img)
    assert w.telemetry.state.state() == "DONE" and "converged after" in w.statusBar().currentMessage()
    assert not w.parameters.report_view.isVisible()
    for name, counts in histograms(w.current_image).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
