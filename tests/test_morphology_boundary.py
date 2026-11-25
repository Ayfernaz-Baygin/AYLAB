"""Sınır çıkarma β(A) = A − (A ⊖ B) (Chapter 4, slayt 53–55).

Bağımsız referanslar OpenCV kullanmaz:
* ``ref_boundary``  — erosion'ı ötelemeyle (görüntü dışı arka plan) kurar, küme farkı alır.
* ``loop_boundary`` — piksel döngüsü: ön plan pikseli, SE'nin (orijine göre)
  kapsadığı hücrelerden biri arka plan ya da görüntü dışıysa sınırdır.
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
from tests.test_morphology_hat import ref_black_hat, ref_top_hat

# --------------------------------------------------------------------------
# Referanslar ve yardımcılar
# --------------------------------------------------------------------------


def ref_boundary(a, se):
    return a & ~ref_erode(a, se.mask, se.anchor)


def loop_boundary(a, se):
    h, w = a.shape
    out = np.zeros_like(a)
    for r in range(h):
        for c in range(w):
            if not a[r, c]:
                continue
            for dy, dx in se.offsets:
                rr, cc = r + dy, c + dx
                if not (0 <= rr < h and 0 <= cc < w) or not a[rr, cc]:
                    out[r, c] = True
                    break
    return out


def se(shape="rect", w=3, h=3, ax=-1, ay=-1):
    return M.structuring_element(shape, w, h, ax, ay)


def components(mask, connectivity=8):
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)[0] - 1


def rect_mask(shape, r0, r1, c0, c1):
    a = np.zeros(shape, bool)
    a[r0:r1, c0:c1] = True
    return a


def ring(size=21, r_out=8, r_in=4):
    y, x = np.mgrid[:size, :size] - size // 2
    d2 = x * x + y * y
    return (d2 <= r_out * r_out) & (d2 > r_in * r_in)


def disk(size=21, r=8):
    y, x = np.mgrid[:size, :size] - size // 2
    return x * x + y * y <= r * r


# --------------------------------------------------------------------------
# Elle kurulmuş diziler — tam beklenen sonuç
# --------------------------------------------------------------------------


def test_filled_rectangle_gives_one_pixel_perimeter():
    a = rect_mask((9, 11), 2, 7, 2, 9)          # 5 × 7 dikdörtgen
    expected = a.copy()
    expected[3:6, 3:8] = False                  # iç kısım (3 × 5) erosion'da kalır
    out = M.boundary_mask(a, se())
    assert np.array_equal(out, expected)
    assert out.sum() == 2 * (5 + 7) - 4 == 20


def test_exact_small_matrix():
    a = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0]], bool)
    expected = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 1, 1, 1, 1, 0],
                         [0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0]], bool)
    assert np.array_equal(M.boundary_mask(a, se()), expected)


def test_thicker_se_gives_thicker_boundary():
    a = rect_mask((13, 15), 2, 11, 2, 13)       # 9 × 11
    out5 = M.boundary_mask(a, se("rect", 5, 5))
    assert out5.sum() == 9 * 11 - 5 * 7          # iç 5 × 7 kalır → kalınlık 2
    assert out5[2:4, 2:13].all() and not out5[4:9, 4:11].any()
    out7 = M.boundary_mask(a, se("rect", 7, 7))
    assert out7.sum() == 9 * 11 - 3 * 5
    assert M.boundary_mask(a, se()).sum() < out5.sum() < out7.sum()


def test_line_se_extracts_only_edges_across_its_direction():
    a = rect_mask((9, 11), 2, 7, 2, 9)
    horizontal = M.boundary_mask(a, se("rect", 3, 1))     # yalnızca sol/sağ kenarlar
    expected = np.zeros_like(a)
    expected[2:7, 2] = expected[2:7, 8] = True
    assert np.array_equal(horizontal, expected)
    vertical = M.boundary_mask(a, se("rect", 1, 3))       # yalnızca üst/alt kenarlar
    expected = np.zeros_like(a)
    expected[2, 2:9] = expected[6, 2:9] = True
    assert np.array_equal(vertical, expected)


def test_square_vs_cross_connectivity_on_a_disk():
    """3×3 kare SE → 4-bağlantılı (kalın) sınır; artı SE → 8-bağlantılı (ince) sınır."""
    a = disk()
    b_rect, b_cross = M.boundary_mask(a, se("rect")), M.boundary_mask(a, se("cross"))
    assert not (b_cross & ~b_rect).any() and b_cross.sum() < b_rect.sum()
    assert components(b_rect, 4) == 1 and components(b_cross, 8) == 1
    assert components(b_cross, 4) > 1            # artı SE sınırı yalnızca köşegenlerle bağlı
    assert not np.array_equal(b_rect, b_cross)


def test_ring_has_outer_and_inner_boundary():
    a = ring()
    out = M.boundary_mask(a, se())
    assert components(out) == 2                  # dış ve iç kontur
    y, x = np.mgrid[:21, :21] - 10
    d = np.sqrt(x * x + y * y)
    assert out[(d > 3.5) & (d < 5.2) & a].any() and out[(d > 7) & a].any()
    assert not out[(d > 5.5) & (d < 6.5)].any()   # halkanın ortası sınır değil


def test_multiple_disconnected_objects():
    a = np.zeros((12, 30), bool)
    a[2:7, 2:7] = True
    a[3:10, 12:16] = True
    a[1:11, 22:28] = True
    out = M.boundary_mask(a, se())
    assert components(out) == 3
    for r0, r1, c0, c1 in [(2, 7, 2, 7), (3, 10, 12, 16), (1, 11, 22, 28)]:
        part = out[r0:r1, c0:c1]
        h, w = r1 - r0, c1 - c0
        assert part.sum() == 2 * (h + w) - 4


def test_letter_shaped_object():
    a = np.zeros((60, 60), np.uint8)
    cv2.putText(a, "A", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.9, 255, 6)
    a = a > 0
    out = M.boundary_mask(a, se())
    assert not (out & ~a).any() and out.sum() < a.sum()
    assert components(out) >= 2                  # dış kontur + A'nın iç üçgeni
    assert np.array_equal(out, ref_boundary(a, se()))


def test_anchor_at_corner_moves_the_kept_region():
    """B = 3×3, orijin sol üst: E = {z | z + {0,1,2}² ⊆ A}; sınır alt/sağ kenarda 2 kalınlığında."""
    a = rect_mask((7, 7), 1, 6, 1, 6)
    out = M.boundary_mask(a, se("rect", 3, 3, 0, 0))
    expected = a.copy()
    expected[1:4, 1:4] = False
    assert np.array_equal(out, expected)
    out = M.boundary_mask(a, se("rect", 3, 3, 2, 2))
    expected = a.copy()
    expected[3:6, 3:6] = False
    assert np.array_equal(out, expected)


def test_boundary_is_subset_of_a_even_with_non_member_anchor():
    a = random_binary(5, (25, 25), 0.6)
    b = se("cross", 5, 5, 0, 0)                  # orijin üye değil: A ⊖ B ⊄ A olabilir
    out = M.boundary_mask(a, b)
    assert not (out & ~a).any()
    assert np.array_equal(out, a & ~M.binary_erosion(a, b))


# --------------------------------------------------------------------------
# Bağımsız referans ve özellikler
# --------------------------------------------------------------------------

SE_CASES = [(s, w, h, ax, ay) for s in M.SE_SHAPES
            for (w, h, ax, ay) in ((1, 1, -1, -1), (3, 3, -1, -1), (5, 5, -1, -1), (7, 3, -1, -1),
                                   (3, 7, -1, -1), (5, 5, 0, 0), (3, 5, 2, 4))]


@pytest.mark.parametrize("s", SE_CASES, ids=lambda s: "-".join(map(str, s)))
def test_matches_independent_references(s):
    b = se(*s)
    for seed in range(3):
        a = random_binary(100 + seed, (14, 17), 0.6)
        out = M.boundary_mask(a, b)
        assert np.array_equal(out, ref_boundary(a, b))
        assert np.array_equal(out, loop_boundary(a, b))


@pytest.mark.parametrize("shape", M.SE_SHAPES)
def test_partition_and_relation_to_gradient(shape):
    a = random_binary(9, (30, 30), 0.55)
    b = se(shape, 5, 5)
    beta, e = M.boundary_mask(a, b), M.binary_erosion(a, b)
    assert not (beta & e).any() and np.array_equal(beta | e, a)   # A = β(A) ⊔ (A ⊖ B)
    g = M.binary_gradient(a, b)
    assert np.array_equal(beta, g & a)                              # β = gradyanın iç kuşağı
    assert (g & ~a).any() and not np.array_equal(beta, g)           # gradyan DEĞİL


def test_one_by_one_se_gives_empty_boundary():
    a = random_binary(3)
    assert not M.boundary_mask(a, se("rect", 1, 1)).any()


def test_matches_slide_53_opencv_code_away_from_border():
    """Slayt 53: boundary = img − cv2.erode(img, kernel). OpenCV sınırı farklı ele alır; iç bölgede aynı."""
    a = np.pad(random_binary(17, (30, 30), 0.6), 3)
    u8 = np.where(a, 255, 0).astype(np.uint8)
    for shape in M.SE_SHAPES:
        b = se(shape, 5, 5)
        cv = cv2.subtract(u8, cv2.erode(u8, b.mask)) > 0
        assert np.array_equal(M.boundary_mask(a, b)[2:-2, 2:-2], cv[2:-2, 2:-2])


# --------------------------------------------------------------------------
# Giriş/çıkış, eşikleme, kutupluluk
# --------------------------------------------------------------------------


def test_grayscale_manual_threshold_and_polarity():
    img = np.full((9, 9), 40, np.uint8)
    img[2:7, 2:7] = 200
    bright = M.boundary_extraction(img, binarize="manual", thresh=127)
    expected = rect_mask((9, 9), 2, 7, 2, 7)
    expected[3:6, 3:6] = False
    np.testing.assert_array_equal(bright, np.where(expected, 255, 0))
    # DARK: ön plan = zemin (kare dışı); görüntü dışı arka plan → çerçeve + karenin dış çevresi
    dark = M.boundary_extraction(img, binarize="manual", thresh=127, foreground="dark") > 0
    a_dark = ~rect_mask((9, 9), 2, 7, 2, 7)
    np.testing.assert_array_equal(dark, ref_boundary(a_dark, se()))
    assert dark[0].all() and dark[1, 1:8].all() and not dark[2:7, 2:7].any()


def test_threshold_value_changes_the_object():
    img = np.zeros((11, 11), np.uint8)
    img[1:10, 1:10] = 100
    img[3:8, 3:8] = 220
    assert M.boundary_extraction(img, binarize="manual", thresh=50).astype(bool).sum() == 9 * 4 - 4
    assert M.boundary_extraction(img, binarize="manual", thresh=150).astype(bool).sum() == 5 * 4 - 4


def test_otsu_on_grayscale():
    img = np.full((12, 12), 30, np.uint8)
    img[3:9, 3:9] = 210
    np.testing.assert_array_equal(M.boundary_extraction(img), M.boundary_extraction(img, binarize="manual", thresh=100))


def test_bgr_uses_luminance_and_returns_three_equal_channels():
    img = np.zeros((9, 9, 3), np.uint8)
    img[2:7, 2:7] = (0, 255, 0)         # yeşil Y≈150
    img[4, 4] = (255, 0, 0)             # mavi Y≈29 → T=100 ile arka plan (delik)
    out = M.boundary_extraction(img, binarize="manual", thresh=100)
    assert out.shape == img.shape and np.array_equal(out[..., 0], out[..., 2])
    a = rect_mask((9, 9), 2, 7, 2, 7)
    a[4, 4] = False
    np.testing.assert_array_equal(out[..., 1], np.where(ref_boundary(a, se()), 255, 0))


@pytest.mark.parametrize("shape", [(12, 15), (12, 15, 1), (12, 15, 3), (12, 15, 4)])
def test_layout_alpha_and_immutability(shape):
    img = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)
    before = img.copy()
    out = M.boundary_extraction(img, binarize="manual", thresh=120, shape="ellipse", width=5, height=5)
    np.testing.assert_array_equal(img, before)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.shares_memory(out, img)
    expected = np.where(ref_boundary(M.to_binary(img, "manual", 120), se("ellipse", 5, 5)), 255, 0)
    planes = [out] if out.ndim == 2 else [out[..., c] for c in range(min(3, out.shape[2]))]
    for p in planes:
        np.testing.assert_array_equal(p, expected)
    if len(shape) == 3 and shape[2] == 4:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])
    assert set(np.unique(out[..., :3] if out.ndim == 3 else out)) <= {0, 255}


def test_constant_black_is_empty():
    for img in (np.zeros((8, 8), np.uint8), np.zeros((8, 8, 3), np.uint8)):
        assert not M.boundary_extraction(img, binarize="manual").any()
        assert not M.boundary_extraction(img).any()          # Otsu: T = 0, Y > 0 yok


def test_constant_white_gives_image_frame():
    """Tüm görüntü ön plan; görüntü dışı arka plan olduğundan sınır = görüntü çerçevesi."""
    out = M.boundary_extraction(np.full((6, 8), 255, np.uint8)) > 0
    expected = np.ones((6, 8), bool)
    expected[1:-1, 1:-1] = False
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (6, 1), (2, 2)])
def test_tiny_all_foreground_images_are_all_boundary(shape):
    out = M.boundary_extraction(np.full(shape, 255, np.uint8), width=3, height=3) > 0
    assert out.all()


def test_tiny_images_with_large_se():
    img = np.random.default_rng(4).integers(0, 256, (3, 4), dtype=np.uint8)
    out = M.boundary_extraction(img, binarize="manual", thresh=100, width=31, height=31) > 0
    np.testing.assert_array_equal(out, img > 100)   # SE hiçbir yere sığmaz → tüm ön plan sınırdır


@pytest.mark.parametrize("kwargs", [dict(width=4), dict(height=0), dict(shape="disk"), dict(width=3, anchor_x=3),
                                    dict(anchor_y=-2), dict(width=2.5), dict(binarize="auto"),
                                    dict(foreground="white"), dict(binarize="manual", thresh=300),
                                    dict(binarize="manual", thresh=True)])
def test_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        M.boundary_extraction(np.zeros((5, 5), np.uint8), **kwargs)


@pytest.mark.parametrize("bad", [np.zeros((3, 3), np.float32), np.zeros((0, 3), np.uint8),
                                 np.zeros((3, 3, 2), np.uint8), [[0, 1]]])
def test_invalid_images(bad):
    with pytest.raises(AlgorithmParameterError):
        M.boundary_extraction(bad)


def test_boundary_mask_rejects_non_2d():
    with pytest.raises(AlgorithmParameterError):
        M.boundary_mask(np.zeros((3, 3, 3), bool), se())


# --------------------------------------------------------------------------
# Katalog ve mevcut morfoloji işlemlerinde regresyon
# --------------------------------------------------------------------------


def test_registry_entry():
    registry = build_registry()
    keys = [a.key for a in registry.in_category("morphology")]
    assert keys == ["erode", "dilate", "opening", "closing", "gradient", "hit_or_miss",
                    "top_hat", "black_hat", "hat_correction", "boundary", "hole_filling", "connected_components"]
    algo = registry.get("boundary")
    assert algo.title == "Boundary Extraction" and algo.implemented
    assert algo.summary.startswith("Extracts object boundaries by subtracting the eroded foreground "
                                   "from the original binary image")
    assert "β(A) = A − (A ⊖ B)" in algo.summary
    assert [p.key for p in algo.params] == [p.key for p in registry.get("erode").params[:-1]]   # erosion parametreleri
    assert algo.default_params() == dict(binarize="otsu", thresh=127, foreground="bright", shape="rect",
                                         width=3, height=3, anchor_x=-1, anchor_y=-1)
    assert len(registry.categories()) == 7


def test_registry_run_and_anchor_error():
    algo = build_registry().get("boundary")
    img = np.zeros((10, 10), np.uint8)
    img[2:8, 2:8] = 255
    assert (algo.run(img) > 0).sum() == 20
    with pytest.raises(AlgorithmParameterError, match="Anchor X"):
        algo.run(img, {"width": 3, "anchor_x": 4})


def test_existing_morphology_operations_unchanged():
    """Aşama 1–2 işlemleri hâlâ bağımsız tanımlarla birebir (registry üzerinden)."""
    registry = build_registry()
    img = np.where(random_binary(77, (20, 24), 0.55), 255, 0).astype(np.uint8)
    a, b = img > 127, se()
    pad = 10
    big = lambda fn: (lambda x: fn(np.pad(x, pad))[pad:-pad, pad:-pad])
    expected = {
        "erode": ref_erode(a, b.mask, b.anchor),
        "dilate": ref_dilate(a, b.mask, b.anchor),
        "opening": big(lambda x: ref_dilate(ref_erode(x, b.mask, b.anchor), b.mask, b.anchor))(a),
        "closing": big(lambda x: ref_erode(ref_dilate(x, b.mask, b.anchor), b.mask, b.anchor))(a),
        "gradient": ref_dilate(a, b.mask, b.anchor) & ~ref_erode(a, b.mask, b.anchor),
        "hit_or_miss": ref_hmt(a, M.hmt_kernel("corner")),
    }
    for key, exp in expected.items():
        np.testing.assert_array_equal(registry.get(key).run(img), np.where(exp, 255, 0), err_msg=key)
    gray = np.random.default_rng(3).integers(0, 256, (20, 24), dtype=np.uint8)
    b15 = se("rect", 15, 15)
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
    img = np.full((40, 56, 3), 25, np.uint8)
    cv2.circle(img, (16, 20), 11, (210, 220, 200), -1)
    cv2.circle(img, (16, 20), 4, (25, 25, 25), -1)          # halka
    cv2.rectangle(img, (32, 8), (50, 30), (230, 230, 230), -1)
    path = tmp_path / "Sınır" / "şekiller.png"
    save_image(path, img)
    assert w.load_image(path)
    yield w, img
    w.close()


def test_ui_listed_and_apply(loaded):
    w, img = loaded
    w.navigator.tiles["morphology"].click()
    assert list(w.navigator.operation_rows)[9] == "boundary"  # 10. işlem (Aşama 4 ile 11 işlem)
    assert w.navigator.operation_rows["boundary"].algorithm.title == "Boundary Extraction"
    w.navigator.select_algorithm("boundary")
    assert list(w.parameters.controls) == ["binarize", "thresh", "foreground", "shape", "width", "height",
                                           "anchor_x", "anchor_y"]
    w.parameters.controls["shape"].set_value("cross")
    w.parameters.apply_button.click()
    out = w.current_image
    np.testing.assert_array_equal(out, M.boundary_extraction(img, shape="cross"))
    a = M.to_binary(img, "otsu")
    np.testing.assert_array_equal(out[..., 0] > 0, ref_boundary(a, se("cross")))
    np.testing.assert_array_equal(w.original_image, img)
    assert w.telemetry.state.state() == "DONE"
    for name, counts in histograms(out).items():
        np.testing.assert_array_equal(w.parameters.analysis.scope._current[name], counts)


def test_ui_se_choice_changes_result_and_undo_reset(loaded):
    w, img = loaded
    w.navigator.select_algorithm("boundary")
    w.parameters.apply_button.click()
    thin = w.current_image.copy()
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
    w.parameters.controls["width"].set_value(7)
    w.parameters.controls["height"].set_value(7)
    w.parameters.apply_button.click()
    thick = w.current_image.copy()
    assert (thick > 0).sum() > (thin > 0).sum()
    w.reset_action.trigger()
    np.testing.assert_array_equal(w.current_image, img)
    w.undo_action.trigger()
    np.testing.assert_array_equal(w.current_image, thick)
    np.testing.assert_array_equal(w.original_image, img)


def test_ui_invalid_anchor_uses_error_dialog(loaded, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    w, img = loaded
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append((title, text)))
    w.navigator.select_algorithm("boundary")
    w.parameters.controls["anchor_y"].set_value(6)
    w.parameters.apply_button.click()
    assert shown and shown[0][0] == "Invalid parameters" and "Anchor Y" in shown[0][1]
    assert w.telemetry.state.state() == "ERROR" and len(w.history) == 0
    np.testing.assert_array_equal(w.current_image, img)
