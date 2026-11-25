"""Uzamsal filtreler: bilinen değerler, bağımsız referanslar, kenar durumları."""

import cv2
import numpy as np
import pytest

from processing import filters as F
from processing.catalog import build_registry
from processing.common import luminance
from processing.registry import AlgorithmParameterError

ALL = [
    (F.mean_filter, {"ksize": 3}),
    (F.gaussian_filter, {"ksize": 5, "sigma_x": 1.0}),
    (F.median_filter, {"ksize": 3}),
    (F.sobel_x, {}),
    (F.sobel_y, {}),
    (F.laplacian, {}),
]
IDS = [f.__name__ for f, _ in ALL]


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def vertical_edge(h=9, w=10, left=0, right=100, col=5):
    """Sütun `col`'dan itibaren parlaklığı değişen dikey kenar (yatay geçiş)."""
    img = np.full((h, w), left, np.uint8)
    img[:, col:] = right
    return img


def horizontal_edge(h=10, w=9, top=0, bottom=100, row=5):
    img = np.full((h, w), top, np.uint8)
    img[row:, :] = bottom
    return img


def reflect101_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """OpenCV'den bağımsız 2-B korelasyon (BORDER_REFLECT_101 = np.pad 'reflect')."""
    kh, kw = kernel.shape
    padded = np.pad(image.astype(np.float64), ((kh // 2,) * 2, (kw // 2,) * 2), mode="reflect")
    out = np.zeros(image.shape, np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out


# --------------------------------------------------------------------------
# Ortak davranış
# --------------------------------------------------------------------------


@pytest.mark.parametrize("func,kw", ALL, ids=IDS)
@pytest.mark.parametrize("shape", [(12, 15), (12, 15, 1), (12, 15, 3), (12, 15, 4)])
def test_shape_dtype_and_immutability(func, kw, shape, rng):
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    before = img.copy()
    out = func(img, **kw)
    np.testing.assert_array_equal(img, before)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.shares_memory(out, img)
    if shape[-1] == 4 and len(shape) == 3:
        np.testing.assert_array_equal(out[..., 3], img[..., 3])  # alfa korunur


@pytest.mark.parametrize("func,kw", ALL, ids=IDS)
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (1, 9), (9, 1), (3, 3, 3)])
def test_tiny_images(func, kw, shape, rng):
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    big = dict(kw)
    if "ksize" in big:
        big["ksize"] = 7  # çekirdek görüntüden büyük
    assert func(img, **big).shape == img.shape


@pytest.mark.parametrize("func,kw", ALL[:3], ids=IDS[:3])
def test_smoothing_keeps_constant_image(func, kw):
    img = np.full((8, 8, 3), (12, 130, 250), np.uint8)
    np.testing.assert_array_equal(func(img, **kw), img)


@pytest.mark.parametrize("func", [F.sobel_x, F.sobel_y, F.laplacian], ids=["sobel_x", "sobel_y", "laplacian"])
def test_derivatives_of_constant_image(func):
    img = np.full((8, 8), 200, np.uint8)
    with np.errstate(all="raise"):  # sıfıra bölme yok
        assert np.all(func(img, display="abs") == 0)
        assert np.all(func(img, display="signed") == 128)


@pytest.mark.parametrize("func,kw", ALL, ids=IDS)
@pytest.mark.parametrize("bad", [np.zeros((4, 4), np.float32), np.zeros((0, 4), np.uint8),
                                 np.zeros((4, 4, 2), np.uint8), [[1, 2], [3, 4]]])
def test_rejects_invalid_images(func, kw, bad):
    with pytest.raises(AlgorithmParameterError):
        func(bad, **kw)


# --------------------------------------------------------------------------
# Çekirdek boyutu / sigma doğrulaması
# --------------------------------------------------------------------------


@pytest.mark.parametrize("func", [F.mean_filter, F.gaussian_filter, F.median_filter])
@pytest.mark.parametrize("ksize", [0, -3, 4, 2.5, "5", None, True])
def test_invalid_kernel_sizes(func, ksize, rng):
    img = rng.integers(0, 256, (6, 6), dtype=np.uint8)
    with pytest.raises(AlgorithmParameterError):
        func(img, ksize=ksize)


def test_integral_float_kernel_size_accepted(rng):
    img = rng.integers(0, 256, (6, 6), dtype=np.uint8)
    np.testing.assert_array_equal(F.mean_filter(img, ksize=3.0), F.mean_filter(img, ksize=3))


def test_median_requires_at_least_3():
    with pytest.raises(AlgorithmParameterError, match="≥ 3"):
        F.median_filter(np.zeros((5, 5), np.uint8), ksize=1)


@pytest.mark.parametrize("sigma", [0, -1.0, float("nan"), float("inf"), None, "1"])
def test_gaussian_rejects_invalid_sigma(sigma):
    with pytest.raises(AlgorithmParameterError):
        F.gaussian_filter(np.zeros((5, 5), np.uint8), ksize=5, sigma_x=sigma)


@pytest.mark.parametrize("func", [F.sobel_x, F.sobel_y])
@pytest.mark.parametrize("ksize", [9, 2, 0, -1])
def test_sobel_invalid_aperture(func, ksize):
    with pytest.raises(AlgorithmParameterError):
        func(np.zeros((5, 5), np.uint8), ksize=ksize)


@pytest.mark.parametrize("kwargs", [{"display": "raw"}, {"neighbours": "6"}])
def test_laplacian_invalid_options(kwargs):
    with pytest.raises(AlgorithmParameterError):
        F.laplacian(np.zeros((5, 5), np.uint8), **kwargs)


# --------------------------------------------------------------------------
# A. Mean
# --------------------------------------------------------------------------


def test_mean_known_impulse():
    img = np.zeros((7, 7), np.uint8)
    img[3, 3] = 90
    out = F.mean_filter(img, ksize=3)
    expected = np.zeros((7, 7), np.uint8)
    expected[2:5, 2:5] = 10  # 90 / 9
    np.testing.assert_array_equal(out, expected)


def test_mean_known_small_array():
    img = np.arange(25, dtype=np.uint8).reshape(5, 5) * 10  # doğrusal rampa
    out = F.mean_filter(img, ksize=3)
    # Doğrusal bir fonksiyonun simetrik ortalaması iç bölgede kendisidir
    np.testing.assert_array_equal(out[1:-1, 1:-1], img[1:-1, 1:-1])
    # Köşe (REFLECT_101): komşuluk {0,10,50,60} tekrarlarla → (0+10+10+50+60+60+50+60+60)/9
    assert out[0, 0] == round((0 + 10 + 10 + 50 + 60 + 60 + 50 + 60 + 60) / 9)


def test_mean_matches_independent_reference(rng):
    img = rng.integers(0, 256, (20, 23), dtype=np.uint8)
    ref = reflect101_correlate(img, np.full((5, 5), 1 / 25))
    out = F.mean_filter(img, ksize=5).astype(int)
    assert np.abs(out - np.rint(ref)).max() <= 1
    assert (out == np.rint(ref)).mean() > 0.99


def test_mean_larger_kernel_smooths_more(rng):
    noisy = np.clip(128 + rng.normal(0, 30, (64, 64)), 0, 255).astype(np.uint8)
    stds = [F.mean_filter(noisy, ksize=k).std() for k in (1, 3, 5, 9)]
    assert stds[0] == pytest.approx(noisy.std())
    assert stds[0] > stds[1] > stds[2] > stds[3]


def test_mean_color_channels_filtered_independently(rng):
    img = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
    out = F.mean_filter(img, ksize=3)
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], F.mean_filter(img[..., c], ksize=3))


# --------------------------------------------------------------------------
# B. Gaussian
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ksize,sigma", [(5, 1.0), (7, 1.5), (3, 0.6)])
def test_gaussian_kernel_formula(ksize, sigma):
    k = F.gaussian_kernel(ksize, sigma)
    x = np.arange(ksize) - ksize // 2
    ref = np.exp(-(x ** 2) / (2 * sigma ** 2))
    np.testing.assert_allclose(k, ref / ref.sum(), rtol=1e-12)
    assert k.sum() == pytest.approx(1.0)


def test_gaussian_matches_independent_reference(rng):
    img = rng.integers(0, 256, (24, 30), dtype=np.uint8)
    g = F.gaussian_kernel(7, 1.5)
    ref = reflect101_correlate(img, np.outer(g, g))
    out = F.gaussian_filter(img, ksize=7, sigma_x=1.5).astype(int)
    assert np.abs(out - np.rint(ref)).max() <= 1
    assert (out == np.rint(ref)).mean() > 0.99


def test_gaussian_impulse_response_is_the_kernel():
    img = np.zeros((11, 11), np.uint8)
    img[5, 5] = 255
    out = F.gaussian_filter(img, ksize=5, sigma_x=1.0)
    g = F.gaussian_kernel(5, 1.0)
    expected = np.rint(255 * np.outer(g, g))
    np.testing.assert_array_equal(out[3:8, 3:8], expected)
    assert out[5, 5] == out.max() and out[5, 4] == out[4, 5]  # tepe merkezde, izotropik


def test_gaussian_smoothing_increases_with_sigma(rng):
    noisy = np.clip(128 + rng.normal(0, 30, (64, 64)), 0, 255).astype(np.uint8)
    s1 = F.gaussian_filter(noisy, ksize=15, sigma_x=0.8).std()
    s2 = F.gaussian_filter(noisy, ksize=15, sigma_x=2.0).std()
    assert noisy.std() > s1 > s2
    assert abs(F.gaussian_filter(noisy, 15, 2.0).mean() - noisy.mean()) < 1.0  # ortalama korunur


def test_gaussian_no_color_shift_on_uniform_hue():
    img = np.zeros((16, 16, 3), np.uint8)
    img[:, :8] = (40, 120, 200)
    img[:, 8:] = (20, 60, 100)  # aynı ton, yarı parlaklık
    out = F.gaussian_filter(img, ksize=5, sigma_x=1.5).astype(float)
    # Her piksel iki rengin dışbükey birleşimi: B:G:R oranı 1:3:5 korunur
    np.testing.assert_allclose(out[..., 1] / out[..., 0], 3.0, atol=0.15)
    np.testing.assert_allclose(out[..., 2] / out[..., 0], 5.0, atol=0.25)


# --------------------------------------------------------------------------
# C. Median
# --------------------------------------------------------------------------


def test_median_known_values():
    img = np.array([[1, 2, 3], [4, 100, 6], [7, 8, 9]], dtype=np.uint8)
    assert F.median_filter(img, 3)[1, 1] == 6  # {1,2,3,4,6,7,8,9,100} → 6
    img5 = np.zeros((5, 5), np.uint8)
    img5[2, 2] = 255  # tek bir "tuz" pikseli
    np.testing.assert_array_equal(F.median_filter(img5, 3), np.zeros((5, 5)))


def test_median_matches_numpy_reference(rng):
    img = rng.integers(0, 256, (12, 14), dtype=np.uint8)
    padded = np.pad(img, 2, mode="edge")  # medianBlur: BORDER_REPLICATE
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5))
    ref = np.median(windows.reshape(*img.shape, 25), axis=-1)
    np.testing.assert_array_equal(F.median_filter(img, 5), ref.astype(np.uint8))


def test_median_removes_salt_and_pepper_better_than_mean(rng):
    clean = np.tile(np.linspace(60, 190, 64), (64, 1)).astype(np.uint8)
    noisy = clean.copy()
    mask = rng.random(clean.shape)
    noisy[mask < 0.04] = 0
    noisy[mask > 0.96] = 255

    def bad_pixels(img):
        return (np.abs(img.astype(int) - clean.astype(int)) > 10).mean()

    assert bad_pixels(noisy) > 0.07
    assert bad_pixels(F.median_filter(noisy, 3)) < 0.005
    assert bad_pixels(F.mean_filter(noisy, 3)) > 10 * bad_pixels(F.median_filter(noisy, 3))


def test_median_preserves_step_edge():
    edge = vertical_edge(right=200)
    np.testing.assert_array_equal(F.median_filter(edge, 5), edge)  # medyan basamağı bulanıklaştırmaz
    assert not np.array_equal(F.mean_filter(edge, 5), edge)


# --------------------------------------------------------------------------
# D/E. Sobel X / Sobel Y
# --------------------------------------------------------------------------


def test_sobel_x_vertical_edge_signed_values():
    g = F.sobel_response(vertical_edge(), "x", 3)
    # [-1 0 1; -2 0 2; -1 0 1]: kenarın iki yanındaki sütunlarda 100·(1+2+1) = 400
    np.testing.assert_array_equal(g[:, 4], 400)
    np.testing.assert_array_equal(g[:, 5], 400)
    assert np.all(g[:, [0, 1, 2, 3, 6, 7, 8, 9]] == 0)
    assert np.all(F.sobel_response(vertical_edge(), "y", 3) == 0)  # Gy dikey kenarı görmez


def test_sobel_y_horizontal_edge_signed_values():
    g = F.sobel_response(horizontal_edge(), "y", 3)
    np.testing.assert_array_equal(g[4, :], 400)
    np.testing.assert_array_equal(g[5, :], 400)
    assert np.all(np.delete(g, [4, 5], axis=0) == 0)
    assert np.all(F.sobel_response(horizontal_edge(), "x", 3) == 0)


def test_sobel_matches_textbook_kernel():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (15, 17), dtype=np.uint8)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    np.testing.assert_allclose(F.sobel_response(img, "x", 3), reflect101_correlate(img, kx), atol=1e-9)
    np.testing.assert_allclose(F.sobel_response(img, "y", 3), reflect101_correlate(img, kx.T), atol=1e-9)


def test_sobel_keeps_negative_derivatives():
    falling = vertical_edge(left=100, right=0)  # açıktan koyuya
    g = F.sobel_response(falling, "x", 3)
    assert g.min() == -400 and g.max() == 0  # işaret kaybolmadı
    # Büyüklük gösterimi kutupluluktan bağımsız: yükselen ve düşen kenar aynı görünür
    np.testing.assert_array_equal(F.sobel_x(falling), F.sobel_x(vertical_edge()))
    # Doğrudan uint8'e çevirmek negatif kenarı yok ederdi (OpenCV saturate → 0)
    assert np.all(cv2.Sobel(falling, cv2.CV_8U, 1, 0, ksize=3) == 0)
    assert F.sobel_x(falling).max() == 255


def test_sobel_display_modes():
    rising, falling = vertical_edge(), vertical_edge(left=100, right=0)
    out = F.sobel_x(rising, display="abs")
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    assert set(np.unique(out)) == {0, 255}
    s_up, s_down = F.sobel_x(rising, display="signed"), F.sobel_x(falling, display="signed")
    assert s_up[0, 0] == 128 and s_up[0, 4] == 255  # sıfır → orta gri, pozitif → açık
    assert s_down[0, 4] == 0  # negatif → koyu


def test_sobel_x_and_y_are_different_operations(rng):
    img = rng.integers(0, 256, (20, 20), dtype=np.uint8)
    gx, gy = F.sobel_response(img, "x", 3), F.sobel_response(img, "y", 3)
    assert not np.allclose(gx, gy)
    np.testing.assert_allclose(F.sobel_response(img.T.copy(), "x", 3), gy.T, atol=1e-9)  # Gx(fᵀ) = Gy(f)ᵀ
    assert not np.array_equal(F.sobel_x(vertical_edge()), F.sobel_y(vertical_edge()))


@pytest.mark.parametrize("ksize", [1, 3, 5, 7])
def test_sobel_apertures(ksize):
    out = F.sobel_x(vertical_edge(w=20, col=10), ksize=ksize)
    assert out.max() == 255 and out[:, 0].max() == 0


def test_derivatives_use_luminance_not_independent_channels():
    # İki bölge: farklı renk, aynı parlaklık → gerçek bir parlaklık kenarı yok
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :5] = (0, 0, 200)  # kırmızı: Y = 0.299·200 = 59.8
    img[:, 5:] = (0, 102, 0)  # yeşil:   Y = 0.587·102 = 59.874
    y = luminance(img)
    assert np.ptp(y) < 0.1
    assert np.abs(F.sobel_response(img, "x", 3)).max() < 0.5
    per_channel = max(np.abs(cv2.Sobel(img[..., c], cv2.CV_64F, 1, 0)).max() for c in range(3))
    assert per_channel > 300  # kanal başına yaklaşım sahte bir kenar görürdü
    out = F.sobel_x(img)
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_luminance_weights():
    px = np.array([[[10, 20, 30]]], np.uint8)  # B, G, R
    assert luminance(px)[0, 0] == pytest.approx(0.114 * 10 + 0.587 * 20 + 0.299 * 30)


# --------------------------------------------------------------------------
# F. Laplacian
# --------------------------------------------------------------------------


def test_laplacian_step_edge_second_derivative():
    r4 = F.laplacian_response(vertical_edge(), "4")
    # Koyu tarafta +100, açık tarafta −100; düz bölgelerde 0 (sıfır geçişi kenarda)
    np.testing.assert_array_equal(r4[:, 4], 100)
    np.testing.assert_array_equal(r4[:, 5], -100)
    assert np.all(np.delete(r4, [4, 5], axis=1) == 0)
    r8 = F.laplacian_response(vertical_edge(), "8")
    np.testing.assert_array_equal(r8[:, 4], 300)
    np.testing.assert_array_equal(r8[:, 5], -300)


def test_laplacian_is_zero_on_linear_ramp():
    ramp = np.tile(np.arange(0, 200, 10, dtype=np.uint8), (8, 1))
    r = F.laplacian_response(ramp, "4")
    assert np.all(r[:, 1:-1] == 0)  # ikinci türev doğrusal rampada sıfırdır


def test_laplacian_isotropic_responds_to_both_directions():
    square = np.zeros((20, 20), np.uint8)
    square[6:14, 6:14] = 200
    r = F.laplacian_response(square, "4")
    assert np.abs(r[6, 10]) > 0 and np.abs(r[10, 6]) > 0  # yatay ve dikey kenarlar
    np.testing.assert_allclose(r, r.T)  # simetrik desen → simetrik yanıt


def test_laplacian_matches_textbook_kernels(rng):
    img = rng.integers(0, 256, (13, 16), dtype=np.uint8)
    for key, kernel in F.LAPLACIAN_KERNELS.items():
        np.testing.assert_allclose(F.laplacian_response(img, key), reflect101_correlate(img, kernel), atol=1e-9)
    np.testing.assert_allclose(F.laplacian_response(img, "4"),
                               cv2.Laplacian(img, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT_101))


def test_laplacian_display_conversion():
    edge = vertical_edge()
    mag = F.laplacian(edge, display="abs")
    np.testing.assert_array_equal(mag[:, 4], 255)  # |+100| → 255
    np.testing.assert_array_equal(mag[:, 5], 255)  # |−100| → 255 (negatif kaybolmadı)
    assert np.all(np.delete(mag, [4, 5], axis=1) == 0)
    sig = F.laplacian(edge, display="signed")
    assert sig[0, 4] == 255 and sig[0, 5] == 0 and sig[0, 0] == 128
    # Kırpma ile doğrudan dönüşüm negatif tarafı silerdi
    assert np.all(np.clip(F.laplacian_response(edge, "4"), 0, 255)[:, 5] == 0)


def test_display_response_scaling():
    r = np.array([[-50.0, 0.0, 25.0, 100.0]])
    np.testing.assert_array_equal(F.display_response(r, "abs"), [[128, 0, 64, 255]])
    np.testing.assert_array_equal(F.display_response(r, "signed"), [[64, 128, 159, 255]])


# --------------------------------------------------------------------------
# Registry / katalog entegrasyonu
# --------------------------------------------------------------------------

SPATIAL_KEYS = ["mean_blur", "gaussian_blur", "median_blur", "sobel_x", "sobel_y", "laplacian"]


def test_spatial_category_contents_and_order():
    registry = build_registry()
    keys = [a.key for a in registry.in_category("spatial")]
    assert keys[:6] == SPATIAL_KEYS
    for key in SPATIAL_KEYS:
        algo = registry.get(key)
        assert algo.implemented and algo.category == "spatial" and algo.summary
        assert all(p.help for p in algo.params), key  # her parametrenin açıklaması var


@pytest.mark.parametrize("key", SPATIAL_KEYS)
def test_runs_with_default_parameters(key, rng):
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = build_registry().get(key).run(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_registry_sanitises_kernel_parameters():
    reg = build_registry()
    assert reg.get("mean_blur").resolve_params({"ksize": 4})["ksize"] == 5  # tek sayıya
    assert reg.get("median_blur").resolve_params({"ksize": 1})["ksize"] == 3  # alt sınır
    assert reg.get("gaussian_blur").resolve_params({"sigma_x": -2})["sigma_x"] == 0.1
    assert reg.get("sobel_x").resolve_params({"ksize": 9})["ksize"] == 7
    assert reg.get("laplacian").resolve_params({"display": "??"})["display"] == "abs"
