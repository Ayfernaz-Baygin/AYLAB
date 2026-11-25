"""Yoğunluk dönüşümleri: bilinen değerler, kenar durumları ve geçersiz girdiler."""

import cv2
import numpy as np
import pytest

from processing import intensity
from processing.catalog import build_registry
from processing.registry import AlgorithmParameterError

RAMP = np.array([[0, 51, 102, 153, 204, 255]], dtype=np.uint8)  # r/255 = 0, .2, .4, .6, .8, 1


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def color_image(rng):
    return rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)


def _unchanged_after(func, image, **kwargs):
    before = image.copy()
    out = func(image, **kwargs)
    np.testing.assert_array_equal(image, before)  # girdi dizisi değişmemeli
    assert out is not image and not np.shares_memory(out, image)
    return out


# --------------------------------------------------------------------------
# Ortak: tür / biçim doğrulaması
# --------------------------------------------------------------------------

ALL_FUNCS = [
    intensity.negative,
    intensity.gamma_correction,
    intensity.contrast_stretch,
    intensity.equalize_histogram,
]


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("bad", [
    np.zeros((4, 4), np.float32),
    np.zeros((4, 4), np.uint16),
    np.zeros((0, 0), np.uint8),
    np.zeros((4, 4, 2), np.uint8),
])
def test_rejects_invalid_images(func, bad):
    with pytest.raises(AlgorithmParameterError):
        func(bad)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_bgra_alpha_preserved_and_shape_kept(func, rng):
    img = rng.integers(0, 256, (10, 12, 4), dtype=np.uint8)
    out = _unchanged_after(func, img)
    assert out.shape == img.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(out[:, :, 3], img[:, :, 3])


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_single_channel_3d_gray(func, rng):
    img = rng.integers(0, 256, (8, 9, 1), dtype=np.uint8)
    out = func(img)
    assert out.shape == (8, 9, 1)


# --------------------------------------------------------------------------
# A. Negative
# --------------------------------------------------------------------------


def test_negative_known_values_and_extremes():
    img = np.array([[0, 1, 127, 128, 254, 255]], dtype=np.uint8)
    out = _unchanged_after(intensity.negative, img)
    np.testing.assert_array_equal(out, [[255, 254, 128, 127, 1, 0]])


def test_negative_color_each_channel(color_image):
    out = _unchanged_after(intensity.negative, color_image)
    np.testing.assert_array_equal(out, 255 - color_image.astype(int))
    np.testing.assert_array_equal(out, cv2.bitwise_not(color_image))


def test_negative_is_involution(color_image):
    np.testing.assert_array_equal(intensity.negative(intensity.negative(color_image)), color_image)


# --------------------------------------------------------------------------
# B. Gamma correction
# --------------------------------------------------------------------------


def test_gamma_known_values():
    # s = 255 · (r/255)^γ
    np.testing.assert_array_equal(intensity.gamma_correction(RAMP, gamma=2.0),
                                  [[0, 10, 41, 92, 163, 255]])
    np.testing.assert_array_equal(intensity.gamma_correction(RAMP, gamma=0.5),
                                  [[0, 114, 161, 198, 228, 255]])


def test_gamma_identity_and_endpoints(color_image):
    np.testing.assert_array_equal(intensity.gamma_correction(color_image, 1.0, 1.0), color_image)
    for g in (0.05, 0.4, 2.5, 10.0):
        lut = intensity.gamma_lut(g)
        assert lut[0] == 0 and lut[255] == 255  # 0 ve 1 güç dönüşümünün sabit noktaları
        assert np.all(np.diff(lut.astype(int)) >= 0)  # monoton artan


def test_gamma_direction(color_image):
    brighter = intensity.gamma_correction(color_image, gamma=0.5)
    darker = intensity.gamma_correction(color_image, gamma=2.0)
    assert brighter.mean() > color_image.mean() > darker.mean()


def test_gamma_gain_clips_instead_of_wrapping():
    img = np.array([[100, 200, 255]], dtype=np.uint8)
    out = intensity.gamma_correction(img, gamma=1.0, c=2.0)
    np.testing.assert_array_equal(out, [[200, 255, 255]])  # 400 → 255, uint8 sarmalanması yok


def test_gamma_matches_float_reference(color_image):
    ref = np.clip(np.rint(255.0 * (color_image / 255.0) ** 0.7), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(intensity.gamma_correction(color_image, gamma=0.7), ref)


@pytest.mark.parametrize("gamma,c", [(0, 1), (-1, 1), (float("nan"), 1), (None, 1), (1, 0), (1, -2)])
def test_gamma_invalid_parameters(gamma, c):
    with pytest.raises(AlgorithmParameterError):
        intensity.gamma_correction(RAMP, gamma=gamma, c=c)


# --------------------------------------------------------------------------
# C. Contrast stretching
# --------------------------------------------------------------------------


def test_stretch_auto_known_values():
    img = np.array([[50, 100, 150]], dtype=np.uint8)
    out = _unchanged_after(intensity.contrast_stretch, img)
    np.testing.assert_array_equal(out, [[0, 128, 255]])  # (r − 50) · 255 / 100


def test_stretch_manual_known_values_and_saturation():
    img = np.array([[50, 100, 150, 200, 250]], dtype=np.uint8)
    out = intensity.contrast_stretch(img, mode="manual", low=100, high=200)
    np.testing.assert_array_equal(out, [[0, 0, 128, 255, 255]])


def test_stretch_uses_full_range_and_matches_cv2_normalize(rng):
    img = rng.integers(60, 181, (20, 20), dtype=np.uint8)
    out = intensity.contrast_stretch(img)
    assert out.min() == 0 and out.max() == 255
    ref = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1  # yalnızca yuvarlama farkı


def test_stretch_color_uses_joint_limits():
    # Ortak sınırlar: kanallar arası oran (renk dengesi) korunur; kanal başına germe bozardı.
    img = np.zeros((2, 2, 3), np.uint8)
    img[0] = (40, 60, 80)
    img[1] = (60, 90, 120)
    out = intensity.contrast_stretch(img)
    lo, hi = 40, 120
    expected = np.clip(np.rint((img.astype(float) - lo) * 255 / (hi - lo)), 0, 255)
    np.testing.assert_array_equal(out, expected)
    assert not np.array_equal(out[0, 0], [0, 0, 0])  # bağımsız germe ilk satırı siyaha çekerdi


def test_stretch_full_range_image_is_identity(color_image):
    color_image[0, 0] = 0
    color_image[0, 1] = 255
    np.testing.assert_array_equal(intensity.contrast_stretch(color_image), color_image)


def test_stretch_constant_image_no_division_by_zero():
    img = np.full((5, 5), 77, np.uint8)
    with np.errstate(all="raise"):
        out = _unchanged_after(intensity.contrast_stretch, img)
    np.testing.assert_array_equal(out, img)
    np.testing.assert_array_equal(intensity.contrast_stretch(img, saturate=2.0), img)


def test_stretch_saturation_ignores_outliers(rng):
    img = rng.integers(100, 156, (50, 50), dtype=np.uint8)
    img[0, 0], img[0, 1] = 0, 255  # tek piksellik aykırı değerler
    plain = intensity.contrast_stretch(img)
    robust = intensity.contrast_stretch(img, saturate=1.0)
    assert robust.std() > plain.std()


@pytest.mark.parametrize("kwargs", [
    {"mode": "manual", "low": 200, "high": 100},
    {"mode": "manual", "low": 120, "high": 120},
    {"mode": "manual", "low": None, "high": 100},
    {"mode": "manual", "low": -5, "high": 100},
    {"mode": "bogus"},
    {"saturate": 60.0},
])
def test_stretch_invalid_parameters(kwargs):
    with pytest.raises(AlgorithmParameterError):
        intensity.contrast_stretch(RAMP, **kwargs)


# --------------------------------------------------------------------------
# D. Histogram equalization
# --------------------------------------------------------------------------


def test_equalize_two_levels_known_values():
    img = np.array([[10, 10, 20, 20]], dtype=np.uint8)
    np.testing.assert_array_equal(intensity.equalize_histogram(img), [[0, 0, 255, 255]])


@pytest.mark.parametrize("seed", range(5))
def test_equalize_gray_matches_opencv(seed):
    rng = np.random.default_rng(seed)
    # Çarpık (düşük kontrastlı) dağılım
    img = np.clip(rng.normal(90, 12, (64, 80)), 0, 255).astype(np.uint8)
    out = _unchanged_after(intensity.equalize_histogram, img)
    np.testing.assert_array_equal(out, cv2.equalizeHist(img))
    assert out.min() == 0 and out.max() == 255


def test_equalize_flattens_histogram(rng):
    img = np.clip(rng.normal(100, 8, (128, 128)), 0, 255).astype(np.uint8)
    out = intensity.equalize_histogram(img)
    cdf = np.cumsum(np.bincount(out.ravel(), minlength=256)) / out.size
    ideal = np.arange(1, 257) / 256
    assert np.abs(cdf - ideal).max() < 0.08  # eşitlenmiş CDF ≈ doğru (düzgün dağılım)
    assert out.std() > img.std() * 3


def test_equalize_constant_image_unchanged():
    img = np.full((6, 6), 140, np.uint8)
    with np.errstate(all="raise"):
        out = intensity.equalize_histogram(img)
    np.testing.assert_array_equal(out, img)
    color = np.full((6, 6, 3), (30, 60, 90), np.uint8)
    np.testing.assert_array_equal(intensity.equalize_histogram(color), color)


def _low_contrast_color(rng):
    img = np.zeros((40, 40, 3), np.uint8)
    img[:, :20] = (20, 30, 60)  # koyu kırmızımsı
    img[:, 20:] = (40, 50, 90)
    noise = rng.integers(-3, 4, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("space", ["ycrcb", "lab", "hsv"])
def test_equalize_color_changes_only_luminance(rng, space):
    img = _low_contrast_color(rng)
    out = _unchanged_after(intensity.equalize_histogram, img, space=space)
    assert out.shape == img.shape
    # Parlaklık genişler…
    y_in = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0]
    y_out = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[..., 0]
    assert int(y_out.max()) - int(y_out.min()) > 2 * (int(y_in.max()) - int(y_in.min()))
    # …ama renk tonu (hue) korunur.
    hue_in = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    hue_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    diff = np.minimum(np.abs(hue_in - hue_out), 180 - np.abs(hue_in - hue_out))
    assert np.median(diff) <= 4


def test_equalize_ycrcb_keeps_chroma_channels_vs_naive_rgb(rng):
    img = _low_contrast_color(rng)
    ours = intensity.equalize_histogram(img, space="ycrcb")
    naive = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])  # kanal başına (yanlış) yaklaşım

    def chroma_shift(out):
        a = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 1:].astype(int)
        b = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[..., 1:].astype(int)
        return np.abs(a - b).mean()

    assert chroma_shift(ours) < 3
    assert chroma_shift(naive) > 3 * chroma_shift(ours)


def test_equalize_ycrcb_y_channel_equals_gray_equalization(rng):
    img = _low_contrast_color(rng)
    out = intensity.equalize_histogram(img, space="ycrcb")
    y_expected = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0])
    y_out = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)[..., 0].astype(int)
    diff = y_out - y_expected.astype(int)
    # Gamut içindeki piksellerde Y tam olarak gri eşitlemenin sonucudur (± dönüşüm yuvarlaması).
    in_gamut = ((out > 0) & (out < 255)).all(axis=-1)
    assert in_gamut.mean() > 0.5
    assert np.abs(diff[in_gamut]).max() <= 2
    # Gamut dışına taşan pikseller [0, 255]'e kırpılır: 255'te kırpılan Y'yi düşürür,
    # 0'da kırpılan yükseltir. Bu sapma azınlıkta kalır ve ortalamada küçüktür.
    assert np.abs(diff).mean() < 2


def test_clahe_matches_opencv(rng):
    img = np.clip(rng.normal(110, 15, (64, 64)), 0, 255).astype(np.uint8)
    out = intensity.equalize_histogram(img, method="clahe", clip=3.0, tiles=4)
    ref = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(img)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("kwargs", [
    {"method": "bogus"},
    {"space": "rgb"},
    {"method": "clahe", "clip": 0},
    {"method": "clahe", "tiles": 0},
])
def test_equalize_invalid_parameters(color_image, kwargs):
    with pytest.raises(AlgorithmParameterError):
        intensity.equalize_histogram(color_image, **kwargs)


# --------------------------------------------------------------------------
# Registry bağlantısı ve parametre çözümleme
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["negative", "gamma", "contrast_stretch", "hist_eq"])
def test_registered_and_runnable_with_defaults(key, color_image):
    algo = build_registry().get(key)
    assert algo.implemented
    out = algo.run(color_image)  # boş parametre → varsayılanlar
    assert out.shape == color_image.shape and out.dtype == np.uint8


def test_registry_sanitises_empty_or_garbage_parameters(color_image):
    gamma = build_registry().get("gamma")
    # Anlamsız/boş değerler varsayılana, aralık dışı değerler sınıra çekilir.
    identity = gamma.run(color_image, {"gamma": "abc", "c": None})
    np.testing.assert_array_equal(identity, color_image)
    clamped = gamma.resolve_params({"gamma": 1e9, "c": -4})
    assert clamped == {"gamma": 10.0, "c": 0.1}


def test_registry_reports_cross_parameter_error(color_image):
    stretch = build_registry().get("contrast_stretch")
    with pytest.raises(AlgorithmParameterError, match="greater than input low"):
        stretch.run(color_image, {"mode": "manual", "low": 220, "high": 30})
