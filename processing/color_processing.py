"""Renk işleme — COMP 4360 Chapter 6 (slayt 24–42) ve Chapter 6 laboratuvarı (Color Lab).

Paket 1: renk modelleri, renk uzayı dönüşümü, kanal görüntüleri, renk histogramları.
Modeller slayt sırasıyla: RGB (26–28), CMY / CMYK (29), HSV / HSL / HSI (30–37), CIELAB (38),
YCbCr (39); OpenCV dönüşümleri (40–41), renk histogramları (42).

Formüllerin kaynağı
-------------------
* Slaytta formülü olanlar: CMY = 1 − RGB (slayt 29), RGB → HSI ve HSI → RGB (slayt 34–35,
  Gonzalez & Woods; üç 120° sektör).
* Slaytta yalnızca adı geçenler (CMYK, HSV, HSL, CIELAB, YCbCr) laboratuvar kodundaki formüllerle
  hesaplanır: CMYK K = min(C, M, Y), C′ = (C − K)/(1 − K); HSV / HSL altıgen koni (hexcone);
  CIELAB sRGB gama → doğrusal → XYZ (3×3) → D65 beyazı → f(t); YCbCr OpenCV BT.601 tam aralık
  (Cr = (R − Y)·0.713 + 128, Cb = (B − Y)·0.564 + 128).
* HSL → RGB laboratuvarda yoktur; yalnızca gidiş-dönüş için standart altıgen koni tersi kullanılır.

Kurallar
--------
* Giriş: gri (R = G = B), BGR ya da BGRA (alfa işlenmez, aynı boyutlu çıktılara aynen eklenir).
  OpenCV düzeni BGR'dir; dönüşümler içeride R, G, B sırasıyla yapılır ve çıktı yine BGR'dir.
* Kanallar gerçek birimlerinde float64 hesaplanır: R, G, B, Y, Cb, Cr ∈ [0, 255]; C, M, Y, K, S, V, L, I
  ∈ [0, 1]; H ∈ [0°, 360°); L* ∈ [0, 100]; a*, b* yaklaşık [−128, 127].
* Kanal görüntüsü (8-bit gösterim, laboratuvar kuralları): H → H/360·255; S, V, L, I → ·255;
  L* → L*·255/100; a*, b* → +128; C, M, Y, K → 255·(1 − değer) ("kâğıda basılmış" mürekkep, laboratuvar);
  R, G, B, Y, Cb, Cr → değerin kendisi. Yuvarlama yarım yukarı (JS Math.round), sonra [0, 255]'e kırpma.
* Tanımsız ton (gri pikseller, R = G = B) H = 0° kabul edilir (laboratuvar); siyahta S = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from processing.common import check_image, match_layout, merge_alpha, split_alpha
from processing.registry import AlgorithmParameterError, ProcessResult

MODELS = ("rgb", "cmy", "cmyk", "hsv", "hsl", "hsi", "lab", "ycbcr")
MODEL_NAMES = {"rgb": "RGB", "cmy": "CMY", "cmyk": "CMYK", "hsv": "HSV", "hsl": "HSL", "hsi": "HSI",
               "lab": "CIELAB", "ycbcr": "YCbCr"}
CONVERSION_OUTPUTS = ("channels", "roundtrip", "roundtrip8")
CHANNEL_STYLES = ("gray", "tint")
CHANNEL_INDEXES = ("1", "2", "3", "4")
HIST_BINS = ("256", "64", "32", "16")
EPS = 1e-9
HUE_MASK_S, HUE_MASK_V = 0.15, 0.1        # laboratuvar ton histogramı: S > 0.15 ve V > 0.1


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    lo: float          # histogram aralığı [lo, hi)
    hi: float
    unit: str
    kind: str          # 8-bit gösterim kuralı: value, ink, hue, unit, lightness, opponent


_RGB = tuple(ChannelSpec(n, 0, 256, "", "value") for n in "RGB")
_INK = lambda n: ChannelSpec(n, 0, 1, "", "ink")
_H = ChannelSpec("H", 0, 360, "°", "hue")
_U = lambda n: ChannelSpec(n, 0, 1, "", "unit")
CHANNELS: dict[str, tuple[ChannelSpec, ...]] = {
    "rgb": _RGB,
    "cmy": tuple(_INK(n) for n in "CMY"),
    "cmyk": tuple(_INK(n) for n in "CMYK"),
    "hsv": (_H, _U("S"), _U("V")),
    "hsl": (_H, _U("S"), _U("L")),
    "hsi": (_H, _U("S"), _U("I")),
    "lab": (ChannelSpec("L*", 0, 100, "", "lightness"), ChannelSpec("a*", -128, 128, "", "opponent"),
            ChannelSpec("b*", -128, 128, "", "opponent")),
    "ycbcr": tuple(ChannelSpec(n, 0, 256, "", "value") for n in ("Y", "Cb", "Cr")),
}
DISPLAY_RULE = {"value": "value", "ink": "255·(1 − v)", "hue": "H/360°·255", "unit": "v·255",
                "lightness": "L*·255/100", "opponent": "v + 128"}

# CIELAB (laboratuvar: sRGB, D65)
M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.072175],
                      [0.0193339, 0.119192, 0.9503041]])
M_XYZ2RGB = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.969266, 1.8760108, 0.041556],
                      [0.0556434, -0.2040259, 1.0572252]])
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
E3 = 6 / 29


# ------------------------------------------------------------------ doğrulama ve yardımcılar
def _choice(value, name: str, options) -> str:
    if isinstance(value, bool) or str(value) not in options:
        raise AlgorithmParameterError(f"{name} must be one of {', '.join(options)} (got {value!r}).")
    return str(value)


def js_round(v) -> np.ndarray:
    """JS Math.round (yarım yukarı): floor(v) + [v − floor(v) ≥ 0.5] — laboratuvarın c8 yuvarlaması."""
    v = np.asarray(v, dtype=np.float64)
    fl = np.floor(v)
    return fl + (v - fl >= 0.5)


def c8(v) -> np.ndarray:
    """Laboratuvar ``c8``: [0, 255]'e kırp, yarım yukarı yuvarla → uint8."""
    return js_round(np.clip(v, 0, 255)).astype(np.uint8)


def rgb_input(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(H×W×3 float64 R, G, B ∈ [0, 255], alfa). Gri girdi → R = G = B; BGR → RGB sırası."""
    check_image(image)
    color, alpha = split_alpha(image)
    if color.ndim == 2:
        color = color[:, :, None]
    if color.shape[2] == 1:
        rgb = np.repeat(color, 3, axis=2)
    else:
        rgb = color[:, :, ::-1]
    return rgb.astype(np.float64), alpha


def to_bgr8(rgb: np.ndarray) -> np.ndarray:
    """Gerçel R, G, B ∈ [0, 255] → uint8 BGR (c8 yuvarlama)."""
    return c8(rgb[:, :, ::-1])


# ------------------------------------------------------------------ dönüşümler (gerçek birimler)
def _hue_hexcone(r, g, b, v, d):
    """Laboratuvar rgb2hsv tonu: max = R → 60(G − B)/d; max = G → 120 + 60(B − R)/d; aksi 240 + 60(R − G)/d."""
    safe = np.where(d > EPS, d, 1.0)
    h = np.where(v == r, 60 * (g - b) / safe, np.where(v == g, 120 + 60 * (b - r) / safe, 240 + 60 * (r - g) / safe))
    h = np.where(h < 0, h + 360, h)
    return np.where(d > EPS, h, 0.0)


def rgb_to_model(rgb: np.ndarray, model: str) -> np.ndarray:
    """R, G, B ∈ [0, 255] (H×W×3) → seçilen modelin kanalları (H×W×k, gerçek birimler)."""
    r, g, b = (rgb[..., i] / 255 for i in range(3))
    if model == "rgb":
        return rgb.astype(np.float64).copy()
    if model == "cmy":                                            # slayt 29
        return np.stack([1 - r, 1 - g, 1 - b], -1)
    if model == "cmyk":                                           # laboratuvar rgb2cmyk
        c, m, y = 1 - r, 1 - g, 1 - b
        k = np.minimum(np.minimum(c, m), y)
        den = 1 - k
        safe = np.where(den < EPS, 1.0, den)
        return np.stack([np.where(den < EPS, 0, (c - k) / safe), np.where(den < EPS, 0, (m - k) / safe),
                         np.where(den < EPS, 0, (y - k) / safe), k], -1)
    if model in ("hsv", "hsl"):
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        d = mx - mn
        h = _hue_hexcone(r, g, b, mx, d)
        if model == "hsv":
            s = np.where(mx < EPS, 0.0, d / np.where(mx < EPS, 1.0, mx))
            return np.stack([h, s, mx], -1)
        light = (mx + mn) / 2
        den = 1 - np.abs(2 * light - 1)
        s = np.where(d < EPS, 0.0, d / np.where(den < EPS, 1.0, den))
        return np.stack([h, s, light], -1)
    if model == "hsi":                                            # slayt 34
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g) * (r - g) + (r - b) * (g - b))
        theta = (np.arccos(np.clip(num / np.where(den < EPS, 1.0, den), -1, 1)) * 180) / np.pi   # laboratuvar sırası
        h = np.where(den < EPS, 0.0, np.where(b <= g, theta, 360 - theta))
        total = r + g + b
        mn = np.minimum(np.minimum(r, g), b)
        s = np.where(total < EPS, 0.0, 1 - 3 / np.where(total < EPS, 1.0, total) * mn)
        return np.stack([h, s, total / 3], -1)
    if model == "lab":                                            # laboratuvar rgb2lab (sRGB, D65)
        srgb = np.stack([r, g, b], -1)
        lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        t = (lin @ M_RGB2XYZ.T) / WHITE_D65
        f = np.where(t > E3 ** 3, np.cbrt(t), t / (3 * E3 * E3) + 4 / 29)
        return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
    if model == "ycbcr":                                          # laboratuvar rgb2ycrcb (OpenCV BT.601)
        R, G, B = (rgb[..., i].astype(np.float64) for i in range(3))
        y = 0.299 * R + 0.587 * G + 0.114 * B
        return np.stack([y, (B - y) * 0.564 + 128, (R - y) * 0.713 + 128], -1)
    raise AlgorithmParameterError(f"Color model must be one of {', '.join(MODELS)} (got {model!r}).")


def model_to_rgb(v: np.ndarray, model: str) -> np.ndarray:
    """Model kanalları → R, G, B (float, 0…255; kırpılmamış — çıktıda c8 kırpar). CIELAB: laboratuvar gibi
    doğrusal RGB [0, 1]'e kırpılır (gamut dışı)."""
    if model == "rgb":
        return v.astype(np.float64).copy()
    if model == "cmy":
        return (1 - v) * 255
    if model == "cmyk":
        k = v[..., 3:4]
        return (1 - (v[..., :3] * (1 - k) + k)) * 255
    if model == "hsv":                                            # laboratuvar hsv2rgb
        h = (v[..., 0] % 360) / 60
        s, val = v[..., 1], v[..., 2]
        i = np.floor(h).astype(int) % 6
        fr = h - np.floor(h)
        p, q, t = val * (1 - s), val * (1 - s * fr), val * (1 - s * (1 - fr))
        table = np.stack([np.stack(c, -1) for c in ((val, t, p), (q, val, p), (p, val, t), (p, q, val),
                                                    (t, p, val), (val, p, q))], 0)
        return np.take_along_axis(table, i[None, ..., None].repeat(3, -1), 0)[0] * 255
    if model == "hsl":                                            # standart altıgen koni tersi (laboratuvarda yok)
        h, s, light = v[..., 0] % 360, v[..., 1], v[..., 2]
        c = (1 - np.abs(2 * light - 1)) * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = light - c / 2
        z = np.zeros_like(h)
        sec = np.floor(h / 60).astype(int) % 6
        table = np.stack([np.stack(t, -1) for t in ((c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x))], 0)
        return (np.take_along_axis(table, sec[None, ..., None].repeat(3, -1), 0)[0] + m[..., None]) * 255
    if model == "hsi":                                            # slayt 35: RG, GB, BR sektörleri
        h, s, i_ = ((v[..., 0] % 360) + 360) % 360, v[..., 1], v[..., 2]    # laboratuvar: ((H % 360) + 360) % 360
        hs = np.where(h < 120, h, np.where(h < 240, h - 120, h - 240))
        rad = lambda d: (d * np.pi) / 180                                    # laboratuvar rad()
        f = i_ * (1 + s * np.cos(rad(hs)) / np.cos(rad(60 - hs)))
        low = i_ * (1 - s)
        rest = 3 * i_ - (f + low)
        r = np.where(h < 120, f, np.where(h < 240, low, rest))
        g = np.where(h < 120, rest, np.where(h < 240, f, low))
        b = np.where(h < 120, low, np.where(h < 240, rest, f))
        return np.stack([r, g, b], -1) * 255
    if model == "lab":                                            # laboratuvar lab2rgb
        fy = (v[..., 0] + 16) / 116
        fx, fz = fy + v[..., 1] / 500, fy - v[..., 2] / 200
        inv = lambda t: np.where(t > E3, t ** 3, 3 * E3 * E3 * (t - 4 / 29))
        xyz = np.stack([inv(fx), inv(fy), inv(fz)], -1) * WHITE_D65
        lin = np.clip(xyz @ M_XYZ2RGB.T, 0, 1)
        return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255
    if model == "ycbcr":                                          # laboratuvar ycrcb2rgb
        y, cb, cr = v[..., 0], v[..., 1] - 128, v[..., 2] - 128
        return np.stack([y + 1.403 * cr, y - 0.714 * cr - 0.344 * cb, y + 1.773 * cb], -1)
    raise AlgorithmParameterError(f"Color model must be one of {', '.join(MODELS)} (got {model!r}).")


def encode8(values: np.ndarray, model: str) -> np.ndarray:
    """Kanallar → 8-bit gösterim kodları (H×W×k uint8), laboratuvarın kanal görüntüsü kuralları."""
    out = np.empty(values.shape, np.uint8)
    for k, spec in enumerate(CHANNELS[model]):
        x = values[..., k]
        out[..., k] = c8({"value": x, "ink": 255 * (1 - x), "hue": x / 360 * 255, "unit": x * 255,
                          "lightness": x * 255 / 100, "opponent": x + 128}[spec.kind])
    return out


def decode8(codes: np.ndarray, model: str) -> np.ndarray:
    """8-bit gösterim kodları → kanal değerleri (``encode8``in tersi; nicemleme hatası kalır)."""
    c = codes.astype(np.float64)
    out = np.empty(c.shape)
    for k, spec in enumerate(CHANNELS[model]):
        x = c[..., k]
        out[..., k] = {"value": x, "ink": 1 - x / 255, "hue": x / 255 * 360, "unit": x / 255,
                       "lightness": x * 100 / 255, "opponent": x - 128}[spec.kind]
    return out


# ------------------------------------------------------------------ raporlar
@dataclass(frozen=True)
class ColorReport:
    title: str
    detail: str

    def summary(self) -> str:
        return f"{self.title}: {self.detail}"


@dataclass(frozen=True)
class ChannelHistogram:
    name: str
    unit: str
    lo: float
    hi: float
    counts: np.ndarray      # int64, uzunluk = kutu sayısı
    mean: float
    minimum: float
    maximum: float

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def edges(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, self.counts.size + 1)


@dataclass(frozen=True)
class ColorHistogramReport:
    model: str
    bins: int
    pixels: int
    channels: tuple[ChannelHistogram, ...]
    note: str = ""

    def summary(self) -> str:
        parts = ", ".join(f"{c.name} μ = {c.mean:.3g}{c.unit}" for c in self.channels)
        return (f"{MODEL_NAMES[self.model]} histograms: {self.bins} bins per channel, {self.pixels} px; {parts}"
                + (f"; {self.note}" if self.note else ""))


def _stats_text(values: np.ndarray, model: str) -> str:
    return ", ".join(f"{s.name} {values[..., k].min():.4g}…{values[..., k].max():.4g}{s.unit}"
                     for k, s in enumerate(CHANNELS[model]))


# ------------------------------------------------------------------ işlemler
def color_space_conversion(image: np.ndarray, model: str = "hsv", output: str = "channels") -> ProcessResult:
    """Renk uzayı dönüşümü (slayt 24–41). CHANNELS: modelin kanalları yan yana (8-bit gösterim, aralarında
    3 px beyaz şerit; boyut girdiden farklı). ROUNDTRIP: RGB → model → RGB (gerçel). ROUNDTRIP8: kanallar
    8-bit gösterim kodlarına nicemlenip geri dönüştürülür (nicemleme hatası)."""
    _choice(model, "Color model", MODELS)
    _choice(output, "Output", CONVERSION_OUTPUTS)
    rgb, alpha = rgb_input(image)
    vals = rgb_to_model(rgb, model)
    name = MODEL_NAMES[model]
    if output == "channels":
        codes = encode8(vals, model)
        h = codes.shape[0]
        sep = np.full((h, 3), 255, np.uint8)
        parts = []
        for k in range(codes.shape[2]):
            parts += [codes[..., k], sep]
        mont = np.hstack(parts[:-1])
        out = mont if image.ndim == 2 else (mont[:, :, None] if image.shape[2] == 1 else np.repeat(mont[:, :, None], 3, 2))
        rules = ", ".join(f"{s.name}: {DISPLAY_RULE[s.kind]}" for s in CHANNELS[model])
        return ProcessResult(out, ColorReport(f"RGB → {name}", f"channels left to right ({rules}); "
                                                                f"{_stats_text(vals, model)}"))
    back_vals = decode8(encode8(vals, model), model) if output == "roundtrip8" else vals
    back = to_bgr8(model_to_rgb(back_vals, model))
    src = to_bgr8(rgb)
    err = np.abs(back.astype(int) - src.astype(int))
    changed = int(err.max(axis=2).astype(bool).sum())
    per = ", ".join(f"{c} {int(err[..., k].max())}" for k, c in zip((2, 1, 0), "RGB"))
    via = f"8-bit {name} channels" if output == "roundtrip8" else name
    out = merge_alpha(back, alpha)                                # her zaman BGR (gri girdide renk kayması görünür)
    mean_err = float(err.mean())
    return ProcessResult(out, ColorReport(f"RGB → {via} → RGB", f"max |Δ| {per}; mean |Δ| {mean_err:.4f}; "
                                                                f"{changed} of {err.shape[0] * err.shape[1]} px changed"))


def channel_visualization(image: np.ndarray, model: str = "hsv", channel: str = "1",
                          style: str = "gray") -> ProcessResult:
    """Tek kanal, tam boyut. GRAY: 8-bit gösterim kuralı. TINT (laboratuvar): RGB kanalı kendi renginde;
    CMY/CMYK mürekkebi beyaz kâğıt üzerinde kendi renginde (K gri)."""
    _choice(model, "Color model", MODELS)
    _choice(channel, "Channel", CHANNEL_INDEXES)
    _choice(style, "Style", CHANNEL_STYLES)
    specs = CHANNELS[model]
    k = int(channel) - 1
    if k >= len(specs):
        raise AlgorithmParameterError(f"{MODEL_NAMES[model]} has {len(specs)} channels "
                                      f"({', '.join(s.name for s in specs)}); channel {channel} does not exist.")
    if style == "tint" and model not in ("rgb", "cmy", "cmyk"):
        raise AlgorithmParameterError("Tinted display exists for RGB, CMY and CMYK only; use Gray for this model.")
    rgb, alpha = rgb_input(image)
    vals = rgb_to_model(rgb, model)
    spec = specs[k]
    code = encode8(vals, model)[..., k]
    x = vals[..., k]
    rule = DISPLAY_RULE[spec.kind]
    report = ColorReport(f"{MODEL_NAMES[model]} channel {spec.name}",
                         f"{spec.name} = {x.min():.4g}…{x.max():.4g}{spec.unit} (mean {x.mean():.4g}{spec.unit}); "
                         f"shown as {rule}" + (" (tinted)" if style == "tint" else ""))
    if style == "gray":
        return ProcessResult(match_layout(code, image), report)
    bgr = np.full(code.shape + (3,), 255, np.uint8)
    if model == "rgb":
        bgr[:] = 0
        bgr[..., 2 - k] = code                                     # R → BGR kanal 2
    elif spec.name == "K":
        bgr[:] = code[..., None]
    else:
        bgr[..., 2 - k] = code                                     # C: kırmızı ışık azalır (laboratuvar)
    return ProcessResult(merge_alpha(bgr, alpha), report)


def channel_histogram(x: np.ndarray, spec: ChannelSpec, bins: int, mask: np.ndarray | None = None) -> ChannelHistogram:
    """Laboratuvar ``hist``: kutu = ⌊(v − lo)·bins/(hi − lo)⌋, [0, bins − 1]'e kırpılır. Değerler önce 1e-9'a
    yuvarlanır (ör. gri pikselde Y = 99.99999999999999 → 100)."""
    v = np.round(x if mask is None else x[mask], 9).ravel()
    idx = np.clip(np.floor((v - spec.lo) * bins / (spec.hi - spec.lo)), 0, bins - 1).astype(np.int64)
    counts = np.bincount(idx, minlength=bins).astype(np.int64)
    if v.size:
        return ChannelHistogram(spec.name, spec.unit, spec.lo, spec.hi, counts, float(v.mean()), float(v.min()), float(v.max()))
    return ChannelHistogram(spec.name, spec.unit, spec.lo, spec.hi, counts, 0.0, 0.0, 0.0)


def color_histogram(image: np.ndarray, model: str = "rgb", bins: str = "256", hue_mask: bool = False) -> ProcessResult:
    """Renk histogramları (slayt 42): h_c(k) = #{(x, y) : f_c(x, y) kutu k'da}, her kanal ayrı, gerçek
    aralıklarla. Görüntü değişmez (B = A); histogramlar sağ paneldeki COLOR HISTOGRAM bölümünde çizilir.
    ``hue_mask`` (laboratuvar ton histogramı): H yalnızca HSV S > 0.15 ve V > 0.1 olan piksellerden."""
    _choice(model, "Color model", MODELS)
    nb = int(_choice(bins, "Bins", HIST_BINS))
    if not isinstance(hue_mask, (bool, np.bool_)):
        raise AlgorithmParameterError(f"Hue mask must be True or False (got {hue_mask!r}).")
    rgb, _ = rgb_input(image)
    vals = rgb_to_model(rgb, model)
    mask, note = None, ""
    if hue_mask and model in ("hsv", "hsl", "hsi"):
        hsv = rgb_to_model(rgb, "hsv")
        mask = (hsv[..., 1] > HUE_MASK_S) & (hsv[..., 2] > HUE_MASK_V)
        note = f"H counts {int(mask.sum())} chromatic px (S > {HUE_MASK_S}, V > {HUE_MASK_V})"
    elif hue_mask:
        raise AlgorithmParameterError("The hue mask applies to HSV, HSL and HSI only.")
    chans = tuple(channel_histogram(vals[..., k], s, nb, mask if s.kind == "hue" else None)
                  for k, s in enumerate(CHANNELS[model]))
    report = ColorHistogramReport(model, nb, int(rgb.shape[0] * rgb.shape[1]), chans, note)
    return ProcessResult(image.copy(), report, changes_image=False)    # yalnızca ölçüm: B ve geçmiş değişmez


# ====================================================================================================
# Paket 2: pseudo-color (sözde renk) — slayt 43–50, laboratuvar 'Intensity Slicing' ve 'Color Coding'.
# Gri görüntüden başlanır; renkler yapaydır (slayt 44, 46). Renkli girdi önce laboratuvar toGray ile griye
# çevrilir: f = c8(0.299 R + 0.587 G + 0.114 B). Çıktı her zaman BGR (BGRA girdide alfa eklenir).
# ====================================================================================================
SLICE_COUNTS = ("2", "4", "8")
# laboratuvar PAL8 (RGB) ve renk sayısına göre seçimi (colorsFor)
PAL8 = ((20, 20, 110), (30, 80, 220), (0, 190, 220), (40, 170, 60), (240, 220, 40), (250, 140, 20), (220, 30, 30),
        (255, 255, 255))
SLICE_COLORS = {2: ((30, 80, 220), (250, 200, 30)), 4: (PAL8[0], PAL8[3], PAL8[4], PAL8[6]), 8: PAL8}
# laboratuvar CMAPS: parçalı doğrusal duraklar (t, RGB) — sıra laboratuvardaki gibi
COLORMAP_STOPS = {
    "jet": ((0, (0, 0, 143)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)), (0.625, (255, 255, 0)),
            (0.875, (255, 0, 0)), (1, (128, 0, 0))),
    "hot": ((0, (10, 0, 0)), (0.375, (255, 0, 0)), (0.75, (255, 255, 0)), (1, (255, 255, 255))),
    "parula": ((0, (53, 42, 135)), (0.15, (15, 92, 221)), (0.3, (18, 125, 216)), (0.45, (7, 156, 207)),
               (0.6, (21, 177, 180)), (0.72, (89, 189, 140)), (0.84, (165, 190, 107)), (0.93, (225, 185, 82)),
               (1, (249, 251, 14))),
    "viridis": ((0, (68, 1, 84)), (0.13, (71, 44, 122)), (0.25, (59, 81, 139)), (0.38, (44, 113, 142)),
                (0.5, (33, 144, 141)), (0.63, (39, 173, 129)), (0.75, (92, 200, 99)), (0.88, (170, 220, 50)),
                (1, (253, 231, 37))),
    "bone": ((0, (0, 0, 0)), (0.375, (84, 84, 116)), (0.75, (169, 200, 200)), (1, (255, 255, 255))),
}
COLORMAPS = ("jet", "hot", "parula", "viridis", "hsv", "bone", "gray")          # laboratuvar NAMES sırası
COLORMAP_NAMES = {"jet": "Jet", "hot": "Hot", "parula": "Parula", "viridis": "Viridis", "hsv": "HSV",
                  "bone": "Bone", "gray": "Gray"}
SLICE_PALETTES = ("lab",) + COLORMAPS


@dataclass(frozen=True)
class PseudoColorReport:
    """Gri seviye → renk tablosu (256×3 RGB uint8) + varsa dilim aralıkları (lo, hi, RGB), hi hariç."""

    title: str
    detail: str
    lut: np.ndarray
    slices: tuple[tuple[int, int, tuple[int, int, int]], ...] = ()
    gray_from_color: bool = False

    def summary(self) -> str:
        return f"{self.title}: {self.detail}" + ("; color input → Y = 0.299R + 0.587G + 0.114B" if self.gray_from_color else "")


def gray_levels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """(f uint8 H×W, alfa, renkli miydi). Gri girdi aynen; renkli girdi laboratuvar toGray: c8(luma)."""
    rgb, alpha = rgb_input(image)
    color, _ = split_alpha(image)
    is_color = color.ndim == 3 and color.shape[2] == 3
    if not is_color:
        return rgb[..., 0].astype(np.uint8), alpha, False
    return c8(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]), alpha, True


def colormap_lut(name: str) -> np.ndarray:
    """Laboratuvar ``lutOf``: v = 0…255 için c8(CMAPS[name](v / 255)) → 256×3 RGB uint8.
    Duraklar arasında doğrusal ara değer: c₀ + (c₁ − c₀)·(t − t₀)/(t₁ − t₀), ilk t ≤ t₁ aralığı."""
    _choice(name, "Color map", COLORMAPS)
    t = np.arange(256) / 255
    if name == "gray":
        vals = np.stack([t * 255] * 3, -1)
    elif name == "hsv":                                                          # laboratuvar: hsv2rgb(300·t, 1, 1)
        vals = model_to_rgb(np.stack([t * 300, np.ones(256), np.ones(256)], -1)[None], "hsv")[0]
    else:
        stops = COLORMAP_STOPS[name]
        ts = np.array([s[0] for s in stops], float)
        cs = np.array([s[1] for s in stops], float)
        k = np.clip(np.searchsorted(ts, t, side="left") - 1, 0, len(ts) - 2)    # ilk t ≤ t₁ olan aralık
        u = (t - ts[k]) / (ts[k + 1] - ts[k])
        vals = cs[k] + (cs[k + 1] - cs[k]) * u[:, None]
    return c8(vals)


def slice_thresholds(n: int, plane: int) -> tuple[int, ...]:
    """Laboratuvar thrsFor: 2 renk → (l_i,); 4 / 8 renk → round(k·256/n), k = 1…n−1 (eşit aralıklar)."""
    return (plane,) if n == 2 else tuple(int(js_round(k * 256 / n)) for k in range(1, n))


def slice_index(f: np.ndarray, thresholds) -> np.ndarray:
    """Laboratuvar sliceOf: f ≥ eşik olan eşik sayısı → dilim no (0…n−1). f = l_i üst dilime düşer."""
    k = np.zeros(f.shape, np.int64)
    for t in thresholds:
        k += f >= t
    return k


def _colorize(f: np.ndarray, lut: np.ndarray, alpha) -> np.ndarray:
    return merge_alpha(np.ascontiguousarray(lut[f][..., ::-1]), alpha)      # RGB tablo → BGR görüntü


def intensity_slicing(image: np.ndarray, colors: str = "2", plane: int = 128, palette: str = "lab") -> ProcessResult:
    """Intensity slicing (slayt 47–49; laboratuvar): yoğunluk fonksiyonu yatay düzlemlerle kesilir; iki düzlem
    arasındaki her V_k aralığı kendi rengini alır: g(x, y) = c_k, f(x, y) ∈ V_k. Tek düzlemde (2 renk)
    f < l_i → c₁, f ≥ l_i → c₂. LAB paleti laboratuvar renkleri; bir renk haritası seçilirse c_k, haritanın
    aralık ortasındaki rengidir."""
    n = int(_choice(colors, "Colors", SLICE_COUNTS))
    if isinstance(plane, bool) or not isinstance(plane, (int, np.integer)) or not 1 <= int(plane) <= 255:
        raise AlgorithmParameterError(f"Slicing plane l_i must be an integer in 1…255 (got {plane!r}).")
    _choice(palette, "Palette", SLICE_PALETTES)
    f, alpha, from_color = gray_levels(image)
    th = slice_thresholds(n, int(plane))
    edges = (0,) + th + (256,)
    if palette == "lab":
        cols = SLICE_COLORS[n]
    else:
        cmap = colormap_lut(palette)
        cols = tuple(tuple(int(c) for c in cmap[min(255, (lo + hi - 1) // 2)]) for lo, hi in zip(edges[:-1], edges[1:]))
    lut = np.array([cols[k] for k in slice_index(np.arange(256), th)], np.uint8)
    slices = tuple((lo, hi, tuple(int(c) for c in col)) for lo, hi, col in zip(edges[:-1], edges[1:], cols))
    counts = np.bincount(slice_index(f, th).ravel(), minlength=n)
    text = ", ".join(f"[{lo}, {hi}) {int(c)} px" for (lo, hi, _), c in zip(slices, counts))
    pal = "lab palette" if palette == "lab" else f"{COLORMAP_NAMES[palette]} colors"
    report = PseudoColorReport("Intensity slicing", f"{n} colors ({pal}), planes at {', '.join(map(str, th))}: {text}",
                               lut, slices, from_color)
    return ProcessResult(_colorize(f, lut, alpha), report)


def color_coding(image: np.ndarray, cmap: str = "jet") -> ProcessResult:
    """Color coding (slayt 44, 50; laboratuvar): R = f_R(f), G = f_G(f), B = f_B(f) — üç bağımsız aktarım
    fonksiyonu, 256 girdili bir arama tablosu (LUT) olarak: g(x, y) = LUT[f(x, y)]."""
    lut = colormap_lut(cmap)
    f, alpha, from_color = gray_levels(image)
    lo, hi = int(f.min()), int(f.max())
    report = PseudoColorReport(f"Color coding · {COLORMAP_NAMES[cmap]}",
                               f"256-entry LUT, f = {lo}…{hi} → RGB {tuple(int(c) for c in lut[lo])}…"
                               f"{tuple(int(c) for c in lut[hi])}", lut, (), from_color)
    return ProcessResult(_colorize(f, lut, alpha), report)


# ====================================================================================================
# Paket 3: full-color (tam renkli) işleme — slayt 51–56, laboratuvar 'Color Transformations', 'Color Slicing',
# 'Tone & Color Corrections', 'Color Balancing & White Balance', 'Histogram Equalization in HSI'.
# Her piksel bir renk vektörüdür; hesap R, G, B ∈ [0, 1] (ya da modelin kendi birimleri) üzerinde yapılır,
# sonuç laboratuvar setpx gibi c8 ile yuvarlanır. Gri girdi R = G = B kabul edilir; çıktı BGR (+ alfa).
# Kayan nokta işlem sırası laboratuvardakiyle aynıdır (yuvarlama sınırlarında aynı sonuç için).
# ====================================================================================================
TRANSFORM_MODELS = ("rgb", "cmy", "cmyk", "hsi")
TRANSFORM_K_RANGE = (0.1, 1.0)                     # laboratuvar k kaydırıcısı
SLICE_SHAPES = ("cube", "sphere")
SLICE_SIZE_RANGE = (10, 255)                       # laboratuvar W ve R₀ kaydırıcıları
SLICE_NEUTRAL = 128                                # dışarıdaki pikseller: 0.5 → c8(127.5) = 128
TONE_CURVES = ("s", "gamma")
TONE_STRENGTH_RANGE = (1.0, 16.0)                  # laboratuvar S-eğrisi gücü
TONE_GAMMA_RANGE = (0.3, 3.0)                      # laboratuvar γ
BALANCE_METHODS = ("white", "gray", "neutral", "gains", "curves")
BALANCE_GAIN_RANGE = (0.0, 4.0)                    # AYLAB: elle kazanç
BALANCE_GAMMA_RANGE = (0.4, 2.5)                   # laboratuvar Şekil 6.36 kanal eğrileri
EQ_METHODS = ("hsi", "rgb")
EQ_SAT_RANGE = (0.5, 1.6)                          # laboratuvar doygunluk × kaydırıcısı
CHANNEL_INK = {"R": (224, 49, 49), "G": (47, 158, 68), "B": (59, 91, 219)}     # laboratuvar eğri renkleri
IDENTITY_INK = (134, 142, 150)


@dataclass(frozen=True)
class TransferCurve:
    """Bir aktarım fonksiyonu: values[v] = T(v / 255), v = 0…255, çıkış [0, 1] birimiyle (çizim için)."""

    name: str
    rgb: tuple[int, int, int] | None       # çizgi rengi (None → vurgu rengi)
    values: np.ndarray


@dataclass(frozen=True)
class FullColorReport:
    """Tam renkli işlem raporu: özet, (varsa) aktarım eğrileri, ek satırlar ve (varsa) örnek renk."""

    title: str
    detail: str
    curves: tuple[TransferCurve, ...] = ()
    rows: tuple[str, ...] = ()
    swatch: tuple[int, int, int] | None = None
    section: str = ""                      # sağ panel başlığı (boşsa: eğri varsa TRANSFER CURVES, yoksa COLOR SLICE)

    def summary(self) -> str:
        return f"{self.title}: {self.detail}"


def _number(value, name: str, lo: float, hi: float, integer: bool = False):
    kinds = (int, np.integer) if integer else (int, float, np.integer, np.floating)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, kinds) or not np.isfinite(value) \
            or not lo <= value <= hi:
        kind = "an integer" if integer else "a number"
        raise AlgorithmParameterError(f"{name} must be {kind} in {lo:g}…{hi:g} (got {value!r}).")
    return int(value) if integer else float(value)


def _rgb8(r, g, b, name: str) -> tuple[int, int, int]:
    return tuple(_number(v, f"{name} {c}", 0, 255, integer=True) for v, c in zip((r, g, b), "RGB"))


def _out(rgb: np.ndarray, alpha) -> np.ndarray:
    """Gerçel R, G, B ∈ [0, 255] → BGR uint8 (laboratuvar setpx: kırp + yarım yukarı yuvarla) + alfa."""
    return merge_alpha(np.ascontiguousarray(to_bgr8(rgb)), alpha)


def _line(f) -> np.ndarray:
    return np.asarray(f(np.arange(256) / 255), dtype=np.float64)


def _identity(name: str) -> TransferCurve:
    return TransferCurve(name, IDENTITY_INK, _line(lambda x: x))


# ------------------------------------------------------------------ slayt 52: color transformations
def transform_rgb(rgb: np.ndarray, model: str, k: float) -> np.ndarray:
    """Slayt 52 / laboratuvar: yoğunluğu k ile azaltan aynı etki, farklı modellerde (R, G, B ∈ [0, 255] →
    [0, 255], kırpılmamış). RGB s_i = k r_i; CMY s_i = k r_i + (1 − k); CMYK (Şekil 6.29 d–e) s_4 = k r_4 + (1 − k),
    C, M, Y aynı; HSI s_3 = k r_3, H, S aynı."""
    n = rgb / 255
    if model == "rgb":                                             # laboratuvar: k·v, sonra ·255
        return (k * n) * 255
    if model == "cmy":                                             # laboratuvar: 1 − (k·c + (1 − k))
        return (1 - (k * (1 - n) + (1 - k))) * 255
    if model == "cmyk":                                            # slayt 52 (laboratuvarda yok)
        v = rgb_to_model(rgb, "cmyk")
        v[..., 3] = k * v[..., 3] + (1 - k)
        return model_to_rgb(v, "cmyk")
    if model == "hsi":                                             # laboratuvar: hsi2rgb(H, S, k·I)
        v = rgb_to_model(rgb, "hsi")
        v[..., 2] = k * v[..., 2]
        return model_to_rgb(v, "hsi")
    raise AlgorithmParameterError(f"Color model must be one of {', '.join(TRANSFORM_MODELS)} (got {model!r}).")


TRANSFORM_FORMULAS = {"rgb": "s_i = k·r_i (R, G, B)", "cmy": "s_i = k·r_i + (1 − k) (C, M, Y)",
                      "cmyk": "s_4 = k·r_4 + (1 − k) (K), C, M, Y kept", "hsi": "s_3 = k·r_3 (I), H, S kept"}


def color_transformation(image: np.ndarray, model: str = "rgb", k: float = 0.7) -> ProcessResult:
    """Color transformations (slayt 51–52, Şekil 6.29; laboratuvar): s_i = T_i(r_1, …, r_n). Aynı etki — yoğunluğu
    k katına indirmek — seçilen modelde uygulanır; rapor dört modelin sonuçları arasındaki en büyük farkı verir
    (yalnızca yuvarlama farkı beklenir)."""
    _choice(model, "Color model", TRANSFORM_MODELS)
    k = _number(k, "k", *TRANSFORM_K_RANGE)
    rgb, alpha = rgb_input(image)
    outs = {m: to_bgr8(transform_rgb(rgb, m, k)) for m in TRANSFORM_MODELS}
    diffs = {m: int(np.abs(outs[m].astype(np.int16) - outs[model]).max(initial=0)) for m in TRANSFORM_MODELS}
    scale = lambda x: k * x
    shift = lambda x: k * x + 1 - k
    curves = {"rgb": (TransferCurve("R, G, B", None, _line(scale)),),
              "cmy": (TransferCurve("C, M, Y", None, _line(shift)),),
              "cmyk": (_identity("C, M, Y"), TransferCurve("K", None, _line(shift))),
              "hsi": (_identity("H, S"), TransferCurve("I", None, _line(scale)))}
    others = ", ".join(f"{MODEL_NAMES[m]} {d}" for m, d in diffs.items() if m != model)
    report = FullColorReport(
        f"Color transformation · {MODEL_NAMES[model]}",
        f"{TRANSFORM_FORMULAS[model]}, k = {k:g}; largest difference from the other models: {others} gray level(s)",
        curves[model], tuple(f"{MODEL_NAMES[m]}: {TRANSFORM_FORMULAS[m]}" for m in TRANSFORM_MODELS)
        + ("Same effect in every model — only the form differs (rounding only).",))
    return ProcessResult(merge_alpha(np.ascontiguousarray(outs[model]), alpha), report)


# ------------------------------------------------------------------ slayt 53: color slicing
def slice_keep(rgb: np.ndarray, shape: str, a, width: float, radius: float) -> np.ndarray:
    """Slayt 53 / laboratuvar: küp → tüm j için |r_j − a_j| ≤ W/2; küre → Σ(r_j − a_j)² ≤ R₀²."""
    d = rgb - np.asarray(a, np.float64)
    if shape == "cube":
        return np.all(np.abs(d) <= width / 2, axis=-1)
    return np.sum(d * d, axis=-1) <= radius * radius


def color_slicing(image: np.ndarray, shape: str = "cube", width: int = 110, radius: int = 80,
                  proto_r: int = 190, proto_g: int = 25, proto_b: int = 45) -> ProcessResult:
    """Color slicing (slayt 53; laboratuvar): prototip renk a etrafındaki küp (genişlik W) ya da küre (yarıçap R₀)
    içindeki renkler korunur, diğerleri nötr griye (0.5, 0.5, 0.5) gider:
    s_i = 0.5 eğer bir j için |r_j − a_j| > W/2 (küre: Σ(r_j − a_j)² > R₀²), aksi hâlde r_i."""
    _choice(shape, "Region", SLICE_SHAPES)
    width = _number(width, "Width W", *SLICE_SIZE_RANGE, integer=True)
    radius = _number(radius, "Radius R0", *SLICE_SIZE_RANGE, integer=True)
    a = _rgb8(proto_r, proto_g, proto_b, "Prototype")
    rgb, alpha = rgb_input(image)
    keep = slice_keep(rgb, shape, a, width, radius)
    out = np.where(keep[..., None], rgb, float(SLICE_NEUTRAL))
    n, kept = keep.size, int(keep.sum())
    if shape == "cube":
        rule = f"cube W = {width}: |r_j − a_j| ≤ {width / 2:g}"
        rows = tuple(f"{c} ∈ [{max(0, a[j] - width / 2):g}, {min(255, a[j] + width / 2):g}]" for j, c in enumerate("RGB"))
    else:
        rule = f"sphere R₀ = {radius}: Σ(r_j − a_j)² ≤ {radius * radius}"
        rows = (f"√Σ(r_j − a_j)² ≤ {radius} around a",)
    report = FullColorReport(
        f"Color slicing · {shape}", f"prototype a = RGB {a[0]}, {a[1]}, {a[2]}; {rule}; kept {kept} of {n} px "
        f"({100 * kept / max(1, n):.1f} %), others → gray (128, 128, 128)",
        rows=rows + ("Click the image to pick a from B (the input of Apply).",), swatch=a)
    return ProcessResult(_out(out, alpha), report)


# ------------------------------------------------------------------ slayt 54: tone corrections
def s_curve(a: float):
    """Laboratuvar sCurve: s(x) = 1 / (1 + e^(−a(x − 0.5))), T(x) = (s(x) − s(0)) / (s(1) − s(0))."""
    s = lambda x: 1 / (1 + np.exp(-a * (x - 0.5)))
    lo, hi = s(0.0), s(1.0)
    return lambda x: (s(x) - lo) / (hi - lo)


def tone_lut(curve: str, strength: float, gamma: float) -> tuple[np.ndarray, object]:
    """(256 girdili LUT = c8(255·T(v / 255)), T)."""
    f = s_curve(strength) if curve == "s" else (lambda x: np.power(x, gamma))
    return c8(255 * _line(f)), f


def _hue_shift_text(before: np.ndarray, after: np.ndarray) -> str:
    """Kromatik piksellerde (S > 0.1, her iki görüntüde) HSI tonunun ortalama ve en büyük kayması."""
    h0, h1 = rgb_to_model(before, "hsi"), rgb_to_model(after, "hsi")
    mask = (h0[..., 1] > 0.1) & (h1[..., 1] > 0.1)
    if not mask.any():
        return "no chromatic pixels (S ≤ 0.1)"
    d = np.abs((h1[..., 0] - h0[..., 0] + 540) % 360 - 180)[mask]
    return f"hue shift on chromatic px: mean {d.mean():.2f}°, max {d.max():.1f}°"


def tone_correction(image: np.ndarray, curve: str = "s", strength: float = 9.0, gamma: float = 1.0) -> ProcessResult:
    """Tone corrections (slayt 54, Şekil 6.33; laboratuvar): aynı aktarım fonksiyonu R, G ve B'ye eşit uygulanır:
    s_i = T(r_i). Düz (düşük kontrast) → S-eğrisi; açık → T(r) = r^γ, γ > 1; koyu → γ < 1."""
    _choice(curve, "Curve", TONE_CURVES)
    strength = _number(strength, "S-curve strength", *TONE_STRENGTH_RANGE)
    gamma = _number(gamma, "Gamma", *TONE_GAMMA_RANGE)
    lut, f = tone_lut(curve, strength, gamma)
    rgb, alpha = rgb_input(image)
    out = lut[rgb.astype(np.uint8)]
    name = f"S-curve, strength a = {strength:g}" if curve == "s" else f"power law T(r) = r^γ, γ = {gamma:g}"
    report = FullColorReport("Tone correction", f"{name}, same curve for R, G and B; "
                             f"{_hue_shift_text(rgb, out.astype(np.float64))}",
                             (TransferCurve("R, G, B", None, _line(f)),),
                             ("s_i = T(r_i), i = R, G, B (one curve for all three channels)",
                              "Lab cases (slide 54): flat → S-curve a = 9, too light → γ = 2, too dark → γ = 0.45"))
    return ProcessResult(merge_alpha(np.ascontiguousarray(out[..., ::-1]), alpha), report)


# ------------------------------------------------------------------ slayt 55: color balancing / white balance
def balance_gains(rgb: np.ndarray, method: str, neutral=(128, 128, 128), gains=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, str]:
    """Laboratuvar kazançları (köşegen / von Kries): gri dünya k_c = m̄ / m_c; beyaz yama k_c = 255 / max(1, max_c);
    nötr piksel p: k_c = p̄ / max(1, p_c). Ortalaması 0 olan kanalın kazancı 1 alınır (laboratuvar 0/0 üretir)."""
    flat = rgb.reshape(-1, 3)
    if method == "gray":
        mean = flat.sum(axis=0) / flat.shape[0]
        g = (mean[0] + mean[1] + mean[2]) / 3
        k = np.array([g / m if m > EPS else 1.0 for m in mean])
        why = f"channel means {', '.join(f'{m:.1f}' for m in mean)} → m̄ = {g:.1f}"
    elif method == "white":
        mx = flat.max(axis=0)
        k = 255 / np.maximum(1, mx)
        why = f"channel maxima {', '.join(f'{m:g}' for m in mx)} → 255"
    elif method == "neutral":
        p = np.asarray(neutral, np.float64)
        g = (p[0] + p[1] + p[2]) / 3
        k = g / np.maximum(1, p)
        why = f"pixel RGB {', '.join(f'{v:g}' for v in p)} → gray {g:.1f}"
    else:                                                          # AYLAB: elle kazanç
        k = np.asarray(gains, np.float64)
        why = "set by hand (AYLAB)"
    return k, why


BALANCE_NAMES = {"white": "white patch", "gray": "grey world", "neutral": "neutral pixel", "gains": "channel gains",
                 "curves": "channel curves"}


def color_balance(image: np.ndarray, method: str = "white", neutral_r: int = 128, neutral_g: int = 128,
                  neutral_b: int = 128, gain_r: float = 1.0, gain_g: float = 1.0, gain_b: float = 1.0,
                  gamma_r: float = 1.0, gamma_g: float = 1.0, gamma_b: float = 1.0) -> ProcessResult:
    """Color balancing / white balance (slayt 55, Şekil 6.36; laboratuvar): her kanal ayrı ölçeklenir ya da ayrı bir
    eğriden geçer, böylece nötr nesneler yeniden nötr olur. R′ = k_R R, G′ = k_G G, B′ = k_B B (WHITE PATCH, GREY
    WORLD, NEUTRAL PIXEL, GAINS) ya da c′ = 255·(c / 255)^γ_c (CURVES, Şekil 6.36: γ > 1 kanalı zayıflatır)."""
    _choice(method, "Method", BALANCE_METHODS)
    neutral = _rgb8(neutral_r, neutral_g, neutral_b, "Neutral pixel")
    gains = tuple(_number(v, f"Gain k_{c}", *BALANCE_GAIN_RANGE) for v, c in zip((gain_r, gain_g, gain_b), "RGB"))
    gammas = tuple(_number(v, f"Gamma γ_{c}", *BALANCE_GAMMA_RANGE) for v, c in zip((gamma_r, gamma_g, gamma_b), "RGB"))
    rgb, alpha = rgb_input(image)
    if method == "curves":
        fs = [lambda x, g=g: np.power(x, g) for g in gammas]
        luts = [c8(255 * _line(f)) for f in fs]                    # laboratuvar: c8(255·(v / 255)^γ_c)
        out = np.stack([luts[c][rgb[..., c].astype(np.uint8)] for c in range(3)], -1).astype(np.float64)
        curves = tuple(TransferCurve(c, CHANNEL_INK[c], _line(f)) for c, f in zip("RGB", fs))
        detail = "c′ = 255·(c / 255)^γ_c, " + ", ".join(f"γ_{c} = {g:g}" for c, g in zip("RGB", gammas))
        rows = ("Fig. 6.36: one curve per channel; γ > 1 weakens a channel, γ < 1 strengthens it.",)
    else:
        k, why = balance_gains(rgb, method, neutral, gains)
        out = rgb * k                                               # laboratuvar: c8(r·k_R), c8(g·k_G), c8(b·k_B)
        curves = tuple(TransferCurve(c, CHANNEL_INK[c], _line(lambda x, g=g: np.minimum(1, g * x))) for c, g in zip("RGB", k))
        detail = f"{why}; gains " + ", ".join(f"k_{c} = {g:.3f}" for c, g in zip("RGB", k))
        rows = ("R′ = k_R·R, G′ = k_G·G, B′ = k_B·B (diagonal / von Kries correction)",)
    out8 = to_bgr8(out)
    before = rgb.reshape(-1, 3).mean(axis=0)
    after = out8.reshape(-1, 3)[:, ::-1].mean(axis=0)
    rows += (f"means before: R {before[0]:.1f}, G {before[1]:.1f}, B {before[2]:.1f}",
             f"means after:  R {after[0]:.1f}, G {after[1]:.1f}, B {after[2]:.1f}")
    report = FullColorReport(f"Color balance · {BALANCE_NAMES[method]}", detail, curves, rows,
                             neutral if method == "neutral" else None)
    return ProcessResult(merge_alpha(np.ascontiguousarray(out8), alpha), report)


# ------------------------------------------------------------------ slayt 56: histogram equalization in HSI
def equalization_lut(values: np.ndarray, levels: int = 256) -> np.ndarray:
    """Laboratuvar eqLUT: T(k) = c8((L − 1)·Σ_{j ≤ k} h(j) / N)."""
    h = np.bincount(values.ravel(), minlength=levels)[:levels]
    return c8((levels - 1) * np.cumsum(h) / max(1, values.size))


def equalize_hsi(rgb: np.ndarray, saturation: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slayt 56 / laboratuvar: RGB → HSI, I8 = c8(255·I), I′ = T(I8) / 255, S′ = min(1, S·sat), H aynı → RGB.
    Döner: (HSI giriş, HSI çıkış (H, S′, I′), R, G, B ∈ [0, 255] kırpılmamış)."""
    hsi = rgb_to_model(rgb, "hsi")
    i8 = c8(hsi[..., 2] * 255)
    new = hsi.copy()
    new[..., 2] = equalization_lut(i8)[i8] / 255
    new[..., 1] = np.minimum(1, hsi[..., 1] * saturation)
    return hsi, new, model_to_rgb(new, "hsi")


def hsi_equalization(image: np.ndarray, method: str = "hsi", saturation: float = 1.0) -> ProcessResult:
    """Histogram equalization in HSI (slayt 56; laboratuvar): yalnızca I eşitlenir, H ve S korunur —
    (H, S, I) → (H, S, T(I)) → RGB, T(I) = (L − 1) Σ_{j ≤ I} p_I(j). RGB yöntemi laboratuvardaki karşılaştırmadır
    (R, G, B ayrı ayrı eşitlenir, tonlar kayabilir). Doygunluk × (laboratuvar): S′ = min(1, S·sat), yalnızca HSI."""
    _choice(method, "Method", EQ_METHODS)
    saturation = _number(saturation, "Saturation ×", *EQ_SAT_RANGE)
    rgb, alpha = rgb_input(image)
    if method == "hsi":
        hsi, new, out = equalize_hsi(rgb, saturation)
        lut = equalization_lut(c8(hsi[..., 2] * 255))
        clipped = int(np.any((out > 255 * 1.0001) | (out < -255e-4), axis=-1).sum())   # laboratuvar 'out' ölçütü
        med0, med1 = np.median(hsi[..., 2]), np.median(new[..., 2])
        curves = (_identity("H"), TransferCurve("I", None, lut / 255))
        curves += ((TransferCurve("S", (150, 110, 200), _line(lambda x: np.minimum(1, saturation * x))),)
                   if saturation != 1 else (_identity("S"),))
        detail = (f"only I equalized, H kept, S × {saturation:g}; median I {med0:.2f} → {med1:.2f}; {clipped} px out "
                  f"of the RGB gamut (clipped)")
        rows = ("I′ = T(I) = (L − 1)·Σ_{j ≤ I} p_I(j), I8 = c8(255·I)", "(H, S, I) → (H, S, I′) → RGB")
        report = FullColorReport("HSI equalization", detail, curves, rows)
    else:
        u8 = rgb.astype(np.uint8)
        luts = [equalization_lut(u8[..., c]) for c in range(3)]
        out = np.stack([luts[c][u8[..., c]] for c in range(3)], -1).astype(np.float64)
        report = FullColorReport("HSI equalization · RGB comparison",
                                 f"R, G and B equalized separately (lab comparison, not HSI); {_hue_shift_text(rgb, out)}",
                                 tuple(TransferCurve(c, CHANNEL_INK[c], luts[j] / 255) for j, c in enumerate("RGB")),
                                 ("Three different curves T_R, T_G, T_B: hues can shift.",))
    return ProcessResult(_out(out, alpha), report)


# ====================================================================================================
# Paket 4: renkli görüntü yumuşatma ve keskinleştirme — slayt 57–58 (Şekil 6.38–6.39), laboratuvar 'Color Image
# Smoothing & Sharpening'. Komşuluk işlemi her pikseli bir renk vektörü sayar: (a) her RGB kanalı ayrı filtrelenir
# ya da (b) yalnızca HSI'nın I bileşeni filtrelenir, H ve S korunur. Laboratuvar kuralları aynen: kenarlar
# tekrarlanır (replicate), filtre çıktısı ve I·255 Float32Array'de saklanır (float32 yuvarlaması), keskinleştirme
# her zaman 3×3 Laplacian'dır (Kernel seçimi yalnızca yumuşatmaya uygulanır).
# ====================================================================================================
SMOOTH_OPERATIONS = ("smooth", "sharpen")
SMOOTH_KERNELS = ("3", "5")
SMOOTH_PROCESSES = ("rgb", "hsi")
SMOOTH_DISPLAYS = ("result", "difference")
SMOOTH_DIFF_GAIN = 8                                       # laboratuvar: |kanal başına − HSI| × 8
LAPLACE_SHARPEN = (0, -1, 0, -1, 4, -1, 0, -1, 0)          # laboratuvar LAPL: f + (4f − 4 komşu) = f − ∇²f


def _windows(x: np.ndarray, k: int):
    """k×k pencere elemanları satır sırasıyla (laboratuvar filt: dy, sonra dx), kenarlar tekrarlanır."""
    r = k // 2
    p = np.pad(x, r, mode="edge")
    h, w = x.shape
    for dy in range(k):
        for dx in range(k):
            yield p[dy:dy + h, dx:dx + w]


def lab_box(x: np.ndarray, k: int) -> np.ndarray:
    """Laboratuvar boxFn: pencere toplamı (soldan sağa, çift duyarlık) / k², sonuç float32 saklanır."""
    acc = np.zeros(x.shape, np.float64)
    for v in _windows(x.astype(np.float64), k):
        acc = acc + v
    return (acc / (k * k)).astype(np.float32)


def lab_laplace(x: np.ndarray) -> np.ndarray:
    """Laboratuvar lapFn: Σ LAPL[i]·w_i (sırayla), = 4·merkez − 4 komşu = −∇²f; float32 saklanır."""
    acc = np.zeros(x.shape, np.float64)
    for coef, v in zip(LAPLACE_SHARPEN, _windows(x.astype(np.float64), 3)):
        acc = acc + coef * v
    return acc.astype(np.float32)


def smooth_sharpen_rgb(rgb: np.ndarray, operation: str, k: int, process: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Laboratuvar process(): R, G, B ∈ [0, 255] → sonuç R, G, B uint8 (RGB sırası) ve HSI yolunda (H, S, I, I′)."""
    if process == "rgb":
        chans = []
        for c in range(3):
            g = rgb[..., c]
            if operation == "smooth":
                chans.append(lab_box(g, k).astype(np.float64))                    # c̄ = (1/K) Σ c(s, t)
            else:
                chans.append(g + lab_laplace(g).astype(np.float64))               # c + (4c − 4 komşu)
        return c8(np.stack(chans, -1)), None
    hsi = rgb_to_model(rgb, "hsi")
    ig = (hsi[..., 2] * 255).astype(np.float32)                                    # laboratuvar Ig (Float32Array)
    if operation == "smooth":
        new_i = lab_box(ig, k).astype(np.float64) / 255
    else:
        new_i = (ig.astype(np.float64) + lab_laplace(ig).astype(np.float64)) / 255
    new_i = np.maximum(0, np.minimum(1, new_i))                                    # I′ ∈ [0, 1]
    out = hsi.copy()
    out[..., 2] = new_i                                                             # H, S aynen
    return c8(model_to_rgb(out, "hsi")), np.stack([hsi[..., 0], hsi[..., 1], hsi[..., 2], new_i], -1)


def smoothing_difference(rgb_result: np.ndarray, hsi_result: np.ndarray) -> np.ndarray:
    """Laboratuvar D: c8(8 · max_c |RGB_c − HSI_c|) — tek kanallı fark görüntüsü."""
    d = np.abs(rgb_result.astype(np.int16) - hsi_result.astype(np.int16)).max(axis=-1)
    return c8(d * SMOOTH_DIFF_GAIN)


def color_smoothing_sharpening(image: np.ndarray, operation: str = "smooth", kernel: str = "3", process: str = "rgb",
                               display: str = "result") -> ProcessResult:
    """Color image smoothing & sharpening (slayt 57–58; laboratuvar). SMOOTHING: k×k ortalama,
    c̄(x, y) = [(1/K) Σ R, (1/K) Σ G, (1/K) Σ B] (her RGB kanalı) ya da yalnızca I. SHARPENING: 3×3 Laplacian,
    g = f − ∇²f = f + (4f − 4 komşu) her RGB kanalına ya da yalnızca I'ya; I′ [0, 1]'e kırpılır. DIFFERENCE:
    iki yöntemin farkı c8(8 · max_c |RGB_c − HSI_c|) (laboratuvar), gri olarak gösterilir."""
    _choice(operation, "Operation", SMOOTH_OPERATIONS)
    k = int(_choice(kernel, "Kernel", SMOOTH_KERNELS)) if operation == "smooth" else 3
    if operation == "sharpen":
        _choice(kernel, "Kernel", SMOOTH_KERNELS)                                   # geçerli olmalı ama kullanılmaz
    _choice(process, "Process", SMOOTH_PROCESSES)
    _choice(display, "Display", SMOOTH_DISPLAYS)
    rgb, alpha = rgb_input(image)
    by_rgb, _ = smooth_sharpen_rgb(rgb, operation, k, "rgb")
    by_hsi, hsi = smooth_sharpen_rgb(rgb, operation, k, "hsi")
    diff = smoothing_difference(by_rgb, by_hsi)
    raw = np.abs(by_rgb.astype(np.int16) - by_hsi.astype(np.int16)).max(axis=-1)
    chosen = by_rgb if process == "rgb" else by_hsi
    shown = np.repeat(diff[..., None], 3, -1) if display == "difference" else chosen
    name = f"{k}×{k} average" if operation == "smooth" else "3×3 Laplacian sharpening"
    how = "each RGB channel" if process == "rgb" else "only I of HSI (H, S kept)"
    detail = (f"{name}, {how}; |RGB − HSI|: max {int(raw.max(initial=0))}, mean {raw.mean():.2f}, "
              f"{int((raw > 0).sum())} of {raw.size} px differ")
    if display == "difference":
        detail = f"difference × {SMOOTH_DIFF_GAIN} = c8(8 · max_c |RGB_c − HSI_c|) of the two methods; " + detail
    kern = (f"kernel (1/{k * k})·ones({k}×{k}), K = {k * k}" if operation == "smooth"
            else "kernel [0 −1 0; −1 4 −1; 0 −1 0]: g = f + (4f − 4 neighbours) = f − ∇²f")
    rows = [kern, "border: replicate (lab); smoothing kernel 3×3 / 5×5, sharpening always 3×3",
            "c̄ = [(1/K)ΣR, (1/K)ΣG, (1/K)ΣB]" if operation == "smooth" else "∇²c = [∇²R, ∇²G, ∇²B]"]
    if raw.size:                                                                    # örnek piksel: en büyük fark
        y, x = np.unravel_index(int(np.argmax(raw)), raw.shape)
        p, a, b = (tuple(int(v) for v in im[y, x]) for im in (rgb, by_rgb, by_hsi))
        H, S, I0, I1 = hsi[y, x]
        rows += [f"pixel ({x}, {y}) RGB {p}: each channel → {a}",
                 f"  HSI: H {H:.1f}°, S {S:.3f} kept, I {I0:.3f} → {I1:.3f} → {b}"]
    report = FullColorReport(f"Color {'smoothing' if operation == 'smooth' else 'sharpening'} · "
                             f"{'RGB' if process == 'rgb' else 'HSI I'}", detail, (), tuple(rows),
                             section="SMOOTHING & SHARPENING")
    return ProcessResult(merge_alpha(np.ascontiguousarray(shown[..., ::-1]), alpha), report)


# ====================================================================================================
# Paket 5: renkli bölütleme — slayt 59–63, laboratuvar 'Hue Thresholding in HSV', 'k-Means Clustering in Lab',
# 'Skin Regions in YCbCr', 'Color Edge Detection'. RGB'de bölütleme zordur (renk ve parlaklık karışık); HSV'de ton
# eşikleme, Lab'da k-means, YCbCr'de ten bölgeleri (slayt 59). Laboratuvar kuralları aynen: OpenCV HSV ölçeği
# (H 0–179, S, V 0–255), Lab ve gradyan ara değerleri Float32Array'de saklanır, Sobel kenarları tekrarlanır.
# ====================================================================================================
HUE_TARGETS = {"blue": (90, 130, 50, 50), "green": (35, 85, 60, 40), "yellow": (20, 34, 80, 80),
               "red": (170, 8, 90, 60)}                 # laboratuvar HUE_PRE: (H min, H max, S min, V min)
HUE_CHOICES = tuple(HUE_TARGETS) + ("custom",)
SEG_DISPLAYS = ("segmented", "mask")
KMEANS_FEATURES = ("lab", "ab", "rgb")
KMEANS_K_RANGE = (2, 6)
KMEANS_SEED_RANGE = (1, 9999)
KMEANS_MAX_IT = 30                                      # laboratuvar kmeans(…, maxIt = 30)
SKIN_RULE = (133, 173, 77, 127)                         # Cr min, Cr max, Cb min, Cb max (Chai & Ngan)
EDGE_METHODS = ("gray", "v", "rgb", "dizenzo")
EDGE_NAMES = {"gray": "Grayscale", "v": "HSV V channel", "rgb": "RGB channels combined (max)",
              "dizenzo": "Vector gradient (Di Zenzo)"}
EDGE_T_RANGE = (10, 600)
EDGE_DISPLAYS = ("edges", "magnitude")
SOBEL_X = (-1, 0, 1, -2, 0, 2, -1, 0, 1)
SOBEL_Y = (-1, -2, -1, 0, 0, 0, 1, 2, 1)


def _segmented(rgb: np.ndarray, mask: np.ndarray, display: str, alpha) -> np.ndarray:
    """Laboratuvar: maske 0/255 (gri) ya da bölütlenmiş görüntü (maske dışı siyah); çıktı BGR."""
    out = np.repeat(np.where(mask, 255, 0).astype(np.uint8)[..., None], 3, -1) if display == "mask" \
        else np.where(mask[..., None], c8(rgb), 0).astype(np.uint8)
    return merge_alpha(np.ascontiguousarray(out[..., ::-1]), alpha)


def _px_text(rgb, y, x) -> str:
    return f"pixel ({x}, {y}) RGB {tuple(int(v) for v in rgb[y, x])}"


# ------------------------------------------------------------------ slayt 59–60: hue thresholding in HSV
def hsv_cv(rgb: np.ndarray) -> np.ndarray:
    """Laboratuvar rgb2hsvCV: H = round(H° / 2) mod 180, S = c8(255·S), V = c8(255·V) (OpenCV ölçeği)."""
    v = rgb_to_model(rgb, "hsv")
    return np.stack([js_round(v[..., 0] / 2) % 180, c8(v[..., 1] * 255), c8(v[..., 2] * 255)], -1).astype(np.int64)


def hue_in_range(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """lo ≤ hi: lo ≤ H ≤ hi; lo > hi (kırmızı, 0'dan sarar): H ≥ lo veya H ≤ hi — iki aralık."""
    return (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)


def hue_thresholding(image: np.ndarray, target: str = "blue", h_min: int = 90, h_max: int = 130, s_min: int = 50,
                     v_min: int = 50, display: str = "segmented") -> ProcessResult:
    """Hue thresholding in HSV (slayt 59–60; laboratuvar; cv2.inRange): mask = 255 eğer H_lo ≤ H ≤ H_hi, S ≥ S_min,
    V ≥ V_min. OpenCV ölçeği H 0–179. H_lo > H_hi ise aralık 0'dan sarar (kırmızı): iki aralık [H_lo, 179] ∪ [0, H_hi].
    Hedef seçimi laboratuvar hazır ayarlarını kullanır; CUSTOM aşağıdaki eşikleri kullanır."""
    _choice(target, "Target", HUE_CHOICES)
    vals = (_number(h_min, "H min", 0, 179, integer=True), _number(h_max, "H max", 0, 179, integer=True),
            _number(s_min, "S min", 0, 255, integer=True), _number(v_min, "V min", 0, 255, integer=True))
    lo, hi, smin, vmin = HUE_TARGETS.get(target, vals)
    _choice(display, "Display", SEG_DISPLAYS)
    rgb, alpha = rgb_input(image)
    hsv = hsv_cv(rgb)
    h_ok, s_ok, v_ok = hue_in_range(hsv[..., 0], lo, hi), hsv[..., 1] >= smin, hsv[..., 2] >= vmin
    mask = h_ok & s_ok & v_ok
    n, kept = mask.size, int(mask.sum())
    rng = f"H ∈ [{lo}, {hi}]" if lo <= hi else f"H ∈ [{lo}, 179] ∪ [0, {hi}] (wraps around 0)"
    rows = [f"mask = 255 if {rng}, S ≥ {smin}, V ≥ {vmin}  (OpenCV H 0–179, S, V 0–255)",
            f"hue matches but S or V too low: {int((h_ok & ~(s_ok & v_ok)).sum())} px (unreliable hue)"]
    if n:
        cand = np.argwhere(h_ok & ~s_ok)                         # laboratuvar notu: ton uyuyor, doygunluk düşük
        y, x = (cand[0] if len(cand) else np.argwhere(mask)[0] if kept else (rgb.shape[0] // 2, rgb.shape[1] // 2))
        H, S, V = (int(v) for v in hsv[y, x])
        ok = lambda b: "✓" if b else "✗"
        rows += [_px_text(rgb, y, x) + f": H {H} {ok(h_ok[y, x])}, S {S} ≥ {smin} {ok(s_ok[y, x])}, "
                                     f"V {V} ≥ {vmin} {ok(v_ok[y, x])} → mask {255 if mask[y, x] else 0}"]
    name = "custom" if target == "custom" else f"{target} (lab preset)"
    report = FullColorReport(f"Hue thresholding · {name}", f"{rng}, S ≥ {smin}, V ≥ {vmin}: {kept} of {n} px "
                             f"({100 * kept / max(1, n):.1f} %) in the mask", (), tuple(rows), section="COLOR SEGMENTATION")
    return ProcessResult(_segmented(rgb, mask, display, alpha), report)


# ------------------------------------------------------------------ slayt 61: k-means in Lab
def kmeans_features(rgb: np.ndarray, space: str) -> np.ndarray:
    """Laboratuvar feats: piksel başına (L*, a*, b*), (a*, b*) ya da (R, G, B); Float32Array saklaması."""
    if space == "rgb":
        x = rgb.reshape(-1, 3)
    else:
        lab = rgb_to_model(rgb, "lab").reshape(-1, 3)
        x = lab[:, 1:] if space == "ab" else lab
    return x.astype(np.float32).astype(np.float64)


def lab_kmeans(X: np.ndarray, K: int, seed: int, max_it: int = KMEANS_MAX_IT) -> dict:
    """Laboratuvar kmeans: LCG (s·1664525 + 1013904223 mod 2³²) ile k-means++ başlangıcı; atama c(z) = argmin_k
    ‖z − μ_k‖², güncelleme μ_k = ortalama; değişen piksel yoksa ya da merkezler kımıldamıyorsa (|Δ| ≤ 1e−6) dur.
    Boş kümenin merkezi yerinde kalır."""
    n, dim = X.shape
    s = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 4294967296

    def dist2(c):
        d = np.zeros(n)
        for j in range(dim):                                   # laboratuvar sırası: Σ_j (z_j − c_j)²
            t = X[:, j] - c[j]
            d = d + t * t
        return d

    first = int(rnd() * n)
    init_idx, centers = [first], [X[first].copy()]
    while len(centers) < K:
        d = dist2(centers[0])
        for c in centers[1:]:
            d = np.minimum(d, dist2(c))
        cum = np.cumsum(d)                                     # laboratuvar: tot sıralı toplam, r −= d[i] ≤ 0
        r = rnd() * cum[-1]
        hit = np.nonzero(r - cum <= 0)[0]
        pick = int(hit[0]) if len(hit) else 0
        init_idx.append(pick)
        centers.append(X[pick].copy())
    cur = np.array(centers)
    labels = np.full(n, -1)
    its = []
    for _ in range(max_it):
        dists = np.stack([dist2(c) for c in cur], 0)
        new = np.argmin(dists, axis=0)                          # ilk en küçük (laboratuvar: d < bd)
        changed = int((new != labels).sum())
        cnt = np.bincount(new, minlength=K)
        sums = np.stack([np.bincount(new, weights=X[:, j], minlength=K) for j in range(dim)], -1)
        nxt = np.where(cnt[:, None] > 0, sums / np.maximum(cnt, 1)[:, None], cur)
        its.append({"changed": changed, "counts": cnt})
        moved = bool(np.any(np.abs(nxt - cur) > 1e-6))
        labels, cur = new, nxt
        if not changed or not moved:
            break
    return {"init_idx": init_idx, "labels": labels, "centers": cur, "its": its}


def kmeans_lab(image: np.ndarray, k: int = 4, features: str = "lab", seed: int = 3) -> ProcessResult:
    """k-Means clustering in Lab (slayt 61; laboratuvar): pikseller renge göre K kümeye ayrılır — atama
    c(z) = argmin_k ‖z − μ_k‖², güncelleme μ_k = (1/|C_k|) Σ z, atamalar değişmeyince yakınsar. Lab'da Öklid uzaklığı
    algılanan renk farkına uyar. Her küme kendi piksellerinin ortalama RGB rengiyle boyanır (slayt 61)."""
    k = _number(k, "K", *KMEANS_K_RANGE, integer=True)
    _choice(features, "Features", KMEANS_FEATURES)
    seed = _number(seed, "Seed", *KMEANS_SEED_RANGE, integer=True)
    rgb, alpha = rgb_input(image)
    X = kmeans_features(rgb, features)
    n = X.shape[0]
    if n < k:
        raise AlgorithmParameterError(f"K = {k} needs at least {k} pixels (image has {n}).")
    km = lab_kmeans(X, k, seed)
    labels = km["labels"]
    flat = rgb.reshape(-1, 3)
    cnt = np.bincount(labels, minlength=k)
    mean = np.stack([np.bincount(labels, weights=flat[:, c], minlength=k) for c in range(3)], -1) / np.maximum(cnt, 1)[:, None]
    mean = np.where(cnt[:, None] > 0, mean, 128.0)             # laboratuvar meanRGB: boş küme gri
    out = c8(mean[labels]).reshape(rgb.shape)
    it = len(km["its"])
    conv = it < KMEANS_MAX_IT or km["its"][-1]["changed"] == 0          # 30 yinelemeden önce durduysa yakınsadı
    feat = {"lab": "(L*, a*, b*)", "ab": "(a*, b*) — color only, lightness ignored", "rgb": "(R, G, B)"}[features]
    rows = ["assignment c(z) = argmin_k ‖z − μ_k‖²,  update μ_k = (1/|C_k|) Σ z",
            f"features {feat}; k-means++ start (seed {seed}), initial pixels {km['init_idx']}",
            f"{'converged' if conv else 'stopped (max 30 iterations)'} after {it} iteration(s); last step: "
            f"{km['its'][-1]['changed']} px changed cluster"]
    for j in range(k):
        mu = ", ".join(f"{v:.1f}" for v in km["centers"][j])
        col = tuple(int(v) for v in c8(mean[j]))
        rows.append(f"cluster {j + 1}: {int(cnt[j])} px, μ = ({mu}), mean RGB {col}")
    i0 = min(n - 1, (rgb.shape[0] // 2) * rgb.shape[1] + rgb.shape[1] // 2)   # örnek piksel: görüntü ortası
    d = np.sqrt([((X[i0] - c) ** 2).sum() for c in km["centers"]])
    y0, x0 = divmod(i0, rgb.shape[1])
    rows.append(_px_text(rgb, y0, x0) + ": ‖z − μ‖ = " + ", ".join(f"{v:.1f}" for v in d) + f" → cluster {int(labels[i0]) + 1}")
    report = FullColorReport(f"k-Means · {features.upper() if features != 'ab' else 'a*b*'}, K = {k}",
                             f"{k} clusters in {feat.split(' —')[0]}, {it} iteration(s), sizes "
                             + ", ".join(str(int(c)) for c in cnt), (), tuple(rows), section="COLOR SEGMENTATION")
    return ProcessResult(merge_alpha(np.ascontiguousarray(out[..., ::-1]), alpha), report)


# ------------------------------------------------------------------ slayt 62: skin regions in YCbCr
def skin_regions(image: np.ndarray, cr_min: int = 133, cr_max: int = 173, cb_min: int = 77, cb_max: int = 127,
                 display: str = "segmented") -> ProcessResult:
    """Skin regions in YCbCr (slayt 62; laboratuvar; Chai & Ngan): ten ⇔ 133 ≤ Cr ≤ 173 ve 77 ≤ Cb ≤ 127. Y yok
    sayılır, bu yüzden çok farklı parlaklıktaki tenler aynı küçük Cb–Cr kutusuna düşer. Cr = (R − Y)·0.713 + 128,
    Cb = (B − Y)·0.564 + 128, Y = 0.299R + 0.587G + 0.114B (OpenCV YCrCb)."""
    lims = [_number(v, n, 0, 255, integer=True) for v, n in ((cr_min, "Cr min"), (cr_max, "Cr max"),
                                                             (cb_min, "Cb min"), (cb_max, "Cb max"))]
    _choice(display, "Display", SEG_DISPLAYS)
    rgb, alpha = rgb_input(image)
    ycc = rgb_to_model(rgb, "ycbcr")                            # (Y, Cb, Cr) — Y kullanılmaz
    cb, cr = ycc[..., 1], ycc[..., 2]
    mask = (cr >= lims[0]) & (cr <= lims[1]) & (cb >= lims[2]) & (cb <= lims[3])
    n, kept = mask.size, int(mask.sum())
    rows = [f"skin ⇔ {lims[0]} ≤ Cr ≤ {lims[1]} and {lims[2]} ≤ Cb ≤ {lims[3]}  (Y not used)",
            "Cr = (R − Y)·0.713 + 128,  Cb = (B − Y)·0.564 + 128"]
    if n:
        y, x = np.argwhere(mask)[0] if kept else (rgb.shape[0] // 2, rgb.shape[1] // 2)
        Y, Cb, Cr = ycc[y, x]
        ok = lambda b: "✓" if b else "✗"
        rows.append(_px_text(rgb, y, x) + f": Y {Y:.1f} (not used), Cr {Cr:.1f} {ok(lims[0] <= Cr <= lims[1])}, "
                                         f"Cb {Cb:.1f} {ok(lims[2] <= Cb <= lims[3])} → {'skin' if mask[y, x] else 'not skin'}")
    report = FullColorReport("Skin regions · YCbCr", f"{lims[0]} ≤ Cr ≤ {lims[1]}, {lims[2]} ≤ Cb ≤ {lims[3]}: "
                             f"{kept} of {n} px ({100 * kept / max(1, n):.1f} %) skin", (), tuple(rows),
                             section="COLOR SEGMENTATION")
    return ProcessResult(_segmented(rgb, mask, display, alpha), report)


# ------------------------------------------------------------------ slayt 63: color edge detection
def lab_sobel(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Laboratuvar sobel(windowAt(·, 3)): Σ SX·w, Σ SY·w (satır sırası, kenarlar tekrarlanır), float32 saklanır."""
    gx = np.zeros(g.shape)
    gy = np.zeros(g.shape)
    for cx, cy, v in zip(SOBEL_X, SOBEL_Y, _windows(g.astype(np.float64), 3)):
        gx = gx + cx * v
        gy = gy + cy * v
    return gx.astype(np.float32).astype(np.float64), gy.astype(np.float32).astype(np.float64)


def edge_gradient(rgb: np.ndarray, method: str) -> tuple[np.ndarray, list, dict]:
    """Laboratuvar edgeData: kanal(lar), Sobel gradyanları ve |∇| (float32). GRAY: Y = c8(luma); V: max(R, G, B);
    RGB: max_c √(gx_c² + gy_c²); DI ZENZO: g_xx = Σ gx², g_yy = Σ gy², g_xy = Σ gx·gy, θ = ½ atan2(2g_xy, g_xx − g_yy),
    F(θ) = √(½[(g_xx + g_yy) + (g_xx − g_yy) cos 2θ + 2 g_xy sin 2θ])."""
    if method == "gray":
        chans = [c8(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)]
    elif method == "v":
        chans = [rgb.max(axis=-1)]
    else:
        chans = [rgb[..., c] for c in range(3)]
    grads = [lab_sobel(c) for c in chans]
    extra = {}
    if method == "dizenzo":
        gxx = sum(gx * gx for gx, _ in grads)
        gyy = sum(gy * gy for _, gy in grads)
        gxy = sum(gx * gy for gx, gy in grads)
        th = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
        mag = np.sqrt(np.maximum(0, 0.5 * ((gxx + gyy) + (gxx - gyy) * np.cos(2 * th) + 2 * gxy * np.sin(2 * th))))
        extra = {"gxx": gxx, "gyy": gyy, "gxy": gxy, "th": th}
    elif method == "rgb":
        mag = np.max([np.hypot(gx, gy) for gx, gy in grads], axis=0)
    else:
        mag = np.hypot(*grads[0])
    return mag.astype(np.float32).astype(np.float64), grads, extra


def color_edge_detection(image: np.ndarray, method: str = "gray", t: int = 120, display: str = "edges") -> ProcessResult:
    """Color edge detection (slayt 63; laboratuvar): 3×3 Sobel gradyanı gri tonlamada, HSV V'de, her RGB kanalında
    (maks ile birleşik) ya da renk vektöründe (Di Zenzo). Eşit parlaklıktaki renkler arasındaki kenar gri tonlamada
    kaybolur, renkli yöntemlerle bulunur. EDGES: |∇| > T → 255; MAGNITUDE: c8(|∇| / 4)."""
    _choice(method, "Gradient of", EDGE_METHODS)
    t = _number(t, "T", *EDGE_T_RANGE, integer=True)
    _choice(display, "Display", EDGE_DISPLAYS)
    rgb, alpha = rgb_input(image)
    mag, grads, extra = edge_gradient(rgb, method)
    edges = mag > t
    shown = np.where(edges, 255, 0).astype(np.uint8) if display == "edges" else c8(mag / 4)
    n, ne = edges.size, int(edges.sum())
    form = {"gray": "|∇Y| = √(gx² + gy²), Y = 0.299R + 0.587G + 0.114B", "v": "|∇V| = √(gx² + gy²), V = max(R, G, B)",
            "rgb": "|∇| = max(|∇R|, |∇G|, |∇B|)",
            "dizenzo": "F(θ) = √(½[(gxx + gyy) + (gxx − gyy) cos 2θ + 2gxy sin 2θ]), θ = ½ atan2(2gxy, gxx − gyy)"}[method]
    rows = [form, "Sobel 3×3: gx = [−1 0 1; −2 0 2; −1 0 1], gy = transpose; border replicate (lab)",
            f"edge ⇔ |∇| > T = {t}; magnitude shown as c8(|∇| / 4)"]
    if n:
        y, x = np.unravel_index(int(np.argmax(mag)), mag.shape)
        names = ["Y"] if method == "gray" else ["V"] if method == "v" else ["R", "G", "B"]
        parts = ", ".join(f"{nm}: gx {gx[y, x]:.0f}, gy {gy[y, x]:.0f}" for nm, (gx, gy) in zip(names, grads))
        rows.append(_px_text(rgb, y, x) + f" (strongest): {parts}")
        if method == "dizenzo":
            rows.append(f"  gxx {extra['gxx'][y, x]:.0f}, gyy {extra['gyy'][y, x]:.0f}, gxy {extra['gxy'][y, x]:.0f}, "
                        f"θ {np.degrees(extra['th'][y, x]):.1f}° → F = {mag[y, x]:.0f}")
        else:
            rows.append(f"  |∇| = {mag[y, x]:.0f} {'>' if edges[y, x] else '≤'} T = {t}")
    report = FullColorReport(f"Color edges · {EDGE_NAMES[method]}", f"{ne} of {n} px with |∇| > {t}, max |∇| = "
                             f"{mag.max(initial=0):.0f}", (), tuple(rows), section="COLOR SEGMENTATION")
    return ProcessResult(merge_alpha(np.ascontiguousarray(np.repeat(shown[..., None], 3, -1)), alpha), report)
