"""Görüntü ölçümleri: histogram, kanal istatistikleri, teknik özet.

GUI'den bağımsızdır; arayüz yalnızca buradan dönen sayıları çizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    channels: int
    dtype: str
    color_model: str
    nbytes: int


@dataclass(frozen=True)
class ChannelStats:
    name: str
    mean: float
    std: float
    minimum: int
    maximum: int


def channel_names(image: np.ndarray) -> tuple[str, ...]:
    """OpenCV düzenine göre kanal adları (gri: L, renkli: B, G, R [, A])."""
    if image.ndim == 2:
        return ("L",)
    n = image.shape[2]
    if n == 1:
        return ("L",)
    if n == 3:
        return ("B", "G", "R")
    if n == 4:
        return ("B", "G", "R", "A")
    return tuple(f"C{i}" for i in range(n))


def describe(image: np.ndarray) -> ImageInfo:
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    model = {1: "GRAY", 3: "BGR", 4: "BGRA"}.get(channels, f"{channels}-CH")
    return ImageInfo(w, h, channels, str(image.dtype), model, int(image.nbytes))


def _planes(image: np.ndarray, include_alpha: bool = False) -> list[tuple[str, np.ndarray]]:
    names = channel_names(image)
    if image.ndim == 2:
        return [(names[0], image)]
    planes = [(name, image[:, :, i]) for i, name in enumerate(names)]
    if not include_alpha:
        planes = [p for p in planes if p[0] != "A"]
    return planes


def histograms(image: np.ndarray, bins: int = 256) -> dict[str, np.ndarray]:
    """Her renk kanalının (alfa hariç) histogramını döndürür.

    uint8 görüntülerde 256 kutu, değer aralığı [0, 255] varsayılır.
    """
    if image.dtype != np.uint8:
        raise ValueError("Histogram yalnızca uint8 görüntüler için hesaplanır")
    if not 1 <= bins <= 256:
        raise ValueError("bins 1..256 aralığında olmalı")
    result: dict[str, np.ndarray] = {}
    for name, plane in _planes(image):
        counts = np.bincount(plane.ravel(), minlength=256)
        if bins != 256:
            counts = np.add.reduceat(counts, np.linspace(0, 256, bins, endpoint=False).astype(int))
        result[name] = counts.astype(np.int64)
    return result


def channel_stats(image: np.ndarray) -> list[ChannelStats]:
    stats = []
    for name, plane in _planes(image):
        stats.append(
            ChannelStats(
                name=name,
                mean=float(plane.mean()),
                std=float(plane.std()),
                minimum=int(plane.min()),
                maximum=int(plane.max()),
            )
        )
    return stats


def pixel_value(image: np.ndarray, x: int, y: int) -> tuple[int, ...] | None:
    """(x, y) konumundaki piksel değeri; görüntü dışındaysa None."""
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    value = image[y, x]
    return (int(value),) if np.ndim(value) == 0 else tuple(int(v) for v in value)
