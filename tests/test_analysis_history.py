"""Görüntü ölçümleri ve geri alma geçmişi testleri (GUI yok)."""

import numpy as np
import pytest

from app.history import ImageHistory
from processing.analysis import channel_stats, describe, histograms, pixel_value


def test_histograms_gray_and_color():
    gray = np.array([[0, 0, 255], [10, 10, 10]], dtype=np.uint8)
    h = histograms(gray)
    assert list(h) == ["L"]
    assert h["L"][0] == 2 and h["L"][10] == 3 and h["L"][255] == 1
    assert h["L"].sum() == gray.size

    bgra = np.zeros((4, 5, 4), dtype=np.uint8)
    hc = histograms(bgra)
    assert list(hc) == ["B", "G", "R"]  # alfa histogramı çizilmez
    assert all(v.sum() == 20 for v in hc.values())


def test_histogram_rebinning_preserves_total():
    img = np.random.default_rng(0).integers(0, 256, (30, 30), dtype=np.uint8)
    h = histograms(img, bins=16)["L"]
    assert len(h) == 16 and h.sum() == img.size
    with pytest.raises(ValueError):
        histograms(img.astype(np.float32))


def test_describe_and_stats():
    img = np.zeros((30, 50, 3), dtype=np.uint8)
    img[..., 2] = 200
    info = describe(img)
    assert (info.width, info.height, info.channels, info.color_model) == (50, 30, 3, "BGR")
    assert info.nbytes == 30 * 50 * 3
    stats = {s.name: s for s in channel_stats(img)}
    assert stats["R"].mean == 200 and stats["B"].maximum == 0


def test_pixel_value_bounds():
    img = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert pixel_value(img, 2, 1) == (5,)
    assert pixel_value(img, 3, 0) is None
    assert pixel_value(img, -1, 0) is None


def test_history_push_undo_limit_and_copy():
    hist = ImageHistory(limit=2)
    assert not hist.can_undo() and hist.undo() is None
    a = np.zeros((2, 2), np.uint8)
    hist.push(a)
    a[:] = 9  # geçmiş kopya tutmalı
    hist.push(np.full((2, 2), 1, np.uint8))
    hist.push(np.full((2, 2), 2, np.uint8))  # limit aşıldı → ilk kayıt düşer
    assert len(hist) == 2
    assert hist.undo()[0, 0] == 2
    assert hist.undo()[0, 0] == 1
    assert hist.undo() is None
    with pytest.raises(ValueError):
        ImageHistory(limit=0)
