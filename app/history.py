"""Geri alma (undo) altyapısı: sınırlı uzunlukta görüntü geçmişi."""

from __future__ import annotations

from collections import deque

import numpy as np


class ImageHistory:
    """İşlem öncesi görüntülerin kopyalarını saklayan yığın.

    Bellek kontrolü için en fazla ``limit`` adım tutulur; en eski adım atılır.
    """

    def __init__(self, limit: int = 20) -> None:
        if limit < 1:
            raise ValueError("limit en az 1 olmalı")
        self._stack: deque[np.ndarray] = deque(maxlen=limit)

    @property
    def limit(self) -> int:
        return self._stack.maxlen or 0

    def push(self, image: np.ndarray) -> None:
        self._stack.append(image.copy())

    def undo(self) -> np.ndarray | None:
        """Son kaydedilen görüntüyü döndürür (yoksa None)."""
        return self._stack.pop() if self._stack else None

    def can_undo(self) -> bool:
        return bool(self._stack)

    def clear(self) -> None:
        self._stack.clear()

    def __len__(self) -> int:
        return len(self._stack)
