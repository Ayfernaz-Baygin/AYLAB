"""Algoritma kayıt sistemi (GUI'den bağımsız).

Her algoritma; kimliği, kategorisi, açıklaması ve *parametre tanımları* ile
kaydedilir. Arayüz, parametre panelini bu tanımlardan dinamik olarak üretir;
böylece yeni bir algoritma eklemek için GUI koduna dokunmak gerekmez.

Bir algoritmanın asıl işlevi (``func``) daha sonra ``AlgorithmRegistry.bind``
ile bağlanır. İşlevi bağlanmamış algoritmalar arayüzde görünür, parametreleri
düzenlenebilir ama ``run`` çağrıldığında ``AlgorithmNotImplemented`` fırlatır.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Union

import numpy as np

# --------------------------------------------------------------------------
# Parametre tanımları
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IntParam:
    key: str
    label: str
    default: int
    minimum: int
    maximum: int
    step: int = 1
    odd_only: bool = False  # Çekirdek boyutları gibi yalnızca tek sayı alan değerler
    suffix: str = ""
    help: str = ""

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(f"{self.key}: minimum > maximum")
        if self.odd_only and (self.minimum % 2 == 0 or self.default % 2 == 0):
            raise ValueError(f"{self.key}: odd_only parametrede minimum/default tek olmalı")

    @property
    def effective_step(self) -> int:
        return 2 if self.odd_only else max(1, self.step)

    def validate(self, value: Any) -> int:
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            return self.default
        v = max(self.minimum, min(self.maximum, v))
        if self.odd_only and v % 2 == 0:
            v = v + 1 if v + 1 <= self.maximum else v - 1
        return v


@dataclass(frozen=True)
class FloatParam:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float = 0.1
    decimals: int = 2
    suffix: str = ""
    help: str = ""

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(f"{self.key}: minimum > maximum")
        if self.step <= 0:
            raise ValueError(f"{self.key}: step pozitif olmalı")

    def validate(self, value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return self.default
        if not np.isfinite(v):
            return self.default
        return round(max(self.minimum, min(self.maximum, v)), self.decimals)


@dataclass(frozen=True)
class ChoiceParam:
    key: str
    label: str
    options: tuple[tuple[str, str], ...]  # (değer, görünen etiket)
    default: str
    help: str = ""

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"{self.key}: en az bir seçenek gerekli")
        if self.default not in self.values:
            raise ValueError(f"{self.key}: default seçeneklerde yok")

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(v for v, _ in self.options)

    def validate(self, value: Any) -> str:
        return value if value in self.values else self.default


@dataclass(frozen=True)
class BoolParam:
    key: str
    label: str
    default: bool = False
    help: str = ""

    def validate(self, value: Any) -> bool:
        return self.default if value is None else bool(value)


ParamSpec = Union[IntParam, FloatParam, ChoiceParam, BoolParam]


# --------------------------------------------------------------------------
# Kategori ve algoritma
# --------------------------------------------------------------------------


class AlgorithmNotImplemented(NotImplementedError):
    """Tanımı olan ama işlevi henüz bağlanmamış algoritma çalıştırıldığında."""


class AlgorithmParameterError(ValueError):
    """Parametreler (veya parametre–görüntü birleşimi) geçersiz olduğunda.

    Tek tek sınır kontrolleri ``ParamSpec.validate`` ile yapılır; bu hata,
    örneğin "alt sınır ≥ üst sınır" gibi parametreler arası tutarsızlıklar ve
    desteklenmeyen görüntü biçimleri için kullanıcıya gösterilecek mesajı taşır.
    """


@dataclass(frozen=True)
class Category:
    key: str
    index: int
    title: str
    summary: str
    wavelength: int  # nm — arayüzdeki vurgu rengi bu dalga boyundan gelir


AlgorithmFunc = Callable[..., np.ndarray]


@dataclass(frozen=True)
class Algorithm:
    key: str
    title: str
    category: str
    summary: str
    params: tuple[ParamSpec, ...] = ()
    func: AlgorithmFunc | None = field(default=None, compare=False)

    @property
    def implemented(self) -> bool:
        return self.func is not None

    def default_params(self) -> dict[str, Any]:
        return {p.key: p.default for p in self.params}

    def resolve_params(self, values: dict[str, Any] | None = None) -> dict[str, Any]:
        """Eksikleri varsayılanla doldurur, değerleri doğrular, bilinmeyenleri atar."""
        values = values or {}
        return {p.key: p.validate(values.get(p.key, p.default)) for p in self.params}

    def run(self, image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
        if self.func is None:
            raise AlgorithmNotImplemented(f"'{self.title}' henüz uygulanmadı.")
        return self.func(image, **self.resolve_params(params))


class AlgorithmRegistry:
    """Kategorileri ve algoritmaları sıralı biçimde tutar."""

    def __init__(self) -> None:
        self._categories: dict[str, Category] = {}
        self._algorithms: dict[str, Algorithm] = {}

    # --- kayıt ---------------------------------------------------------
    def add_category(self, category: Category) -> Category:
        if category.key in self._categories:
            raise ValueError(f"Kategori zaten kayıtlı: {category.key}")
        self._categories[category.key] = category
        return category

    def add(self, algorithm: Algorithm) -> Algorithm:
        if algorithm.category not in self._categories:
            raise KeyError(f"Bilinmeyen kategori: {algorithm.category}")
        if algorithm.key in self._algorithms:
            raise ValueError(f"Algoritma zaten kayıtlı: {algorithm.key}")
        keys = [p.key for p in algorithm.params]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{algorithm.key}: tekrarlanan parametre anahtarı")
        self._algorithms[algorithm.key] = algorithm
        return algorithm

    def bind(self, key: str, func: AlgorithmFunc) -> Algorithm:
        """Kayıtlı bir algoritmaya çalıştırılabilir işlevini bağlar."""
        bound = replace(self.get(key), func=func)
        self._algorithms[key] = bound
        return bound

    # --- sorgu ---------------------------------------------------------
    def categories(self) -> list[Category]:
        return sorted(self._categories.values(), key=lambda c: c.index)

    def category(self, key: str) -> Category:
        return self._categories[key]

    def get(self, key: str) -> Algorithm:
        try:
            return self._algorithms[key]
        except KeyError:
            raise KeyError(f"Bilinmeyen algoritma: {key}") from None

    def in_category(self, category_key: str) -> list[Algorithm]:
        return [a for a in self._algorithms.values() if a.category == category_key]

    def __contains__(self, key: object) -> bool:
        return key in self._algorithms

    def __len__(self) -> int:
        return len(self._algorithms)
