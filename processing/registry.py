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
    # İsteğe bağlı: seçenekler başka bir parametrenin değerine göre daralır (örn. kanal ← renk modeli).
    # option_sets = ((üst değer, ((değer, etiket), ...)), ...); her küme ``options``un alt kümesidir.
    depends_on: str = ""
    option_sets: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"{self.key}: en az bir seçenek gerekli")
        if self.default not in self.values:
            raise ValueError(f"{self.key}: default seçeneklerde yok")
        for parent, opts in self.option_sets:
            if not opts or not {v for v, _ in opts} <= set(self.values):
                raise ValueError(f"{self.key}: '{parent}' seçenek kümesi geçersiz")

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(v for v, _ in self.options)

    def options_for(self, parent_value: Any) -> tuple[tuple[str, str], ...]:
        """Üst parametrenin değerine göre geçerli seçenekler (küme yoksa tümü)."""
        return dict(self.option_sets).get(parent_value, self.options)

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


@dataclass(frozen=True)
class MaskParam:
    """Hücre hücre düzenlenen küçük ikili maske ve orijini (ör. özel yapı elemanı).

    Değer bir metindir: satırlar ``/`` ile ayrılır, her hücre ``0``/``1``; ``@r,c`` orijinin
    satır ve sütunudur (0'dan). Örnek: ``"010/011/000@1,1"``. Orijin üye olmayan bir hücrede
    de olabilir. En az bir hücre 1 olmalı; boyutlar 1 … ``max_size``.
    """

    key: str
    label: str
    default: str
    max_size: int = 7
    help: str = ""

    def __post_init__(self) -> None:
        if self.decode(self.default, self.max_size) is None:
            raise ValueError(f"{self.key}: geçersiz varsayılan maske")

    @staticmethod
    def decode(value: Any, max_size: int = 7) -> tuple[tuple[tuple[int, ...], ...], tuple[int, int]] | None:
        """``(satırlar, (r, c))`` ya da geçersizse None."""
        if not isinstance(value, str) or value.count("@") != 1:
            return None
        cells, origin = value.split("@")
        rows = cells.split("/")
        if not rows or len(rows) > max_size or any(len(r) != len(rows[0]) for r in rows):
            return None
        if not 1 <= len(rows[0]) <= max_size or any(ch not in "01" for r in rows for ch in r):
            return None
        grid = tuple(tuple(int(ch) for ch in r) for r in rows)
        if not any(any(r) for r in grid):
            return None
        try:
            r, c = (int(x) for x in origin.split(","))
        except ValueError:
            return None
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return None
        return grid, (r, c)

    @staticmethod
    def encode(grid, origin: tuple[int, int]) -> str:
        return "/".join("".join("1" if v else "0" for v in row) for row in grid) + f"@{origin[0]},{origin[1]}"

    def validate(self, value: Any) -> str:
        parsed = self.decode(value, self.max_size)
        return self.default if parsed is None else self.encode(*parsed)


ParamSpec = Union[IntParam, FloatParam, ChoiceParam, BoolParam, MaskParam]


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


@dataclass(frozen=True)
class ProcessResult:
    """Görüntüye ek olarak sayısal bir rapor da üreten algoritmaların dönüş tipi.

    ``report`` GUI'den bağımsız bir nesnedir (örn. bağlantılı bileşen tablosu).
    ``Algorithm.run`` yalnızca ``image``'ı döndürür; raporu isteyen çağıran
    ``Algorithm.execute`` kullanır. Böylece görüntü döndüren mevcut tüm
    algoritmalar ve çağrılar değişmeden çalışır.
    """

    image: np.ndarray
    report: Any = None
    # False: yalnızca ölçüm (örn. Color Histogram) — arayüz B görüntüsünü ve geri alma geçmişini değiştirmez
    changes_image: bool = True


AlgorithmFunc = Callable[..., Union[np.ndarray, ProcessResult]]


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

    def execute(self, image: np.ndarray, params: dict[str, Any] | None = None) -> ProcessResult:
        """Algoritmayı çalıştırır; görüntüyü ve (varsa) raporu birlikte döndürür."""
        if self.func is None:
            raise AlgorithmNotImplemented(f"'{self.title}' henüz uygulanmadı.")
        result = self.func(image, **self.resolve_params(params))
        return result if isinstance(result, ProcessResult) else ProcessResult(result)

    def run(self, image: np.ndarray, params: dict[str, Any] | None = None) -> np.ndarray:
        return self.execute(image, params).image


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
