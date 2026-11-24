"""Algoritma kayıt sistemi ve katalog testleri (GUI yok)."""

import numpy as np
import pytest

from processing.catalog import CATEGORIES, build_registry
from processing.registry import (
    Algorithm,
    AlgorithmNotImplemented,
    AlgorithmRegistry,
    BoolParam,
    Category,
    ChoiceParam,
    FloatParam,
    IntParam,
)

EXPECTED_ORDER = [
    "Image Transformations",
    "Intensity Transformations",
    "Spatial Filtering",
    "Morphological Processing",
    "Image Segmentation",
    "Color Processing",
    "Feature Extraction",
]


def test_catalog_has_seven_categories_in_course_order():
    registry = build_registry()
    assert [c.title for c in registry.categories()] == EXPECTED_ORDER
    assert [c.index for c in registry.categories()] == list(range(1, 8))


def test_every_category_has_operations_and_unique_wavelength():
    registry = build_registry()
    for cat in registry.categories():
        assert registry.in_category(cat.key), cat.key
    assert len({c.wavelength for c in CATEGORIES}) == len(CATEGORIES)


def test_catalog_defaults_are_valid():
    registry = build_registry()
    for cat in registry.categories():
        for algo in registry.in_category(cat.key):
            assert algo.resolve_params({}) == algo.default_params(), algo.key


def test_int_param_clamps_and_forces_odd():
    p = IntParam("k", "Kernel", 3, 1, 31, odd_only=True)
    assert p.validate(100) == 31
    assert p.validate(-5) == 1
    assert p.validate(4) == 5
    assert p.validate("7") == 7
    assert p.validate("abc") == 3
    assert p.effective_step == 2


def test_float_choice_bool_validation():
    f = FloatParam("g", "Gamma", 1.0, 0.1, 5.0, decimals=2)
    assert f.validate(9) == 5.0
    assert f.validate(float("nan")) == 1.0
    assert f.validate(0.123) == 0.12
    c = ChoiceParam("m", "Mode", (("a", "A"), ("b", "B")), "a")
    assert c.validate("b") == "b"
    assert c.validate("zzz") == "a"
    assert BoolParam("x", "X").validate(1) is True


def test_invalid_param_definitions_rejected():
    with pytest.raises(ValueError):
        IntParam("k", "K", 2, 1, 9, odd_only=True)
    with pytest.raises(ValueError):
        ChoiceParam("m", "M", (("a", "A"),), "b")
    with pytest.raises(ValueError):
        FloatParam("f", "F", 1.0, 5.0, 1.0)


def test_unimplemented_algorithm_raises_and_bind_makes_it_runnable():
    registry = AlgorithmRegistry()
    registry.add_category(Category("c", 1, "Cat", "", 500))
    registry.add(Algorithm("inv", "Invert", "c", "", (IntParam("offset", "Offset", 0, 0, 10),)))
    img = np.full((2, 2), 10, dtype=np.uint8)

    with pytest.raises(AlgorithmNotImplemented):
        registry.get("inv").run(img)

    registry.bind("inv", lambda image, offset: (255 - image - offset).astype(np.uint8))
    algo = registry.get("inv")
    assert algo.implemented
    out = algo.run(img, {"offset": 99, "unknown": 1})  # 99 → 10'a kırpılır, bilinmeyen atılır
    assert np.all(out == 235)


def test_registry_rejects_duplicates_and_unknown_category():
    registry = AlgorithmRegistry()
    registry.add_category(Category("c", 1, "Cat", "", 500))
    registry.add(Algorithm("a", "A", "c", ""))
    with pytest.raises(ValueError):
        registry.add(Algorithm("a", "A", "c", ""))
    with pytest.raises(KeyError):
        registry.add(Algorithm("b", "B", "missing", ""))
    with pytest.raises(KeyError):
        registry.get("nope")
