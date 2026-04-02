# ============================================================
# Tests — DatasetManager  (pure registry logic — no torch needed)
# api/dataset_manager.py
# ============================================================

import pytest
from unittest.mock import MagicMock, patch
import sys

# dataset_manager.py imports torch at module level — stub it out
# so the pure registry/selection logic can be tested without GPU deps
_torch_stub = MagicMock()
with patch.dict(sys.modules, {
    'torch': _torch_stub,
    'torchvision': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'torch.utils': MagicMock(),
    'torch.utils.data': MagicMock(),
}):
    from api.dataset_manager import (
        select_dataset,
        get_num_classes,
        get_class_names,
        DATASET_REGISTRY,
    )


# ── get_num_classes ───────────────────────────────────────

class TestGetNumClasses:

    def test_cifar10_returns_10(self):
        assert get_num_classes("cifar10") == 10

    def test_mnist_returns_10(self):
        assert get_num_classes("mnist") == 10

    def test_fashionmnist_returns_10(self):
        assert get_num_classes("fashionmnist") == 10

    def test_unknown_dataset_returns_10(self):
        assert get_num_classes("unknown_dataset_xyz") == 10

    def test_returns_int(self):
        assert isinstance(get_num_classes("cifar10"), int)


# ── get_class_names ───────────────────────────────────────

class TestGetClassNames:

    def test_cifar10_has_ten_classes(self):
        names = get_class_names("cifar10")
        assert len(names) == 10

    def test_cifar10_contains_airplane(self):
        assert "airplane" in get_class_names("cifar10")

    def test_mnist_has_digit_strings(self):
        names = get_class_names("mnist")
        assert "0" in names and "9" in names

    def test_fashionmnist_has_ten_classes(self):
        assert len(get_class_names("fashionmnist")) == 10

    def test_fashionmnist_contains_tshirt(self):
        names = get_class_names("fashionmnist")
        assert any("shirt" in n.lower() or "t-shirt" in n.lower()
                   for n in names)

    def test_returns_list(self):
        assert isinstance(get_class_names("cifar10"), list)

    def test_unknown_dataset_returns_list_of_strings(self):
        names = get_class_names("completely_unknown")
        assert isinstance(names, list)
        assert len(names) == 10
        assert all(isinstance(n, str) for n in names)


# ── select_dataset ────────────────────────────────────────

class TestSelectDataset:

    def test_returns_dict(self):
        result = select_dataset("detect potholes", "image")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = select_dataset("detect potholes", "image")
        for key in ("name", "reason", "num_classes"):
            assert key in result, f"Missing key: {key}"

    def test_pothole_keyword_gives_cifar10(self):
        result = select_dataset("detect potholes in roads", "image")
        assert result["name"] == "cifar10"

    def test_spam_keyword_gives_mnist(self):
        result = select_dataset("classify spam emails", "text")
        assert result["name"] == "mnist"

    def test_tumor_keyword_gives_fashionmnist(self):
        result = select_dataset("brain tumor analysis mri", "medical")
        assert result["name"] == "fashionmnist"

    def test_fraud_keyword_gives_cifar10(self):
        result = select_dataset("detect fraud transactions", "security")
        assert result["name"] == "cifar10"

    def test_num_classes_matches_get_num_classes(self):
        result = select_dataset("detect potholes", "image")
        assert result["num_classes"] == get_num_classes(result["name"])

    def test_unknown_problem_falls_back_to_category_default(self):
        result = select_dataset("completely unknown xyz", "medical")
        assert result["name"] == "fashionmnist"

    def test_unknown_problem_image_category_gives_cifar10(self):
        result = select_dataset("completely unknown xyz abc", "image")
        assert result["name"] == "cifar10"

    def test_unknown_problem_text_category_gives_mnist(self):
        result = select_dataset("zzzzzz totally novel problem", "text")
        assert result["name"] == "mnist"

    def test_unknown_category_falls_back_to_cifar10(self):
        result = select_dataset("random problem", "unknown_category")
        assert result["name"] == "cifar10"

    def test_reason_is_string(self):
        result = select_dataset("detect potholes", "image")
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_keyword_in_reason(self):
        result = select_dataset("detect potholes in roads", "image")
        # reason should mention the matched keyword
        assert "pothole" in result["reason"].lower() or \
               "road" in result["reason"].lower() or \
               "default" in result["reason"].lower()


# ── DATASET_REGISTRY sanity checks ───────────────────────

class TestDatasetManagerRegistry:

    def test_registry_is_not_empty(self):
        assert len(DATASET_REGISTRY) > 0

    def test_all_values_are_known_datasets(self):
        valid = {"cifar10", "mnist", "fashionmnist"}
        for kw, ds in DATASET_REGISTRY.items():
            assert ds in valid, \
                f"Unknown dataset '{ds}' for keyword '{kw}'"

    def test_all_keys_are_lowercase_strings(self):
        for key in DATASET_REGISTRY:
            assert isinstance(key, str)
            assert key == key.lower(), \
                f"Registry key '{key}' is not lowercase"
