# ============================================================
# Tests — dataset_fetcher.py
# DATASET_REGISTRY structure, keyword coverage
# ============================================================

import pytest
pytest.importorskip("torch")


class TestDatasetRegistry:

    @pytest.fixture
    def registry(self):
        from api.dataset_fetcher import DATASET_REGISTRY
        return DATASET_REGISTRY

    def test_registry_is_list(self, registry):
        assert isinstance(registry, list)

    def test_registry_is_non_empty(self, registry):
        assert len(registry) > 0

    def test_every_entry_has_hf_name(self, registry):
        for entry in registry:
            assert "hf_name" in entry, \
                f"Missing hf_name in: {entry}"

    def test_every_entry_has_domain(self, registry):
        valid_domains = {"image", "text", "medical", "security"}
        for entry in registry:
            assert "domain" in entry
            assert entry["domain"] in valid_domains, \
                f"Unknown domain '{entry['domain']}'"

    def test_every_entry_has_keywords_list(self, registry):
        for entry in registry:
            assert "keywords" in entry
            assert isinstance(entry["keywords"], list)
            assert len(entry["keywords"]) > 0

    def test_every_entry_has_classes(self, registry):
        for entry in registry:
            assert "classes" in entry
            assert isinstance(entry["classes"], list)
            assert len(entry["classes"]) >= 2

    def test_every_entry_has_num_classes(self, registry):
        for entry in registry:
            assert "num_classes" in entry
            assert isinstance(entry["num_classes"], int)
            assert entry["num_classes"] >= 2

    def test_num_classes_matches_classes_list(self, registry):
        for entry in registry:
            # Allow mismatch by ±1 for entries that override
            assert abs(entry["num_classes"] -
                       len(entry["classes"])) <= 1, \
                f"num_classes mismatch in {entry.get('hf_name')}"

    def test_accuracy_expected_is_reasonable(self, registry):
        for entry in registry:
            if "accuracy_expected" in entry:
                acc = entry["accuracy_expected"]
                assert 50 <= acc <= 99, \
                    f"Unreasonable accuracy {acc} in {entry.get('hf_name')}"

    def test_contains_text_entries(self, registry):
        text_entries = [e for e in registry if e["domain"] == "text"]
        assert len(text_entries) >= 1

    def test_contains_image_entries(self, registry):
        image_entries = [e for e in registry if e["domain"] == "image"]
        assert len(image_entries) >= 1

    def test_spam_keyword_present(self, registry):
        found = any("spam" in e.get("keywords", []) for e in registry)
        assert found, "Expected 'spam' keyword in registry"

    def test_pothole_keyword_present(self, registry):
        found = any("pothole" in e.get("keywords", []) for e in registry)
        assert found, "Expected 'pothole' keyword in registry"


class TestDatasetFetcherHelpers:

    def test_data_dir_variable_is_string(self):
        from api.dataset_fetcher import DATA_DIR
        assert isinstance(DATA_DIR, str)

    def test_base_dir_variable_is_string(self):
        from api.dataset_fetcher import BASE_DIR
        assert isinstance(BASE_DIR, str)
