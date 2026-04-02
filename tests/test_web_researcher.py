# ============================================================
# Tests — web_researcher.py
# WebResearcher: registry lookup, fallback terms, cache,
# VERIFIED_REGISTRY, MODEL_MAP, ACC_MAP
# ============================================================

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Module-level constants ────────────────────────────────────

class TestVerifiedRegistry:

    @pytest.fixture
    def registry(self):
        from api.brain.web_researcher import VERIFIED_REGISTRY
        return VERIFIED_REGISTRY

    def test_is_dict(self, registry):
        assert isinstance(registry, dict)

    def test_non_empty(self, registry):
        assert len(registry) > 0

    def test_all_values_are_strings(self, registry):
        for k, v in registry.items():
            assert isinstance(v, str), \
                f"Value for '{k}' is not a string"

    def test_pothole_maps_to_dataset(self, registry):
        assert "pothole" in registry
        assert len(registry["pothole"]) > 0

    def test_fire_maps_to_dataset(self, registry):
        assert "fire" in registry

    def test_spam_maps_to_dataset(self, registry):
        assert "spam" in registry


class TestModelMap:

    def test_all_domains_covered(self):
        from api.brain.web_researcher import MODEL_MAP
        for domain in ("image", "medical", "text", "security"):
            assert domain in MODEL_MAP

    def test_values_are_strings(self):
        from api.brain.web_researcher import MODEL_MAP
        for v in MODEL_MAP.values():
            assert isinstance(v, str)


class TestAccMap:

    def test_all_domains_covered(self):
        from api.brain.web_researcher import ACC_MAP
        for domain in ("image", "medical", "text", "security"):
            assert domain in ACC_MAP

    def test_values_contain_percent(self):
        from api.brain.web_researcher import ACC_MAP
        for v in ACC_MAP.values():
            assert "%" in v


# ── WebResearcher instance ────────────────────────────────────

def _make_researcher(tmp_path):
    """Create WebResearcher with cache dir patched to tmp_path."""
    with patch("api.brain.web_researcher.Path",
               side_effect=lambda *a: (
                   tmp_path / "/".join(str(x) for x in a)
                   if a else Path())):
        pass

    # Simpler: patch ddgs import to avoid network
    with patch.dict("sys.modules",
                    {"ddgs": MagicMock(), "duckduckgo_search": MagicMock()}):
        from api.brain.web_researcher import WebResearcher
        researcher = WebResearcher.__new__(WebResearcher)
        researcher.groq_key  = ""
        researcher.cache_dir = tmp_path / "research_cache"
        researcher.cache_dir.mkdir(parents=True, exist_ok=True)
        researcher._ddg      = None
    return researcher


class TestCheckRegistry:

    @pytest.fixture
    def researcher(self, tmp_path):
        return _make_researcher(tmp_path)

    def test_pothole_problem_hits_registry(self, researcher):
        ds_id = researcher._check_registry("detect potholes in roads")
        assert ds_id != ""

    def test_unknown_problem_returns_empty(self, researcher):
        ds_id = researcher._check_registry("solve the halting problem")
        assert ds_id == ""

    def test_case_insensitive_match(self, researcher):
        ds_id = researcher._check_registry("FIRE detection system")
        assert ds_id != ""

    def test_spam_detection_hits_registry(self, researcher):
        ds_id = researcher._check_registry("classify spam emails")
        assert ds_id != ""


class TestFallbackTerms:

    @pytest.fixture
    def researcher(self, tmp_path):
        return _make_researcher(tmp_path)

    def test_returns_list_of_3(self, researcher):
        terms = researcher._fallback_terms("detect potholes", "image")
        assert isinstance(terms, list)
        assert len(terms) == 3

    def test_terms_are_non_empty_strings(self, researcher):
        terms = researcher._fallback_terms("classify spam messages", "text")
        for t in terms:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_stop_words_excluded_from_keywords(self, researcher):
        terms = researcher._fallback_terms("detect the fire", "image")
        # "the" is a stop word — should not appear as isolated keyword
        for t in terms:
            assert " the " not in t


class TestCache:

    @pytest.fixture
    def researcher(self, tmp_path):
        return _make_researcher(tmp_path)

    def test_cache_miss_returns_none(self, researcher):
        result = researcher._check_cache("totally new problem xyz")
        assert result is None

    def test_save_and_load_cache(self, researcher):
        from datetime import datetime
        problem  = "detect fire in forests"
        approach = {
            "best_model":   "ResNet18",
            "best_dataset": "pyronear/openfire",
            "searched_at":  datetime.now().isoformat(),   # required for age check
        }

        researcher._save_cache(problem, approach)
        loaded = researcher._check_cache(problem)

        assert loaded is not None
        assert loaded["best_model"]   == "ResNet18"
        assert loaded["best_dataset"] == "pyronear/openfire"

    def test_cache_miss_after_different_problem(self, researcher):
        from datetime import datetime
        researcher._save_cache("problem A",
                               {"data": 1, "searched_at": datetime.now().isoformat()})
        assert researcher._check_cache("problem B") is None
