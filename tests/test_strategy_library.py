# ============================================================
# Tests — StrategyLibrary
# api/brain/strategy_library.py
# ============================================================

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def empty_library(tmp_path):
    """StrategyLibrary with no strategies and no file I/O."""
    import api.brain.strategy_library as mod
    with patch.object(mod, 'STRATEGY_FILE', str(tmp_path / 'strategies.json')):
        lib = mod.StrategyLibrary.__new__(mod.StrategyLibrary)
        lib.strategies = {}
        lib._file = str(tmp_path / 'strategies.json')
    return lib, mod, tmp_path


@pytest.fixture
def seeded_library(tmp_path):
    """StrategyLibrary seeded with real SEED_STRATEGIES."""
    import api.brain.strategy_library as mod
    with patch.object(mod, 'STRATEGY_FILE', str(tmp_path / 'strategies.json')):
        lib = mod.StrategyLibrary()
    return lib, mod, tmp_path


# ── get_stats — empty strategies ─────────────────────────

class TestGetStatsEmpty:

    def test_best_strategy_is_none_not_string(self, empty_library):
        lib, _, _ = empty_library
        stats = lib.get_stats()
        # Fix #2 — must be None, not the string "none"
        assert stats["best_strategy"] is None, \
            'best_strategy must be None when empty, not "none"'

    def test_best_strategy_falsy(self, empty_library):
        lib, _, _ = empty_library
        stats = lib.get_stats()
        assert not stats["best_strategy"], \
            '"none" string evaluates to True — breaks guard logic'

    def test_total_strategies_zero(self, empty_library):
        lib, _, _ = empty_library
        assert lib.get_stats()["total_strategies"] == 0

    def test_avg_accuracy_zero(self, empty_library):
        lib, _, _ = empty_library
        assert lib.get_stats()["avg_accuracy"] == 0.0


# ── get_stats — with strategies ───────────────────────────

class TestGetStatsFilled:

    def test_best_strategy_returns_highest_accuracy(self, empty_library):
        lib, _, _ = empty_library
        lib.strategies = {
            "low":  {"agents": ["image"], "avg_accuracy": 55.0, "uses": 1, "auto_learned": False},
            "high": {"agents": ["image"], "avg_accuracy": 92.0, "uses": 1, "auto_learned": False},
        }
        stats = lib.get_stats()
        assert stats["best_strategy"] == "high"

    def test_auto_learned_count(self, empty_library):
        lib, _, _ = empty_library
        lib.strategies = {
            "seed":    {"agents": ["image"], "avg_accuracy": 70.0, "uses": 1, "auto_learned": False},
            "learned": {"agents": ["text"],  "avg_accuracy": 80.0, "uses": 1, "auto_learned": True},
        }
        stats = lib.get_stats()
        assert stats["auto_learned"] == 1
        assert stats["seed_strategies"] == 1

    def test_total_problems_sums_uses(self, empty_library):
        lib, _, _ = empty_library
        lib.strategies = {
            "a": {"agents": ["image"],    "avg_accuracy": 70.0, "uses": 3, "auto_learned": False},
            "b": {"agents": ["security"], "avg_accuracy": 80.0, "uses": 5, "auto_learned": False},
        }
        assert lib.get_stats()["total_problems"] == 8


# ── find_best_strategy ────────────────────────────────────

class TestFindBestStrategy:

    def test_returns_default_when_no_strategies(self, empty_library):
        lib, _, _ = empty_library
        result = lib.find_best_strategy("detect potholes", "image")
        assert "agents" in result
        assert result["agents"] == ["image"]

    def test_domain_match_boosts_score(self, seeded_library):
        lib, _, _ = seeded_library
        result = lib.find_best_strategy("detect potholes in road images", "image")
        assert "image" in result["agents"]

    def test_keyword_match_increases_score(self, seeded_library):
        lib, _, _ = seeded_library
        result = lib.find_best_strategy("classify spam emails", "text")
        assert "text" in result["agents"]

    def test_result_has_required_keys(self, seeded_library):
        lib, _, _ = seeded_library
        result = lib.find_best_strategy("detect fraud", "security")
        for key in ("strategy_name", "agents", "confidence", "avg_accuracy"):
            assert key in result, f"Missing key: {key}"

    def test_confidence_between_0_and_1(self, seeded_library):
        lib, _, _ = seeded_library
        result = lib.find_best_strategy("classify images", "image")
        assert 0.0 <= result["confidence"] <= 1.0


# ── learn() ───────────────────────────────────────────────

class TestStrategyLibraryLearn:

    def test_learn_updates_existing_strategy(self, seeded_library):
        lib, _, _ = seeded_library
        strategy_name = list(lib.strategies.keys())[0]
        original_uses = lib.strategies[strategy_name]["uses"]
        lib.learn("test problem", strategy_name, 85.0, ["image"], success=True)
        assert lib.strategies[strategy_name]["uses"] == original_uses + 1

    def test_learn_creates_new_strategy(self, seeded_library):
        lib, _, _ = seeded_library
        count_before = len(lib.strategies)
        lib.learn("brand new unique problem xyz", "new_strat_xyz", 75.0, ["text"])
        assert len(lib.strategies) > count_before

    def test_new_strategy_marked_auto_learned(self, seeded_library):
        lib, _, _ = seeded_library
        count_before = len(lib.strategies)
        lib.learn("unique problem abc", "auto_strat_abc", 80.0, ["security"])
        new_strategies = [
            s for s in lib.strategies.values()
            if s.get("auto_learned") and "unique problem" in s.get("description", "")
        ]
        assert len(new_strategies) >= 1


# ── _extract_keywords ─────────────────────────────────────

class TestExtractKeywords:

    def test_filters_short_words(self, seeded_library):
        lib, _, _ = seeded_library
        kws = lib._extract_keywords("detect if the car is ok")
        assert "if" not in kws
        assert "the" not in kws
        assert "is" not in kws

    def test_returns_list(self, seeded_library):
        lib, _, _ = seeded_library
        assert isinstance(lib._extract_keywords("classify medical images"), list)

    def test_max_eight_keywords(self, seeded_library):
        lib, _, _ = seeded_library
        long_problem = " ".join([f"word{i}" for i in range(20)])
        assert len(lib._extract_keywords(long_problem)) <= 8


# ── _default_strategy ─────────────────────────────────────

class TestDefaultStrategy:

    def test_agents_match_domain(self, seeded_library):
        lib, _, _ = seeded_library
        for domain in ("image", "text", "medical", "security"):
            result = lib._default_strategy(domain)
            assert domain in result["agents"]

    def test_accuracy_is_50(self, seeded_library):
        lib, _, _ = seeded_library
        assert lib._default_strategy("image")["avg_accuracy"] == 50.0

    def test_confidence_is_half(self, seeded_library):
        lib, _, _ = seeded_library
        assert lib._default_strategy("text")["confidence"] == 0.5
