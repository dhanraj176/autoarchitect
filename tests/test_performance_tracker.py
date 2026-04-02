# ============================================================
# Tests — PerformanceTracker
# api/brain/performance_tracker.py
# ============================================================

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def tracker(tmp_path):
    """Fresh PerformanceTracker backed by a temp file."""
    import api.brain.performance_tracker as mod
    with patch.object(mod, 'BRAIN_DIR', str(tmp_path)), \
         patch.object(mod, 'HISTORY_FILE', str(tmp_path / 'history.json')):
        t = mod.PerformanceTracker()
    return t, mod, tmp_path


# ── get_insights on empty history ─────────────────────────

class TestGetInsightsEmpty:

    def test_returns_message_key(self, tracker):
        t, _, _ = tracker
        assert t.get_insights() == {"message": "No history yet"}

    def test_does_not_raise_on_empty(self, tracker):
        t, _, _ = tracker
        # Fix #1 — this must never raise ZeroDivisionError
        try:
            t.get_insights()
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError on empty history — Fix #1 regression")


# ── get_insights with data ────────────────────────────────

class TestGetInsightsWithData:

    def _fill(self, tracker, entries):
        t, mod, tmp = tracker
        t.history = entries

    def test_avg_accuracy_correct(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": 60.0, "time_taken": 1.0, "from_cache": False,
             "success": True, "strategy": "s1", "agents": ["image"]},
            {"accuracy": 80.0, "time_taken": 3.0, "from_cache": False,
             "success": True, "strategy": "s1", "agents": ["text"]},
        ]
        result = t.get_insights()
        assert result["avg_accuracy"] == 70.0

    def test_avg_time_correct(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": 70.0, "time_taken": 2.0, "from_cache": False,
             "success": True, "strategy": "s1", "agents": ["image"]},
            {"accuracy": 70.0, "time_taken": 4.0, "from_cache": True,
             "success": True, "strategy": "s1", "agents": ["image"]},
        ]
        result = t.get_insights()
        assert result["avg_time"] == 3.0

    def test_avg_metrics_always_float(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": 75.0, "time_taken": 2.0, "from_cache": False,
             "success": True, "strategy": "s1", "agents": ["image"]},
        ]
        result = t.get_insights()
        assert isinstance(result["avg_accuracy"], float)
        assert isinstance(result["avg_time"], float)

    def test_cache_rate_calculation(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": 70.0, "time_taken": 1.0, "from_cache": True,
             "success": True, "strategy": "s1", "agents": []},
            {"accuracy": 70.0, "time_taken": 1.0, "from_cache": False,
             "success": True, "strategy": "s1", "agents": []},
        ]
        result = t.get_insights()
        assert result["cache_hits"] == 1
        assert result["cache_rate"] == 50.0

    def test_success_rate_calculation(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": 80.0, "time_taken": 1.0, "from_cache": False,
             "success": True,  "strategy": "s1", "agents": []},
            {"accuracy": 30.0, "time_taken": 1.0, "from_cache": False,
             "success": False, "strategy": "s1", "agents": []},
        ]
        result = t.get_insights()
        assert result["success_rate"] == 50.0

    def test_agent_usage_counts(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": 70.0, "time_taken": 1.0, "from_cache": False,
             "success": True, "strategy": "s1", "agents": ["image", "text"]},
            {"accuracy": 70.0, "time_taken": 1.0, "from_cache": False,
             "success": True, "strategy": "s1", "agents": ["image"]},
        ]
        result = t.get_insights()
        assert result["agent_usage"]["image"] == 2
        assert result["agent_usage"]["text"] == 1

    def test_best_strategy_chosen(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": 90.0, "time_taken": 1.0, "from_cache": False,
             "success": True, "strategy": "winner", "agents": []},
            {"accuracy": 50.0, "time_taken": 1.0, "from_cache": False,
             "success": False, "strategy": "loser", "agents": []},
        ]
        result = t.get_insights()
        assert result["best_strategy"] == "winner"

    def test_recent_capped_at_five(self, tracker):
        t, _, _ = tracker
        t.history = [
            {"accuracy": float(i), "time_taken": 1.0, "from_cache": False,
             "success": True, "strategy": "s", "agents": []}
            for i in range(10)
        ]
        result = t.get_insights()
        assert len(result["recent"]) == 5


# ── record() ──────────────────────────────────────────────

class TestRecord:

    def test_record_appends_to_history(self, tmp_path):
        import api.brain.performance_tracker as mod
        with patch.object(mod, 'BRAIN_DIR', str(tmp_path)), \
             patch.object(mod, 'HISTORY_FILE', str(tmp_path / 'history.json')):
            t = mod.PerformanceTracker()
            t.record("detect potholes", "visual_detection",
                     ["image"], 74.5, 3.2)
        assert len(t.history) == 1
        assert t.history[0]["accuracy"] == 74.5

    def test_record_success_flag_above_50(self, tmp_path):
        import api.brain.performance_tracker as mod
        with patch.object(mod, 'BRAIN_DIR', str(tmp_path)), \
             patch.object(mod, 'HISTORY_FILE', str(tmp_path / 'history.json')):
            t = mod.PerformanceTracker()
            t.record("test", "s", [], 51.0, 1.0)
        assert t.history[0]["success"] is True

    def test_record_success_flag_below_50(self, tmp_path):
        import api.brain.performance_tracker as mod
        with patch.object(mod, 'BRAIN_DIR', str(tmp_path)), \
             patch.object(mod, 'HISTORY_FILE', str(tmp_path / 'history.json')):
            t = mod.PerformanceTracker()
            t.record("test", "s", [], 49.9, 1.0)
        assert t.history[0]["success"] is False
