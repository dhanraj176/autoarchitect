# ============================================================
# Tests — base_agent.py
# AgentMemory, BaseAgent (via concrete stub)
# ============================================================

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Minimal concrete agent for testing BaseAgent ABC ─────────

def _make_concrete_class():
    from api.agents.base_agent import BaseAgent

    class ConcreteAgent(BaseAgent):
        def predict(self, input_data) -> dict:
            return {"label": "test_label", "confidence": 90.0,
                    "action": "alert"}

    return ConcreteAgent


# ── AgentMemory tests ─────────────────────────────────────────

class TestAgentMemory:

    @pytest.fixture
    def memory(self, tmp_path):
        with patch("api.agents.base_agent.AGENTS_DIR", str(tmp_path)):
            from api.agents.base_agent import AgentMemory
            return AgentMemory("test_agent_001")

    def test_initially_empty(self, memory):
        assert memory.get_all() == []

    def test_store_adds_entry(self, memory):
        memory.store({"label": "cat", "confidence": 85.0})
        assert len(memory.get_all()) == 1

    def test_get_all_returns_all_entries(self, memory):
        for i in range(5):
            memory.store({"label": f"cls{i}", "confidence": float(i * 10)})
        assert len(memory.get_all()) == 5

    def test_get_recent_limits_count(self, memory):
        for i in range(20):
            memory.store({"label": "x", "confidence": 50.0})
        recent = memory.get_recent(n=5)
        assert len(recent) == 5

    def test_mark_processed_tracks_file(self, memory):
        memory.mark_processed("/path/to/file.jpg")
        assert "/path/to/file.jpg" in memory.get_processed_files()

    def test_get_hard_cases_filters_by_confidence(self, memory):
        memory.store({"label": "a", "confidence": 30.0})
        memory.store({"label": "b", "confidence": 90.0})
        hard = memory.get_hard_cases(conf_threshold=60.0)
        assert len(hard) == 1
        assert hard[0]["label"] == "a"

    def test_get_mistakes_filters_incorrect(self, memory):
        memory.store({"label": "a", "correct": False})
        memory.store({"label": "b", "correct": True})
        mistakes = memory.get_mistakes()
        assert len(mistakes) == 1

    def test_stats_total_matches_stored(self, memory):
        for _ in range(4):
            memory.store({"label": "x", "confidence": 80.0})
        stats = memory.stats()
        assert stats["total"] == 4

    def test_stats_accuracy_calculation(self, memory):
        memory.store({"label": "a", "correct": True,  "confidence": 80.0})
        memory.store({"label": "b", "correct": True,  "confidence": 80.0})
        memory.store({"label": "c", "correct": False, "confidence": 80.0})
        stats = memory.stats()
        assert stats["labeled"]  == 3
        assert stats["correct"]  == 2
        assert abs(stats["accuracy"] - 66.7) < 1.0

    def test_persists_to_disk(self, tmp_path):
        with patch("api.agents.base_agent.AGENTS_DIR", str(tmp_path)):
            from api.agents.base_agent import AgentMemory
            mem = AgentMemory("persist_test")
            mem.store({"label": "dog", "confidence": 77.0})

        # Re-load from disk
        with patch("api.agents.base_agent.AGENTS_DIR", str(tmp_path)):
            from api.agents.base_agent import AgentMemory
            mem2 = AgentMemory("persist_test")
            assert len(mem2.get_all()) == 1
            assert mem2.get_all()[0]["label"] == "dog"


# ── BaseAgent tests ───────────────────────────────────────────

class TestBaseAgent:

    @pytest.fixture
    def agent(self, tmp_path):
        with patch("api.agents.base_agent.AGENTS_DIR", str(tmp_path)):
            ConcreteAgent = _make_concrete_class()
            return ConcreteAgent(
                problem="detect cats",
                category="image",
                classes=["cat", "dog"],
                accuracy=80.0,
            )

    def test_agent_has_agent_id(self, agent):
        assert isinstance(agent.agent_id, str)
        assert len(agent.agent_id) > 0

    def test_agent_info_has_required_keys(self, agent):
        info = agent.info()
        for k in ("agent_id", "problem", "category", "classes",
                  "accuracy", "is_running", "is_trained",
                  "predictions", "memory_size", "actions",
                  "network_connections", "memory_accuracy", "created_at"):
            assert k in info, f"Missing key: {k}"

    def test_not_running_by_default(self, agent):
        assert agent.is_running is False

    def test_stop_sets_is_running_false(self, agent):
        agent.is_running = True
        agent.stop()
        assert agent.is_running is False

    def test_register_action_added_to_actions(self, agent):
        agent.register_action("alert", lambda p, i: {"sent": True})
        assert "alert" in agent._actions

    def test_act_calls_registered_actions(self, agent):
        calls = []
        agent.register_action("log", lambda p, i: calls.append(p) or {})
        agent.act({"label": "cat", "confidence": 90.0}, "input")
        assert len(calls) == 1

    def test_remember_increments_total_predictions(self, agent):
        agent.remember("input", {"label": "cat", "confidence": 80.0})
        assert agent.total_predictions == 1

    def test_remember_with_ground_truth_updates_correct_count(self, agent):
        agent.remember("input",
                       {"label": "cat", "confidence": 80.0},
                       ground_truth="cat")
        assert agent.correct_predictions == 1

    def test_remember_wrong_prediction_does_not_increment_correct(self, agent):
        agent.remember("input",
                       {"label": "dog", "confidence": 70.0},
                       ground_truth="cat")
        assert agent.correct_predictions == 0

    def test_learn_returns_false_with_insufficient_memories(self, agent):
        # Less than 20 memories → should return False
        result = agent.learn(min_examples=20)
        assert result is False

    def test_learn_returns_true_with_enough_memories(self, agent):
        with patch("api.agents.base_agent.AGENTS_DIR",
                   str(Path(agent._memory.mem_file).parent)):
            for i in range(25):
                agent._memory.store(
                    {"label": "cat", "correct": True,
                     "confidence": 80.0})
            result = agent.learn(min_examples=20)
            assert result is True

    def test_predict_returns_label_and_confidence(self, agent):
        result = agent.predict("some input")
        assert "label"      in result
        assert "confidence" in result

    def test_memory_accuracy_defaults_to_accuracy_when_no_labeled(self, agent):
        acc = agent._get_memory_accuracy()
        assert acc == agent.accuracy

    def test_run_async_returns_thread(self, agent):
        import threading
        t = agent.run_async(interval=1, source=None)
        assert isinstance(t, threading.Thread)
        agent.stop()

    def test_perceive_returns_empty_when_no_source(self, agent):
        inputs = agent.perceive(source=None)
        assert inputs == []
