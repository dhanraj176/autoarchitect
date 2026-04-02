# ============================================================
# Tests — DynamicAgent  (api/agents/dynamic_agent.py)
# init, run(), predict() fallback, act(), learn(), status()
# ============================================================

import pytest
torch = pytest.importorskip("torch")
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixture ───────────────────────────────────────────────────

@pytest.fixture
def agent(tmp_path):
    with patch("api.agents.dynamic_agent.AGENTS_DIR", tmp_path):
        from api.agents.dynamic_agent import DynamicAgent
        return DynamicAgent(
            agent_name  = "pothole_detector_agent",
            class_name  = "PotholeDetectorAgent",
            problem     = "detect potholes in streets",
            domain      = "image",
            model_type  = "resnet18",
            num_classes = 2,
            classes     = ["no_damage", "damage"],
        )


# ── Initialisation ────────────────────────────────────────────

class TestDynamicAgentInit:

    def test_agent_name_stored(self, agent):
        assert agent.agent_name == "pothole_detector_agent"

    def test_class_name_stored(self, agent):
        assert agent.class_name == "PotholeDetectorAgent"

    def test_domain_stored(self, agent):
        assert agent.domain == "image"

    def test_model_not_loaded_initially(self, agent):
        assert agent.model_loaded is False
        assert agent.model is None

    def test_predictions_zero(self, agent):
        assert agent.predictions == 0

    def test_memory_empty(self, agent):
        assert agent.memory == []

    def test_is_running_false(self, agent):
        assert agent.is_running is False

    def test_classes_stored(self, agent):
        assert agent.classes == ["no_damage", "damage"]

    def test_name_property(self, agent):
        assert agent.NAME == "PotholeDetectorAgent"


# ── run() ────────────────────────────────────────────────────

class TestDynamicAgentRun:

    NAS = {"architecture": [], "parameters": 1000, "search_time": 0.1}

    def test_run_returns_dict(self, agent):
        with patch("api.nas_engine.run_quick_nas",
                   return_value=self.NAS):
            result = agent.run("detect potholes")
        assert isinstance(result, dict)

    def test_run_status_success(self, agent):
        with patch("api.nas_engine.run_quick_nas",
                   return_value=self.NAS):
            result = agent.run("detect potholes")
        assert result["status"] == "success"

    def test_run_has_architecture(self, agent):
        with patch("api.nas_engine.run_quick_nas",
                   return_value=self.NAS):
            result = agent.run("detect potholes")
        assert "architecture" in result

    def test_run_elapsed_non_negative(self, agent):
        with patch("api.nas_engine.run_quick_nas",
                   return_value=self.NAS):
            result = agent.run("detect potholes")
        assert result.get("elapsed", 0) >= 0

    def test_run_domain_in_result(self, agent):
        with patch("api.nas_engine.run_quick_nas",
                   return_value=self.NAS):
            result = agent.run("detect potholes")
        assert result["domain"] == "image"


# ── predict() — fallback mode (no model) ─────────────────────

class TestDynamicAgentPredictFallback:

    def test_predict_returns_dict(self, agent):
        result = agent.predict("some_input.jpg")
        assert isinstance(result, dict)

    def test_predict_fallback_has_label(self, agent):
        result = agent.predict("some_input.jpg")
        assert "label" in result

    def test_predict_fallback_confidence_zero(self, agent):
        result = agent.predict("some_input.jpg")
        assert result["confidence"] == 0.0

    def test_predict_fallback_mode_key(self, agent):
        result = agent.predict("some_input.jpg")
        assert result.get("mode") == "fallback_no_model"

    def test_predict_increments_predictions(self, agent):
        agent.predict("input")
        assert agent.predictions == 1

    def test_predict_stores_to_memory(self, agent):
        agent.predict("input")
        assert len(agent.memory) == 0  # fallback_predict does not call _remember

    def test_predict_label_is_first_class(self, agent):
        result = agent.predict("input")
        assert result["label"] == "no_damage"


# ── act() ────────────────────────────────────────────────────

class TestDynamicAgentAct:

    def test_high_confidence_gives_alert(self, agent):
        result = agent.act({"label": "damage", "confidence": 0.90})
        assert result.get("action") == "alert"

    def test_medium_confidence_gives_log(self, agent):
        result = agent.act({"label": "damage", "confidence": 0.70})
        assert result.get("action") == "log"

    def test_low_confidence_gives_monitor(self, agent):
        result = agent.act({"label": "damage", "confidence": 0.40})
        assert result.get("action") == "monitor"

    def test_act_returns_same_dict(self, agent):
        d      = {"label": "x", "confidence": 0.5}
        result = agent.act(d)
        assert result is d


# ── learn() ──────────────────────────────────────────────────

class TestDynamicAgentLearn:

    def test_learn_returns_false_with_few_examples(self, agent):
        result = agent.learn(min_examples=20)
        assert result is False

    def test_learn_returns_true_with_enough_examples(self, agent):
        for _ in range(25):
            agent.memory.append({"label": "damage", "confidence": 0.8})
        result = agent.learn(min_examples=20)
        assert result is True


# ── status() / info() ─────────────────────────────────────────

class TestDynamicAgentStatus:

    def test_status_has_required_keys(self, agent):
        status = agent.status()
        for k in ("agent_name", "class_name", "problem", "domain",
                  "accuracy", "model_loaded", "predictions",
                  "memory_size", "classes"):
            assert k in status, f"Missing: {k}"

    def test_info_equals_status(self, agent):
        assert agent.info() == agent.status()

    def test_status_model_loaded_false(self, agent):
        assert agent.status()["model_loaded"] is False

    def test_status_accuracy_zero(self, agent):
        assert agent.status()["accuracy"] == 0.0

    def test_status_memory_size_after_predict(self, agent):
        agent.predict("input_x")
        assert agent.status()["memory_size"] == 0  # fallback_predict does not call _remember


# ── evaluate() compatibility ──────────────────────────────────

class TestDynamicAgentEvaluate:

    def test_returns_dict(self, agent):
        result = agent.evaluate({}, "some problem")
        assert isinstance(result, dict)

    def test_has_avg_score(self, agent):
        result = agent.evaluate({}, "some problem")
        assert "avg_score" in result

    def test_has_verdict(self, agent):
        result = agent.evaluate({}, "some problem")
        assert "verdict" in result


# ── load_model failure ────────────────────────────────────────

class TestDynamicAgentLoadModel:

    def test_load_model_nonexistent_returns_false(self, agent):
        result = agent.load_model("/nonexistent/path/model.pth")
        assert result is False
        assert agent.model_loaded is False

    def test_load_model_bad_file_does_not_raise(self, tmp_path, agent):
        bad_path = tmp_path / "bad.pth"
        bad_path.write_bytes(b"not a valid pth file")
        result = agent.load_model(str(bad_path))
        assert result is False
