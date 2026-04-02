# ============================================================
# Tests — WorkflowGenerator
# api/brain/workflow_generator.py
# ============================================================

import pytest
pytest.importorskip("torch")
from unittest.mock import patch, MagicMock
from pathlib import Path


@pytest.fixture
def generator(tmp_path):
    """WorkflowGenerator backed by temp files; meta-learner patched to
    return 'no prediction' so tests don't depend on torch."""
    import api.brain.workflow_generator as mod
    import api.brain.strategy_library as sl_mod
    import api.brain.performance_tracker as pt_mod

    no_pred = {"predicted": False}

    with patch.object(mod, 'get_meta_learner') as mock_ml, \
         patch.object(sl_mod, 'STRATEGY_FILE',
                      str(tmp_path / 'strategies.json')), \
         patch.object(pt_mod, 'HISTORY_FILE',
                      str(tmp_path / 'history.json')):
        ml_instance = MagicMock()
        ml_instance.predict.return_value = no_pred
        ml_instance.get_insights.return_value = {
            "status": "learning", "examples": 0, "trained": False,
            "avg_accuracy": 0, "best_combo": None, "best_dataset": None,
            "accuracy_trend": [], "until_retrain": 3,
            "combo_performance": {}
        }
        mock_ml.return_value = ml_instance
        wg = mod.WorkflowGenerator()
    return wg


# ── generate() ────────────────────────────────────────────

class TestGenerate:

    def test_returns_dict(self, generator):
        result = generator.generate("detect potholes", "image")
        assert isinstance(result, dict)

    def test_has_required_keys(self, generator):
        result = generator.generate("detect potholes", "image")
        for key in ("type", "agents", "strategy_name",
                    "steps", "expected_accuracy", "source"):
            assert key in result, f"Missing key: {key}"

    def test_type_is_single_or_multi(self, generator):
        result = generator.generate("detect potholes", "image")
        assert result["type"] in ("single", "multi")

    def test_agents_is_non_empty_list(self, generator):
        result = generator.generate("detect potholes", "image")
        assert isinstance(result["agents"], list)
        assert len(result["agents"]) >= 1

    def test_steps_is_list(self, generator):
        result = generator.generate("detect potholes", "image")
        assert isinstance(result["steps"], list)

    def test_steps_mention_evaluator(self, generator):
        result = generator.generate("detect potholes", "image")
        steps_text = " ".join(result["steps"]).lower()
        assert "evaluator" in steps_text

    def test_steps_mention_cache(self, generator):
        result = generator.generate("detect potholes", "image")
        steps_text = " ".join(result["steps"]).lower()
        assert "cache" in steps_text

    def test_source_is_strategy_library_when_meta_not_confident(
            self, generator):
        result = generator.generate("detect potholes", "image")
        assert result["source"] == "strategy_library"

    def test_expected_accuracy_is_positive(self, generator):
        result = generator.generate("detect potholes", "image")
        assert result["expected_accuracy"] >= 0

    def test_multi_agent_strategy_includes_fusion_step(self, generator):
        result = generator.generate(
            "detect fraud and classify suspicious emails", "security")
        if result["type"] == "multi":
            steps_text = " ".join(result["steps"]).lower()
            assert "fusion" in steps_text

    def test_no_raise_on_unknown_domain(self, generator):
        try:
            generator.generate("some problem", "unknown_domain_xyz")
        except Exception as e:
            pytest.fail(f"generate() raised on unknown domain: {e}")


# ── get_brain_status() ────────────────────────────────────

class TestGetBrainStatus:

    def test_returns_dict(self, generator):
        assert isinstance(generator.get_brain_status(), dict)

    def test_has_strategies_known(self, generator):
        status = generator.get_brain_status()
        assert "strategies_known" in status

    def test_has_problems_solved(self, generator):
        status = generator.get_brain_status()
        assert "problems_solved" in status

    def test_has_meta_learner_section(self, generator):
        status = generator.get_brain_status()
        assert "meta_learner" in status

    def test_strategies_known_is_int(self, generator):
        status = generator.get_brain_status()
        assert isinstance(status["strategies_known"], int)

    def test_meta_learner_has_status_key(self, generator):
        status = generator.get_brain_status()
        assert "status" in status["meta_learner"]


# ── _agent_description() ─────────────────────────────────

class TestAgentDescription:

    def test_image_returns_visual_description(self, generator):
        desc = generator._agent_description("image", "detect potholes")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_text_returns_classify_description(self, generator):
        desc = generator._agent_description("text", "classify spam")
        assert "text" in desc.lower() or "classif" in desc.lower()

    def test_medical_returns_medical_description(self, generator):
        desc = generator._agent_description("medical", "detect tumor")
        assert "medical" in desc.lower() or "analyz" in desc.lower()

    def test_security_returns_threat_description(self, generator):
        desc = generator._agent_description("security", "detect fraud")
        assert "threat" in desc.lower() or "detect" in desc.lower() or \
               "anomal" in desc.lower()

    def test_unknown_agent_returns_string(self, generator):
        desc = generator._agent_description("unknown_xyz", "some problem")
        assert isinstance(desc, str)


# ── learn_from_result() ───────────────────────────────────

class TestLearnFromResult:

    def test_does_not_raise(self, generator):
        workflow = {
            "strategy_name": "visual_detection",
            "agents":        ["image"],
        }
        try:
            generator.learn_from_result(
                problem    = "detect potholes",
                workflow   = workflow,
                accuracy   = 74.5,
                time_taken = 12.3,
            )
        except Exception as e:
            pytest.fail(f"learn_from_result raised: {e}")

    def test_from_cache_skips_meta_learner(self, generator):
        workflow = {"strategy_name": "visual_detection", "agents": ["image"]}
        # Meta-learner.learn should NOT be called for cache hits
        generator.learn_from_result(
            problem    = "detect potholes",
            workflow   = workflow,
            accuracy   = 74.5,
            time_taken = 1.0,
            from_cache = True,
        )
        generator.meta.learn.assert_not_called()

    def test_real_training_calls_meta_learner(self, generator):
        workflow = {"strategy_name": "visual_detection", "agents": ["image"]}
        generator.learn_from_result(
            problem    = "detect potholes",
            workflow   = workflow,
            accuracy   = 74.5,
            time_taken = 12.3,
            from_cache = False,
        )
        generator.meta.learn.assert_called_once()
