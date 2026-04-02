# ============================================================
# Tests — Domain Agents
# ImageAgent, TextAgent, MedicalAgent, SecurityAgent
# ============================================================

import pytest
pytest.importorskip("torch")
from unittest.mock import patch, MagicMock


# ── Shared NAS mock ───────────────────────────────────────────

NAS_RESULT = {
    "architecture": [
        {"cell": 1, "operations": [
            {"operation": "conv3x3", "confidence": 80.0,
             "weights": {"skip": 0.1, "conv3x3": 0.8,
                         "conv5x5": 0.05, "maxpool": 0.03,
                         "avgpool": 0.02}}
        ]}
    ],
    "parameters":   105000,
    "search_time":  0.5,
}


# ── ImageAgent ────────────────────────────────────────────────

class TestImageAgent:

    @pytest.fixture
    def agent(self):
        with patch("api.agents.image_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            from api.agents.image_agent import ImageAgent
            return ImageAgent()

    def test_run_returns_dict(self, agent):
        with patch("api.agents.image_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect potholes")
        assert isinstance(result, dict)

    def test_run_status_is_success(self, agent):
        with patch("api.agents.image_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect potholes")
        assert result["status"] == "success"

    def test_run_has_architecture_key(self, agent):
        with patch("api.agents.image_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect potholes")
        assert "architecture" in result

    def test_run_has_parameters_key(self, agent):
        with patch("api.agents.image_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect potholes")
        assert "parameters" in result

    def test_run_elapsed_non_negative(self, agent):
        with patch("api.agents.image_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect potholes")
        assert result.get("elapsed", 0) >= 0

    def test_run_type_is_image_detection(self, agent):
        with patch("api.agents.image_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("any image problem")
        assert result["type"] == "image_detection"

    def test_predict_image_without_model_returns_unknown(self, agent):
        result = agent.predict_image("nonexistent.jpg")
        assert result["label"]      == "unknown"
        assert result["confidence"] == 0.0

    def test_load_trained_model_bad_path_does_not_raise(self, agent):
        """Loading a nonexistent model should print a warning, not crash."""
        agent.load_trained_model("/nonexistent/model.pth",
                                  ["cat", "dog"], 2)
        assert agent.trained_model is None


# ── TextAgent ─────────────────────────────────────────────────

class TestTextAgent:

    @pytest.fixture
    def agent(self):
        with patch("api.agents.text_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            from api.agents.text_agent import TextAgent
            return TextAgent()

    def test_run_returns_dict(self, agent):
        with patch("api.agents.text_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("classify spam emails")
        assert isinstance(result, dict)

    def test_run_status_is_success(self, agent):
        with patch("api.agents.text_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("classify spam emails")
        assert result["status"] == "success"

    def test_run_has_classes(self, agent):
        with patch("api.agents.text_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("classify spam emails")
        assert "classes" in result
        assert isinstance(result["classes"], list)
        assert len(result["classes"]) > 0

    def test_run_type_is_text_classification(self, agent):
        with patch("api.agents.text_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("classify text")
        assert result["type"] == "text_classification"

    def test_classes_has_10_entries(self, agent):
        from api.agents.text_agent import TextAgent
        assert len(TextAgent.CLASSES) == 10

    def test_run_has_dataset_key(self, agent):
        with patch("api.agents.text_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("any text problem")
        assert "dataset" in result


# ── MedicalAgent ─────────────────────────────────────────────

class TestMedicalAgent:

    @pytest.fixture
    def agent(self):
        with patch("api.agents.medical_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            from api.agents.medical_agent import MedicalAgent
            return MedicalAgent()

    def test_run_returns_dict(self, agent):
        with patch("api.agents.medical_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("analyze xray for pneumonia")
        assert isinstance(result, dict)

    def test_run_status_is_success(self, agent):
        with patch("api.agents.medical_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("analyze xray for pneumonia")
        assert result["status"] == "success"

    def test_run_has_disclaimer(self, agent):
        with patch("api.agents.medical_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("analyze xray for pneumonia")
        assert "disclaimer" in result

    def test_run_type_is_medical_analysis(self, agent):
        with patch("api.agents.medical_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("any medical problem")
        assert result["type"] == "medical_analysis"

    def test_predict_scan_without_model_returns_fallback(self, agent):
        result = agent._predict_scan("data:image/png;base64,abc==")
        assert "label"      in result
        assert "confidence" in result
        assert result["confidence"] > 0

    def test_default_classes_are_medical(self, agent):
        assert "Normal" in agent.trained_classes
        assert len(agent.trained_classes) == 10

    def test_load_trained_model_bad_path_does_not_raise(self, agent):
        agent.load_trained_model("/nonexistent/model.pth",
                                  ["Normal", "Abnormal"], 2)
        assert agent.trained_model is None


# ── SecurityAgent ─────────────────────────────────────────────

class TestSecurityAgent:

    @pytest.fixture
    def agent(self):
        with patch("api.agents.security_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            from api.agents.security_agent import SecurityAgent
            return SecurityAgent()

    def test_run_returns_dict(self, agent):
        with patch("api.agents.security_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect network intrusions")
        assert isinstance(result, dict)

    def test_run_status_is_success(self, agent):
        with patch("api.agents.security_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect network intrusions")
        assert result["status"] == "success"

    def test_run_has_threat_levels(self, agent):
        with patch("api.agents.security_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("detect network intrusions")
        assert "threat_levels" in result
        assert len(result["threat_levels"]) == 10

    def test_run_type_is_security_analysis(self, agent):
        with patch("api.agents.security_agent.run_quick_nas",
                   return_value=NAS_RESULT):
            result = agent.run("any security problem")
        assert result["type"] == "security_analysis"

    def test_threat_levels_have_label_code_color(self, agent):
        from api.agents.security_agent import SecurityAgent
        for k, v in SecurityAgent.THREAT_LEVELS.items():
            label, code, color = v
            assert isinstance(label, str)
            assert isinstance(code,  str)
            assert color.startswith("#")

    def test_predict_threat_without_model_returns_unknown(self, agent):
        result = agent.predict_threat("suspicious login attempt")
        assert result["label"]      == "unknown"
        assert result["confidence"] == 0.0

    def test_load_trained_model_bad_path_does_not_raise(self, agent):
        agent.load_trained_model("/nonexistent/model.pth",
                                  ["Safe", "Threat"], 2)
        assert agent.trained_model is None
