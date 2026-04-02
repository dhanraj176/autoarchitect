# ============================================================
# Tests — ProblemAnalyzer (BERT domain classifier)
# api/analyzer.py
# ============================================================

import pickle
import pytest
pytest.importorskip("torch")
from unittest.mock import MagicMock, patch, mock_open


# ── Helpers ──────────────────────────────────────────────────

def _make_analyzer():
    """Instantiate ProblemAnalyzer with fully mocked BERT stack."""
    import api.analyzer  # ensure submodule is loaded before patch resolves it
    import torch

    # Fake label encoder with 4 classes
    fake_le = MagicMock()
    fake_le.classes_ = ["image", "medical", "security", "text"]

    # Fake model outputs
    logits = torch.zeros(1, 4)
    logits[0][0] = 5.0          # image wins
    fake_output = MagicMock()
    fake_output.logits = logits

    fake_model = MagicMock()
    fake_model.return_value = fake_output
    fake_model.eval.return_value = None

    # Fake tokenizer output
    fake_encoding = {
        "input_ids":      torch.zeros(1, 64, dtype=torch.long),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
    }
    fake_tokenizer = MagicMock(return_value=fake_encoding)

    with patch("api.analyzer.BertTokenizer.from_pretrained",
               return_value=fake_tokenizer), \
         patch("api.analyzer.BertForSequenceClassification.from_pretrained",
               return_value=fake_model), \
         patch("builtins.open", mock_open()), \
         patch("pickle.load", return_value=fake_le):
        from api.analyzer import ProblemAnalyzer
        analyzer = ProblemAnalyzer()

    # Stash mocks for assertion in tests
    analyzer._fake_model    = fake_model
    analyzer._fake_le       = fake_le
    return analyzer


# ── Tests ────────────────────────────────────────────────────

class TestProblemAnalyzerAnalyze:

    @pytest.fixture
    def analyzer(self):
        return _make_analyzer()

    def test_returns_dict(self, analyzer):
        result = analyzer.analyze("detect potholes in streets")
        assert isinstance(result, dict)

    def test_has_required_keys(self, analyzer):
        result = analyzer.analyze("classify medical xray images")
        for key in ("problem", "category", "type", "confidence",
                    "certain", "all_scores"):
            assert key in result, f"Missing key: {key}"

    def test_problem_echoed(self, analyzer):
        problem = "detect potholes"
        result  = analyzer.analyze(problem)
        assert result["problem"] == problem

    def test_category_is_valid_domain(self, analyzer):
        result = analyzer.analyze("some problem")
        assert result["category"] in ("image", "medical", "security", "text")

    def test_confidence_is_float_between_0_and_100(self, analyzer):
        result = analyzer.analyze("some problem")
        assert 0.0 <= result["confidence"] <= 100.0

    def test_certain_is_bool(self, analyzer):
        result = analyzer.analyze("some problem")
        assert isinstance(result["certain"], bool)

    def test_all_scores_keys_match_classes(self, analyzer):
        result  = analyzer.analyze("some problem")
        classes = set(str(c) for c in analyzer._fake_le.classes_)
        assert set(result["all_scores"].keys()) == classes

    def test_all_scores_sum_roughly_100(self, analyzer):
        result = analyzer.analyze("classify images")
        total  = sum(result["all_scores"].values())
        assert abs(total - 100.0) < 1.0

    def test_type_from_pipeline_config(self, analyzer):
        result = analyzer.analyze("some image problem")
        assert isinstance(result["type"], str)
        assert len(result["type"]) > 0

    def test_input_size_and_num_classes_present(self, analyzer):
        result = analyzer.analyze("any problem")
        assert "input_size"  in result
        assert "num_classes" in result

    def test_pipelines_dict_has_four_domains(self, analyzer):
        assert set(analyzer.pipelines.keys()) == {
            "image", "medical", "text", "security"
        }

    def test_unknown_category_falls_back_to_image(self, analyzer):
        """pipelines.get() defaults to image when label unknown."""
        config = analyzer.pipelines.get("nonexistent",
                                         analyzer.pipelines["image"])
        assert config["type"] == "Image Classification"
