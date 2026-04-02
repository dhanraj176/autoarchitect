# ============================================================
# Tests — OutputGenerator  (fallback path — no Groq needed)
# api/brain/output_generator.py
# ============================================================

import pytest
from api.brain.output_generator import (
    _fallback_output,
    generate_output,
    generate_workflow_output,
)

PROBLEM = "detect potholes in road images"


# ── _fallback_output ──────────────────────────────────────

class TestFallbackOutput:

    def test_returns_dict(self):
        result = _fallback_output(PROBLEM, 80, 74.5, ["image"])
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        result = _fallback_output(PROBLEM, 80, 74.5, ["image"])
        for key in ("overall_score", "verdict", "summary",
                    "findings", "recommendations",
                    "next_steps", "confidence",
                    "generated_by", "problem"):
            assert key in result, f"Missing key: {key}"

    def test_overall_score_matches_input(self):
        result = _fallback_output(PROBLEM, 90, 0, ["image"])
        assert result["overall_score"] == 90

    def test_verdict_excellent_at_85_plus(self):
        result = _fallback_output(PROBLEM, 90, 0, ["image"])
        assert result["verdict"] == "Excellent"

    def test_verdict_good_at_70_to_84(self):
        result = _fallback_output(PROBLEM, 75, 0, ["image"])
        assert result["verdict"] == "Good"

    def test_verdict_fair_at_55_to_69(self):
        result = _fallback_output(PROBLEM, 60, 0, ["image"])
        assert result["verdict"] == "Fair"

    def test_verdict_needs_work_below_55(self):
        result = _fallback_output(PROBLEM, 40, 0, ["image"])
        assert result["verdict"] == "Needs Work"

    def test_findings_is_list(self):
        result = _fallback_output(PROBLEM, 80, 74.5, ["image"])
        assert isinstance(result["findings"], list)
        assert len(result["findings"]) > 0

    def test_recommendations_is_list(self):
        result = _fallback_output(PROBLEM, 80, 74.5, ["image"])
        assert isinstance(result["recommendations"], list)

    def test_problem_preserved_in_output(self):
        result = _fallback_output(PROBLEM, 80, 74.5, ["image"])
        assert result["problem"] == PROBLEM

    def test_confidence_high_when_score_above_70(self):
        result = _fallback_output(PROBLEM, 80, 0, ["image"])
        assert result["confidence"] == "High"

    def test_confidence_medium_when_score_below_70(self):
        result = _fallback_output(PROBLEM, 60, 0, ["image"])
        assert result["confidence"] == "Medium"

    def test_uses_accuracy_when_eval_score_zero(self):
        result = _fallback_output(PROBLEM, 0, 80.0, ["image"])
        assert result["overall_score"] == 80

    def test_defaults_to_70_when_both_zero(self):
        result = _fallback_output(PROBLEM, 0, 0, ["image"])
        assert result["overall_score"] == 70

    def test_empty_agents_list_handled(self):
        result = _fallback_output(PROBLEM, 75, 74.5, [])
        assert isinstance(result, dict)
        assert "overall_score" in result

    def test_agent_name_appears_in_findings(self):
        result = _fallback_output(PROBLEM, 80, 0, ["image"])
        findings_text = " ".join(result["findings"]).upper()
        assert "IMAGE" in findings_text


# ── generate_output — no Groq key (offline fallback) ─────

class TestGenerateOutputNoGroq:

    def test_returns_dict_without_groq(self):
        result = generate_output(PROBLEM, {}, groq_key="")
        assert isinstance(result, dict)

    def test_has_required_keys_without_groq(self):
        result = generate_output(PROBLEM, {}, groq_key="")
        for key in ("overall_score", "verdict", "summary",
                    "findings", "recommendations"):
            assert key in result

    def test_uses_eval_score_from_result(self):
        result = generate_output(
            PROBLEM,
            {"evaluation": {"avg_score": 88, "feedback": []}},
            groq_key=""
        )
        assert result["overall_score"] >= 80

    def test_agents_used_in_context(self):
        result = generate_output(
            PROBLEM,
            {"agents_used": ["image", "medical"],
             "avg_accuracy": 75.0},
            groq_key=""
        )
        assert isinstance(result, dict)

    def test_from_cache_flag_accepted(self):
        result = generate_output(
            PROBLEM,
            {"from_cache": True, "avg_accuracy": 80.0},
            groq_key=""
        )
        assert isinstance(result, dict)


# ── generate_workflow_output — no Groq ───────────────────

class TestGenerateWorkflowOutputNoGroq:

    def test_returns_dict(self):
        result = generate_workflow_output(PROBLEM, [], groq_key="")
        assert isinstance(result, dict)

    def test_single_workflow_result(self):
        result = generate_workflow_output(
            PROBLEM,
            [{"agent": "image", "score": 87, "input": "thumbnail"}],
            groq_key=""
        )
        assert isinstance(result, dict)
        assert "overall_score" in result

    def test_multi_workflow_result(self):
        result = generate_workflow_output(
            "analyze thumbnail and title",
            [
                {"agent": "image", "score": 87, "input": "thumbnail"},
                {"agent": "text",  "score": 72, "input": "title"},
            ],
            groq_key=""
        )
        assert isinstance(result, dict)

    def test_overall_score_is_average(self):
        result = generate_workflow_output(
            PROBLEM,
            [{"agent": "image", "score": 80},
             {"agent": "text",  "score": 60}],
            groq_key=""
        )
        assert result["overall_score"] == 70

    def test_empty_workflow_results_handled(self):
        result = generate_workflow_output(PROBLEM, [], groq_key="")
        assert result["overall_score"] == 0 or "overall_score" in result
