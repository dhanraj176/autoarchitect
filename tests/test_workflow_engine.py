# ============================================================
# Tests — WorkflowEngine
# api/workflow_engine.py
# ============================================================

import pytest
from api.workflow_engine import WorkflowEngine, MULTI_DOMAIN_PATTERNS


@pytest.fixture
def engine():
    return WorkflowEngine()


# ── build_workflow — single agent ────────────────────────

class TestBuildWorkflowSingle:

    def test_returns_dict(self, engine):
        result = engine.build_workflow("detect potholes", "image")
        assert isinstance(result, dict)

    def test_type_is_single_for_simple_problem(self, engine):
        result = engine.build_workflow("detect potholes", "image")
        assert result["type"] == "single"

    def test_agents_list_length_one(self, engine):
        result = engine.build_workflow("detect potholes", "image")
        assert len(result["agents"]) == 1

    def test_agents_contains_primary_domain(self, engine):
        result = engine.build_workflow("detect potholes", "image")
        assert "image" in result["agents"]

    def test_steps_is_list(self, engine):
        result = engine.build_workflow("detect potholes", "image")
        assert isinstance(result["steps"], list)

    def test_single_workflow_has_three_steps(self, engine):
        result = engine.build_workflow("classify spam", "text")
        assert len(result["steps"]) == 3

    def test_steps_mention_domain(self, engine):
        result = engine.build_workflow("detect potholes", "image")
        assert any("IMAGE" in s or "image" in s.lower()
                   for s in result["steps"])

    def test_unknown_domain_falls_back_to_primary(self, engine):
        result = engine.build_workflow("do something", "security")
        assert result["agents"] == ["security"]
        assert result["type"] == "single"


# ── build_workflow — multi agent ─────────────────────────

class TestBuildWorkflowMulti:

    def test_multi_domain_keywords_trigger_multi_type(self, engine):
        # "detect classify and" matches MULTI_DOMAIN_PATTERNS
        result = engine.build_workflow("detect classify and", "image")
        assert result["type"] == "multi"

    def test_xray_report_pattern_gives_image_medical(self, engine):
        result = engine.build_workflow("xray report analysis", "medical")
        assert result["type"] == "multi"
        assert "image" in result["agents"]
        assert "medical" in result["agents"]

    def test_phishing_detect_gives_text_security(self, engine):
        result = engine.build_workflow("phishing detect emails", "security")
        assert result["type"] == "multi"
        assert "text" in result["agents"]
        assert "security" in result["agents"]

    def test_multi_agents_list_longer_than_one(self, engine):
        result = engine.build_workflow("xray report analysis", "medical")
        assert len(result["agents"]) > 1

    def test_multi_workflow_has_fusion_and_evaluator_steps(self, engine):
        result = engine.build_workflow("xray report analysis", "medical")
        steps_text = " ".join(result["steps"]).lower()
        assert "fusion" in steps_text
        assert "evaluator" in steps_text or "cache" in steps_text

    def test_and_conjunction_triggers_multi_when_domains_detected(self, engine):
        result = engine.build_workflow(
            "detect image defects and classify text descriptions", "image")
        assert result["type"] == "multi"


# ── _detect_domains ───────────────────────────────────────

class TestDetectDomains:

    def test_returns_list(self, engine):
        result = engine._detect_domains("detect potholes", "image")
        assert isinstance(result, list)

    def test_primary_returned_when_no_match(self, engine):
        result = engine._detect_domains("totally ambiguous", "text")
        assert result == ["text"]

    def test_multi_pattern_match_overrides_primary(self, engine):
        result = engine._detect_domains("email fraud detection", "text")
        domains = result
        assert len(domains) > 1

    def test_scan_analyze_report_gives_multi(self, engine):
        result = engine._detect_domains("scan analyze report findings", "image")
        assert len(result) > 1

    def test_mri_detect_gives_image_medical(self, engine):
        result = engine._detect_domains("mri detect tumor", "medical")
        assert "image" in result
        assert "medical" in result

    def test_no_duplicate_domains(self, engine):
        for domains, keywords in MULTI_DOMAIN_PATTERNS:
            problem = " ".join(keywords)
            result = engine._detect_domains(problem, domains[0])
            assert len(result) == len(set(result)), \
                f"Duplicate domains for pattern {keywords}: {result}"
