# ============================================================
# Tests — Orchestrator (pure logic — no BERT/torch needed)
# api/orchestrator.py
# ============================================================

import pytest
import sys
from unittest.mock import patch, MagicMock

# orchestrator.py → analyzer.py → torch at module level; stub the chain
_torch_stub = MagicMock()
with patch.dict(sys.modules, {
    'torch': _torch_stub,
    'torch.nn': MagicMock(),
    'transformers': MagicMock(),
    'api.analyzer': MagicMock(),
}):
    from api.orchestrator import AutoArchitectOrchestrator, LLM_KEYWORDS


# ── Fixture — minimal orchestrator with everything patched ─

@pytest.fixture
def orc():
    """Orchestrator with all heavy dependencies mocked out."""
    with patch('api.orchestrator.ProblemAnalyzer'), \
         patch('api.orchestrator.WorkflowEngine'), \
         patch('api.orchestrator.check_cache',
               return_value={"found": False}), \
         patch('api.orchestrator.find_similar_cached',
               return_value=None):
        o = AutoArchitectOrchestrator.__new__(AutoArchitectOrchestrator)
        import threading
        o._agents        = {}
        o._agents_lock   = threading.Lock()
        o.groq_key       = ""
        o.brain_enabled         = False
        o.brain                 = None
        o.topology_enabled      = False
        o.topology_designer     = None
        o.network_zip_enabled   = False
        o.network_zip           = None
        o.researcher_enabled    = False
        o.researcher            = None
        o.self_evaluator_enabled = False
        o.self_evaluator        = None
        o.output_enabled        = False
        o._generate_output      = None
        o._last_workflow_result = {}
        o._last_problem         = ""
        o._last_topology        = {}
        o._last_research        = {}
        o._last_eval_result     = {}
        o._last_embedding       = None
    return o


# ── _needs_llm ────────────────────────────────────────────

class TestNeedsLlm:

    def test_write_a_triggers_llm(self, orc):
        assert orc._needs_llm("write a poem about the sea") is True

    def test_write_me_triggers_llm(self, orc):
        assert orc._needs_llm("write me a story") is True

    def test_pros_and_cons_triggers_llm(self, orc):
        assert orc._needs_llm("pros and cons of electric cars") is True

    def test_detect_potholes_does_not_trigger_llm(self, orc):
        assert orc._needs_llm("detect potholes in road images") is False

    def test_classify_spam_does_not_trigger_llm(self, orc):
        assert orc._needs_llm("classify spam emails") is False

    def test_medical_problem_does_not_trigger_llm(self, orc):
        assert orc._needs_llm("detect tumor in MRI scan") is False

    def test_case_insensitive(self, orc):
        assert orc._needs_llm("WRITE A POEM") is True

    def test_all_llm_keywords_trigger(self, orc):
        for kw in LLM_KEYWORDS:
            assert orc._needs_llm(kw + " test input") is True, \
                f"LLM keyword '{kw}' did not trigger"


# ── _run_llm — no Groq key (fallback) ────────────────────

class TestRunLlmNoGroq:

    def test_returns_dict(self, orc):
        result = orc._run_llm("write a poem about AI")
        assert isinstance(result, dict)

    def test_status_is_success(self, orc):
        result = orc._run_llm("write a poem about AI")
        assert result["status"] == "success"

    def test_type_is_llm_generation(self, orc):
        result = orc._run_llm("write a story")
        assert result["type"] == "llm_generation"

    def test_output_is_string(self, orc):
        result = orc._run_llm("write a poem")
        assert isinstance(result["output"], str)
        assert len(result["output"]) > 0

    def test_model_is_fallback(self, orc):
        result = orc._run_llm("write a poem")
        assert result["model"] == "fallback"


# ── LLM_KEYWORDS sanity ───────────────────────────────────

class TestLlmKeywords:

    def test_keywords_is_list(self):
        assert isinstance(LLM_KEYWORDS, list)

    def test_keywords_not_empty(self):
        assert len(LLM_KEYWORDS) > 0

    def test_all_keywords_are_lowercase_strings(self):
        for kw in LLM_KEYWORDS:
            assert isinstance(kw, str)
            assert kw == kw.lower(), \
                f"LLM keyword '{kw}' is not lowercase"


# ── _sleep_agent ──────────────────────────────────────────

class TestSleepAgent:

    def test_removes_agent_by_key(self, orc):
        orc._agents["image_detect potholes"] = MagicMock()
        orc._sleep_agent("image", "detect potholes")
        assert "image_detect potholes" not in orc._agents

    def test_removes_agent_by_domain_fallback(self, orc):
        orc._agents["image"] = MagicMock()
        orc._sleep_agent("image")
        assert "image" not in orc._agents

    def test_no_raise_when_agent_not_present(self, orc):
        try:
            orc._sleep_agent("image", "nonexistent problem")
        except Exception as e:
            pytest.fail(f"_sleep_agent raised: {e}")

    def test_other_agents_not_affected(self, orc):
        orc._agents["text"] = MagicMock()
        orc._agents["image"] = MagicMock()
        orc._sleep_agent("image")
        assert "text" in orc._agents


# ── _wake_agent ───────────────────────────────────────────

class TestWakeAgent:

    def test_returns_existing_agent_without_recreating(self, orc):
        fake = MagicMock()
        orc._agents["image"] = fake
        with patch('api.orchestrator.AutoArchitectOrchestrator'
                   '._wake_agent', wraps=orc._wake_agent):
            result = orc._wake_agent("image")
        assert result is fake

    def test_creates_agent_when_not_present(self, orc):
        fake_agent = MagicMock()
        fake_factory = MagicMock()
        fake_factory.create.return_value = fake_agent
        orc._agents = {}
        with patch('api.agents.agent_factory.get_factory',
                   return_value=fake_factory):
            result = orc._wake_agent("image", "detect potholes")
        assert result is fake_agent
