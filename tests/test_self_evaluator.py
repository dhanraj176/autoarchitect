# ============================================================
# Tests — SelfEvaluator
# api/brain/self_evaluator.py
# ============================================================

import io
import json
import zipfile
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def evaluator(tmp_path):
    from api.brain.self_evaluator import SelfEvaluator
    with patch.object(SelfEvaluator, '_load_history', return_value=[]):
        ev = SelfEvaluator.__new__(SelfEvaluator)
        ev.eval_dir  = tmp_path / 'self_eval'
        ev.eval_log  = tmp_path / 'self_eval' / 'eval_log.jsonl'
        ev.history   = []
        ev.eval_dir.mkdir(parents=True, exist_ok=True)
    return ev


def _make_zip(files: dict) -> bytes:
    """Build an in-memory ZIP from {filename: content} dict."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


MINIMAL_NETWORK_PY = """
class AgentNetwork:
    def predict(self, x): pass
    def run(self, x): pass
    def status(self): pass
    memory = {}
import time
time.sleep(1)
"""

FULL_ZIP_FILES = {
    'run_network.py': 'import network',
    'network.py':     MINIMAL_NETWORK_PY,
    'requirements.txt': 'torch\nflask\n',
    'README.md':      '# Network\n',
    'agents/image_agent.py': 'class ImageAgent: pass',
    'models/agent_model.pth': b'\x00' * 10,
}


# ── evaluate() — corrupted / empty ZIP ────────────────────

class TestEvaluateCorruptZip:

    def test_empty_bytes_returns_low_score(self, evaluator):
        result = evaluator.evaluate(b'', "test", {}, "image")
        assert result["score"] < 65

    def test_corrupted_bytes_does_not_raise(self, evaluator):
        try:
            evaluator.evaluate(b'not a zip file', "test", {}, "image")
        except Exception as e:
            pytest.fail(f"evaluate() raised on corrupt zip: {e}")

    def test_corrupted_zip_returns_failed_grade(self, evaluator):
        result = evaluator.evaluate(b'garbage', "test", {}, "image")
        assert result["score"] < 65
        assert result.get("passed") is False


# ── evaluate() — valid ZIP ────────────────────────────────

class TestEvaluateValidZip:

    def test_returns_score_key(self, evaluator):
        zb = _make_zip(FULL_ZIP_FILES)
        result = evaluator.evaluate(zb, "detect potholes",
                                    {"agents": ["image"]}, "image")
        assert "score" in result

    def test_score_between_0_and_100(self, evaluator):
        zb = _make_zip(FULL_ZIP_FILES)
        result = evaluator.evaluate(zb, "detect potholes",
                                    {"agents": ["image"]}, "image")
        assert 0 <= result["score"] <= 100

    def test_result_has_grade(self, evaluator):
        zb = _make_zip(FULL_ZIP_FILES)
        result = evaluator.evaluate(zb, "test", {"agents": ["image"]}, "image")
        assert "grade" in result
        assert result["grade"] in (
            "excellent", "good", "acceptable", "needs_improvement")

    def test_result_has_passed_key(self, evaluator):
        zb = _make_zip(FULL_ZIP_FILES)
        result = evaluator.evaluate(zb, "test", {"agents": ["image"]}, "image")
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    def test_passed_true_when_score_above_threshold(self, evaluator):
        zb = _make_zip(FULL_ZIP_FILES)
        result = evaluator.evaluate(zb, "detect potholes",
                                    {"agents": ["image"]}, "image")
        if result["score"] >= 65:
            assert result["passed"] is True

    def test_feedback_is_list(self, evaluator):
        zb = _make_zip(FULL_ZIP_FILES)
        result = evaluator.evaluate(zb, "test", {"agents": ["image"]}, "image")
        assert isinstance(result.get("feedback", []), list)

    def test_elapsed_non_negative(self, evaluator):
        zb = _make_zip(FULL_ZIP_FILES)
        result = evaluator.evaluate(zb, "test", {"agents": ["image"]}, "image")
        assert result.get("elapsed", 0) >= 0


# ── _check_zip_integrity ──────────────────────────────────

class TestCheckZipIntegrity:

    def test_full_zip_scores_higher_than_empty(self, evaluator):
        full_zip = _make_zip(FULL_ZIP_FILES)
        empty_zip = _make_zip({})
        full_result  = evaluator._check_zip_integrity(full_zip,  ["image"])
        empty_result = evaluator._check_zip_integrity(empty_zip, ["image"])
        assert full_result["score"] >= empty_result["score"]

    def test_missing_required_file_reduces_score(self, evaluator):
        partial = dict(FULL_ZIP_FILES)
        del partial['run_network.py']
        zb_partial = _make_zip(partial)
        zb_full    = _make_zip(FULL_ZIP_FILES)

        score_partial = evaluator._check_zip_integrity(zb_partial, ["image"])["score"]
        score_full    = evaluator._check_zip_integrity(zb_full,    ["image"])["score"]
        assert score_partial < score_full

    def test_returns_score_and_feedback(self, evaluator):
        result = evaluator._check_zip_integrity(_make_zip(FULL_ZIP_FILES), ["image"])
        assert "score" in result
        assert "feedback" in result


# ── _check_network_completeness ───────────────────────────

class TestCheckNetworkCompleteness:

    def test_network_with_all_methods_scores_high(self, evaluator):
        zb = _make_zip({'network.py': MINIMAL_NETWORK_PY})
        result = evaluator._check_network_completeness(zb)
        assert result["score"] >= 50

    def test_empty_network_scores_low(self, evaluator):
        zb = _make_zip({'network.py': ''})
        result = evaluator._check_network_completeness(zb)
        assert result["score"] < 100

    def test_returns_improvements_list(self, evaluator):
        zb = _make_zip({'network.py': ''})
        result = evaluator._check_network_completeness(zb)
        assert isinstance(result.get("improvements", []), list)


# ── stats() ──────────────────────────────────────────────

class TestStats:

    def test_empty_history_returns_zeros(self, evaluator):
        stats = evaluator.stats()
        assert stats["total_evaluations"] == 0
        assert stats["avg_score"] == 0

    def test_stats_has_required_keys(self, evaluator):
        # Empty history only returns the 3 summary keys
        empty_stats = evaluator.stats()
        for key in ("total_evaluations", "avg_score", "pass_rate"):
            assert key in empty_stats
        # best_score / worst_score appear only when there is history
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr("network.py", "class AgentNetwork: pass")
        evaluator.evaluate(buf.getvalue(), "t", {}, "image")
        full_stats = evaluator.stats()
        for key in ("total_evaluations", "avg_score", "pass_rate",
                    "best_score", "worst_score"):
            assert key in full_stats
