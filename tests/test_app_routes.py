# ============================================================
# Tests — Flask App Routes
# app.py
# ============================================================

import json
import pytest
pytest.importorskip("torch")
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def client():
    """Flask test client with all heavy deps mocked at import time."""
    import sys

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = {
        "category":   "image",
        "confidence": 0.95,
        "domain":     "image",
        "method":     "bert",
    }

    mock_orc = MagicMock()
    mock_orc.solve.return_value = {
        "status":   "success",
        "domain":   "image",
        "type":     "single",
        "agents_used": ["image"],
        "architecture": [],
        "parameters":   105910,
        "search_time":  1.2,
        "avg_accuracy": 74.5,
        "elapsed":      1.5,
        "message":      "Done",
        "readable_output": {},
        "topology":        {},
    }

    mock_cache_stats = {
        "total_cached": 5, "categories": {},
        "total_uses": 10, "avg_accuracy": 74.5,
    }

    mock_nas_result = {
        "architecture": [
            {"cell": 1, "operations": [
                {"operation": "conv3x3", "confidence": 80.0,
                 "weights": {"skip": 0.1, "conv3x3": 0.8,
                             "conv5x5": 0.05, "maxpool": 0.03,
                             "avgpool": 0.02}}]}
        ],
        "parameters":   105910,
        "search_time":  1.2,
        "status":       "success",
    }

    # Pre-import modules so patch() can resolve targets
    import api.analyzer, api.orchestrator, api.cache_manager, api.nas_engine

    with patch('api.analyzer.ProblemAnalyzer',
               return_value=mock_analyzer), \
         patch('api.orchestrator.AutoArchitectOrchestrator',
               return_value=mock_orc), \
         patch('api.cache_manager.get_cache_stats',
               return_value=mock_cache_stats), \
         patch('api.cache_manager.check_cache',
               return_value={"found": False}), \
         patch('api.cache_manager.save_to_cache'), \
         patch('api.cache_manager.increment_use_count'), \
         patch('api.cache_manager.find_similar_cached',
               return_value=None), \
         patch('api.nas_engine.run_quick_nas',
               return_value=mock_nas_result):

        import autoarchitect.app as app_module
        app_module.analyzer     = mock_analyzer
        app_module.orchestrator = mock_orc
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c


# ── /api/analyze ──────────────────────────────────────────

class TestAnalyzeRoute:

    def test_returns_200(self, client):
        resp = client.post(
            "/api/analyze",
            data=json.dumps({"problem": "detect potholes"}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_returns_json(self, client):
        resp = client.post(
            "/api/analyze",
            data=json.dumps({"problem": "detect potholes"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert isinstance(data, dict)

    def test_response_has_category(self, client):
        resp = client.post(
            "/api/analyze",
            data=json.dumps({"problem": "detect potholes"}),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert "category" in data

    def test_empty_problem_returns_400(self, client):
        resp = client.post(
            "/api/analyze",
            data=json.dumps({"problem": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_missing_problem_returns_400(self, client):
        resp = client.post(
            "/api/analyze",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ── /api/cache/stats ──────────────────────────────────────

class TestCacheStatsRoute:

    def test_returns_200(self, client):
        resp = client.get("/api/cache/stats")
        assert resp.status_code == 200

    def test_returns_json(self, client):
        data = json.loads(client.get("/api/cache/stats").data)
        assert isinstance(data, dict)

    def test_response_has_total_cached(self, client):
        data = json.loads(client.get("/api/cache/stats").data)
        assert "total_cached" in data


# ── /api/search ───────────────────────────────────────────

class TestSearchRoute:

    def test_returns_200_with_valid_payload(self, client):
        resp = client.post(
            "/api/search",
            data=json.dumps({
                "problem":     "detect potholes",
                "category":    "image",
                "num_classes": 10,
                "confidence":  0.95,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_response_has_status_success(self, client):
        resp = client.post(
            "/api/search",
            data=json.dumps({
                "problem":  "detect potholes",
                "category": "image",
            }),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert data.get("status") == "success"

    def test_response_has_architecture(self, client):
        resp = client.post(
            "/api/search",
            data=json.dumps({
                "problem":  "detect potholes",
                "category": "image",
            }),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert "architecture" in data

    def test_response_has_parameters(self, client):
        resp = client.post(
            "/api/search",
            data=json.dumps({
                "problem":  "detect potholes",
                "category": "image",
            }),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert "parameters" in data


# ── /api/explain ──────────────────────────────────────────

class TestExplainRoute:

    def test_returns_200(self, client):
        resp = client.post(
            "/api/explain",
            data=json.dumps({
                "problem":      "detect potholes",
                "architecture": [],
                "parameters":   105910,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_response_has_explanation_key(self, client):
        resp = client.post(
            "/api/explain",
            data=json.dumps({
                "problem":      "detect potholes",
                "architecture": [],
                "parameters":   105910,
            }),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert "explanation" in data

    def test_explanation_is_string(self, client):
        resp = client.post(
            "/api/explain",
            data=json.dumps({
                "problem":      "detect potholes",
                "architecture": [],
                "parameters":   105910,
            }),
            content_type="application/json",
        )
        data = json.loads(resp.data)
        assert isinstance(data["explanation"], str)
