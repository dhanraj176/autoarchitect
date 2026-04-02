# ============================================================
# Tests — Agents (FusionAgent, EvaluatorAgent)
# api/agents/fusion_agent.py
# api/agents/evaluator_agent.py
# ============================================================

import pytest


# ============================================================
# FusionAgent
# ============================================================

class TestFusionAgentFuse:

    @pytest.fixture
    def agent(self):
        from api.agents.fusion_agent import FusionAgent
        return FusionAgent()

    @pytest.fixture
    def single_result(self):
        return [{
            "domain": "image",
            "architecture": [
                {"cell": 1, "operations": [
                    {"operation": "conv3x3", "confidence": 80.0,
                     "weights": {"skip": 0.1, "conv3x3": 0.8,
                                 "conv5x5": 0.05, "maxpool": 0.03,
                                 "avgpool": 0.02}}
                ]}
            ],
            "parameters": 100000,
            "accuracy": 74.0,
            "status": "success",
        }]

    @pytest.fixture
    def multi_result(self):
        return [
            {
                "domain": "image",
                "architecture": [
                    {"cell": 1, "operations": [
                        {"operation": "conv3x3", "confidence": 80.0,
                         "weights": {"skip": 0.1, "conv3x3": 0.8,
                                     "conv5x5": 0.05, "maxpool": 0.03,
                                     "avgpool": 0.02}}
                    ]}
                ],
                "parameters": 100000,
                "accuracy": 74.0,
                "status": "success",
            },
            {
                "domain": "text",
                "architecture": [
                    {"cell": 1, "operations": [
                        {"operation": "conv5x5", "confidence": 90.0,
                         "weights": {"skip": 0.01, "conv3x3": 0.05,
                                     "conv5x5": 0.90, "maxpool": 0.02,
                                     "avgpool": 0.02}}
                    ]}
                ],
                "parameters": 80000,
                "accuracy": 82.0,
                "status": "success",
            }
        ]

    def test_empty_results_returns_error(self, agent):
        result = agent.fuse([], "test problem")
        assert "error" in result

    def test_single_result_returns_success(self, agent, single_result):
        result = agent.fuse(single_result, "detect potholes")
        assert result["status"] == "success"

    def test_multi_result_returns_success(self, agent, multi_result):
        result = agent.fuse(multi_result, "classify and detect")
        assert result["status"] == "success"

    def test_fused_result_has_required_keys(self, agent, multi_result):
        result = agent.fuse(multi_result, "classify and detect")
        for key in ("status", "architecture", "parameters", "domains_combined"):
            assert key in result, f"Missing key: {key}"

    def test_domains_combined_lists_all_inputs(self, agent, multi_result):
        result = agent.fuse(multi_result, "classify and detect")
        domains = result.get("domains_combined", [])
        assert "image" in domains
        assert "text" in domains

    def test_fused_architecture_is_list(self, agent, multi_result):
        result = agent.fuse(multi_result, "test")
        assert isinstance(result["architecture"], list)

    def test_fused_architecture_longer_than_single(self, agent, multi_result):
        """Fusing 2 results should produce more cells than either alone."""
        result = agent.fuse(multi_result, "test")
        combined_cells = len(multi_result[0]["architecture"]) + \
                         len(multi_result[1]["architecture"])
        # Fusion adds a fusion layer, so arch >= combined
        assert len(result["architecture"]) >= combined_cells

    def test_type_is_multi_agent_fusion(self, agent, multi_result):
        result = agent.fuse(multi_result, "test")
        assert result.get("type") == "multi_agent_fusion"

    def test_elapsed_is_positive(self, agent, multi_result):
        result = agent.fuse(multi_result, "test")
        assert result.get("elapsed", 0) >= 0


# ── _find_best_ops ─────────────────────────────────────────

class TestFindBestOps:

    def test_returns_most_common_op(self):
        from api.agents.fusion_agent import FusionAgent
        agent = FusionAgent()
        results = [
            {"architecture": [{"cell": 1, "operations": [
                {"operation": "conv3x3", "confidence": 80.0, "weights": {}}
            ]}]},
            {"architecture": [{"cell": 1, "operations": [
                {"operation": "conv3x3", "confidence": 70.0, "weights": {}}
            ]}]},
            {"architecture": [{"cell": 1, "operations": [
                {"operation": "conv5x5", "confidence": 90.0, "weights": {}}
            ]}]},
        ]
        best = agent._find_best_ops(results)
        assert best == "conv3x3"

    def test_defaults_to_conv5x5_on_empty(self):
        from api.agents.fusion_agent import FusionAgent
        agent = FusionAgent()
        assert agent._find_best_ops([]) == "conv5x5"


# ============================================================
# EvaluatorAgent
# ============================================================

class TestEvaluatorAgentEvaluate:

    @pytest.fixture
    def evaluator(self):
        from api.agents.evaluator_agent import EvaluatorAgent
        return EvaluatorAgent()

    @pytest.fixture
    def fusion_result(self):
        return {
            "status": "success",
            "type": "multi_agent_fusion",
            "domains_combined": ["image", "text"],
            "architecture": [
                {"cell": 1, "operations": [
                    {"operation": "conv3x3", "confidence": 80.0}
                ]},
                {"cell": 2, "operations": [
                    {"operation": "conv5x5", "confidence": 85.0}
                ]},
            ],
            "parameters": 120000,
            "accuracy": 78.0,
        }

    def test_returns_success_status(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        assert result["status"] == "success"

    def test_result_has_required_keys(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        for key in ("status", "scores", "avg_score", "verdict", "feedback"):
            assert key in result

    def test_scores_dict_has_all_metrics(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        for metric in ("complexity", "coverage", "depth", "diversity", "innovation"):
            assert metric in result["scores"]

    def test_all_scores_between_0_and_100(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        for metric, score in result["scores"].items():
            assert 0 <= score <= 100, \
                f"{metric} score {score} out of range [0, 100]"

    def test_avg_score_between_0_and_100(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        assert 0 <= result["avg_score"] <= 100

    def test_verdict_is_valid_string(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        assert result["verdict"] in ("excellent", "good", "needs_improvement")

    def test_excellent_verdict_at_high_score(self, evaluator):
        """A large multi-domain architecture should score excellent."""
        big_result = {
            "status": "success",
            "domains_combined": ["image", "text", "medical"],
            "architecture": [{"cell": i, "operations": [
                {"operation": op, "confidence": 90.0}
                for op in ("conv3x3", "conv5x5", "maxpool", "avgpool")
            ]} for i in range(1, 8)],
            "parameters": 100000,
            "accuracy": 90.0,
        }
        result = evaluator.evaluate(big_result, "multi-domain detection")
        assert result["avg_score"] >= 70

    def test_ready_to_cache_true_when_score_above_threshold(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        if result["avg_score"] >= 70:
            assert result["ready_to_cache"] is True

    def test_feedback_is_list(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        assert isinstance(result["feedback"], list)

    def test_elapsed_is_non_negative(self, evaluator, fusion_result):
        result = evaluator.evaluate(fusion_result, "detect potholes")
        assert result.get("elapsed", 0) >= 0
