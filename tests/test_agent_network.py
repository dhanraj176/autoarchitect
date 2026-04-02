# ============================================================
# Tests — AgentNetwork  (api/agents/agent_network.py)
# ============================================================

import pytest
import importlib.util as _iutil
from unittest.mock import patch, MagicMock

_NO_TORCH = _iutil.find_spec("torch") is None


# ── Helpers ───────────────────────────────────────────────────

def _make_network(tmp_path, name="test_network"):
    with patch("api.agents.agent_network.AGENTS_DIR", str(tmp_path)):
        from api.agents.agent_network import AgentNetwork
        return AgentNetwork(name=name, description="test")


def _make_mock_agent(label="cat", confidence=80.0, category="image"):
    agent = MagicMock()
    agent.agent_id = "agt_001"
    agent.category = category
    agent.problem  = "test problem"
    agent.total_predictions = 0
    agent._get_memory_accuracy = MagicMock(return_value=0.0)
    agent.predict  = MagicMock(return_value={
        "label": label, "confidence": confidence, "action": "alert"})
    agent.act      = MagicMock(return_value={})
    agent.remember = MagicMock()
    agent.stop     = MagicMock()
    agent.run_async = MagicMock(return_value=None)

    # Memory stub
    from unittest.mock import MagicMock as MM
    agent._memory = MM()
    agent._memory.get_all = MagicMock(return_value=[])
    return agent


# ── AgentNetwork construction ─────────────────────────────────

class TestAgentNetworkInit:

    def test_has_network_id(self, tmp_path):
        net = _make_network(tmp_path)
        assert isinstance(net.network_id, str)
        assert len(net.network_id) > 0

    def test_name_stored(self, tmp_path):
        net = _make_network(tmp_path, name="my_net")
        assert net.name == "my_net"

    def test_initially_no_agents(self, tmp_path):
        net = _make_network(tmp_path)
        assert len(net.agents) == 0

    def test_not_running_by_default(self, tmp_path):
        net = _make_network(tmp_path)
        assert net.is_running is False

    def test_total_runs_zero(self, tmp_path):
        net = _make_network(tmp_path)
        assert net.total_runs == 0


# ── add_agent ─────────────────────────────────────────────────

class TestAddAgent:

    def test_agent_added_to_dict(self, tmp_path):
        net   = _make_network(tmp_path)
        agent = _make_mock_agent()
        net.add_agent(agent, role="image")
        assert len(net.agents) == 1

    def test_agent_role_stored(self, tmp_path):
        net   = _make_network(tmp_path)
        agent = _make_mock_agent()
        net.add_agent(agent, role="medical")
        entry = list(net.agents.values())[0]
        assert entry["role"] == "medical"

    def test_legacy_agent_gets_agent_id(self, tmp_path):
        net = _make_network(tmp_path)
        # Agent without agent_id attribute
        legacy = MagicMock(spec=[])
        legacy.category = "image"
        net.add_agent(legacy, role="image")
        assert hasattr(legacy, "agent_id")

    def test_two_agents_stored(self, tmp_path):
        net    = _make_network(tmp_path)
        agent1 = _make_mock_agent(label="cat", category="image")
        agent2 = _make_mock_agent(label="dog", category="text")
        agent2.agent_id = "agt_002"
        net.add_agent(agent1)
        net.add_agent(agent2)
        assert len(net.agents) == 2


# ── add_pipeline ──────────────────────────────────────────────

class TestAddPipeline:

    def test_pipeline_added(self, tmp_path):
        net = _make_network(tmp_path)
        net.add_pipeline(["agt_001", "agt_002"], name="main")
        assert len(net.pipelines) == 1

    def test_pipeline_name_stored(self, tmp_path):
        net = _make_network(tmp_path)
        net.add_pipeline(["agt_001"], name="detection")
        assert net.pipelines[0]["name"] == "detection"

    def test_multiple_pipelines(self, tmp_path):
        net = _make_network(tmp_path)
        net.add_pipeline(["a"], name="p1")
        net.add_pipeline(["b"], name="p2")
        assert len(net.pipelines) == 2


# ── _combine_predictions ──────────────────────────────────────

class TestCombinePredictions:

    @pytest.fixture
    def net(self, tmp_path):
        return _make_network(tmp_path)

    def test_empty_predictions_returns_unknown(self, net):
        result = net._combine_predictions([])
        assert result["label"]     == "unknown"
        assert result["confidence"] == 0
        assert result["success"]   is False

    def test_single_prediction_passthrough(self, net):
        preds = [{"agent_id": "a1", "role": "image",
                  "prediction": {"label": "cat", "confidence": 90.0}}]
        result = net._combine_predictions(preds)
        assert result["label"]      == "cat"
        assert result["confidence"] == 90.0

    def test_majority_vote_wins(self, net):
        preds = [
            {"agent_id": "a1", "role": "image",
             "prediction": {"label": "cat", "confidence": 80.0}},
            {"agent_id": "a2", "role": "text",
             "prediction": {"label": "cat", "confidence": 70.0}},
            {"agent_id": "a3", "role": "medical",
             "prediction": {"label": "dog", "confidence": 90.0}},
        ]
        result = net._combine_predictions(preds)
        assert result["label"] == "cat"

    def test_all_votes_returned(self, net):
        preds = [
            {"agent_id": "a1", "role": "image",
             "prediction": {"label": "cat", "confidence": 80.0}},
            {"agent_id": "a2", "role": "text",
             "prediction": {"label": "dog", "confidence": 70.0}},
        ]
        result = net._combine_predictions(preds)
        assert "all_votes" in result
        assert "cat" in result["all_votes"]
        assert "dog" in result["all_votes"]

    def test_n_agents_count(self, net):
        preds = [
            {"agent_id": f"a{i}", "role": "image",
             "prediction": {"label": "cat", "confidence": 70.0}}
            for i in range(3)
        ]
        result = net._combine_predictions(preds)
        assert result["n_agents"] == 3


# ── info ──────────────────────────────────────────────────────

class TestNetworkInfo:

    def test_info_has_required_keys(self, tmp_path):
        net = _make_network(tmp_path)
        info = net.info()
        for k in ("network_id", "name", "agents", "total_runs",
                  "pipelines", "is_running", "created_at"):
            assert k in info

    def test_info_agents_dict(self, tmp_path):
        net   = _make_network(tmp_path)
        agent = _make_mock_agent()
        net.add_agent(agent, role="image")
        info  = net.info()
        assert isinstance(info["agents"], dict)
        assert len(info["agents"]) == 1


# ── stop ──────────────────────────────────────────────────────

class TestNetworkStop:

    def test_stop_sets_is_running_false(self, tmp_path):
        net   = _make_network(tmp_path)
        agent = _make_mock_agent()
        net.add_agent(agent)
        net.is_running = True
        net.stop()
        assert net.is_running is False

    def test_stop_calls_stop_on_agents(self, tmp_path):
        net   = _make_network(tmp_path)
        agent = _make_mock_agent()
        net.add_agent(agent)
        net.stop()
        agent.stop.assert_called_once()


# ── build_network_from_problem ────────────────────────────────

@pytest.mark.skipif(_NO_TORCH, reason="requires torch")
class TestBuildNetworkFromProblem:

    def test_returns_agent_network(self, tmp_path):
        with patch("api.agents.agent_network.AGENTS_DIR", str(tmp_path)), \
             patch("api.agents.image_agent.run_quick_nas",
                   return_value={"architecture": [], "parameters": 0,
                                 "search_time": 0}):
            from api.agents.agent_network import build_network_from_problem
            from api.agents.agent_network import AgentNetwork
            net = build_network_from_problem(
                "detect potholes", ["image"])
        assert isinstance(net, AgentNetwork)

    def test_network_has_one_agent_per_domain(self, tmp_path):
        with patch("api.agents.agent_network.AGENTS_DIR", str(tmp_path)), \
             patch("api.agents.image_agent.run_quick_nas",
                   return_value={"architecture": [], "parameters": 0,
                                 "search_time": 0}), \
             patch("api.agents.text_agent.run_quick_nas",
                   return_value={"architecture": [], "parameters": 0,
                                 "search_time": 0}):
            from api.agents.agent_network import build_network_from_problem
            net = build_network_from_problem(
                "classify and detect", ["image", "text"])
        assert len(net.agents) == 2

    def test_multi_domain_network_has_pipeline(self, tmp_path):
        with patch("api.agents.agent_network.AGENTS_DIR", str(tmp_path)), \
             patch("api.agents.image_agent.run_quick_nas",
                   return_value={"architecture": [], "parameters": 0,
                                 "search_time": 0}), \
             patch("api.agents.text_agent.run_quick_nas",
                   return_value={"architecture": [], "parameters": 0,
                                 "search_time": 0}):
            from api.agents.agent_network import build_network_from_problem
            net = build_network_from_problem(
                "multi-domain task", ["image", "text"])
        assert len(net.pipelines) >= 1
