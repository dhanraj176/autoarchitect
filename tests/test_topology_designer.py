# ============================================================
# Tests — TopologyDesigner
# api/brain/topology_designer.py
# ============================================================

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def designer(tmp_path):
    from api.brain.topology_designer import TopologyDesigner
    with patch.object(TopologyDesigner, '_load_history', return_value=[]):
        td = TopologyDesigner.__new__(TopologyDesigner)
        td.data_dir     = tmp_path
        td.topology_log = tmp_path / 'topology_history.json'
        td.history      = []
    return td


# ── Path resolution (Fix #3) ──────────────────────────────

class TestPathResolution:

    def test_data_dir_built_from_file_not_cwd(self):
        """data_dir must be __file__-relative, not CWD-relative."""
        import api.brain.topology_designer as mod
        source = Path(mod.__file__).read_text(encoding='utf-8')
        assert '__file__' in source, \
            "__file__ not used for path resolution — Fix #3 regression"
        assert 'Path("brain_data")' not in source, \
            'Hardcoded relative Path("brain_data") still present'

    def test_data_dir_ends_with_brain_data(self):
        from api.brain.topology_designer import TopologyDesigner
        with patch.object(TopologyDesigner, '_load_history', return_value=[]):
            with patch('pathlib.Path.mkdir'):
                td = TopologyDesigner()
        assert td.data_dir.name == "brain_data"

    def test_data_dir_is_absolute(self):
        from api.brain.topology_designer import TopologyDesigner
        with patch.object(TopologyDesigner, '_load_history', return_value=[]):
            with patch('pathlib.Path.mkdir'):
                td = TopologyDesigner()
        assert td.data_dir.is_absolute()


# ── design() ─────────────────────────────────────────────

class TestDesign:

    def test_returns_dict(self, designer):
        result = designer.design("detect potholes", "image")
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, designer):
        result = designer.design("classify spam", "text")
        for key in ("agents", "topology", "connections"):
            assert key in result, f"Missing key: {key}"

    def test_agents_is_list(self, designer):
        result = designer.design("detect fraud", "security")
        assert isinstance(result["agents"], list)

    def test_agents_list_not_empty(self, designer):
        result = designer.design("detect potholes", "image")
        assert len(result["agents"]) > 0

    def test_domain_agent_included(self, designer):
        """The primary domain agent must always appear in the topology."""
        for domain in ("image", "text", "medical", "security"):
            result = designer.design(f"solve a {domain} problem", domain)
            agent_names = [a.get("name", a) if isinstance(a, dict) else a
                           for a in result["agents"]]
            assert any(domain in str(a).lower() for a in agent_names), \
                f"Domain '{domain}' agent missing from topology"

    def test_topology_type_is_valid_string(self, designer):
        result = designer.design("detect something", "image")
        assert isinstance(result["topology"], str)
        assert len(result["topology"]) > 0

    def test_connections_is_list(self, designer):
        result = designer.design("classify text", "text")
        assert isinstance(result["connections"], list)

    def test_does_not_raise_on_unknown_domain(self, designer):
        try:
            designer.design("some problem", "unknown_domain")
        except Exception as e:
            pytest.fail(f"design() raised on unknown domain: {e}")


# ── avg_accuracy type (Fix type inconsistency) ────────────

class TestAvgAccuracyType:

    def test_avg_accuracy_is_float_when_history_empty(self, designer):
        """topology stats avg_accuracy must be float, not int 0."""
        if hasattr(designer, 'get_stats'):
            stats = designer.get_stats()
            avg = stats.get("avg_accuracy", 0.0)
            assert isinstance(avg, float), \
                f"avg_accuracy is {type(avg).__name__}, expected float"
