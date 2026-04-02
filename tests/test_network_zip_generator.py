# ============================================================
# Tests — NetworkZipGenerator
# api/brain/network_zip_generator.py
# ============================================================

import io
import json
import zipfile
import pytest
pytest.importorskip("torch")
from unittest.mock import patch, MagicMock
from pathlib import Path


@pytest.fixture
def generator():
    from api.brain.network_zip_generator import NetworkZipGenerator
    return NetworkZipGenerator()


@pytest.fixture
def simple_topology():
    return {
        "agents":      ["image"],
        "topology":    "sequential",
        "connections": [],
    }


@pytest.fixture
def multi_topology():
    return {
        "agents":      ["image", "text"],
        "topology":    "sequential",
        "connections": [{"from": "image", "to": "text", "type": "output"}],
    }


def _open_zip(raw: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(raw))


# ── generate() — ZIP structure ────────────────────────────

class TestGenerateZipStructure:

    def test_returns_bytes(self, generator, simple_topology):
        result = generator.generate("detect potholes", simple_topology)
        assert isinstance(result, bytes)

    def test_result_is_valid_zip(self, generator, simple_topology):
        raw = generator.generate("detect potholes", simple_topology)
        assert zipfile.is_zipfile(io.BytesIO(raw))

    def test_zip_contains_network_py(self, generator, simple_topology):
        raw   = generator.generate("detect potholes", simple_topology)
        names = _open_zip(raw).namelist()
        assert "network.py" in names

    def test_zip_contains_run_network_py(self, generator, simple_topology):
        raw   = generator.generate("detect potholes", simple_topology)
        names = _open_zip(raw).namelist()
        assert "run_network.py" in names

    def test_zip_contains_api_server_py(self, generator, simple_topology):
        raw   = generator.generate("detect potholes", simple_topology)
        names = _open_zip(raw).namelist()
        assert "api_server.py" in names

    def test_zip_contains_readme(self, generator, simple_topology):
        raw   = generator.generate("detect potholes", simple_topology)
        names = _open_zip(raw).namelist()
        assert "README.md" in names

    def test_zip_contains_requirements(self, generator, simple_topology):
        raw   = generator.generate("detect potholes", simple_topology)
        names = _open_zip(raw).namelist()
        assert "requirements.txt" in names

    def test_zip_contains_agent_file(self, generator, simple_topology):
        raw   = generator.generate("detect potholes", simple_topology)
        names = _open_zip(raw).namelist()
        agent_files = [n for n in names if n.startswith("agents/")]
        assert len(agent_files) >= 1

    def test_multi_topology_has_multiple_agents(
            self, generator, multi_topology):
        raw   = generator.generate("detect and classify", multi_topology)
        names = _open_zip(raw).namelist()
        agent_files = [n for n in names if n.startswith("agents/")]
        assert len(agent_files) >= 2

    def test_unknown_agents_default_to_image(self, generator):
        topo = {"agents": ["unknown_xyz"], "topology": "sequential"}
        raw  = generator.generate("some problem", topo)
        assert zipfile.is_zipfile(io.BytesIO(raw))

    def test_empty_agents_defaults_to_image(self, generator):
        topo = {"agents": [], "topology": "sequential"}
        raw  = generator.generate("some problem", topo)
        assert zipfile.is_zipfile(io.BytesIO(raw))


# ── network.py content ────────────────────────────────────

class TestNetworkPyContent:

    def test_network_py_contains_agent_network_class(
            self, generator, simple_topology):
        raw     = generator.generate("detect potholes", simple_topology)
        content = _open_zip(raw).read("network.py").decode()
        assert "class AgentNetwork" in content

    def test_network_py_contains_predict_method(
            self, generator, simple_topology):
        raw     = generator.generate("detect potholes", simple_topology)
        content = _open_zip(raw).read("network.py").decode()
        assert "def predict" in content

    def test_network_py_contains_run_method(
            self, generator, simple_topology):
        raw     = generator.generate("detect potholes", simple_topology)
        content = _open_zip(raw).read("network.py").decode()
        assert "def run" in content

    def test_network_py_contains_status_method(
            self, generator, simple_topology):
        raw     = generator.generate("detect potholes", simple_topology)
        content = _open_zip(raw).read("network.py").decode()
        assert "def status" in content

    def test_parallel_topology_uses_threadpoolexecutor(
            self, generator):
        topo    = {"agents": ["image", "text"], "topology": "parallel"}
        raw     = generator.generate("detect and classify", topo)
        content = _open_zip(raw).read("network.py").decode()
        assert "ThreadPoolExecutor" in content


# ── requirements.txt content ─────────────────────────────

class TestRequirementsTxtContent:

    def test_requirements_contains_torch(
            self, generator, simple_topology):
        raw     = generator.generate("detect potholes", simple_topology)
        content = _open_zip(raw).read("requirements.txt").decode()
        assert "torch" in content

    def test_requirements_contains_flask(
            self, generator, simple_topology):
        raw     = generator.generate("detect potholes", simple_topology)
        content = _open_zip(raw).read("requirements.txt").decode().lower()
        assert "flask" in content


# ── _find_trained_model ───────────────────────────────────

class TestFindTrainedModel:

    def test_returns_none_when_no_model_exists(self, generator, tmp_path):
        mp, meta = generator._find_trained_model(
            "some novel problem xyz", "image")
        assert mp is None or not Path(mp).exists()

    def test_meta_is_dict(self, generator):
        _, meta = generator._find_trained_model(
            "some novel problem xyz", "image")
        assert isinstance(meta, dict)

    def test_uses_provided_trained_models_dict(self, generator, tmp_path):
        fake_model = tmp_path / "model.pth"
        fake_model.write_bytes(b"\x00" * 16)
        mp, meta = generator._find_trained_model(
            "detect potholes", "image",
            trained_models={"image": str(fake_model)})
        assert mp == str(fake_model)


# ── _generate_real_agent_named ────────────────────────────

class TestGenerateRealAgentNamed:

    def test_resnet_code_for_image_domain(self, generator):
        code = generator._generate_real_agent_named(
            class_name  = "PotholeDetectorAgent",
            agent_name  = "pothole_detector",
            domain      = "image",
            problem     = "detect potholes",
            classes     = ["pothole", "normal"],
            accuracy    = 74.5,
            dataset     = "cifar10",
            method      = "transfer_learning_resnet18",
        )
        assert "PotholeDetectorAgent" in code
        assert "ResNet" in code or "resnet18" in code.lower()

    def test_darts_code_for_text_domain(self, generator):
        code = generator._generate_real_agent_named(
            class_name  = "SpamDetectorAgent",
            agent_name  = "spam_detector",
            domain      = "text",
            problem     = "classify spam",
            classes     = ["spam", "ham"],
            accuracy    = 82.0,
            dataset     = "imdb",
            method      = "darts_nas",
        )
        assert "SpamDetectorAgent" in code
        assert "DARTS" in code or "DARTSNet" in code

    def test_code_contains_predict_method(self, generator):
        code = generator._generate_real_agent_named(
            "AgentX", "agent_x", "image",
            "test problem", ["a", "b"], 70.0, "cifar10",
            "transfer_learning_resnet18")
        assert "def predict" in code

    def test_code_contains_status_method(self, generator):
        code = generator._generate_real_agent_named(
            "AgentX", "agent_x", "image",
            "test problem", ["a", "b"], 70.0, "cifar10",
            "transfer_learning_resnet18")
        assert "def status" in code
