# ============================================================
# Tests — AgentFactory  (api/agents/agent_factory.py)
# Naming helpers, create(), generate_agent_code()
# ============================================================

import pytest
pytest.importorskip("torch")
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def factory():
    from api.agents.agent_factory import AgentFactory
    return AgentFactory()


# ── generate_name ─────────────────────────────────────────────

class TestGenerateName:

    def test_returns_string(self, factory):
        name = factory.generate_name("detect potholes in Oakland")
        assert isinstance(name, str)

    def test_ends_with_agent(self, factory):
        name = factory.generate_name("detect potholes in Oakland")
        assert name.endswith("_agent")

    def test_stop_words_excluded(self, factory):
        name = factory.generate_name("detect the fire")
        # "detect" and "the" are stop words — name should not be just "_agent"
        assert len(name) > len("_agent")

    def test_keywords_lowercase(self, factory):
        name = factory.generate_name("Classify SPAM Emails")
        assert name == name.lower()

    def test_pothole_problem(self, factory):
        name = factory.generate_name("detect potholes in street cameras")
        assert "pothole" in name

    def test_very_short_problem(self, factory):
        """Should not crash on short input."""
        name = factory.generate_name("spam")
        assert isinstance(name, str)
        assert name.endswith("_agent")


# ── generate_class_name ───────────────────────────────────────

class TestGenerateClassName:

    def test_returns_string(self, factory):
        cn = factory.generate_class_name("detect potholes in Oakland")
        assert isinstance(cn, str)

    def test_ends_with_agent(self, factory):
        cn = factory.generate_class_name("detect potholes in Oakland")
        assert cn.endswith("Agent")

    def test_title_case_words(self, factory):
        cn = factory.generate_class_name("detect potholes")
        # Every word-part should be capitalized
        assert cn[0].isupper()

    def test_pothole_class_name(self, factory):
        cn = factory.generate_class_name("detect potholes in roads")
        assert "Pothole" in cn

    def test_spam_class_name(self, factory):
        cn = factory.generate_class_name("classify spam messages")
        assert "Spam" in cn


# ── generate_file_name ────────────────────────────────────────

class TestGenerateFileName:

    def test_ends_with_py(self, factory):
        fn = factory.generate_file_name("detect potholes")
        assert fn.endswith(".py")

    def test_contains_agent(self, factory):
        fn = factory.generate_file_name("detect potholes")
        assert "agent" in fn


# ── _model_type_for_domain ────────────────────────────────────

class TestModelTypeForDomain:

    def test_image_domain_resnet18(self, factory):
        assert factory._model_type_for_domain("image") == "resnet18"

    def test_medical_domain_resnet18(self, factory):
        assert factory._model_type_for_domain("medical") == "resnet18"

    def test_text_domain_darts(self, factory):
        assert factory._model_type_for_domain("text") == "darts"

    def test_security_domain_darts(self, factory):
        assert factory._model_type_for_domain("security") == "darts"

    def test_unknown_domain_darts(self, factory):
        assert factory._model_type_for_domain("unknown") == "darts"


# ── create ────────────────────────────────────────────────────

class TestCreate:

    def test_returns_dynamic_agent(self, factory):
        from api.agents.dynamic_agent import DynamicAgent
        agent = factory.create(
            problem="detect potholes",
            domain="image",
            classes=["damage", "no_damage"],
        )
        assert isinstance(agent, DynamicAgent)

    def test_agent_has_correct_domain(self, factory):
        agent = factory.create("detect fire", domain="image",
                               classes=["fire", "no_fire"])
        assert agent.domain == "image"

    def test_agent_classes_stored(self, factory):
        agent = factory.create("classify spam", domain="text",
                               classes=["ham", "spam"])
        assert agent.classes == ["ham", "spam"]

    def test_model_not_loaded_when_no_path(self, factory):
        agent = factory.create("detect potholes", domain="image",
                               classes=["a", "b"])
        assert agent.model_loaded is False

    def test_num_classes_defaults_to_len_classes(self, factory):
        agent = factory.create("problem x", domain="text",
                               classes=["a", "b", "c"])
        assert agent.num_classes == 3

    def test_create_from_trained_sets_accuracy(self, factory):
        trained = {
            "model_path":    None,
            "classes":       ["cat", "dog"],
            "test_accuracy": 82.5,
            "dataset":       "cifar10",
            "method":        "resnet18",
        }
        agent = factory.create_from_trained(
            "classify animals", "image", trained)
        assert agent.accuracy == 82.5


# ── get_factory singleton ─────────────────────────────────────

class TestGetFactory:

    def test_returns_same_instance(self):
        from api.agents.agent_factory import get_factory
        f1 = get_factory()
        f2 = get_factory()
        assert f1 is f2


# ── generate_agent_code ───────────────────────────────────────

class TestFactoryGenerateAgentCode:

    def test_resnet_code_contains_class_name(self, factory):
        code = factory.generate_agent_code(
            problem="detect potholes in roads",
            domain="image",
            classes=["damage", "no_damage"],
            accuracy=85.0,
            dataset="taroii/pothole-detection",
            method="transfer_learning_resnet18",
        )
        class_name = factory.generate_class_name("detect potholes in roads")
        assert class_name in code

    def test_darts_code_contains_class_name(self, factory):
        code = factory.generate_agent_code(
            problem="classify spam messages",
            domain="text",
            classes=["ham", "spam"],
            accuracy=95.0,
            dataset="sms_spam",
            method="darts",
        )
        class_name = factory.generate_class_name("classify spam messages")
        assert class_name in code

    def test_resnet_code_has_predict_method(self, factory):
        code = factory.generate_agent_code(
            "detect fire", "image", ["fire", "no_fire"],
            88.0, "openfire", "resnet18")
        assert "def predict(" in code

    def test_darts_code_has_predict_method(self, factory):
        code = factory.generate_agent_code(
            "detect threats", "security", ["safe", "threat"],
            80.0, "synthetic", "darts")
        assert "def predict(" in code

    def test_code_contains_accuracy(self, factory):
        code = factory.generate_agent_code(
            "detect fire", "image", ["fire", "no_fire"],
            88.0, "openfire", "resnet18")
        assert "88.0" in code

    def test_code_contains_classes(self, factory):
        code = factory.generate_agent_code(
            "classify spam", "text", ["ham", "spam"],
            95.0, "sms_spam", "darts")
        assert "ham" in code
        assert "spam" in code
