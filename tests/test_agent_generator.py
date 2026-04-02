# ============================================================
# Tests — Agent Generator
# api/brain/agent_generator.py
# ============================================================

import pytest
from api.brain.agent_generator import (
    generate_agent_code,
    generate_api_server,
    generate_predict_cli,
    generate_readme,
    generate_requirements,
)

PROBLEM  = "detect potholes in road images"
CATEGORY = "image"
CLASSES  = ["pothole", "normal", "crack"]
ACCURACY = 74.5
METHOD   = "transfer_learning"
AGENTS   = ["image"]


# ── generate_agent_code ───────────────────────────────────

class TestGenerateAgentCode:

    def test_returns_string(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, CLASSES, ACCURACY, METHOD, AGENTS)
        assert isinstance(code, str)

    def test_contains_problem(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, CLASSES, ACCURACY, METHOD, AGENTS)
        assert PROBLEM in code

    def test_contains_accuracy(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, CLASSES, ACCURACY, METHOD, AGENTS)
        assert str(ACCURACY) in code

    def test_contains_autoarchitect_class(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, CLASSES, ACCURACY, METHOD, AGENTS)
        assert "AutoArchitectAgent" in code

    def test_contains_predict_method(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, CLASSES, ACCURACY, METHOD, AGENTS)
        assert "def predict" in code

    def test_contains_classes_list(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, CLASSES, ACCURACY, METHOD, AGENTS)
        assert "pothole" in code

    def test_category_image_has_predict_image(self):
        code = generate_agent_code(
            PROBLEM, "image", CLASSES, ACCURACY, METHOD, AGENTS)
        assert "_predict_image" in code

    def test_category_text_has_predict_text(self):
        code = generate_agent_code(
            "classify spam", "text", ["spam", "ham"],
            82.0, "darts_nas", ["text"])
        assert "_predict_text" in code

    def test_method_title_case_in_code(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, CLASSES, ACCURACY,
            "transfer_learning", AGENTS)
        assert "Transfer Learning" in code

    def test_non_empty_classes_appear_in_code(self):
        code = generate_agent_code(
            PROBLEM, CATEGORY, ["alpha", "beta"],
            70.0, METHOD, AGENTS)
        assert "alpha" in code
        assert "beta" in code


# ── generate_api_server ───────────────────────────────────

class TestGenerateApiServer:

    def test_returns_string(self):
        assert isinstance(generate_api_server(), str)

    def test_contains_flask(self):
        assert "Flask" in generate_api_server()

    def test_contains_predict_route(self):
        assert "/predict" in generate_api_server()

    def test_contains_health_route(self):
        assert "/health" in generate_api_server()

    def test_contains_info_route(self):
        assert "/info" in generate_api_server()

    def test_contains_autoarchitect_agent_import(self):
        assert "AutoArchitectAgent" in generate_api_server()


# ── generate_predict_cli ──────────────────────────────────

class TestGeneratePredictCli:

    def test_returns_string(self):
        code = generate_predict_cli(PROBLEM, CATEGORY, CLASSES)
        assert isinstance(code, str)

    def test_contains_problem_reference(self):
        code = generate_predict_cli(PROBLEM, CATEGORY, CLASSES)
        assert PROBLEM in code

    def test_image_category_uses_jpg_example(self):
        code = generate_predict_cli(PROBLEM, "image", CLASSES)
        assert ".jpg" in code

    def test_text_category_uses_text_example(self):
        code = generate_predict_cli("classify spam", "text", ["spam"])
        assert ".jpg" not in code

    def test_contains_main_function(self):
        code = generate_predict_cli(PROBLEM, CATEGORY, CLASSES)
        assert "def main" in code

    def test_medical_category_uses_jpg_example(self):
        code = generate_predict_cli("detect tumor", "medical", ["tumor"])
        assert ".jpg" in code


# ── generate_readme ───────────────────────────────────────

class TestGenerateReadme:

    def test_returns_string(self):
        md = generate_readme(PROBLEM, CATEGORY, CLASSES,
                             ACCURACY, METHOD, AGENTS)
        assert isinstance(md, str)

    def test_contains_problem(self):
        md = generate_readme(PROBLEM, CATEGORY, CLASSES,
                             ACCURACY, METHOD, AGENTS)
        assert PROBLEM in md

    def test_contains_accuracy(self):
        md = generate_readme(PROBLEM, CATEGORY, CLASSES,
                             ACCURACY, METHOD, AGENTS)
        assert str(ACCURACY) in md

    def test_contains_all_classes(self):
        md = generate_readme(PROBLEM, CATEGORY, CLASSES,
                             ACCURACY, METHOD, AGENTS)
        for cls in CLASSES:
            assert cls in md

    def test_contains_quick_start(self):
        md = generate_readme(PROBLEM, CATEGORY, CLASSES,
                             ACCURACY, METHOD, AGENTS)
        assert "Quick Start" in md

    def test_contains_pip_install(self):
        md = generate_readme(PROBLEM, CATEGORY, CLASSES,
                             ACCURACY, METHOD, AGENTS)
        assert "pip install" in md

    def test_contains_autoarchitect_branding(self):
        md = generate_readme(PROBLEM, CATEGORY, CLASSES,
                             ACCURACY, METHOD, AGENTS)
        assert "AutoArchitect" in md


# ── generate_requirements ─────────────────────────────────

class TestGenerateRequirements:

    def test_returns_string(self):
        assert isinstance(generate_requirements(), str)

    def test_contains_torch(self):
        assert "torch" in generate_requirements()

    def test_contains_flask(self):
        assert "flask" in generate_requirements().lower()

    def test_contains_pillow(self):
        reqs = generate_requirements().lower()
        assert "pillow" in reqs or "pil" in reqs

    def test_each_line_is_package_name(self):
        lines = [l.strip() for l in generate_requirements().splitlines()
                 if l.strip()]
        for line in lines:
            assert len(line) > 0
            assert " " not in line or ">=" in line or "==" in line
