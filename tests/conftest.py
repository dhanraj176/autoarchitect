# ============================================================
# conftest.py — shared pytest fixtures
# ============================================================

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make autoarchitect importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "autoarchitect"))


# ── Shared tmp directory ──────────────────────────────────
@pytest.fixture
def tmp_dir(tmp_path):
    """A fresh temp directory for each test."""
    return tmp_path


# ── Minimal BERT embedding stub ───────────────────────────
@pytest.fixture
def fake_embedding():
    """768-dim unit vector — stands in wherever BERT embeddings are needed."""
    import math
    v = [1.0 / math.sqrt(768)] * 768
    return v


# ── Stub agent result (single domain) ────────────────────
@pytest.fixture
def image_agent_result():
    return {
        "domain": "image",
        "architecture": [
            {"cell": 1, "operations": [
                {"operation": "conv3x3", "confidence": 80.0,
                 "weights": {"skip": 0.05, "conv3x3": 0.80,
                             "conv5x5": 0.05, "maxpool": 0.05,
                             "avgpool": 0.05}}
            ]},
        ],
        "parameters": 105910,
        "accuracy": 74.5,
        "status": "success",
    }


@pytest.fixture
def text_agent_result():
    return {
        "domain": "text",
        "architecture": [
            {"cell": 1, "operations": [
                {"operation": "conv5x5", "confidence": 90.0,
                 "weights": {"skip": 0.01, "conv3x3": 0.05,
                             "conv5x5": 0.90, "maxpool": 0.02,
                             "avgpool": 0.02}}
            ]},
        ],
        "parameters": 88000,
        "accuracy": 82.0,
        "status": "success",
    }
