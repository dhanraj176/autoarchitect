# ============================================================
# Tests — data_discovery_engine.py
# VERIFIED_HF registry, cache behaviour, constants
# ============================================================

import json
import hashlib
import pytest
pytest.importorskip("torch")
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestVerifiedHF:

    @pytest.fixture
    def verified(self):
        from api.brain.data_discovery_engine import VERIFIED_HF
        return VERIFIED_HF

    def test_is_dict(self, verified):
        assert isinstance(verified, dict)

    def test_non_empty(self, verified):
        assert len(verified) > 0

    def test_all_values_are_strings(self, verified):
        for k, v in verified.items():
            assert isinstance(v, str), f"Value for '{k}' is not a string"

    def test_pothole_present(self, verified):
        assert "pothole" in verified

    def test_garbage_present(self, verified):
        assert "garbage" in verified

    def test_fire_present(self, verified):
        assert "fire" in verified


class TestDiscoveryConstants:

    def test_accuracy_threshold_between_0_and_1(self):
        from api.brain.data_discovery_engine import ACCURACY_THRESHOLD
        assert 0.0 < ACCURACY_THRESHOLD < 1.0

    def test_openimages_max_positive_integer(self):
        from api.brain.data_discovery_engine import OPENIMAGES_MAX
        assert isinstance(OPENIMAGES_MAX, int)
        assert OPENIMAGES_MAX > 0

    def test_openimages_batch_positive_integer(self):
        from api.brain.data_discovery_engine import OPENIMAGES_BATCH
        assert isinstance(OPENIMAGES_BATCH, int)
        assert OPENIMAGES_BATCH > 0

    def test_cache_dir_is_path(self):
        from api.brain.data_discovery_engine import CACHE_DIR
        assert isinstance(CACHE_DIR, Path)


class TestDataDiscoveryCache:
    """Test the discovery engine's cache read/write via DataDiscoveryEngine."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("api.brain.data_discovery_engine.CACHE_DIR", tmp_path):
            from api.brain.data_discovery_engine import DataDiscoveryEngine
            eng = DataDiscoveryEngine.__new__(DataDiscoveryEngine)
            eng.cache_dir  = tmp_path
            eng.groq_key   = ""
            eng.meta       = {}
            eng.meta_file  = tmp_path / "discovery_meta.json"
        return eng, tmp_path

    def _cache_key(self, problem: str, domain: str = "image") -> str:
        import hashlib
        return hashlib.md5(
            f"{problem.lower().strip()}_{domain}".encode()
        ).hexdigest()[:12]

    def test_cache_miss_returns_none(self, engine):
        eng, tmp = engine
        result = eng._check_local_cache("novel problem 9999", "image")
        assert result is None

    def test_save_then_load_cache(self, engine):
        eng, tmp = engine
        problem = "test cache problem"
        payload = {"dataset": "ds/test", "accuracy": 0.85}

        eng._save_local_cache(problem, "image", payload)
        loaded = eng._check_local_cache(problem, "image")

        assert loaded is not None
        assert loaded["dataset"]  == "ds/test"
        assert loaded["accuracy"] == 0.85

    def test_different_problem_is_cache_miss(self, engine):
        eng, tmp = engine
        eng._save_local_cache("problem A", "image", {"x": 1})
        assert eng._check_local_cache("problem B completely different", "image") is None

    def test_cache_file_is_json(self, engine):
        eng, tmp = engine
        problem = "json check problem"
        eng._save_local_cache(problem, "image", {"key": "value"})
        key  = self._cache_key(problem, "image")
        path = tmp / f"{key}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["key"] == "value"


class TestVerifiedHFLookup:
    """Test helper that checks if a problem matches verified HF datasets."""

    @pytest.fixture
    def engine(self, tmp_path):
        with patch("api.brain.data_discovery_engine.CACHE_DIR", tmp_path):
            from api.brain.data_discovery_engine import DataDiscoveryEngine
            eng = DataDiscoveryEngine.__new__(DataDiscoveryEngine)
            eng.cache_dir = tmp_path
            eng.groq_key  = ""
        return eng

    def test_pothole_problem_finds_verified_dataset(self, engine):
        ds_id = engine._check_verified_registry("detect potholes in streets")
        assert ds_id is not None and ds_id != ""

    def test_unknown_problem_returns_none(self, engine):
        ds_id = engine._check_verified_registry("quantum teleportation research")
        assert not ds_id

    def test_case_insensitive_lookup(self, engine):
        ds_id = engine._check_verified_registry("FIRE detection in forests")
        assert ds_id
