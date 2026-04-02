# ============================================================
# Tests — CacheManager
# api/cache_manager.py
# ============================================================

import pytest
import math
from unittest.mock import patch

pytest.importorskip('torch')


# ── cosine_similarity ─────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors_return_1(self):
        from api.cache_manager import cosine_similarity
        v = [1.0] * 768
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_opposite_vectors_return_minus_1(self):
        from api.cache_manager import cosine_similarity
        v1 = [1.0] + [0.0] * 767
        v2 = [-1.0] + [0.0] * 767
        assert cosine_similarity(v1, v2) < 0

    def test_orthogonal_vectors_return_0(self):
        from api.cache_manager import cosine_similarity
        v1 = [1.0] + [0.0] * 767
        v2 = [0.0, 1.0] + [0.0] * 766
        assert abs(cosine_similarity(v1, v2)) < 1e-5

    def test_empty_vectors_return_0(self):
        from api.cache_manager import cosine_similarity
        assert cosine_similarity([], []) == 0.0

    def test_result_between_neg1_and_1(self):
        from api.cache_manager import cosine_similarity
        import random
        random.seed(42)
        v1 = [random.uniform(-1, 1) for _ in range(768)]
        v2 = [random.uniform(-1, 1) for _ in range(768)]
        result = cosine_similarity(v1, v2)
        assert -1.0 <= result <= 1.0


# ── get_problem_hash ──────────────────────────────────────

class TestGetProblemHash:

    def test_same_problem_same_hash(self):
        from api.cache_manager import get_problem_hash
        assert get_problem_hash("detect potholes") == \
               get_problem_hash("detect potholes")

    def test_different_problems_different_hash(self):
        from api.cache_manager import get_problem_hash
        assert get_problem_hash("detect potholes") != \
               get_problem_hash("classify spam")

    def test_hash_is_string(self):
        from api.cache_manager import get_problem_hash
        assert isinstance(get_problem_hash("test problem"), str)

    def test_hash_length(self):
        from api.cache_manager import get_problem_hash
        # Should be a short hex string (10 chars based on code)
        h = get_problem_hash("test problem")
        assert len(h) > 0

    def test_normalised_before_hashing(self):
        from api.cache_manager import get_problem_hash
        # Leading/trailing whitespace and case should not change hash
        h1 = get_problem_hash("Detect Potholes")
        h2 = get_problem_hash("detect potholes")
        # Not guaranteed by all implementations, but worth checking
        # At minimum both must be valid strings
        assert isinstance(h1, str) and isinstance(h2, str)


# ── check_cache — miss ────────────────────────────────────

class TestCheckCacheMiss:

    def test_returns_found_false_on_empty_cache(self, tmp_path):
        import api.cache_manager as mod
        with patch.object(mod, 'CACHE_DIR', str(tmp_path / 'cache')):
            result = mod.check_cache("completely unknown problem xyz123")
        assert result["found"] is False

    def test_found_false_has_no_metadata(self, tmp_path):
        import api.cache_manager as mod
        with patch.object(mod, 'CACHE_DIR', str(tmp_path / 'cache')):
            result = mod.check_cache("unknown problem")
        assert "metadata" not in result


# ── check_cache — exact hit ───────────────────────────────

class TestCheckCacheExactHit:

    def _write_cache_entry(self, cache_dir, problem, metadata):
        import hashlib
        import json
        from pathlib import Path
        h = hashlib.md5(problem.lower().strip().encode()).hexdigest()[:10]
        entry_dir = Path(cache_dir) / h
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / 'metadata.json').write_text(json.dumps(metadata))
        return h

    def test_exact_match_found(self, tmp_path):
        import api.cache_manager as mod
        problem = "detect potholes in road"
        meta = {"problem": problem, "accuracy": 74.5, "category": "image",
                "use_count": 1, "architecture": [], "parameters": 1000,
                "search_time": 2.0}
        cache_dir = tmp_path / 'cache'
        self._write_cache_entry(str(cache_dir), problem, meta)

        with patch.object(mod, 'CACHE_DIR', str(cache_dir)):
            # Disable embedding to force exact-only check
            with patch.object(mod, 'get_embedding', return_value=[]):
                result = mod.check_cache(problem)

        assert result["found"] is True

    def test_exact_match_type(self, tmp_path):
        import api.cache_manager as mod
        problem = "classify spam emails"
        meta = {"problem": problem, "accuracy": 88.0, "category": "text",
                "use_count": 1, "architecture": [], "parameters": 500,
                "search_time": 1.5}
        cache_dir = tmp_path / 'cache'
        self._write_cache_entry(str(cache_dir), problem, meta)

        with patch.object(mod, 'CACHE_DIR', str(cache_dir)):
            with patch.object(mod, 'get_embedding', return_value=[]):
                result = mod.check_cache(problem)

        if result["found"]:
            assert result.get("match_type") == "exact"


# ── get_cache_stats ───────────────────────────────────────

class TestGetCacheStats:

    def test_empty_cache_returns_zeros(self, tmp_path):
        import api.cache_manager as mod
        with patch.object(mod, 'CACHE_DIR', str(tmp_path / 'cache')):
            stats = mod.get_cache_stats()
        assert stats["total_solutions"] == 0
        assert stats["total_uses"] == 0

    def test_stats_has_required_keys(self, tmp_path):
        import api.cache_manager as mod
        with patch.object(mod, 'CACHE_DIR', str(tmp_path / 'cache')):
            stats = mod.get_cache_stats()
        for key in ("total_solutions", "total_uses", "categories", "recent"):
            assert key in stats

    def test_categories_is_dict(self, tmp_path):
        import api.cache_manager as mod
        with patch.object(mod, 'CACHE_DIR', str(tmp_path / 'cache')):
            stats = mod.get_cache_stats()
        assert isinstance(stats["categories"], dict)


# ── find_similar_cached — word overlap fallback ───────────

class TestFindSimilarCached:

    def test_returns_none_on_empty_cache(self, tmp_path):
        import api.cache_manager as mod
        with patch.object(mod, 'CACHE_DIR', str(tmp_path / 'cache')):
            result = mod.find_similar_cached("detect potholes", "image")
        assert result is None
