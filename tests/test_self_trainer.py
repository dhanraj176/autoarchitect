# ============================================================
# Tests — SelfTrainer  (pure helper functions — no torch needed)
# api/self_trainer.py
# ============================================================

import pytest
import sys
from unittest.mock import MagicMock, patch

# self_trainer.py imports torch and nas_engine at module level
_torch_stub = MagicMock()
with patch.dict(sys.modules, {
    'torch': _torch_stub,
    'torch.nn': MagicMock(),
    'torch.optim': MagicMock(),
    'api.nas_engine': MagicMock(),
}):
    from api.self_trainer import _correct_domain, _problem_hash, DOMAIN_OVERRIDES


# ── _correct_domain ───────────────────────────────────────

class TestCorrectDomain:

    def test_returns_string(self):
        result = _correct_domain("detect potholes", "image")
        assert isinstance(result, str)

    def test_no_keyword_returns_bert_domain(self):
        result = _correct_domain("some random problem", "text")
        assert result == "text"

    def test_pothole_corrects_to_image(self):
        result = _correct_domain("detect potholes in roads", "text")
        assert result == "image"

    def test_xray_corrects_to_medical(self):
        result = _correct_domain("analyze xray scan", "image")
        assert result == "medical"

    def test_mri_corrects_to_medical(self):
        result = _correct_domain("detect tumor in mri scan", "image")
        assert result == "medical"

    def test_fraud_corrects_to_security(self):
        result = _correct_domain("detect fraud in transactions", "text")
        assert result == "security"

    def test_spam_corrects_to_text(self):
        result = _correct_domain("classify spam emails", "image")
        assert result == "text"

    def test_sentiment_corrects_to_text(self):
        result = _correct_domain("analyze customer sentiment", "image")
        assert result == "text"

    def test_intrusion_corrects_to_security(self):
        result = _correct_domain("detect network intrusion attacks", "text")
        assert result == "security"

    def test_crack_corrects_to_image(self):
        result = _correct_domain("find crack in pavement", "text")
        assert result == "image"

    def test_malware_corrects_to_security(self):
        result = _correct_domain("detect malware in system", "text")
        assert result == "security"

    def test_case_insensitive_matching(self):
        result = _correct_domain("POTHOLE detection", "text")
        assert result == "image"

    def test_bert_domain_preserved_when_already_correct(self):
        result = _correct_domain("detect potholes", "image")
        assert result == "image"

    def test_all_keywords_in_overrides_produce_valid_domain(self):
        valid_domains = {"image", "text", "medical", "security"}
        for keyword, expected in DOMAIN_OVERRIDES.items():
            result = _correct_domain(keyword, "image")
            assert result in valid_domains, \
                f"Keyword '{keyword}' produced invalid domain '{result}'"


# ── _problem_hash ─────────────────────────────────────────

class TestProblemHash:

    def test_returns_string(self):
        assert isinstance(_problem_hash("detect potholes"), str)

    def test_hash_length_is_ten(self):
        h = _problem_hash("detect potholes")
        assert len(h) == 10

    def test_same_problem_same_hash(self):
        p = "detect potholes in roads"
        assert _problem_hash(p) == _problem_hash(p)

    def test_different_problems_different_hash(self):
        h1 = _problem_hash("detect potholes")
        h2 = _problem_hash("classify spam emails")
        assert h1 != h2

    def test_case_normalised(self):
        assert _problem_hash("Detect Potholes") == \
               _problem_hash("detect potholes")

    def test_punctuation_stripped(self):
        assert _problem_hash("detect potholes!") == \
               _problem_hash("detect potholes")

    def test_extra_spaces_normalised(self):
        assert _problem_hash("detect  potholes") == \
               _problem_hash("detect potholes")

    def test_hash_is_hex_string(self):
        h = _problem_hash("detect potholes")
        int(h, 16)   # raises ValueError if not valid hex

    def test_empty_string_does_not_raise(self):
        try:
            _problem_hash("")
        except Exception as e:
            pytest.fail(f"_problem_hash('') raised: {e}")


# ── DOMAIN_OVERRIDES structure ────────────────────────────

class TestDomainOverrides:

    def test_is_dict(self):
        assert isinstance(DOMAIN_OVERRIDES, dict)

    def test_all_values_are_valid_domains(self):
        valid = {"image", "text", "medical", "security"}
        for kw, domain in DOMAIN_OVERRIDES.items():
            assert domain in valid, \
                f"Override '{kw}' → '{domain}' not a valid domain"

    def test_all_keys_are_strings(self):
        for kw in DOMAIN_OVERRIDES:
            assert isinstance(kw, str)

    def test_not_empty(self):
        assert len(DOMAIN_OVERRIDES) > 0
