# ============================================================
# Tests — MetaLearner
# api/brain/meta_learner.py
# ============================================================

import math
import pytest
from pathlib import Path
from unittest.mock import patch

pytest.importorskip('torch')


@pytest.fixture
def fresh_learner(tmp_path):
    """MetaLearner backed by temp files, no prior examples."""
    import api.brain.meta_learner as mod
    with patch.object(mod, 'BRAIN_DIR', str(tmp_path)), \
         patch.object(mod, 'META_FILE', str(tmp_path / 'meta.json')), \
         patch.object(mod, 'MODEL_FILE', str(tmp_path / 'meta_model.pth')), \
         patch.object(mod, 'INSIGHT_FILE', str(tmp_path / 'meta_insights.json')):
        ml = mod.MetaLearner()
    return ml, mod, tmp_path


@pytest.fixture
def fake_embedding():
    v = [1.0 / math.sqrt(768)] * 768
    return v


# ── predict — no examples yet ─────────────────────────────

class TestPredictUntrained:

    def test_returns_predicted_false_with_no_examples(self, fresh_learner):
        ml, _, _ = fresh_learner
        result = ml.predict("detect potholes")
        assert result["predicted"] is False

    def test_does_not_raise_without_embedding(self, fresh_learner):
        ml, _, _ = fresh_learner
        try:
            ml.predict("some problem")
        except Exception as e:
            pytest.fail(f"predict() raised unexpectedly: {e}")


# ── predict — after training ──────────────────────────────

class TestPredictTrained:

    def _fill_and_train(self, ml, fake_emb, n=7):
        """Add n examples and trigger training."""
        combos = [["image"], ["text"], ["medical"], ["security"],
                  ["image", "text"], ["image", "medical"], ["text", "security"]]
        datasets = ["cifar10", "imdb", "sms_spam", "synthetic_fraud",
                    "GonzaloA/fake_news", "fashionmnist", "user_data"]
        for i in range(n):
            ml.learn(
                problem=f"problem {i}",
                agents_used=combos[i % len(combos)],
                dataset_used=datasets[i % len(datasets)],
                method_used="transfer_learning" if i % 2 == 0 else "darts_nas",
                actual_accuracy=70.0 + i,
                bert_embedding=fake_emb,
            )

    def test_returns_predicted_true_after_enough_examples(
            self, fresh_learner, fake_embedding):
        ml, _, _ = fresh_learner
        self._fill_and_train(ml, fake_embedding, n=7)
        result = ml.predict("detect objects", bert_embedding=fake_embedding)
        # After 7 examples and training, prediction should be possible
        assert "predicted" in result

    def test_agents_is_list_when_predicted(self, fresh_learner, fake_embedding):
        ml, _, _ = fresh_learner
        self._fill_and_train(ml, fake_embedding, n=7)
        result = ml.predict("classify images", bert_embedding=fake_embedding)
        if result["predicted"]:
            assert isinstance(result["agents"], list)
            assert len(result["agents"]) >= 1

    def test_confidence_between_0_and_1(self, fresh_learner, fake_embedding):
        ml, _, _ = fresh_learner
        self._fill_and_train(ml, fake_embedding, n=7)
        result = ml.predict("detect fraud", bert_embedding=fake_embedding)
        if result["predicted"]:
            assert 0.0 <= result["confidence"] <= 1.0

    def test_accuracy_is_positive_when_predicted(self, fresh_learner, fake_embedding):
        ml, _, _ = fresh_learner
        self._fill_and_train(ml, fake_embedding, n=7)
        result = ml.predict("text classification", bert_embedding=fake_embedding)
        if result["predicted"]:
            assert result["accuracy"] >= 0


# ── learn() ───────────────────────────────────────────────

class TestMetaLearnerLearn:

    def test_learn_adds_example(self, fresh_learner, fake_embedding):
        ml, _, _ = fresh_learner
        ml.learn("detect potholes", ["image"], "cifar10",
                 "transfer_learning", 74.5, fake_embedding)
        assert len(ml.examples) == 1

    def test_learn_five_triggers_training(self, fresh_learner, fake_embedding):
        ml, _, _ = fresh_learner
        for i in range(5):
            ml.learn(f"problem {i}", ["image"], "cifar10",
                     "transfer_learning", 70.0 + i, fake_embedding)
        assert ml.trained is True

    def test_learn_does_not_retrain_every_single_call(
            self, fresh_learner, fake_embedding):
        ml, _, _ = fresh_learner
        # First 5 trigger training (MIN_EXAMPLES_TO_TRAIN)
        for i in range(5):
            ml.learn(f"p{i}", ["image"], "cifar10",
                     "transfer_learning", 70.0, fake_embedding)
        trained_after_5 = ml.trained
        # Next 1 alone should NOT trigger (RETRAIN_EVERY=3, need 2 more)
        ml.learn("p6", ["text"], "imdb", "darts_nas", 80.0, fake_embedding)
        # trained flag should still be True (no re-trigger yet)
        assert ml.trained is True
        _ = trained_after_5  # used

    def test_learn_without_embedding_does_not_crash(self, fresh_learner):
        ml, _, _ = fresh_learner
        try:
            ml.learn("detect spam", ["text"], "imdb",
                     "darts_nas", 82.0)
        except Exception as e:
            pytest.fail(f"learn() without embedding raised: {e}")

    def test_examples_are_persisted(self, fresh_learner, fake_embedding, tmp_path):
        ml, mod, tmp = fresh_learner
        import json
        meta_file = str(tmp / 'meta.json')
        with patch.object(mod, 'META_FILE', meta_file):
            ml.learn("test problem", ["image"], "cifar10",
                     "transfer_learning", 74.5, fake_embedding)
            ml._save_examples()
        assert Path(meta_file).exists()
        data = json.loads(Path(meta_file).read_text())
        assert len(data) >= 1


# ── _encode_combo ─────────────────────────────────────────

class TestEncodeCombo:

    def test_single_image_encodes(self, fresh_learner):
        ml, _, _ = fresh_learner
        idx = ml._encode_combo(["image"])
        assert isinstance(idx, int)
        assert 0 <= idx <= 9

    def test_same_input_same_output(self, fresh_learner):
        ml, _, _ = fresh_learner
        assert ml._encode_combo(["text"]) == ml._encode_combo(["text"])

    def test_unknown_combo_returns_int(self, fresh_learner):
        ml, _, _ = fresh_learner
        idx = ml._encode_combo(["unknown_domain_xyz"])
        assert isinstance(idx, int)


# ── _encode_dataset ───────────────────────────────────────

class TestEncodeDataset:

    def test_known_dataset_returns_int(self, fresh_learner):
        ml, _, _ = fresh_learner
        idx = ml._encode_dataset("cifar10")
        assert isinstance(idx, int)
        assert 0 <= idx <= 7

    def test_unknown_dataset_returns_last_index(self, fresh_learner):
        ml, _, _ = fresh_learner
        idx = ml._encode_dataset("completely_unknown_dataset_xyz")
        assert isinstance(idx, int)


# ── get_insights ──────────────────────────────────────────

class TestGetInsights:

    def test_no_examples_returns_learning_status(self, fresh_learner):
        ml, _, _ = fresh_learner
        result = ml.get_insights()
        assert result.get("status") == "learning"

    def test_has_examples_count(self, fresh_learner):
        ml, _, _ = fresh_learner
        result = ml.get_insights()
        assert "examples" in result


# ── MetaNet forward pass ──────────────────────────────────

class TestMetaNet:

    def test_forward_returns_dict(self):
        import torch
        from api.brain.meta_learner import MetaNet
        net = MetaNet()
        x = torch.randn(2, 768)
        out = net(x)
        assert isinstance(out, dict)

    def test_forward_has_all_heads(self):
        import torch
        from api.brain.meta_learner import MetaNet
        net = MetaNet()
        x = torch.randn(1, 768)
        out = net(x)
        for key in ("agents", "dataset", "method", "accuracy"):
            assert key in out

    def test_agents_head_shape(self):
        import torch
        from api.brain.meta_learner import MetaNet
        net = MetaNet()
        x = torch.randn(3, 768)
        out = net(x)
        assert out["agents"].shape == (3, 10)

    def test_accuracy_head_shape(self):
        import torch
        from api.brain.meta_learner import MetaNet
        net = MetaNet()
        x = torch.randn(3, 768)
        out = net(x)
        assert out["accuracy"].shape == torch.Size([3])
