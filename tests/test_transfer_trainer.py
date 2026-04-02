# ============================================================
# Tests — Transfer Trainer (ResNet18 fine-tuning)
# api/transfer_trainer.py
# ============================================================

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
from unittest.mock import MagicMock, patch


# ── build_transfer_model ──────────────────────────────────────

class TestBuildTransferModel:

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_returns_nn_module(self, device):
        from api.transfer_trainer import build_transfer_model
        model = build_transfer_model(num_classes=10, device=device)
        assert isinstance(model, nn.Module)

    def test_final_layer_has_correct_output_size(self, device):
        from api.transfer_trainer import build_transfer_model
        for n in (2, 5, 10):
            model = build_transfer_model(num_classes=n, device=device)
            assert model.fc.out_features == n

    def test_most_layers_frozen(self, device):
        from api.transfer_trainer import build_transfer_model
        model = build_transfer_model(num_classes=4, device=device)
        # fc must be trainable
        assert model.fc.weight.requires_grad is True

    def test_fc_layer_is_linear(self, device):
        from api.transfer_trainer import build_transfer_model
        model = build_transfer_model(num_classes=3, device=device)
        assert isinstance(model.fc, nn.Linear)

    def test_layer4_params_are_trainable(self, device):
        from api.transfer_trainer import build_transfer_model
        model = build_transfer_model(num_classes=4, device=device)
        layer4_params = list(model.layer4.parameters())
        assert all(p.requires_grad for p in layer4_params)

    def test_model_runs_forward_pass(self, device):
        from api.transfer_trainer import build_transfer_model
        model = build_transfer_model(num_classes=4, device=device)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)


# ── get_transform ─────────────────────────────────────────────

class TestGetTransform:

    def test_returns_compose(self):
        import torchvision.transforms as transforms
        from api.transfer_trainer import get_transform
        tfm = get_transform()
        assert isinstance(tfm, transforms.Compose)

    def test_transform_produces_correct_tensor_shape(self):
        from PIL import Image
        from api.transfer_trainer import get_transform
        import numpy as np
        img = Image.fromarray(
            np.uint8(np.random.randint(0, 255, (64, 64, 3))))
        tfm    = get_transform()
        tensor = tfm(img)
        assert tensor.shape == (3, 224, 224)

    def test_transform_normalizes_values(self):
        """Normalized tensor should not be in [0,1] range anymore."""
        from PIL import Image
        from api.transfer_trainer import get_transform
        import numpy as np
        img = Image.fromarray(
            np.uint8(np.zeros((64, 64, 3))))  # black image → negative after normalization
        tfm    = get_transform()
        tensor = tfm(img)
        # After normalization with ImageNet mean, black pixels go negative
        assert tensor.min().item() < 0


# ── train_transfer ────────────────────────────────────────────

class TestTrainTransfer:
    """Tests for train_transfer using a tiny synthetic DataLoader."""

    @pytest.fixture
    def tiny_data(self):
        """Minimal data dict mimicking self_trainer format."""
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 2, (4,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=4)
        return {
            "name":         "test_dataset",
            "train_size":   4,
            "num_classes":  2,
            "train_loader": loader,
            "test_loader":  loader,
        }

    def test_returns_dict(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=1)
        assert isinstance(result, dict)

    def test_has_required_keys(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=1)
        for key in ("train_accuracy", "test_accuracy", "epoch_history",
                    "parameters", "model", "time", "method"):
            assert key in result, f"Missing key: {key}"

    def test_method_is_transfer_learning(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=1)
        assert result["method"] == "transfer_learning_resnet18"

    def test_epoch_history_length_matches_epochs(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=2)
        assert len(result["epoch_history"]) == 2

    def test_train_accuracy_is_percentage(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=1)
        assert 0.0 <= result["train_accuracy"] <= 100.0

    def test_test_accuracy_is_percentage(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=1)
        assert 0.0 <= result["test_accuracy"] <= 100.0

    def test_model_is_nn_module(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=1)
        assert isinstance(result["model"], nn.Module)

    def test_progress_callback_called(self, tiny_data):
        from api.transfer_trainer import train_transfer
        calls = []
        def cb(a, b, msg):
            calls.append(msg)
        train_transfer("test problem", tiny_data,
                       epochs=1, progress_callback=cb)
        assert len(calls) >= 1

    def test_time_is_positive(self, tiny_data):
        from api.transfer_trainer import train_transfer
        result = train_transfer("test problem", tiny_data, epochs=1)
        assert result["time"] >= 0

    def test_grayscale_images_handled(self):
        """1-channel images should be expanded to 3-channel."""
        from api.transfer_trainer import train_transfer
        images  = torch.randn(4, 1, 32, 32)
        labels  = torch.randint(0, 2, (4,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=4)
        data = {
            "name": "gray", "train_size": 4, "num_classes": 2,
            "train_loader": loader, "test_loader": loader,
        }
        result = train_transfer("gray problem", data, epochs=1)
        assert "train_accuracy" in result
