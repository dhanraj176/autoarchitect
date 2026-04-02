# ============================================================
# Tests — NAS Engine (DARTSNet / MixedOp)
# api/nas_engine.py
# ============================================================

import pytest
torch = pytest.importorskip('torch')


# ── MixedOp ──────────────────────────────────────────────

class TestMixedOp:

    def test_output_shape_matches_input(self):
        from api.nas_engine import MixedOp
        op = MixedOp(C=16)
        x = torch.randn(2, 16, 8, 8)
        out = op(x)
        assert out.shape == x.shape

    def test_has_five_operations(self):
        from api.nas_engine import MixedOp
        op = MixedOp(C=16)
        assert len(op.ops) == 5

    def test_arch_weights_are_learnable(self):
        from api.nas_engine import MixedOp
        op = MixedOp(C=16)
        assert op.arch_weights.requires_grad

    def test_arch_weights_shape(self):
        from api.nas_engine import MixedOp
        op = MixedOp(C=16)
        assert op.arch_weights.shape == (5,)

    def test_output_is_weighted_sum(self):
        """Arch weights are applied via softmax — output must change as weights change."""
        from api.nas_engine import MixedOp
        op = MixedOp(C=4)
        x = torch.ones(1, 4, 4, 4)
        out1 = op(x).detach().clone()
        with torch.no_grad():
            op.arch_weights.data = torch.tensor([10.0, -10.0, -10.0, -10.0, -10.0])
        out2 = op(x).detach().clone()
        assert not torch.allclose(out1, out2)


# ── DARTSCell ─────────────────────────────────────────────

class TestDARTSCell:

    def test_output_shape(self):
        from api.nas_engine import DARTSCell
        cell = DARTSCell(C=16)
        x = torch.randn(2, 16, 8, 8)
        assert cell(x).shape == x.shape

    def test_has_four_mixed_ops(self):
        from api.nas_engine import DARTSCell
        cell = DARTSCell(C=16)
        assert len(cell.ops) == 4


# ── DARTSNet ──────────────────────────────────────────────

class TestDARTSNet:

    def test_output_shape_default(self):
        from api.nas_engine import DARTSNet
        model = DARTSNet()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_output_shape_custom_classes(self):
        from api.nas_engine import DARTSNet
        model = DARTSNet(num_classes=5)
        x = torch.randn(1, 3, 32, 32)
        assert model(x).shape == (1, 5)

    def test_custom_channels(self):
        from api.nas_engine import DARTSNet
        model = DARTSNet(C=8, num_cells=2, num_classes=3)
        x = torch.randn(1, 3, 32, 32)
        assert model(x).shape == (1, 3)

    def test_parameter_count_positive(self):
        from api.nas_engine import DARTSNet
        model = DARTSNet()
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_arch_weights_separate_from_network_weights(self):
        from api.nas_engine import DARTSNet
        model = DARTSNet()
        arch_params = [n for n, _ in model.named_parameters()
                       if 'arch_weights' in n]
        net_params  = [n for n, _ in model.named_parameters()
                       if 'arch_weights' not in n]
        assert len(arch_params) > 0
        assert len(net_params) > 0

    def test_forward_does_not_raise(self):
        from api.nas_engine import DARTSNet
        model = DARTSNet()
        x = torch.randn(4, 3, 32, 32)
        try:
            model(x)
        except Exception as e:
            pytest.fail(f"DARTSNet forward raised: {e}")


# ── get_architecture ──────────────────────────────────────

class TestGetArchitecture:

    def test_returns_list_of_cells(self):
        from api.nas_engine import DARTSNet, get_architecture
        model = DARTSNet(num_cells=3)
        arch = get_architecture(model)
        assert isinstance(arch, list)
        assert len(arch) == 3

    def test_each_cell_has_operations_key(self):
        from api.nas_engine import DARTSNet, get_architecture
        arch = get_architecture(DARTSNet())
        for cell in arch:
            assert "operations" in cell
            assert "cell" in cell

    def test_operation_has_expected_keys(self):
        from api.nas_engine import DARTSNet, get_architecture
        arch = get_architecture(DARTSNet())
        op = arch[0]["operations"][0]
        for key in ("operation", "confidence", "weights"):
            assert key in op, f"Missing key: {key}"

    def test_operation_name_is_valid(self):
        from api.nas_engine import DARTSNet, get_architecture
        VALID_OPS = {"skip", "conv3x3", "conv5x5", "maxpool", "avgpool"}
        arch = get_architecture(DARTSNet())
        for cell in arch:
            for op in cell["operations"]:
                assert op["operation"] in VALID_OPS

    def test_confidence_between_0_and_100(self):
        from api.nas_engine import DARTSNet, get_architecture
        arch = get_architecture(DARTSNet())
        for cell in arch:
            for op in cell["operations"]:
                assert 0.0 <= op["confidence"] <= 100.0

    def test_weights_sum_to_1(self):
        from api.nas_engine import DARTSNet, get_architecture
        arch = get_architecture(DARTSNet())
        for cell in arch:
            for op in cell["operations"]:
                total = sum(op["weights"].values())
                assert abs(total - 1.0) < 1e-4, \
                    f"Weights sum {total}, expected ~1.0"


# ── run_quick_nas ─────────────────────────────────────────

class TestRunQuickNas:

    def test_returns_success_status(self):
        from api.nas_engine import run_quick_nas
        result = run_quick_nas(num_classes=5)
        assert result["status"] == "success"

    def test_result_has_required_keys(self):
        from api.nas_engine import run_quick_nas
        result = run_quick_nas(num_classes=5)
        for key in ("architecture", "parameters", "search_time", "status"):
            assert key in result

    def test_parameters_positive(self):
        from api.nas_engine import run_quick_nas
        result = run_quick_nas(num_classes=5)
        assert result["parameters"] > 0

    def test_search_time_positive(self):
        from api.nas_engine import run_quick_nas
        result = run_quick_nas(num_classes=5)
        assert result["search_time"] > 0

    def test_progress_callback_called(self):
        from api.nas_engine import run_quick_nas
        calls = []
        run_quick_nas(num_classes=3,
                      progress_callback=lambda e, t: calls.append((e, t)))
        assert len(calls) > 0

    def test_progress_callback_epochs_ascending(self):
        from api.nas_engine import run_quick_nas
        calls = []
        run_quick_nas(num_classes=3,
                      progress_callback=lambda e, t: calls.append(e))
        epochs = calls
        assert epochs == sorted(epochs)
