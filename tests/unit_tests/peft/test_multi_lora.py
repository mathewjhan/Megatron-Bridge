# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for multi-LoRA PEFT components."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft.multi_lora_state import (
    get_active_adapter_idx,
    get_lora_num_tokens,
    reset_state,
    set_lora_num_tokens,
)
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockLinearWithTupleReturn(nn.Module):
    """Mock linear module that returns tuples like Megatron layers."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        return self.linear(x), None  # (output, bias)


class MockLinearWithTripleReturn(nn.Module):
    """Mock linear module that returns (output, bias, layernorm_output)."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        return self.linear(x), None, x  # (output, bias, layernorm_output)


class MockParallelLinearAdapter(nn.Module):
    """Mock ParallelLinearAdapter for testing MultiLoRALinear."""

    def __init__(self, in_features=10, out_features=10, dim=8, alpha=32):
        super().__init__()
        self.linear_in = nn.Linear(in_features, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_features, bias=False)
        self.dim = dim
        self.alpha = alpha
        nn.init.zeros_(self.linear_out.weight)

    def _get_init_fn(self, init_method):
        if init_method == "xavier":
            return nn.init.xavier_normal_
        elif init_method == "zero":
            return lambda t: nn.init.constant_(t, 0.0)
        return nn.init.xavier_normal_

    def forward(self, x):
        out = self.linear_out(self.linear_in(x))
        return out * (self.alpha / self.dim)


# ---------------------------------------------------------------------------
# TestMultiLoRAState
# ---------------------------------------------------------------------------


class TestMultiLoRAState:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_state()
        yield
        reset_state()

    def test_set_and_get(self):
        t = torch.tensor([0, 10, 0], dtype=torch.int32)
        set_lora_num_tokens(t, reset_reference=True)
        assert torch.equal(get_lora_num_tokens(), t)

    def test_get_before_set_raises(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            get_lora_num_tokens()

    def test_in_place_copy(self):
        t = torch.tensor([0, 10, 0], dtype=torch.int32)
        set_lora_num_tokens(t, reset_reference=True)

        t2 = torch.tensor([5, 0, 5], dtype=torch.int32)
        set_lora_num_tokens(t2)

        # Should have been copied in-place into the original tensor
        assert torch.equal(get_lora_num_tokens(), t2)
        assert get_lora_num_tokens().data_ptr() == t.data_ptr()

    def test_reset_reference(self):
        t1 = torch.tensor([0, 10, 0], dtype=torch.int32)
        set_lora_num_tokens(t1, reset_reference=True)
        ptr1 = get_lora_num_tokens().data_ptr()

        t2 = torch.tensor([5, 0, 5], dtype=torch.int32)
        set_lora_num_tokens(t2, reset_reference=True)
        ptr2 = get_lora_num_tokens().data_ptr()

        assert ptr1 != ptr2

    def test_get_active_adapter_idx(self):
        set_lora_num_tokens(torch.tensor([0, 0, 42, 0]), reset_reference=True)
        assert get_active_adapter_idx() == 2

    def test_get_active_adapter_idx_first(self):
        set_lora_num_tokens(torch.tensor([100, 0, 0]), reset_reference=True)
        assert get_active_adapter_idx() == 0

    def test_reset_state(self):
        set_lora_num_tokens(torch.tensor([1, 2, 3]), reset_reference=True)
        reset_state()
        with pytest.raises(RuntimeError):
            get_lora_num_tokens()


# ---------------------------------------------------------------------------
# TestMultiLoRALinear
# ---------------------------------------------------------------------------


class TestMultiLoRALinear:
    N_ADAPTERS = 3
    IN_FEATURES = 10
    OUT_FEATURES = 10

    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_state()
        yield
        reset_state()

    @pytest.fixture
    def mock_linear(self):
        return MockLinearWithTupleReturn(self.IN_FEATURES, self.OUT_FEATURES)

    @pytest.fixture
    def mock_adapters(self):
        return nn.ModuleList(
            [MockParallelLinearAdapter(self.IN_FEATURES, self.OUT_FEATURES) for _ in range(self.N_ADAPTERS)]
        )

    @pytest.fixture
    def multi_lora(self, mock_linear, mock_adapters):
        return MultiLoRALinear(mock_linear, mock_adapters, self.N_ADAPTERS)

    def _set_tokens(self, *counts):
        """Set lora_num_tokens from a variable number of per-adapter counts."""
        t = torch.tensor(counts, dtype=torch.int32)
        set_lora_num_tokens(t, reset_reference=True)

    # --- Init ---

    def test_init(self, multi_lora, mock_linear, mock_adapters):
        assert multi_lora.to_wrap is mock_linear
        assert multi_lora.n_adapters == self.N_ADAPTERS
        assert len(multi_lora.adapters) == self.N_ADAPTERS
        assert multi_lora._adapter_enabled is True

    def test_scaling_initialized(self, multi_lora, mock_adapters):
        for i, adapter in enumerate(mock_adapters):
            assert multi_lora.scaling[i] == adapter.alpha / adapter.dim

    # --- Forward (single adapter) ---

    def test_forward_single_adapter(self, multi_lora, mock_linear, mock_adapters):
        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(0, 5, 0)

        output, bias = multi_lora(x)

        base_output, _ = mock_linear(x)
        adapter_output = mock_adapters[1](x.contiguous())
        expected = base_output + adapter_output
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_different_adapters_different_output(self, multi_lora, mock_adapters):
        nn.init.normal_(mock_adapters[0].linear_out.weight)

        x = torch.randn(5, self.IN_FEATURES)

        self._set_tokens(5, 0, 0)
        output_0, _ = multi_lora(x)

        self._set_tokens(0, 5, 0)
        output_1, _ = multi_lora(x)

        assert not torch.allclose(output_0, output_1)

    # --- Forward (mixed adapters) ---

    def test_forward_mixed_adapters(self, multi_lora, mock_linear, mock_adapters):
        """Tokens from two adapters in the same micro-batch."""
        n0, n1, n2 = 3, 2, 0
        total = n0 + n1 + n2
        x = torch.randn(total, self.IN_FEATURES)
        self._set_tokens(n0, n1, n2)

        output, bias = multi_lora(x)

        # Manually compute expected
        base_output, _ = mock_linear(x)
        a0_out = mock_adapters[0](x[:n0].contiguous())
        a1_out = mock_adapters[1](x[n0:n0 + n1].contiguous())
        expected = base_output.clone()
        expected[:n0] += a0_out
        expected[n0:n0 + n1] += a1_out
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_all_adapters_active(self, multi_lora, mock_linear, mock_adapters):
        """All three adapters have tokens."""
        n0, n1, n2 = 2, 3, 4
        total = n0 + n1 + n2
        x = torch.randn(total, self.IN_FEATURES)
        self._set_tokens(n0, n1, n2)

        output, bias = multi_lora(x)
        assert output.shape == (total, self.OUT_FEATURES)

    def test_forward_empty_adapter_in_middle(self, multi_lora, mock_linear, mock_adapters):
        """Adapter 1 has zero tokens, adapters 0 and 2 have tokens."""
        n0, n1, n2 = 3, 0, 4
        total = n0 + n1 + n2
        x = torch.randn(total, self.IN_FEATURES)
        self._set_tokens(n0, n1, n2)

        output, bias = multi_lora(x)

        base_output, _ = mock_linear(x)
        a0_out = mock_adapters[0](x[:n0].contiguous())
        a2_out = mock_adapters[2](x[n0:n0 + n2].contiguous())
        expected = base_output.clone()
        expected[:n0] += a0_out
        expected[n0:n0 + n2] += a2_out
        assert torch.allclose(output, expected, atol=1e-6)

    # --- Forward (disabled / edge cases) ---

    def test_forward_disabled(self, multi_lora, mock_linear):
        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(5, 0, 0)

        multi_lora.disable_adapter_layers()
        output, bias = multi_lora(x)

        base_output, _ = mock_linear(x)
        assert torch.allclose(output, base_output, atol=1e-6)

    def test_forward_re_enable(self, multi_lora):
        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(5, 0, 0)

        multi_lora.disable_adapter_layers()
        disabled_out, _ = multi_lora(x)

        multi_lora.enable_adapter_layers()
        enabled_out, _ = multi_lora(x)

        assert enabled_out.shape == disabled_out.shape

    def test_forward_with_triple_return(self, mock_adapters):
        base = MockLinearWithTripleReturn(self.IN_FEATURES, self.OUT_FEATURES)
        multi_lora = MultiLoRALinear(base, mock_adapters, self.N_ADAPTERS)

        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(2, 3, 0)

        output, bias = multi_lora(x)
        assert output.shape == (5, self.OUT_FEATURES)

    # --- Scaling ---

    def test_set_scaling(self, multi_lora):
        multi_lora.set_scaling(1, alpha=64, rank=16)
        assert multi_lora.scaling[1] == 64 / 16

    def test_forward_with_custom_scaling(self, multi_lora, mock_linear, mock_adapters):
        nn.init.normal_(mock_adapters[0].linear_out.weight)
        multi_lora.set_scaling(0, alpha=128, rank=8)

        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(5, 0, 0)

        output, _ = multi_lora(x)

        base_output, _ = mock_linear(x)
        adapter_raw = mock_adapters[0](x.contiguous())
        built_in_scale = mock_adapters[0].alpha / mock_adapters[0].dim
        correction = (128 / 8) / built_in_scale
        expected = base_output + adapter_raw * correction

        assert torch.allclose(output, expected, atol=1e-5)

    # --- Reset ---

    def test_reset_adapter(self, multi_lora, mock_adapters):
        # Dirty the adapter weights
        nn.init.normal_(mock_adapters[1].linear_in.weight)
        nn.init.normal_(mock_adapters[1].linear_out.weight)
        multi_lora.set_scaling(1, alpha=999, rank=1)

        multi_lora.reset_adapter(1)

        # B should be zeros
        assert torch.allclose(mock_adapters[1].linear_out.weight, torch.zeros_like(mock_adapters[1].linear_out.weight))
        # A should be non-zero (xavier)
        assert not torch.allclose(
            mock_adapters[1].linear_in.weight, torch.zeros_like(mock_adapters[1].linear_in.weight)
        )
        # Scaling should be reset to default
        assert multi_lora.scaling[1] == mock_adapters[1].alpha / mock_adapters[1].dim

    def test_reset_one_adapter_leaves_others_unchanged(self, multi_lora, mock_adapters):
        nn.init.normal_(mock_adapters[0].linear_out.weight)
        original_weight = mock_adapters[0].linear_out.weight.clone()

        multi_lora.reset_adapter(1)

        assert torch.equal(mock_adapters[0].linear_out.weight, original_weight)

    # --- Per-adapter parameters ---

    def test_named_parameters_for_adapter(self, multi_lora):
        params = list(multi_lora.named_parameters_for_adapter(1))
        names = [n for n, _ in params]

        assert any("adapters.1." in n for n in names)
        assert not any("adapters.0." in n for n in names)
        assert not any("adapters.2." in n for n in names)
        assert len(params) > 0

    def test_named_parameters_for_adapter_all_require_grad(self, multi_lora):
        for _, param in multi_lora.named_parameters_for_adapter(0):
            assert param.requires_grad

    # --- State dict ---

    def test_state_dict_for_adapter(self, multi_lora):
        sd = multi_lora.state_dict_for_adapter(1, prefix="layer.")
        assert len(sd) > 0
        for key in sd:
            assert key.startswith("layer.adapters.1.")

    def test_state_dict_contains_all_adapters(self, multi_lora):
        sd = multi_lora.state_dict()
        adapter_keys = [k for k in sd if "adapters." in k]
        for i in range(self.N_ADAPTERS):
            assert any(f"adapters.{i}." in k for k in adapter_keys), f"Missing adapter {i}"

    # --- Load adapter ---

    def test_load_adapter(self, multi_lora, mock_adapters):
        # Create a state dict with known weights
        known_weight = torch.ones_like(mock_adapters[2].linear_in.weight) * 42.0
        sd = {"linear_in.weight": known_weight, "linear_out.weight": torch.zeros_like(mock_adapters[2].linear_out.weight)}

        multi_lora.load_adapter(2, sd)

        assert torch.equal(mock_adapters[2].linear_in.weight.data, known_weight)


# ---------------------------------------------------------------------------
# TestMultiLoRA (PEFT class)
# ---------------------------------------------------------------------------


class TestMultiLoRA:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_state()
        yield
        reset_state()

    def test_transform_wraps_matching_modules(self, monkeypatch):
        from megatron.bridge.peft import multi_lora as multi_lora_module
        from megatron.bridge.peft.multi_lora import MultiLoRA
        from megatron.bridge.peft.utils import AdapterAttributes

        # Mock out Megatron parallel linear types so our mock matches
        fake_attrs = AdapterAttributes(
            input_is_parallel=False,
            in_features=10,
            out_features=10,
            disable_tensor_parallel_comm=False,
            disable_sequence_parallel_comm=True,
            base_linear_is_parallel=True,
        )
        monkeypatch.setattr(multi_lora_module, "get_adapter_attributes_from_linear", lambda m, **kw: fake_attrs)
        monkeypatch.setattr(multi_lora_module, "is_expert_linear", lambda fqn: False)

        # Create a mock ParallelLinearAdapter factory
        def mock_pla(*args, **kwargs):
            return MockParallelLinearAdapter(10, 10)

        monkeypatch.setattr(multi_lora_module, "ParallelLinearAdapter", mock_pla)

        # Build a model with a module named "linear_qkv"
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_qkv = MockLinearWithTupleReturn()
                self.linear_proj = MockLinearWithTupleReturn()
                self.other = nn.Linear(10, 10)

        model = SimpleModel()
        multi_lora = MultiLoRA(target_modules=["linear_qkv", "linear_proj"], n_adapters=3, dim=8, alpha=16)
        transformed = multi_lora(model, training=True)

        assert isinstance(transformed.linear_qkv, MultiLoRALinear)
        assert isinstance(transformed.linear_proj, MultiLoRALinear)
        # nn.Linear should NOT be wrapped (we skip plain nn.Linear)
        assert not isinstance(transformed.other, MultiLoRALinear)

    def test_transform_skips_already_transformed(self, monkeypatch):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        multi_lora = MultiLoRA(n_adapters=2)

        adapters = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])
        already_wrapped = MultiLoRALinear(MockLinearWithTupleReturn(), adapters, 2)

        result = multi_lora.transform(already_wrapped, name="test")
        assert result is already_wrapped

    def test_transform_skips_nn_linear(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        multi_lora = MultiLoRA(n_adapters=2)
        linear = nn.Linear(10, 10)

        result = multi_lora.transform(linear, name="linear_qkv")
        assert result is linear
        assert not isinstance(result, MultiLoRALinear)

    def test_transform_skips_expert_linears(self, monkeypatch):
        from megatron.bridge.peft import multi_lora as multi_lora_module
        from megatron.bridge.peft.multi_lora import MultiLoRA
        from megatron.bridge.peft.utils import AdapterAttributes

        fake_attrs = AdapterAttributes(
            input_is_parallel=False,
            in_features=10,
            out_features=10,
            disable_tensor_parallel_comm=False,
            disable_sequence_parallel_comm=True,
            base_linear_is_parallel=True,
        )
        monkeypatch.setattr(multi_lora_module, "get_adapter_attributes_from_linear", lambda m, **kw: fake_attrs)
        monkeypatch.setattr(multi_lora_module, "is_expert_linear", lambda fqn: True)

        multi_lora = MultiLoRA(target_modules=["linear_fc1"], n_adapters=2)
        module = MockLinearWithTupleReturn()

        result = multi_lora.transform(module, name="linear_fc1", prefix="mlp.experts.0")
        assert result is module

    def test_reset_adapter_across_model(self, monkeypatch):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        adapters_a = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])
        adapters_b = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])

        class TwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_a, 2)
                self.layer_b = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_b, 2)

        model = TwoLayerModel()
        multi_lora = MultiLoRA(n_adapters=2)

        # Dirty adapter 0 in both layers
        nn.init.normal_(adapters_a[0].linear_out.weight)
        nn.init.normal_(adapters_b[0].linear_out.weight)

        multi_lora.reset_adapter(model, 0)

        assert torch.allclose(adapters_a[0].linear_out.weight, torch.zeros_like(adapters_a[0].linear_out.weight))
        assert torch.allclose(adapters_b[0].linear_out.weight, torch.zeros_like(adapters_b[0].linear_out.weight))

    def test_set_adapter_scaling_across_model(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        adapters_a = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])
        adapters_b = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])

        class TwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_a, 2)
                self.layer_b = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_b, 2)

        model = TwoLayerModel()
        multi_lora = MultiLoRA(n_adapters=2)

        multi_lora.set_adapter_scaling(model, 1, alpha=64, rank=16)

        assert model.layer_a.scaling[1] == 64 / 16
        assert model.layer_b.scaling[1] == 64 / 16

    def test_named_parameters_for_adapter_across_model(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        adapters_a = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])
        adapters_b = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])

        class TwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_a, 2)
                self.layer_b = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_b, 2)

        model = TwoLayerModel()
        multi_lora = MultiLoRA(n_adapters=2)

        params = list(multi_lora.named_parameters_for_adapter(model, 0))
        names = [n for n, _ in params]

        assert any("layer_a" in n for n in names)
        assert any("layer_b" in n for n in names)
        assert all("adapters.0." in n for n in names)
        assert not any("adapters.1." in n for n in names)

    def test_state_dict_for_adapter_across_model(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        adapters_a = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])
        adapters_b = nn.ModuleList([MockParallelLinearAdapter() for _ in range(2)])

        class TwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_a, 2)
                self.layer_b = MultiLoRALinear(MockLinearWithTupleReturn(), adapters_b, 2)

        model = TwoLayerModel()
        multi_lora = MultiLoRA(n_adapters=2)

        sd = multi_lora.state_dict_for_adapter(model, 1)

        assert len(sd) > 0
        for key in sd:
            assert "adapters.1." in key

    def test_adapter_key_filter(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        multi_lora = MultiLoRA(n_adapters=2)
        assert multi_lora.adapter_key_filter("layer.adapters.0.linear_in.weight") is True
        assert multi_lora.adapter_key_filter("layer.to_wrap.weight") is False

    def test_adapter_key_filter_tuple(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        multi_lora = MultiLoRA(n_adapters=2)

        trainable_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        frozen_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        assert multi_lora.adapter_key_filter(("key", trainable_param)) is True
        assert multi_lora.adapter_key_filter(("key", frozen_param)) is False
