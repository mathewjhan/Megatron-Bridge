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

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import megatron.core.parallel_state as parallel_state
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.peft.multi_lora_state import multi_lora_state
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, SimpleMultiLoRALinear


# ---------------------------------------------------------------------------
# TestMultiLoRAState
# ---------------------------------------------------------------------------


class TestMultiLoRAState:
    @pytest.fixture(autouse=True)
    def _reset(self):
        multi_lora_state.reset()
        yield
        multi_lora_state.reset()

    def test_init(self):
        multi_lora_state.init(n_adapters=3)
        assert multi_lora_state.lora_num_tokens.shape == (3,)
        assert multi_lora_state.scaling_factors.shape == (3,)

    def test_get_before_init_raises(self):
        with pytest.raises(AssertionError):
            multi_lora_state.get_lora_num_tokens()

    def test_get_scaling_before_init_raises(self):
        with pytest.raises(AssertionError):
            multi_lora_state.get_scaling_factors()

    def test_in_place_update(self):
        multi_lora_state.init(n_adapters=3)
        ptr = multi_lora_state.lora_num_tokens.data_ptr()
        multi_lora_state.lora_num_tokens.copy_(torch.tensor([5, 0, 5], dtype=torch.int32))
        assert multi_lora_state.lora_num_tokens.data_ptr() == ptr

    def test_reset(self):
        multi_lora_state.init(n_adapters=3)
        multi_lora_state.reset()
        assert multi_lora_state.lora_num_tokens is None
        assert multi_lora_state.scaling_factors is None


# ---------------------------------------------------------------------------
# TestSimpleMultiLoRALinear
# ---------------------------------------------------------------------------


class TestSimpleMultiLoRALinear:
    N_ADAPTERS = 3
    IN_FEATURES = 10
    OUT_FEATURES = 10

    @pytest.fixture(autouse=True)
    def _reset(self):
        multi_lora_state.reset()
        multi_lora_state.init(n_adapters=self.N_ADAPTERS)
        # Default scaling
        multi_lora_state.scaling_factors.copy_(
            torch.tensor([32.0 / 8] * self.N_ADAPTERS, dtype=multi_lora_state.scaling_factors.dtype)
        )
        yield
        multi_lora_state.reset()

    @pytest.fixture
    def orig_linear(self):
        return nn.Linear(self.IN_FEATURES, self.OUT_FEATURES)

    @pytest.fixture
    def multi_lora(self, orig_linear):
        return SimpleMultiLoRALinear(orig_linear, n_adapters=self.N_ADAPTERS, dim=8, alpha=32)

    def _set_tokens(self, *counts):
        multi_lora_state.lora_num_tokens.copy_(torch.tensor(counts, dtype=torch.int32))

    # --- Init ---

    def test_init(self, multi_lora, orig_linear):
        assert multi_lora.n_adapters == self.N_ADAPTERS
        assert len(multi_lora.adapters) == self.N_ADAPTERS
        assert not multi_lora.weight.requires_grad
        assert torch.equal(multi_lora.weight, orig_linear.weight)

    # --- Forward (single adapter) ---

    def test_forward_single_adapter(self, multi_lora):
        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(0, 5, 0)

        output = multi_lora(x)
        assert output.shape == (5, self.OUT_FEATURES)

    def test_forward_different_adapters_different_output(self, multi_lora):
        nn.init.normal_(multi_lora.adapters[0].linear_out.weight)

        x = torch.randn(5, self.IN_FEATURES)

        self._set_tokens(5, 0, 0)
        output_0 = multi_lora(x)

        self._set_tokens(0, 5, 0)
        output_1 = multi_lora(x)

        assert not torch.allclose(output_0, output_1)

    # --- Forward (mixed adapters) ---

    def test_forward_mixed_adapters(self, multi_lora):
        nn.init.normal_(multi_lora.adapters[0].linear_out.weight)
        n0, n1, n2 = 3, 2, 0
        total = n0 + n1 + n2
        x = torch.randn(total, self.IN_FEATURES)
        self._set_tokens(n0, n1, n2)

        output = multi_lora(x)

        # Manually compute expected
        base_out = torch.nn.functional.linear(x, multi_lora.weight, multi_lora.bias)
        a0_out = multi_lora.adapters[0](x[:n0], apply_scaling=False)
        a1_out = multi_lora.adapters[1](x[n0:n0 + n1], apply_scaling=False)
        sf = multi_lora_state.get_scaling_factors()
        expected = base_out.clone()
        expected[:n0] += a0_out * sf[0]
        expected[n0:n0 + n1] += a1_out * sf[1]
        assert torch.allclose(output, expected, atol=1e-5)

    def test_forward_all_adapters_active(self, multi_lora):
        n0, n1, n2 = 2, 3, 4
        total = n0 + n1 + n2
        x = torch.randn(total, self.IN_FEATURES)
        self._set_tokens(n0, n1, n2)

        output = multi_lora(x)
        assert output.shape == (total, self.OUT_FEATURES)

    def test_forward_empty_adapter_in_middle(self, multi_lora):
        n0, n1, n2 = 3, 0, 4
        total = n0 + n1 + n2
        x = torch.randn(total, self.IN_FEATURES)
        self._set_tokens(n0, n1, n2)

        output = multi_lora(x)
        assert output.shape == (total, self.OUT_FEATURES)

    # --- Forward (disabled) ---

    def test_forward_disabled(self, multi_lora):
        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(5, 0, 0)

        multi_lora.disable_adapter_layers()
        output = multi_lora(x)

        base_out = torch.nn.functional.linear(x, multi_lora.weight, multi_lora.bias)
        assert torch.allclose(output, base_out, atol=1e-6)

    # --- Scaling ---

    def test_scaling_from_global_state(self, multi_lora):
        nn.init.normal_(multi_lora.adapters[0].linear_out.weight)
        multi_lora_state.scaling_factors.copy_(
            torch.tensor([16.0, 4.0, 4.0], dtype=multi_lora_state.scaling_factors.dtype)
        )

        x = torch.randn(5, self.IN_FEATURES)
        self._set_tokens(5, 0, 0)

        output = multi_lora(x)

        base_out = torch.nn.functional.linear(x, multi_lora.weight, multi_lora.bias)
        adapter_raw = multi_lora.adapters[0](x, apply_scaling=False)
        expected = base_out + adapter_raw * 16.0
        assert torch.allclose(output, expected, atol=1e-5)

    # --- Reset ---

    def test_reset_adapter(self, multi_lora):
        nn.init.normal_(multi_lora.adapters[1].linear_in.weight)
        nn.init.normal_(multi_lora.adapters[1].linear_out.weight)

        multi_lora.reset_adapter(1)

        assert torch.allclose(
            multi_lora.adapters[1].linear_out.weight, torch.zeros_like(multi_lora.adapters[1].linear_out.weight)
        )
        assert not torch.allclose(
            multi_lora.adapters[1].linear_in.weight, torch.zeros_like(multi_lora.adapters[1].linear_in.weight)
        )

    def test_reset_one_adapter_leaves_others_unchanged(self, multi_lora):
        nn.init.normal_(multi_lora.adapters[0].linear_out.weight)
        original_weight = multi_lora.adapters[0].linear_out.weight.clone()

        multi_lora.reset_adapter(1)

        assert torch.equal(multi_lora.adapters[0].linear_out.weight, original_weight)

    # --- Per-adapter parameters ---

    def test_named_parameters_for_adapter(self, multi_lora):
        params = list(multi_lora.named_parameters_for_adapter(1))
        names = [n for n, _ in params]

        assert any("adapters.1." in n for n in names)
        assert not any("adapters.0." in n for n in names)
        assert len(params) > 0

    # --- State dict ---

    def test_state_dict_for_adapter(self, multi_lora):
        sd = multi_lora.state_dict_for_adapter(1, prefix="layer.")
        assert len(sd) > 0
        for key in sd:
            assert key.startswith("layer.adapters.1.")

    # --- Load adapter ---

    def test_load_adapter(self, multi_lora):
        known_weight = torch.ones_like(multi_lora.adapters[2].linear_in.weight) * 42.0
        sd = {
            "linear_in.weight": known_weight,
            "linear_out.weight": torch.zeros_like(multi_lora.adapters[2].linear_out.weight),
        }
        multi_lora.load_adapter(2, sd)
        assert torch.equal(multi_lora.adapters[2].linear_in.weight.data, known_weight)


# ---------------------------------------------------------------------------
# TestMultiLoRA (PEFT class)
# ---------------------------------------------------------------------------


class TestMultiLoRA:
    @pytest.fixture(autouse=True)
    def _reset(self):
        multi_lora_state.reset()
        yield
        multi_lora_state.reset()

    def test_transform_wraps_nn_linear(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(10, 10)
                self.k_proj = nn.Linear(10, 10)
                self.other = nn.Linear(10, 10)

        model = SimpleModel()
        multi_lora = MultiLoRA(target_modules=["q_proj", "k_proj"], n_adapters=3, dim=8, alpha=16)
        transformed = multi_lora(model, training=True)

        assert isinstance(transformed.q_proj, SimpleMultiLoRALinear)
        assert isinstance(transformed.k_proj, SimpleMultiLoRALinear)
        assert not isinstance(transformed.other, SimpleMultiLoRALinear)

    def test_transform_skips_already_transformed(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        multi_lora = MultiLoRA(n_adapters=2)
        orig = nn.Linear(10, 10)
        wrapped = SimpleMultiLoRALinear(orig, n_adapters=2)

        result = multi_lora.transform(wrapped, name="test")
        assert result is wrapped

    def test_reset_adapter_across_model(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        class TwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = SimpleMultiLoRALinear(nn.Linear(10, 10), n_adapters=2)
                self.layer_b = SimpleMultiLoRALinear(nn.Linear(10, 10), n_adapters=2)

        model = TwoLayerModel()
        multi_lora = MultiLoRA(n_adapters=2)

        nn.init.normal_(model.layer_a.adapters[0].linear_out.weight)
        nn.init.normal_(model.layer_b.adapters[0].linear_out.weight)

        multi_lora.reset_adapter(model, 0)

        assert torch.allclose(
            model.layer_a.adapters[0].linear_out.weight,
            torch.zeros_like(model.layer_a.adapters[0].linear_out.weight),
        )
        assert torch.allclose(
            model.layer_b.adapters[0].linear_out.weight,
            torch.zeros_like(model.layer_b.adapters[0].linear_out.weight),
        )

    def test_named_parameters_for_adapter_across_model(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        class TwoLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = SimpleMultiLoRALinear(nn.Linear(10, 10), n_adapters=2)
                self.layer_b = SimpleMultiLoRALinear(nn.Linear(10, 10), n_adapters=2)

        model = TwoLayerModel()
        multi_lora = MultiLoRA(n_adapters=2)

        params = list(multi_lora.named_parameters_for_adapter(model, 0))
        names = [n for n, _ in params]

        assert any("layer_a" in n for n in names)
        assert any("layer_b" in n for n in names)
        assert all("adapters.0." in n for n in names)
        assert not any("adapters.1." in n for n in names)

    def test_adapter_key_filter(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        multi_lora = MultiLoRA(n_adapters=2)
        assert multi_lora.adapter_key_filter("layer.adapters.0.linear_in.weight") is True
        assert multi_lora.adapter_key_filter("layer.to_wrap.weight") is False

    def test_adapter_key_filter_tuple(self):
        from megatron.bridge.peft.multi_lora import MultiLoRA

        multi_lora = MultiLoRA(n_adapters=2)

        trainable = nn.Parameter(torch.zeros(1), requires_grad=True)
        frozen = nn.Parameter(torch.zeros(1), requires_grad=False)

        assert multi_lora.adapter_key_filter(("key", trainable)) is True
        assert multi_lora.adapter_key_filter(("key", frozen)) is False


# ---------------------------------------------------------------------------
# TestMultiLoRAMegatronIntegration
# ---------------------------------------------------------------------------


class TestMultiLoRAMegatronIntegration:
    """Integration tests for MultiLoRA with real Megatron models."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        multi_lora_state.reset()

        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            dist.init_process_group(
                backend="nccl" if device_count > 0 else "gloo",
                world_size=1,
                rank=0,
                timeout=datetime.timedelta(minutes=30),
            )

        assert dist.is_initialized()
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        assert parallel_state.model_parallel_is_initialized()
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.bridge.training.initialize import _set_random_seed

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=pg_collection,
        )

        yield

        multi_lora_state.reset()
        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    def test_multi_lora_with_gpt_model(self):
        """Test MultiLoRA application to a real Megatron GPT model."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.bridge.peft.multi_lora import MultiLoRA
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            vocab_size=1000,
            ffn_hidden_size=256,
        )
        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        multi_lora = MultiLoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            n_adapters=3,
            dim=8,
            alpha=16,
        )

        def multi_lora_hook(model: list[MegatronModule]) -> list[MegatronModule]:
            return multi_lora(model, training=True)

        model_provider.register_pre_wrap_hook(multi_lora_hook)
        model_provider.finalize()

        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)

        assert isinstance(adapted_model, list)
        assert len(adapted_model) > 0

        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Verify MultiLoRALinear modules were created
        found_modules = []
        for chunk in adapted_model:
            for name, module in chunk.named_modules():
                if isinstance(module, MultiLoRALinear):
                    found_modules.append(name)

        assert len(found_modules) > 0, f"No MultiLoRALinear modules found"

        # Verify parameter efficiency
        total_params = sum(p.numel() for chunk in adapted_model for p in chunk.parameters())
        trainable_params = sum(p.numel() for chunk in adapted_model for p in chunk.parameters() if p.requires_grad)

        assert trainable_params < total_params
        assert trainable_params > 0

    def test_multi_lora_forward_with_gpt_model(self):
        """Test that MultiLoRA-wrapped GPT model can run a forward pass."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.bridge.peft.multi_lora import MultiLoRA
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=100,
            ffn_hidden_size=128,
        )
        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        n_adapters = 2
        multi_lora = MultiLoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            n_adapters=n_adapters,
            dim=4,
            alpha=8,
        )

        def multi_lora_hook(model):
            return multi_lora(model, training=True)

        model_provider.register_pre_wrap_hook(multi_lora_hook)
        model_provider.finalize()

        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Set up global state
        seq_len = 8
        n0, n1 = 3, 5
        multi_lora_state.init(n_adapters=n_adapters, device="cuda")
        multi_lora_state.lora_num_tokens.copy_(torch.tensor([n0, n1], dtype=torch.int32))
        multi_lora_state.scaling_factors.copy_(torch.tensor([8.0 / 4, 8.0 / 4]))

        # Create input tokens
        tokens = torch.randint(0, 100, (1, seq_len), device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

        # Forward pass through first chunk (TP=1, PP=1 so there's one chunk)
        model = adapted_model[0]
        model.eval()

        with torch.no_grad():
            output = model(tokens, position_ids, None)

        assert output is not None

    def test_multi_lora_reset_with_gpt_model(self):
        """Test adapter reset on a Megatron GPT model."""
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.bridge.peft.multi_lora import MultiLoRA
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider = GPTModelProvider(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=2,
            vocab_size=100,
            ffn_hidden_size=128,
        )
        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        multi_lora = MultiLoRA(
            target_modules=["linear_qkv"],
            n_adapters=2,
            dim=4,
            alpha=8,
        )

        def multi_lora_hook(model):
            return multi_lora(model, training=True)

        model_provider.register_pre_wrap_hook(multi_lora_hook)
        model_provider.finalize()

        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        adapted_model = [chunk.cuda() for chunk in adapted_model]

        # Dirty adapter 0 weights
        for chunk in adapted_model:
            for module in chunk.modules():
                if isinstance(module, MultiLoRALinear):
                    nn.init.normal_(module.multi_adapter.weight_B.data[0])

        # Reset adapter 0
        multi_lora.reset_adapter(adapted_model, 0)

        # Verify B weights for adapter 0 are zeros
        for chunk in adapted_model:
            for module in chunk.modules():
                if isinstance(module, MultiLoRALinear):
                    assert torch.allclose(
                        module.multi_adapter.weight_B.data[0],
                        torch.zeros_like(module.multi_adapter.weight_B.data[0]),
                    )
