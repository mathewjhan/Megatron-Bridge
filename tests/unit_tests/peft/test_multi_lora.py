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

from megatron.bridge.peft import multi_lora_state
from megatron.bridge.peft.multi_lora import MultiLoRA
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
        multi_lora_state.init(n_adapters=3, device=torch.device("cpu"))
        assert multi_lora_state.tokens_per_adapter.shape == (3,)
        assert multi_lora_state.alpha.shape == (3,)
        assert multi_lora_state.rank.shape == (3,)

    def test_get_before_init_raises(self):
        with pytest.raises(AssertionError):
            multi_lora_state.get_tokens_per_adapter()

    def test_scaling_factors_computed(self):
        multi_lora_state.init(n_adapters=2, device=torch.device("cpu"))
        multi_lora_state.alpha.copy_(torch.tensor([32.0, 16.0]))
        multi_lora_state.rank.copy_(torch.tensor([8.0, 4.0]))
        sf = multi_lora_state.get_scaling_factors()
        assert torch.allclose(sf, torch.tensor([4.0, 4.0]))

    def test_reset(self):
        multi_lora_state.init(n_adapters=3, device=torch.device("cpu"))
        multi_lora_state.reset()
        assert multi_lora_state.tokens_per_adapter is None
        assert multi_lora_state.alpha is None
        assert multi_lora_state.rank is None


# ---------------------------------------------------------------------------
# TestMultiLoRA (registry + routing + transform)
# ---------------------------------------------------------------------------


class TestMultiLoRA:
    N_ADAPTERS = 3
    IN_FEATURES = 10
    OUT_FEATURES = 10

    @pytest.fixture(autouse=True)
    def _reset(self):
        multi_lora_state.reset()
        yield
        multi_lora_state.reset()

    @pytest.fixture
    def multi_lora(self):
        return MultiLoRA(
            target_modules=["q_proj", "k_proj"],
            n_adapters=self.N_ADAPTERS,
            dim=8,
            alpha=32,
        )

    @pytest.fixture
    def simple_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(10, 10)
                self.k_proj = nn.Linear(10, 10)
                self.other = nn.Linear(10, 10)
        return Model()

    @pytest.fixture
    def transformed_model(self, multi_lora, simple_model):
        return multi_lora(simple_model, training=True)

    # --- Transform ---

    def test_transform_wraps_targets(self, transformed_model):
        assert isinstance(transformed_model.q_proj, SimpleMultiLoRALinear)
        assert isinstance(transformed_model.k_proj, SimpleMultiLoRALinear)
        assert not isinstance(transformed_model.other, SimpleMultiLoRALinear)

    def test_transform_inits_state(self, multi_lora, simple_model):
        assert multi_lora_state.tokens_per_adapter is None
        multi_lora(simple_model, training=True)
        assert multi_lora_state.tokens_per_adapter is not None
        assert multi_lora_state.tokens_per_adapter.shape == (self.N_ADAPTERS,)

    def test_transform_idempotent(self, multi_lora, simple_model):
        model = multi_lora(simple_model, training=True)
        ref = model.q_proj
        model = multi_lora(model, training=True)
        assert model.q_proj is ref

    # --- Registry ---

    def test_register_adapter(self, multi_lora, transformed_model):
        idx = multi_lora.register_adapter("math-lora", rank=16, alpha=32)
        assert idx >= 0
        assert multi_lora.get_adapter_idx("math-lora") == idx
        assert multi_lora_state.alpha[idx] == 32
        assert multi_lora_state.rank[idx] == 16

    def test_register_duplicate_raises(self, multi_lora, transformed_model):
        multi_lora.register_adapter("math-lora", rank=16, alpha=32)
        with pytest.raises(ValueError, match="already registered"):
            multi_lora.register_adapter("math-lora", rank=16, alpha=32)

    def test_register_full_raises(self, multi_lora, transformed_model):
        for i in range(self.N_ADAPTERS):
            multi_lora.register_adapter(f"lora-{i}", rank=8, alpha=16)
        with pytest.raises(ValueError, match="No free"):
            multi_lora.register_adapter("extra-lora", rank=8, alpha=16)

    def test_unregister_adapter(self, multi_lora, transformed_model):
        multi_lora.register_adapter("math-lora", rank=16, alpha=32)
        idx = multi_lora.unregister_adapter("math-lora")
        assert "math-lora" not in multi_lora.registered_adapters
        assert multi_lora_state.tokens_per_adapter[idx] == 0

    def test_unregister_frees_slot(self, multi_lora, transformed_model):
        for i in range(self.N_ADAPTERS):
            multi_lora.register_adapter(f"lora-{i}", rank=8, alpha=16)
        multi_lora.unregister_adapter("lora-1")
        idx = multi_lora.register_adapter("new-lora", rank=8, alpha=16)
        assert idx == multi_lora.get_adapter_idx("new-lora")

    def test_unregister_unknown_raises(self, multi_lora, transformed_model):
        with pytest.raises(KeyError):
            multi_lora.unregister_adapter("nonexistent")

    # --- Batch routing ---

    def test_set_batch(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=16)
        multi_lora.register_adapter("b", rank=8, alpha=16)
        multi_lora.set_batch({"a": 100, "b": 200})

        tpa = multi_lora_state.tokens_per_adapter
        idx_a = multi_lora.get_adapter_idx("a")
        idx_b = multi_lora.get_adapter_idx("b")
        assert tpa[idx_a] == 100
        assert tpa[idx_b] == 200
        # Unregistered slots should be zero
        all_idxs = {idx_a, idx_b}
        for i in range(self.N_ADAPTERS):
            if i not in all_idxs:
                assert tpa[i] == 0

    def test_set_batch_zeros_previous(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=16)
        multi_lora.register_adapter("b", rank=8, alpha=16)

        multi_lora.set_batch({"a": 100, "b": 200})
        multi_lora.set_batch({"a": 50})

        idx_a = multi_lora.get_adapter_idx("a")
        idx_b = multi_lora.get_adapter_idx("b")
        assert multi_lora_state.tokens_per_adapter[idx_a] == 50
        assert multi_lora_state.tokens_per_adapter[idx_b] == 0

    # --- Forward ---

    def test_forward_single_adapter(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=32)
        multi_lora.set_batch({"a": 5})

        x = torch.randn(5, self.IN_FEATURES)
        output = transformed_model.q_proj(x)
        assert output.shape == (5, self.OUT_FEATURES)

    def test_forward_mixed_adapters(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=32)
        multi_lora.register_adapter("b", rank=8, alpha=32)
        nn.init.normal_(transformed_model.q_proj.adapters[multi_lora.get_adapter_idx("a")].linear_out.weight)

        multi_lora.set_batch({"a": 3, "b": 2})
        x = torch.randn(5, self.IN_FEATURES)
        output = transformed_model.q_proj(x)
        assert output.shape == (5, self.OUT_FEATURES)

    def test_forward_different_adapters_different_output(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=32)
        multi_lora.register_adapter("b", rank=8, alpha=32)
        nn.init.normal_(transformed_model.q_proj.adapters[multi_lora.get_adapter_idx("a")].linear_out.weight)

        x = torch.randn(5, self.IN_FEATURES)

        multi_lora.set_batch({"a": 5})
        out_a = transformed_model.q_proj(x).clone()

        multi_lora.set_batch({"b": 5})
        out_b = transformed_model.q_proj(x).clone()

        assert not torch.allclose(out_a, out_b)

    # --- Weight lifecycle ---

    def test_reset_adapter(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=32)
        idx = multi_lora.get_adapter_idx("a")
        nn.init.normal_(transformed_model.q_proj.adapters[idx].linear_out.weight)

        multi_lora.reset_adapter(transformed_model, "a")

        assert torch.allclose(
            transformed_model.q_proj.adapters[idx].linear_out.weight,
            torch.zeros_like(transformed_model.q_proj.adapters[idx].linear_out.weight),
        )

    def test_named_parameters_for_adapter(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=32)
        params = list(multi_lora.named_parameters_for_adapter(transformed_model, "a"))
        assert len(params) > 0
        names = [n for n, _ in params]
        idx = multi_lora.get_adapter_idx("a")
        assert all(f"adapters.{idx}." in n for n in names)

    def test_state_dict_for_adapter(self, multi_lora, transformed_model):
        multi_lora.register_adapter("a", rank=8, alpha=32)
        sd = multi_lora.state_dict_for_adapter(transformed_model, "a")
        assert len(sd) > 0

    # --- Checkpoint filtering ---

    def test_adapter_key_filter(self, multi_lora):
        assert multi_lora.adapter_key_filter("layer.adapters.0.linear_in.weight") is True
        assert multi_lora.adapter_key_filter("layer.weight_A.0") is True
        assert multi_lora.adapter_key_filter("layer.to_wrap.weight") is False

    def test_adapter_key_filter_tuple(self, multi_lora):
        trainable = nn.Parameter(torch.zeros(1), requires_grad=True)
        frozen = nn.Parameter(torch.zeros(1), requires_grad=False)
        assert multi_lora.adapter_key_filter(("key", trainable)) is True
        assert multi_lora.adapter_key_filter(("key", frozen)) is False


# ---------------------------------------------------------------------------
# TestMultiLoRAMegatronIntegration
# ---------------------------------------------------------------------------


class TestMultiLoRAMegatronIntegration:
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
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider = GPTModelProvider(
            num_layers=2, hidden_size=128, num_attention_heads=2,
            vocab_size=1000, ffn_hidden_size=256,
        )
        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        multi_lora = MultiLoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            n_adapters=3, dim=8, alpha=16,
        )

        def hook(model):
            return multi_lora(model, training=True)

        model_provider.register_pre_wrap_hook(hook)
        model_provider.finalize()

        model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        model = [chunk.cuda() for chunk in model]

        found = sum(1 for chunk in model for _, m in chunk.named_modules() if isinstance(m, MultiLoRALinear))
        assert found > 0

        total = sum(p.numel() for chunk in model for p in chunk.parameters())
        trainable = sum(p.numel() for chunk in model for p in chunk.parameters() if p.requires_grad)
        assert 0 < trainable < total

    def test_multi_lora_forward_with_gpt_model(self):
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider = GPTModelProvider(
            num_layers=1, hidden_size=64, num_attention_heads=2,
            vocab_size=100, ffn_hidden_size=128,
        )
        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        multi_lora = MultiLoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            n_adapters=2, dim=4, alpha=8,
        )

        def hook(model):
            return multi_lora(model, training=True)

        model_provider.register_pre_wrap_hook(hook)
        model_provider.finalize()

        model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        model = [chunk.cuda() for chunk in model]

        # Register adapters and set batch
        multi_lora.register_adapter("a", rank=4, alpha=8)
        multi_lora.register_adapter("b", rank=4, alpha=8)
        multi_lora.set_batch({"a": 3, "b": 5})

        tokens = torch.randint(0, 100, (1, 8), device="cuda")
        position_ids = torch.arange(8, device="cuda").unsqueeze(0)

        model[0].eval()
        with torch.no_grad():
            output = model[0](tokens, position_ids, None)
        assert output is not None

    def test_multi_lora_reset_with_gpt_model(self):
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider = GPTModelProvider(
            num_layers=1, hidden_size=64, num_attention_heads=2,
            vocab_size=100, ffn_hidden_size=128,
        )
        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        multi_lora = MultiLoRA(
            target_modules=["linear_qkv"], n_adapters=2, dim=4, alpha=8,
        )

        def hook(model):
            return multi_lora(model, training=True)

        model_provider.register_pre_wrap_hook(hook)
        model_provider.finalize()

        model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        model = [chunk.cuda() for chunk in model]

        multi_lora.register_adapter("a", rank=4, alpha=8)

        # Dirty adapter weights
        for chunk in model:
            for module in chunk.modules():
                if isinstance(module, MultiLoRALinear):
                    nn.init.normal_(module.multi_adapter.weight_B[0].data)

        multi_lora.reset_adapter(model, "a")

        for chunk in model:
            for module in chunk.modules():
                if isinstance(module, MultiLoRALinear):
                    idx = multi_lora.get_adapter_idx("a")
                    assert torch.allclose(
                        module.multi_adapter.weight_B[idx].data,
                        torch.zeros_like(module.multi_adapter.weight_B[idx].data),
                    )
