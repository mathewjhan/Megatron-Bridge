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

"""Tests for MultiLoRALinear hardening (disaggregated multi-LoRA).

Mock-level (no distributed):
  * B7: grouped-GEMM path rejects adapter dropout > 0 (it cannot apply it)
  * B2: expose_adapter_slot / hide_adapters restore the ModuleList even if the body raises
  * B8: MoE expert linears are skipped with a one-time warning, not silently
  * B9: load_adapter raises on a checkpoint/model mismatch in either direction
    (params missing from the checkpoint, or checkpoint tensors no param consumed)

Single-GPU integration (needs CUDA + model-parallel init):
  * grouped-GEMM forward smoke: output shape / finiteness / dtype (no fp32 promotion)
  * B4: reset_adapter re-inits through the model-parallel RNG tracker —
    deterministic given tracker state, mirroring the construction-time init methods
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft import multi_lora as multi_lora_mod
from megatron.bridge.peft import multi_lora_layers as multi_lora_layers_mod
from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import (
    MultiLoRALinear,
    expose_adapter_slot,
    hide_adapters,
    load_adapter,
)
from megatron.bridge.peft.utils import AdapterAttributes


# --------------------------------------------------------------------------- #
# B7: dropout is rejected. The assert is the first statement in __init__, so a
# dummy to_wrap is never touched.
# --------------------------------------------------------------------------- #
def test_dropout_rejected():
    with pytest.raises(AssertionError, match="does not apply adapter dropout"):
        MultiLoRALinear(
            to_wrap=nn.Linear(4, 4),
            n_adapters=2,
            dim=8,
            alpha=16,
            full_name="x",
            dropout=0.1,
        )


# --------------------------------------------------------------------------- #
# B2: the exposure/hiding context managers must restore state on exception.
# --------------------------------------------------------------------------- #
class _FakeMultiLoRALinear(MultiLoRALinear):
    """MultiLoRALinear instance without the heavy __init__ (isinstance still holds)."""

    def __init__(self):
        nn.Module.__init__(self)
        self.adapters = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


def test_expose_adapter_slot_restores_on_exception():
    m = _FakeMultiLoRALinear()
    with pytest.raises(RuntimeError):
        with expose_adapter_slot(m, 0):
            # inside the context the slot is exposed and the list is hidden
            assert "adapter" in m._modules
            assert "adapters" not in m._modules
            raise RuntimeError("boom during export")
    # ...and it is fully restored despite the exception
    assert "adapters" in m._modules
    assert "adapter" not in m._modules


def test_hide_adapters_restores_on_exception():
    m = _FakeMultiLoRALinear()
    with pytest.raises(RuntimeError):
        with hide_adapters(m):
            assert "adapters" not in m._modules
            raise RuntimeError("boom during base load")
    assert "adapters" in m._modules


def test_expose_adapter_slot_restores_on_success():
    m = _FakeMultiLoRALinear()
    with expose_adapter_slot(m, 1):
        assert "adapter" in m._modules
    assert "adapters" in m._modules
    assert "adapter" not in m._modules


# --------------------------------------------------------------------------- #
# B8: expert linears are skipped, but with a one-time warning (not silently).
# --------------------------------------------------------------------------- #
def test_expert_skip_warns_once():
    multi_lora_mod._EXPERT_SKIP_WARNED = False
    mlora = MultiLoRA(target_modules=["linear_fc1"], n_adapters=2, dim=8, alpha=16)
    module = nn.Linear(4, 4)
    full = "decoder.layers.0.mlp.experts.linear_fc1"
    with (
        patch.object(multi_lora_mod, "is_expert_linear", return_value=True),
        patch.object(mlora, "match", return_value=(MagicMock(), full)),
        patch.object(multi_lora_mod, "logger") as logmock,
    ):
        out1 = mlora.transform(module, name="linear_fc1", prefix="decoder.layers.0.mlp.experts.")
        out2 = mlora.transform(module, name="linear_fc1", prefix="decoder.layers.0.mlp.experts.")

    # expert modules are returned unwrapped...
    assert out1 is module and out2 is module
    # ...and the warning fires exactly once across both skips
    assert logmock.warning.call_count == 1


# --------------------------------------------------------------------------- #
# B9: load_adapter raises on a checkpoint/model mismatch in either direction.
# --------------------------------------------------------------------------- #
class _AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.adapter = nn.Module()
        self.layer.adapter.linear_in = nn.Linear(4, 8, bias=False)
        self.layer.adapter.linear_out = nn.Linear(8, 4, bias=False)


def test_load_adapter_partial_raises():
    m = _AdapterModel()
    partial = {"layer.adapter.linear_in.weight": torch.zeros(8, 4)}  # linear_out missing
    with pytest.raises(KeyError, match="absent from the checkpoint"):
        load_adapter(m, 0, partial)


def test_load_adapter_full_ok():
    m = _AdapterModel()
    full = {
        "layer.adapter.linear_in.weight": torch.ones(8, 4),
        "layer.adapter.linear_out.weight": torch.ones(4, 8),
    }
    assert load_adapter(m, 0, full) == 2
    assert torch.allclose(m.layer.adapter.linear_in.weight, torch.ones(8, 4))


def test_load_adapter_unused_keys_raises():
    m = _AdapterModel()
    over_full = {
        "layer.adapter.linear_in.weight": torch.ones(8, 4),
        "layer.adapter.linear_out.weight": torch.ones(4, 8),
        # e.g. saved with a larger target_modules set than the resuming model
        "other_layer.adapter.linear_in.weight": torch.ones(8, 4),
    }
    with pytest.raises(KeyError, match="matched no"):
        load_adapter(m, 0, over_full)


# --------------------------------------------------------------------------- #
# Adapter construction wiring (CPU: ParallelLinearAdapter replaced by a fake).
# --------------------------------------------------------------------------- #
class _RecordingAdapter(nn.Module):
    """CPU stand-in for ParallelLinearAdapter; records constructor kwargs."""

    def __init__(self, in_features, out_features, dim, base_linear_name, *, alpha=None, **extra_kwargs):
        super().__init__()
        self.dim = dim
        self.alpha = alpha if alpha is not None else dim
        self.base_linear_name = base_linear_name
        self.linear_in = nn.Linear(in_features, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_features, bias=False)
        self.extra_kwargs = extra_kwargs


def _fake_attrs(module, *args, **kwargs):
    return AdapterAttributes(
        input_is_parallel=False,
        in_features=module.in_features,
        out_features=module.out_features,
        disable_tensor_parallel_comm=False,
        disable_sequence_parallel_comm=True,
        base_linear_is_parallel=True,
    )


def _build_cpu_layer(n_adapters=2, dim=8, alpha=16, to_wrap=None):
    with (
        patch.object(multi_lora_layers_mod, "ParallelLinearAdapter", _RecordingAdapter),
        patch.object(multi_lora_layers_mod, "get_adapter_attributes_from_linear", _fake_attrs),
    ):
        return MultiLoRALinear(
            to_wrap=to_wrap if to_wrap is not None else nn.Linear(16, 32),
            n_adapters=n_adapters,
            dim=dim,
            alpha=alpha,
            full_name="linear_proj",
        )


def test_constructor_forwards_wrapped_module_runtime_config():
    base = nn.Linear(16, 32)
    base.config = object()

    layer = _build_cpu_layer(to_wrap=base)

    for adapter in layer.adapters:
        assert adapter.extra_kwargs["model_parallel_config"] is base.config
        assert adapter.extra_kwargs["disable_tensor_parallel_comm"] is False
        assert adapter.extra_kwargs["base_linear_is_parallel"] is True


# --------------------------------------------------------------------------- #
# Forward smoke / B4 reset: single-GPU integration through a real
# ColumnParallelLinear.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU + model-parallel init")
class TestMultiLoRALinearGPU:
    @pytest.fixture(autouse=True)
    def _mp(self):
        import megatron.core.parallel_state as parallel_state
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29555")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.cuda.set_device(0)
            dist.init_process_group(backend="nccl", world_size=1, rank=0)
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        yield
        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    def _build(self, dim=8, n_adapters=2, alpha=16, column_init_method="xavier"):
        from megatron.core.tensor_parallel import ColumnParallelLinear
        from megatron.core.transformer.transformer_config import TransformerConfig

        from megatron.bridge.peft.utils import init_method_normal

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        base = ColumnParallelLinear(
            16,
            16,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            gather_output=False,
        ).cuda()
        mlora = MultiLoRALinear(
            to_wrap=base,
            n_adapters=n_adapters,
            dim=dim,
            alpha=alpha,
            full_name="linear_qkv",
            column_init_method=column_init_method,
            row_init_method="zero",
            dropout=0.0,
        )
        # Mirror model setup: adapter weights are cast to the compute dtype
        # (the base is bf16).
        mlora.adapters.to(device="cuda", dtype=torch.bfloat16)
        return mlora

    def test_forward_grouped_gemm_smoke(self):
        from megatron.bridge.peft.multi_lora_layers import (
            init_adapter_slot,
            set_tokens_per_adapter_slot,
        )

        mlora = self._build(dim=8, n_adapters=2, alpha=12)
        init_adapter_slot([mlora], 0, rank=6, alpha=12)
        tokens = 4
        set_tokens_per_adapter_slot([mlora], torch.tensor([tokens, 0], dtype=torch.int32, device="cuda"))
        x = torch.randn(tokens, 16, dtype=torch.bfloat16, device="cuda")
        out, _ = mlora(x)
        assert out.shape[0] == tokens
        # scaling must not promote the activation dtype (bf16 * fp32 -> fp32)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    def test_reset_adapter_through_rng_tracker(self):
        mlora = self._build()
        idx = 0
        # perturb so we can see the re-init take effect
        with torch.no_grad():
            mlora.adapters[idx].linear_in.weight.fill_(7.0)
            mlora.adapters[idx].linear_out.weight.fill_(7.0)
        mlora.clear_adapter_slot(idx)  # -> reset_adapter under get_cuda_rng_tracker().fork()
        a = mlora.adapters[idx].linear_in.weight
        b = mlora.adapters[idx].linear_out.weight
        assert not torch.allclose(a, torch.full_like(a, 7.0))  # A re-initialized (xavier)
        assert torch.count_nonzero(b) == 0  # B zero-initialized

    def test_reset_adapter_deterministic_via_rng_tracker(self):
        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        def reseed():
            _set_random_seed(
                seed_=1234,
                data_parallel_random_init=False,
                te_rng_tracker=True,
                inference_rng_tracker=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )

        mlora = self._build()
        idx = 0

        reseed()
        torch.cuda.manual_seed(111)  # a bare nn.init would draw from here...
        mlora.clear_adapter_slot(idx)
        first = mlora.adapters[idx].linear_in.weight.clone()

        reseed()
        torch.cuda.manual_seed(222)  # ...and this seed change would alter its result
        mlora.clear_adapter_slot(idx)
        second = mlora.adapters[idx].linear_in.weight

        # An identically re-seeded tracker must give an identical re-init
        # regardless of the default generator — the draw has to come from the
        # tracker stream (this is what keeps DP replicas equal on slot reuse).
        assert torch.equal(first, second)

    def test_reset_adapter_mirrors_construction_init_methods(self):
        mlora = self._build(column_init_method="zero")
        idx = 0
        with torch.no_grad():
            mlora.adapters[idx].linear_in.weight.fill_(7.0)
        mlora.clear_adapter_slot(idx)
        # Construction used column_init_method="zero"; reset must reuse it (a
        # hardcoded xavier re-init would leave nonzero values here).
        assert torch.count_nonzero(mlora.adapters[idx].linear_in.weight) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
