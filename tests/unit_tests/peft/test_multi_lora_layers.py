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
  * B9: load_adapter raises on a partial checkpoint instead of leaving slots at random init

Single-GPU integration (needs CUDA + model-parallel init):
  * B3: alpha/rank scaling is stored in fp32 and applied through a forward
  * B4: reset_adapter re-inits through the model-parallel RNG tracker (deterministic)
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft import multi_lora as multi_lora_mod
from megatron.bridge.peft import multi_lora_layers as mll
from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import (
    MultiLoRALinear,
    expose_adapter_slot,
    hide_adapters,
    load_adapter,
)


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
# B9: load_adapter raises on a partial checkpoint (missing adapter params).
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


# --------------------------------------------------------------------------- #
# B3 / B4: single-GPU integration through a real ColumnParallelLinear.
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
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
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

    def _build(self, dim=8, n_adapters=2, alpha=16):
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
            column_init_method="xavier",
            row_init_method="zero",
            dropout=0.0,
        )
        # Mirror model setup: adapter weights are cast to the compute dtype
        # (the base is bf16). alpha_values/rank_values are plain tensor
        # attributes (not params/buffers), so this cast leaves them fp32 — which
        # is the whole point of the B3 fix.
        mlora.adapters.to(device="cuda", dtype=torch.bfloat16)
        return mlora

    def test_scaling_stored_in_fp32(self):
        mlora = self._build()
        assert mlora.alpha_values.dtype == torch.float32
        assert mlora.rank_values.dtype == torch.float32

    def test_forward_runs_with_fp32_scaling(self):
        from megatron.bridge.peft.multi_lora_layers import (
            init_adapter_slot,
            set_tokens_per_adapter_slot,
        )

        mlora = self._build(dim=8, n_adapters=2, alpha=12)
        # non-power-of-two alpha/rank ratio (12/6) is exactly where bf16 would bias
        init_adapter_slot([mlora], 0, rank=6, alpha=12)
        tokens = 4
        set_tokens_per_adapter_slot(
            [mlora], torch.tensor([tokens, 0], dtype=torch.int32, device="cuda")
        )
        x = torch.randn(tokens, 16, dtype=torch.bfloat16, device="cuda")
        out, _ = mlora(x)
        assert out.shape[0] == tokens
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
