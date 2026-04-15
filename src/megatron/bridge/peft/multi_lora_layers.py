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

"""Multi-adapter LoRA layer for Megatron parallel linears.

:class:`MultiLoRALinear` wraps a single Megatron parallel linear module with
*N* concurrent LoRA adapters.  The active adapter is selected at forward time
via per-layer ``tokens_per_adapter`` set by :func:`set_batch`.

Two forward implementations are provided:

* **for-loop** (default) — slices tokens by adapter, runs each adapter's
  full ``ParallelLinearAdapter.forward()``.  TP/SP-safe by construction.
* **grouped GEMM** — stacks raw weights and uses ``torch._grouped_mm``
  for a single fused kernel.  Requires manual TP/SP handling.  Experimental.
"""

import math
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.utils import all2all_hp2sp


class SimpleLoRAAdapter(nn.Module):
    """Lightweight LoRA adapter for plain ``nn.Linear`` modules.

    Holds a ``linear_in`` (A) and ``linear_out`` (B) pair with scaling.
    Unlike :class:`ParallelLinearAdapter`, this has no TP/SP communication.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: str = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha

        dtype = lora_dtype
        self.linear_in = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        self.linear_out = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)

        self._init_weights(lora_A_init_method)

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.dropout_position = dropout_position

    def _init_weights(self, lora_A_init_method: str) -> None:
        if lora_A_init_method == "xavier":
            nn.init.xavier_normal_(self.linear_in.weight.data)
        elif lora_A_init_method == "kaiming":
            nn.init.kaiming_uniform_(self.linear_in.weight.data, a=math.sqrt(5))
        else:
            nn.init.xavier_normal_(self.linear_in.weight.data)
        nn.init.zeros_(self.linear_out.weight.data)

    def _get_init_fn(self, init_method: str):
        if init_method == "xavier":
            return nn.init.xavier_normal_
        elif init_method == "kaiming":
            from megatron.bridge.peft.utils import init_method_kaiming_uniform
            return init_method_kaiming_uniform(math.sqrt(5))
        elif init_method == "zero":
            return lambda t: nn.init.constant_(t, 0.0)
        elif init_method == "normal":
            from megatron.bridge.peft.utils import init_method_normal
            return init_method_normal(0.2)
        raise NotImplementedError(f"Unknown init method: {init_method}")

    def forward(self, x: torch.Tensor, apply_scaling: bool = True) -> torch.Tensor:
        if self.dropout_position == "pre":
            x = self.dropout(x)
        out = self.linear_out(self.linear_in(x))
        if apply_scaling:
            out = out * (self.alpha / self.dim)
        if self.dropout_position == "post":
            out = self.dropout(out)
        return out


class SimpleMultiLoRALinear(nn.Linear):
    """Plain ``nn.Linear`` wrapped with *N* concurrent LoRA adapters.

    Extends ``nn.Linear`` (like :class:`LinearAdapter`), copies the original
    weights, freezes them, and adds N :class:`SimpleLoRAAdapter` instances.
    Returns a plain tensor — compatible with HF models.

    Args:
        orig_linear: The original ``nn.Linear`` to adapt.
        n_adapters: Number of adapter slots.
        dim: LoRA rank.
        alpha: LoRA scaling parameter.
        dropout: Dropout probability.
        dropout_position: ``'pre'`` or ``'post'``.
        lora_A_init_method: Init method for the A matrix.
        lora_dtype: Data type for adapter weights.
    """

    def __init__(
        self,
        orig_linear: nn.Linear,
        n_adapters: int,
        dim: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: str = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        assert isinstance(orig_linear, nn.Linear)
        super().__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.n_adapters = n_adapters
        self.column_init_method = lora_A_init_method
        self.row_init_method = "zero"
        self._adapter_enabled = True
        self.tokens_per_adapter: Optional[torch.Tensor] = None
        self.scaling_factors: Optional[torch.Tensor] = None
        self.max_rank = dim

        dtype = lora_dtype or orig_linear.weight.dtype
        device = orig_linear.weight.device
        self.alpha_values = torch.ones(n_adapters, dtype=dtype, device=device)
        self.rank_values = torch.ones(n_adapters, dtype=dtype, device=device)
        self.adapters = nn.ModuleList([
            SimpleLoRAAdapter(
                orig_linear.in_features,
                orig_linear.out_features,
                dim=dim,
                alpha=alpha,
                dropout=dropout,
                dropout_position=dropout_position,
                lora_A_init_method=lora_A_init_method,
                lora_dtype=dtype,
                device=orig_linear.weight.device,
            )
            for _ in range(n_adapters)
        ])

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)

        if not self._adapter_enabled:
            return base_out

        tokens_per_adapter = self.tokens_per_adapter
        x_flat = x.reshape(-1, x.shape[-1])
        offsets = tokens_per_adapter.cumsum(dim=0)
        total = offsets[-1].item()
        assert total == x_flat.shape[0], (
            f"tokens_per_adapter sum {total} != token count {x_flat.shape[0]}"
        )

        adapter_outputs = []
        prev = 0
        for i in range(self.n_adapters):
            cur = offsets[i].item()
            if cur == prev:
                prev = cur
                continue
            out = self.adapters[i](x_flat[prev:cur], apply_scaling=False)
            adapter_outputs.append(out)
            prev = cur

        if not adapter_outputs:
            return base_out

        adapter_output = torch.cat(adapter_outputs, dim=0)

        per_token_scaling = torch.repeat_interleave(self.scaling_factors, tokens_per_adapter).unsqueeze(-1)
        adapter_output = adapter_output * per_token_scaling

        return base_out + adapter_output.reshape(base_out.shape)

    # --- Per-adapter lifecycle (same interface as MultiLoRALinear) ---

    def reset_adapter(self, idx: int) -> None:
        adapter = self.adapters[idx]
        adapter._get_init_fn(self.column_init_method)(adapter.linear_in.weight.data)
        adapter._get_init_fn(self.row_init_method)(adapter.linear_out.weight.data)

    def named_parameters_for_adapter(self, idx: int) -> Iterator[Tuple[str, nn.Parameter]]:
        prefix = f"adapters.{idx}."
        for name, param in self.adapters[idx].named_parameters():
            yield prefix + name, param

    def state_dict_for_adapter(self, idx: int, prefix: str = "") -> Dict[str, Any]:
        sd: Dict[str, Any] = {}
        self.adapters[idx].state_dict(destination=sd, prefix=f"{prefix}adapters.{idx}.")
        return sd

    def load_adapter(self, idx: int, state_dict: Dict[str, torch.Tensor]) -> None:
        self.adapters[idx].load_state_dict(state_dict, strict=True)


class MultiParallelLinearAdapter(nn.Module):
    """Grouped GEMM multi-adapter with TP/SP comms done once, not N times.

    Stores *N* adapters' weights as ``nn.ParameterList`` (one ``nn.Parameter``
    per adapter per matrix) and uses ``torch._grouped_mm`` for fused
    per-adapter matmuls.  SP gather/scatter and TP all-gather/all-reduce
    are performed once around the grouped GEMMs.

    Supports both ``thd`` (packed, input ``[T, H]``) and ``bshd``
    (padded batch, input ``[B, S, H]``) — the forward flattens to 2D
    before grouped GEMM regardless of format.

    Args:
        n_adapters: Number of adapter slots.
        in_features: Full (unsharded) input features of the base linear.
        out_features: Full (unsharded) output features of the base linear.
        dim: LoRA rank.
        alpha: LoRA scaling parameter.
        input_is_parallel: Whether the base linear is RowParallel (input sharded).
        column_init_method: Init method name for A weights.
        row_init_method: Init method name for B weights.
        disable_sequence_parallel_comm: Whether to skip SP gather/scatter.
        use_a2a: Use all-to-all for SP scatter.
        dtype: Parameter dtype.
        device: Parameter device.
    """

    def __init__(
        self,
        n_adapters: int,
        in_features: int,
        out_features: int,
        dim: int,
        alpha: float = 32.0,
        input_is_parallel: bool = False,
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        disable_sequence_parallel_comm: bool = True,
        use_a2a: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.tokens_per_adapter: Optional[torch.Tensor] = None
        self.scaling_factors: Optional[torch.Tensor] = None
        self.max_rank = dim
        self.n_adapters = n_adapters
        self.alpha_values = torch.ones(n_adapters, dtype=dtype or torch.bfloat16, device=device or torch.device("cpu"))
        self.rank_values = torch.ones(n_adapters, dtype=dtype or torch.bfloat16, device=device or torch.device("cpu"))
        self.dim = dim
        self.alpha = alpha
        self.input_is_parallel = input_is_parallel
        self.disable_sequence_parallel_comm = disable_sequence_parallel_comm
        self.use_a2a = use_a2a
        self.column_init_method = column_init_method
        self.row_init_method = row_init_method

        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        if input_is_parallel:
            a_shape = (dim, in_features // tp_size)
        else:
            a_shape = (dim // tp_size, in_features)

        b_shape = (out_features // tp_size, dim)

        self.weight_A = nn.ParameterList([
            nn.Parameter(torch.empty(*a_shape, dtype=dtype, device=device))
            for _ in range(n_adapters)
        ])
        self.weight_B = nn.ParameterList([
            nn.Parameter(torch.empty(*b_shape, dtype=dtype, device=device))
            for _ in range(n_adapters)
        ])

        self._init_weights()

        self._gather_output = input_is_parallel

    def _init_weights(self) -> None:
        from megatron.bridge.peft.utils import ParallelLinearAdapter
        col_fn = ParallelLinearAdapter._get_init_fn(None, self.column_init_method)
        row_fn = ParallelLinearAdapter._get_init_fn(None, self.row_init_method)
        for i in range(self.n_adapters):
            col_fn(self.weight_A[i].data)
            row_fn(self.weight_B[i].data)

    def reset_adapter(self, idx: int) -> None:
        from megatron.bridge.peft.utils import ParallelLinearAdapter
        col_fn = ParallelLinearAdapter._get_init_fn(None, self.column_init_method)
        row_fn = ParallelLinearAdapter._get_init_fn(None, self.row_init_method)
        col_fn(self.weight_A[idx].data)
        row_fn(self.weight_B[idx].data)

    def forward(self, x: torch.Tensor, tokens_per_adapter: torch.Tensor) -> torch.Tensor:
        """Forward with grouped GEMM and proper TP/SP comms.

        Args:
            x: Input from layernorm.
               ``thd``: shape ``[T, H]`` (or ``[T/TP, H]`` with SP).
               ``bshd``: shape ``[B, S, H]``. Batch rows must be sorted by adapter.
            tokens_per_adapter: Token counts per adapter, shape ``[N]``.
               For ``bshd``, counts are in units of tokens (``n_seqs * seq_len``).

        Returns:
            Adapter output matching the shape of ``linear_output`` from the base layer.
        """
        ori_shape = x.shape
        # --- SP gather (once) ---
        if not self.disable_sequence_parallel_comm and not self.input_is_parallel:
            x = gather_from_sequence_parallel_region(x)

        # --- Flatten for grouped GEMM ---
        x_flat = x.reshape(-1, x.shape[-1])
        offsets = tokens_per_adapter.cumsum(dim=0, dtype=torch.int32)

        # --- Stack weights for grouped GEMM ---
        stacked_A = torch.stack(list(self.weight_A))
        stacked_B = torch.stack(list(self.weight_B))

        # --- Grouped GEMM: x @ A^T ---
        mid = torch._grouped_mm(x_flat, stacked_A.transpose(-2, -1), offsets)

        # --- TP comm between A and B ---
        if self.input_is_parallel:
            mid = reduce_from_tensor_model_parallel_region(mid)
        else:
            mid = gather_from_tensor_model_parallel_region(mid)

        # --- Grouped GEMM: mid @ B^T ---
        out = torch._grouped_mm(mid, stacked_B.transpose(-2, -1), offsets)

        # --- TP comm for output ---
        if self._gather_output:
            out = gather_from_tensor_model_parallel_region(out)

        # --- SP scatter (once) ---
        if not self.disable_sequence_parallel_comm and self.input_is_parallel:
            if self.use_a2a:
                out = all2all_hp2sp(out)
            else:
                out = scatter_to_sequence_parallel_region(out)

        per_token_scaling = torch.repeat_interleave(self.scaling_factors, tokens_per_adapter).unsqueeze(-1)
        out = out * per_token_scaling

        return out

    def named_parameters_for_adapter(self, idx: int) -> Iterator[Tuple[str, nn.Parameter]]:
        yield f"weight_A.{idx}", self.weight_A[idx]
        yield f"weight_B.{idx}", self.weight_B[idx]

    def state_dict_for_adapter(self, idx: int, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}weight_A": self.weight_A[idx].data.clone(),
            f"{prefix}weight_B": self.weight_B[idx].data.clone(),
        }

    def load_adapter(self, idx: int, state_dict: Dict[str, torch.Tensor]) -> None:
        for key, value in state_dict.items():
            if "weight_A" in key or "linear_in" in key:
                self.weight_A[idx].data.copy_(value)
            elif "weight_B" in key or "linear_out" in key:
                self.weight_B[idx].data.copy_(value)


class MultiLoRALinear(AdapterWrapper):
    """Megatron parallel linear wrapped with *N* concurrent LoRA adapters.

    Extends :class:`AdapterWrapper`.  Uses a single
    :class:`MultiParallelLinearAdapter` that stores stacked weights and
    performs grouped GEMM with one set of TP/SP comms.

    Args:
        to_wrap: The base Megatron parallel linear module (frozen).
        multi_adapter: :class:`MultiParallelLinearAdapter` holding all adapter weights.
        n_adapters: Number of adapter slots.
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        multi_adapter: MultiParallelLinearAdapter,
        n_adapters: int,
    ) -> None:
        nn.Module.__init__(self)
        self.to_wrap = to_wrap
        self.multi_adapter = multi_adapter
        self._adapter_enabled = True
        self.n_adapters = n_adapters
        self.tokens_per_adapter: Optional[torch.Tensor] = None
        self.scaling_factors: Optional[torch.Tensor] = None
        self.max_rank = multi_adapter.dim
        device = next(to_wrap.parameters()).device
        dtype = next(to_wrap.parameters()).dtype
        self.alpha_values = torch.ones(n_adapters, dtype=dtype, device=device)
        self.rank_values = torch.ones(n_adapters, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)

        if not self._adapter_enabled:
            return linear_output, bias

        tokens_per_adapter = self.tokens_per_adapter
        adapter_output = self.multi_adapter(layernorm_output.contiguous(), tokens_per_adapter)
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias

    # ------------------------------------------------------------------
    # Per-adapter lifecycle (delegates to multi_adapter)
    # ------------------------------------------------------------------

    def reset_adapter(self, idx: int) -> None:
        self.multi_adapter.reset_adapter(idx)

    def named_parameters_for_adapter(self, idx: int) -> Iterator[Tuple[str, nn.Parameter]]:
        yield from self.multi_adapter.named_parameters_for_adapter(idx)

    def state_dict_for_adapter(self, idx: int, prefix: str = "") -> Dict[str, Any]:
        return self.multi_adapter.state_dict_for_adapter(idx, prefix=f"{prefix}multi_adapter.")

    def load_adapter(self, idx: int, state_dict: Dict[str, torch.Tensor]) -> None:
        self.multi_adapter.load_adapter(idx, state_dict)

    # ------------------------------------------------------------------
    # State dict (overrides AdapterWrapper)
    # ------------------------------------------------------------------

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = {}
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.multi_adapter.state_dict(destination=destination, prefix=f"{prefix}multi_adapter.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sharded_sd: Dict[str, Any] = {}
        sharded_sd.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        # MultiParallelLinearAdapter stores raw Parameters, not Megatron parallel layers,
        # so we use nn.Module.state_dict rather than sharded_state_dict.
        for k, v in self.multi_adapter.state_dict(prefix=f"{prefix}multi_adapter.").items():
            sharded_sd[k] = v
        return sharded_sd


# ==================================================================
# Standalone functions for multi-LoRA batch routing and adapter I/O
# ==================================================================

_MULTI_LORA_TYPES = (MultiLoRALinear, SimpleMultiLoRALinear)


def _iter_multi_lora_modules(model):
    """Iterate over all MultiLoRA layer modules in a model."""
    models = model if isinstance(model, list) else [model]
    for model_chunk in models:
        for module in model_chunk.modules():
            if isinstance(module, _MULTI_LORA_TYPES):
                yield module


def set_batch(model, tokens_per_adapter: torch.Tensor) -> None:
    """Set per-micro-batch routing on all MultiLoRA layers.

    Computes scaling factors from each layer's own alpha/rank values.

    Args:
        model: The transformed model (or list of model chunks).
        tokens_per_adapter: Tensor of shape [n_adapters] with token counts per slot.
    """
    for module in _iter_multi_lora_modules(model):
        scaling = module.alpha_values / module.rank_values
        module.tokens_per_adapter = tokens_per_adapter
        module.scaling_factors = scaling
        if isinstance(module, MultiLoRALinear):
            module.multi_adapter.tokens_per_adapter = tokens_per_adapter
            module.multi_adapter.scaling_factors = scaling


def register_adapter(model, idx: int, rank: int, alpha: float) -> None:
    """Register an adapter slot on all MultiLoRA layers.

    Sets the alpha and rank for the given slot index.

    Args:
        model: The transformed model (or list of model chunks).
        idx: Adapter slot index.
        rank: LoRA rank for this adapter.
        alpha: LoRA alpha for this adapter.
    """
    for module in _iter_multi_lora_modules(model):
        module.alpha_values[idx] = alpha
        module.rank_values[idx] = rank


def unregister_adapter(model, idx: int) -> None:
    """Unregister an adapter slot on all MultiLoRA layers.

    Resets alpha/rank and re-initializes weights for the given slot.

    Args:
        model: The transformed model (or list of model chunks).
        idx: Adapter slot index.
    """
    for module in _iter_multi_lora_modules(model):
        module.alpha_values[idx] = 0
        module.rank_values[idx] = 1
        module.reset_adapter(idx)


def load_adapter(model, idx: int, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load weights into a specific adapter slot across all MultiLoRA layers.

    Args:
        model: The transformed model (or list of model chunks).
        idx: Adapter slot index to load into.
        state_dict: Adapter weights keyed by module name with ``adapter.`` prefix.
    """
    for module_name, module in _named_multi_lora_modules(model):
        adapter_prefix = f"{module_name}.adapter."
        local_sd = {}
        for key, value in state_dict.items():
            if key.startswith(adapter_prefix):
                local_sd[key[len(adapter_prefix):]] = value
        if local_sd:
            module.load_adapter(idx, local_sd)


def _named_multi_lora_modules(model):
    models = model if isinstance(model, list) else [model]
    for model_chunk in models:
        for name, module in model_chunk.named_modules():
            if isinstance(module, _MULTI_LORA_TYPES):
                yield name, module
