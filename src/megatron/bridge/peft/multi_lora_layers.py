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
via the global ``lora_num_tokens`` tensor (see :mod:`multi_lora_state`).

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

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.multi_lora_state import get_lora_num_tokens


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout_position == "pre":
            x = self.dropout(x)
        out = self.linear_out(self.linear_in(x))
        out = out * (self.alpha / self.dim)
        if self.dropout_position == "post":
            out = self.dropout(out)
        return out


class MultiLoRALinear(AdapterWrapper):
    """Megatron parallel linear wrapped with *N* concurrent LoRA adapters.

    Extends :class:`AdapterWrapper` to hold multiple adapters via an
    ``nn.ModuleList`` instead of a single adapter.  Inherits
    ``base_linear_forward()``, ``enable/disable_adapter_layers()``.

    Args:
        to_wrap: The base Megatron parallel linear module (frozen).
        adapters: ``nn.ModuleList`` of *N* :class:`ParallelLinearAdapter` instances.
        n_adapters: Number of adapter slots.
        use_grouped_mm: If True, use the experimental grouped GEMM path.
        column_init_method: Init method name for ``linear_in`` (A matrix).
        row_init_method: Init method name for ``linear_out`` (B matrix).
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        adapters: nn.ModuleList,
        n_adapters: int,
        use_grouped_mm: bool = False,
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
    ) -> None:
        nn.Module.__init__(self)
        self.to_wrap = to_wrap
        self.adapters = adapters
        self._adapter_enabled = True
        self.n_adapters = n_adapters
        self.use_grouped_mm = use_grouped_mm

        self.column_init_method = column_init_method
        self.row_init_method = row_init_method

        # Per-adapter scaling, initialised to each adapter's built-in alpha/dim.
        self.scaling: list[float] = [a.alpha / a.dim for a in adapters]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)

        if not self._adapter_enabled:
            return linear_output, bias

        lora_num_tokens = get_lora_num_tokens()

        if self.use_grouped_mm:
            adapter_output = self._forward_grouped_mm(layernorm_output.contiguous(), lora_num_tokens)
        else:
            adapter_output = self._forward_for_loop(layernorm_output.contiguous(), lora_num_tokens)

        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias

    def _forward_for_loop(
        self, x: torch.Tensor, lora_num_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Run each adapter on its token slice. TP/SP-safe by construction.

        Args:
            x: Input tensor, shape ``[total_tokens, features]`` or
               ``[batch, seq_len, features]``.
            lora_num_tokens: Token counts per adapter, shape ``[n_adapters]``.

        Returns:
            Combined adapter output, same shape as *x*.
        """
        ori_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        offsets = lora_num_tokens.cumsum(dim=0)
        total = offsets[-1].item()
        assert total == x_flat.shape[0], (
            f"lora_num_tokens sum {total} != token count {x_flat.shape[0]}"
        )

        adapter_outputs = []
        prev = 0
        for i in range(self.n_adapters):
            cur = offsets[i].item()
            n_tokens = cur - prev
            if n_tokens == 0:
                prev = cur
                continue

            token_slice = x_flat[prev:cur]
            adapter = self.adapters[i]
            out = adapter(token_slice)

            # Correct scaling if per-adapter override differs from built-in
            built_in_scale = adapter.alpha / adapter.dim
            if abs(self.scaling[i] - built_in_scale) > 1e-8:
                out = out * (self.scaling[i] / built_in_scale)

            adapter_outputs.append(out)
            prev = cur

        if not adapter_outputs:
            return torch.zeros_like(x_flat).reshape(ori_shape)

        return torch.cat(adapter_outputs, dim=0).reshape(ori_shape)

    def _forward_grouped_mm(
        self, x: torch.Tensor, lora_num_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Grouped GEMM over all adapters in a single fused kernel.

        .. warning::
            Experimental.  Bypasses ``ParallelLinearAdapter.forward()`` and
            therefore does **not** handle TP/SP communication.  Only use when
            you have verified that the surrounding model handles communication
            correctly, or when running without tensor/sequence parallelism.

        Args:
            x: Input tensor, shape ``[total_tokens, features]`` or
               ``[batch, seq_len, features]``.
            lora_num_tokens: Token counts per adapter, shape ``[n_adapters]``.

        Returns:
            Combined adapter output, same shape as *x*.
        """
        ori_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        offsets = lora_num_tokens.cumsum(dim=0, dtype=torch.int32)

        # Stack raw weight tensors from all adapters
        # linear_in.weight: [dim_local, in_features_local] (may be TP-sharded)
        # linear_out.weight: [out_features_local, dim] (may be TP-sharded)
        stacked_A = torch.stack([a.linear_in.weight for a in self.adapters])
        stacked_B = torch.stack([a.linear_out.weight for a in self.adapters])

        # grouped_mm: x_flat @ A^T per adapter group, then result @ B^T
        mid = torch._grouped_mm(x_flat, stacked_A.transpose(-2, -1), offsets)
        out = torch._grouped_mm(mid, stacked_B.transpose(-2, -1), offsets)

        # Per-token scaling
        scaling_tensor = torch.tensor(self.scaling, device=x.device, dtype=x.dtype)
        per_token_scaling = torch.repeat_interleave(scaling_tensor, lora_num_tokens).unsqueeze(-1)
        out = out * per_token_scaling

        return out.reshape(ori_shape)

    # ------------------------------------------------------------------
    # Per-adapter lifecycle
    # ------------------------------------------------------------------

    def reset_adapter(self, idx: int) -> None:
        """Re-initialise adapter *idx* using the configured init methods."""
        adapter = self.adapters[idx]
        adapter._get_init_fn(self.column_init_method)(adapter.linear_in.weight.data)
        adapter._get_init_fn(self.row_init_method)(adapter.linear_out.weight.data)
        self.scaling[idx] = adapter.alpha / adapter.dim

    def set_scaling(self, idx: int, alpha: float, rank: int) -> None:
        """Override the scaling factor for adapter *idx*."""
        self.scaling[idx] = alpha / rank

    def named_parameters_for_adapter(self, idx: int) -> Iterator[Tuple[str, nn.Parameter]]:
        """Yield ``(name, param)`` pairs for adapter *idx*."""
        prefix = f"adapters.{idx}."
        for name, param in self.adapters[idx].named_parameters():
            yield prefix + name, param

    def state_dict_for_adapter(self, idx: int, prefix: str = "") -> Dict[str, Any]:
        """Return a state dict containing only adapter *idx*'s weights."""
        sd: Dict[str, Any] = {}
        self.adapters[idx].state_dict(destination=sd, prefix=f"{prefix}adapters.{idx}.")
        return sd

    def load_adapter(self, idx: int, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights into adapter slot *idx*.

        *state_dict* keys should match :class:`ParallelLinearAdapter`'s own
        ``state_dict()`` output (e.g. ``linear_in.weight``, ``linear_out.weight``).
        """
        self.adapters[idx].load_state_dict(state_dict, strict=True)

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
        for idx, adapter in enumerate(self.adapters):
            adapter.state_dict(destination=destination, prefix=f"{prefix}adapters.{idx}.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sharded_sd: Dict[str, Any] = {}
        sharded_sd.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        for idx, adapter in enumerate(self.adapters):
            sharded_sd.update(adapter.sharded_state_dict(f"{prefix}adapters.{idx}.", sharded_offsets, metadata))
        return sharded_sd
