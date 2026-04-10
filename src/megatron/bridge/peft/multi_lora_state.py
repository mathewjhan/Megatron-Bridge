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

"""Global routing state for multi-LoRA.

The downstream framework sets routing state before each forward pass.
Every :class:`MultiLoRALinear` layer reads from the singleton
:class:`MultiLoRAState` instance.

Typical usage::

    from megatron.bridge.peft.multi_lora_state import multi_lora_state

    # One-time init
    multi_lora_state.init(n_adapters=4, device="cuda")

    # Before each micro-batch forward
    multi_lora_state.set_batch(
        tokens_per_adapter=torch.tensor([100, 50, 80, 0]),
        scaling_factors=torch.tensor([2.0, 2.0, 1.0, 1.0]),
    )
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MultiLoRAState:
    """Singleton holding per-micro-batch multi-LoRA routing state.

    Attributes:
        tokens_per_adapter: Token counts per adapter, shape ``[n_adapters]``.
            Defines contiguous token ranges in the flattened sequence:
            first ``tokens_per_adapter[0]`` tokens use adapter 0, next
            ``tokens_per_adapter[1]`` use adapter 1, etc.
        scaling_factors: Per-adapter scaling ``alpha / rank``, shape ``[n_adapters]``.
    """

    tokens_per_adapter: Optional[torch.Tensor] = None
    scaling_factors: Optional[torch.Tensor] = None

    def init(
        self,
        n_adapters: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Allocate state tensors. Call once at startup."""
        self.tokens_per_adapter = torch.zeros(n_adapters, dtype=torch.int32, device=device)
        self.scaling_factors = torch.ones(n_adapters, dtype=dtype, device=device)

    def set_batch(
        self,
        tokens_per_adapter: torch.Tensor,
        scaling_factors: Optional[torch.Tensor] = None,
    ) -> None:
        """Update routing state for the next micro-batch.

        Args:
            tokens_per_adapter: Token counts per adapter, shape ``[n_adapters]``.
            scaling_factors: Per-adapter ``alpha / rank``. If None, keeps
                the current values.
        """
        assert self.tokens_per_adapter is not None, "MultiLoRAState not initialized. Call init() first."
        self.tokens_per_adapter.copy_(tokens_per_adapter)
        if scaling_factors is not None:
            self.scaling_factors.copy_(scaling_factors)

    def reset(self) -> None:
        """Clear all state. Useful in tests."""
        self.tokens_per_adapter = None
        self.scaling_factors = None

    def get_tokens_per_adapter(self) -> torch.Tensor:
        assert self.tokens_per_adapter is not None, "MultiLoRAState not initialized. Call init() first."
        return self.tokens_per_adapter

    def get_scaling_factors(self) -> torch.Tensor:
        assert self.scaling_factors is not None, "MultiLoRAState not initialized. Call init() first."
        return self.scaling_factors


# Singleton instance
multi_lora_state = MultiLoRAState()
