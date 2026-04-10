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
    multi_lora_state.lora_num_tokens.copy_(torch.tensor([100, 50, 80, 0]))
    multi_lora_state.scaling_factors.copy_(torch.tensor([2.0, 2.0, 1.0, 1.0]))
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class MultiLoRAState:
    """Singleton holding per-micro-batch multi-LoRA routing state.

    Attributes:
        lora_num_tokens: Token counts per adapter, shape ``[n_adapters]``.
            Defines contiguous token ranges: first ``lora_num_tokens[0]``
            tokens use adapter 0, next ``lora_num_tokens[1]`` use adapter 1, etc.
        scaling_factors: Per-adapter scaling ``alpha / rank``, shape ``[n_adapters]``.
    """

    lora_num_tokens: Optional[torch.Tensor] = None
    scaling_factors: Optional[torch.Tensor] = None

    def init(self, n_adapters: int, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.bfloat16) -> None:
        """Allocate state tensors. Call once at startup."""
        self.lora_num_tokens = torch.zeros(n_adapters, dtype=torch.int32, device=device)
        self.scaling_factors = torch.ones(n_adapters, dtype=dtype, device=device)

    def reset(self) -> None:
        """Clear all state. Useful in tests."""
        self.lora_num_tokens = None
        self.scaling_factors = None

    def get_lora_num_tokens(self) -> torch.Tensor:
        assert self.lora_num_tokens is not None, "MultiLoRAState not initialized. Call init() first."
        return self.lora_num_tokens

    def get_scaling_factors(self) -> torch.Tensor:
        assert self.scaling_factors is not None, "MultiLoRAState not initialized. Call init() first."
        return self.scaling_factors


# Singleton instance
multi_lora_state = MultiLoRAState()
