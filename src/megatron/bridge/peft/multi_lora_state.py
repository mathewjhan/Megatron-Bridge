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

"""Per-batch routing state for multi-LoRA.

The downstream framework calls :func:`init` once at startup, then
:func:`set_batch` before each forward pass.  Every multi-LoRA layer
reads from the module-level state.

Typical usage::

    from megatron.bridge.peft import multi_lora_state

    # One-time init
    multi_lora_state.init(n_adapters=4, device="cuda")

    # Before each micro-batch forward
    multi_lora_state.set_batch(
        tokens_per_adapter=torch.tensor([100, 50, 80, 0]),
        alpha=torch.tensor([32.0, 32.0, 16.0, 16.0]),
        rank=torch.tensor([16, 16, 8, 8]),
    )
"""

from typing import Optional

import torch

# Module-level state
tokens_per_adapter: Optional[torch.Tensor] = None
alpha: Optional[torch.Tensor] = None
rank: Optional[torch.Tensor] = None


def init(
    n_adapters: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Allocate state tensors. Call once at startup."""
    global tokens_per_adapter, alpha, rank
    tokens_per_adapter = torch.zeros(n_adapters, dtype=torch.int32, device=device)
    alpha = torch.ones(n_adapters, dtype=dtype, device=device)
    rank = torch.ones(n_adapters, dtype=dtype, device=device)


def set_batch(
    new_tokens_per_adapter: torch.Tensor,
    new_alpha: Optional[torch.Tensor] = None,
    new_rank: Optional[torch.Tensor] = None,
) -> None:
    """Update routing state for the next micro-batch.

    Args:
        new_tokens_per_adapter: Token counts per adapter, shape ``[n_adapters]``.
        new_alpha: Per-adapter LoRA alpha. If None, keeps current values.
        new_rank: Per-adapter LoRA rank. If None, keeps current values.
    """
    assert tokens_per_adapter is not None, "multi_lora_state not initialized. Call init() first."
    tokens_per_adapter.copy_(new_tokens_per_adapter)
    if new_alpha is not None:
        alpha.copy_(new_alpha)
    if new_rank is not None:
        rank.copy_(new_rank)


def get_tokens_per_adapter() -> torch.Tensor:
    assert tokens_per_adapter is not None, "multi_lora_state not initialized. Call init() first."
    return tokens_per_adapter


def get_scaling_factors() -> torch.Tensor:
    """Compute ``alpha / rank`` per adapter."""
    assert alpha is not None and rank is not None, "multi_lora_state not initialized. Call init() first."
    return alpha / rank


def reset() -> None:
    """Clear all state. Useful in tests."""
    global tokens_per_adapter, alpha, rank
    tokens_per_adapter = None
    alpha = None
    rank = None
