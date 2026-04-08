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

The downstream framework sets ``lora_num_tokens`` before each forward pass
to indicate how many tokens in the current micro-batch belong to each
adapter.  Every :class:`MultiLoRALinear` layer reads this tensor to select
the active adapter.

Typical usage::

    from megatron.bridge.peft.multi_lora_state import (
        set_lora_num_tokens,
        get_active_adapter_idx,
    )

    # Before each micro-batch forward
    num_tokens = torch.zeros(n_adapters, dtype=torch.int32, device="cuda")
    num_tokens[active_adapter] = batch_token_count
    set_lora_num_tokens(num_tokens)
"""

from typing import Optional

import torch

_LORA_NUM_TOKENS: Optional[torch.Tensor] = None


def set_lora_num_tokens(num_tokens: torch.Tensor, reset_reference: bool = False) -> None:
    """Set the number of tokens per adapter for the current micro-batch.

    Args:
        num_tokens: Tensor of shape ``[n_adapters]`` with token counts.
        reset_reference: If True, replace the tensor reference.
            If False, copy values in-place (requires a prior call with
            ``reset_reference=True`` to establish the tensor).
    """
    global _LORA_NUM_TOKENS
    if _LORA_NUM_TOKENS is None or reset_reference:
        _LORA_NUM_TOKENS = num_tokens
    else:
        _LORA_NUM_TOKENS.copy_(num_tokens)


def get_lora_num_tokens() -> torch.Tensor:
    """Return the current ``lora_num_tokens`` tensor.

    Raises:
        RuntimeError: If called before :func:`set_lora_num_tokens`.
    """
    if _LORA_NUM_TOKENS is None:
        raise RuntimeError("lora_num_tokens not initialized. Call set_lora_num_tokens() first.")
    return _LORA_NUM_TOKENS


def get_active_adapter_idx() -> int:
    """Return the index of the adapter with the most tokens.

    This is the single-adapter-per-microbatch fast path: the downstream
    framework guarantees that only one entry in ``lora_num_tokens`` is
    non-zero per micro-batch.
    """
    return get_lora_num_tokens().argmax().item()


def reset_state() -> None:
    """Clear all global multi-LoRA state.  Useful in tests."""
    global _LORA_NUM_TOKENS
    _LORA_NUM_TOKENS = None
