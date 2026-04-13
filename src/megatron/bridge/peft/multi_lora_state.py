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

"""Low-level per-batch routing tensors for multi-LoRA.

These module-level tensors are written by :class:`MultiLoRA` and read by
:class:`MultiLoRALinear` / :class:`MultiParallelLinearAdapter` during forward.
Users should not interact with this module directly — use :class:`MultiLoRA`
instead.
"""

from typing import Optional

import torch

# Token counts per adapter slot, shape [n_adapters].
# Defines contiguous token ranges in the flattened sequence.
tokens_per_adapter: Optional[torch.Tensor] = None

# Per-adapter LoRA alpha, shape [n_adapters].
alpha: Optional[torch.Tensor] = None

# Per-adapter LoRA rank, shape [n_adapters].
rank: Optional[torch.Tensor] = None


def init(n_adapters: int, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> None:
    global tokens_per_adapter, alpha, rank
    tokens_per_adapter = torch.zeros(n_adapters, dtype=torch.int32, device=device)
    alpha = torch.ones(n_adapters, dtype=dtype, device=device)
    rank = torch.ones(n_adapters, dtype=dtype, device=device)


def get_tokens_per_adapter() -> torch.Tensor:
    assert tokens_per_adapter is not None, "multi_lora_state not initialized"
    return tokens_per_adapter


def get_scaling_factors() -> torch.Tensor:
    assert alpha is not None and rank is not None, "multi_lora_state not initialized"
    return alpha / rank


def reset() -> None:
    global tokens_per_adapter, alpha, rank
    tokens_per_adapter = None
    alpha = None
    rank = None
