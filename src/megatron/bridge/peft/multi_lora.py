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

"""Multi-adapter LoRA: transform, registry, and batch routing in one object.

:class:`MultiLoRA` is the single entry point for multi-LoRA.  It handles:

* **Model transform** — wrapping target modules with multi-adapter layers.
* **Adapter registry** — name-based registration / unregistration of adapters.
* **Batch routing** — setting per-batch token-to-adapter mappings.
* **Weight lifecycle** — resetting, loading, saving per-adapter weights.

Example::

    from megatron.bridge.peft.multi_lora import MultiLoRA

    multi_lora = MultiLoRA(n_adapters=4, dim=16, alpha=32)
    model = multi_lora(model, training=True)

    # Adapter lifecycle
    multi_lora.register_adapter("math-lora", rank=16, alpha=32)
    multi_lora.reset_adapter(model, "math-lora")

    # Per-batch routing
    multi_lora.set_batch({"math-lora": 512, "code-lora": 1024})

    # Forward / backward as normal
    output = model(tokens, position_ids, ...)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.transformer.moe.router import TopKRouter

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, MultiParallelLinearAdapter, SimpleMultiLoRALinear
from megatron.bridge.peft.utils import get_adapter_attributes_from_linear, is_expert_linear

logger = logging.getLogger(__name__)


@dataclass
class MultiLoRA(PEFT, ModuleMatcher):
    """Multi-adapter LoRA: transform + registry + routing.

    Args:
        target_modules: Module names or wildcard patterns to apply multi-LoRA to.
        n_adapters: Maximum number of concurrent adapter slots.
        dim: LoRA rank (bottleneck dimension).
        alpha: Default LoRA scaling parameter for new adapters.
        dropout: Dropout probability for the adapter.
        dropout_position: Where to apply dropout (``'pre'`` or ``'post'``).
        lora_A_init_method: Initialisation method for the A matrix.
        lora_B_init_method: Initialisation method for the B matrix.
        a2a_experimental: Enable experimental all-to-all communication.
        lora_dtype: Data type for adapter weights (``None`` = use model dtype).
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    n_adapters: int = 2
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: Optional[torch.dtype] = None

    # --- Internal registry state (not dataclass fields) ---

    def __post_init__(self):
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}
        self._free_slots: set = set(range(self.n_adapters))

    # ==================================================================
    # Transform
    # ==================================================================

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        if isinstance(module, (MultiLoRALinear, SimpleMultiLoRALinear)):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans

            if is_expert_linear(full_name):
                return module
            if isinstance(module, TopKRouter):
                return module

            # --- Plain nn.Linear (HF models) ---
            if isinstance(module, nn.Linear):
                logger.info(f"Adding multi-lora ({self.n_adapters} adapters) to nn.Linear: {full_name}")
                return SimpleMultiLoRALinear(
                    module,
                    n_adapters=self.n_adapters,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    dropout_position=self.dropout_position,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )

            # --- Megatron parallel linears ---
            attrs = get_adapter_attributes_from_linear(module)
            logger.info(f"Adding multi-lora ({self.n_adapters} adapters) to: {full_name}")

            multi_adapter = MultiParallelLinearAdapter(
                n_adapters=self.n_adapters,
                in_features=attrs.in_features,
                out_features=attrs.out_features,
                dim=self.dim,
                alpha=self.alpha,
                input_is_parallel=attrs.input_is_parallel,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                use_a2a=self.a2a_experimental,
                dtype=next(module.parameters()).dtype,
                device=next(module.parameters()).device,
            )

            return MultiLoRALinear(module, multi_adapter, self.n_adapters)

        return module

    # ==================================================================
    # Adapter Registry
    # ==================================================================

    def register_adapter(self, name: str, rank: int, alpha: float) -> int:
        """Register a named adapter, allocating a slot.

        Args:
            name: Unique adapter name (e.g. ``"math-lora"``).
            rank: LoRA rank for this adapter.
            alpha: LoRA alpha for this adapter.

        Returns:
            The allocated slot index.

        Raises:
            ValueError: If name is already registered or no free slots.
        """
        if name in self._name_to_idx:
            raise ValueError(f"Adapter '{name}' is already registered at slot {self._name_to_idx[name]}")
        if not self._free_slots:
            raise ValueError(f"No free adapter slots (max {self.n_adapters})")

        idx = min(self._free_slots)
        self._free_slots.remove(idx)
        self._name_to_idx[name] = idx
        self._idx_to_name[idx] = name

        logger.info(f"Registered adapter '{name}' at slot {idx} (rank={rank}, alpha={alpha})")
        return idx

    def unregister_adapter(self, name: str) -> int:
        """Unregister a named adapter, freeing its slot.

        Args:
            name: Adapter name to unregister.

        Returns:
            The freed slot index.

        Raises:
            KeyError: If name is not registered.
        """
        idx = self._name_to_idx.pop(name)
        del self._idx_to_name[idx]
        self._free_slots.add(idx)

        logger.info(f"Unregistered adapter '{name}' from slot {idx}")
        return idx

    def get_adapter_idx(self, name: str) -> int:
        """Get the slot index for a named adapter."""
        return self._name_to_idx[name]

    @property
    def registered_adapters(self) -> Dict[str, int]:
        """Return a copy of the name → slot mapping."""
        return dict(self._name_to_idx)

    # ==================================================================
    # Weight Lifecycle
    # ==================================================================

    def reset_adapter(self, model, name: str) -> None:
        """Re-initialise adapter weights across all layers.

        Args:
            model: The transformed model (or list of model chunks).
            name: Adapter name to reset.
        """
        idx = self._name_to_idx[name]
        for module in self._iter_multi_lora_modules(model):
            module.reset_adapter(idx)

    def load_adapter(self, model, name: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights into a named adapter slot across all layers.

        Args:
            model: The transformed model (or list of model chunks).
            name: Adapter name to load into.
            state_dict: Adapter weights. Keys use the single-LoRA ``adapter.``
                prefix (e.g. ``decoder.layers.0...adapter.linear_in.weight``).
        """
        idx = self._name_to_idx[name]
        for module_name, module in self._named_multi_lora_modules(model):
            adapter_prefix = f"{module_name}.adapter."
            local_sd = {}
            for key, value in state_dict.items():
                if key.startswith(adapter_prefix):
                    local_sd[key[len(adapter_prefix):]] = value
            if local_sd:
                module.load_adapter(idx, local_sd)

    def named_parameters_for_adapter(self, model, name: str) -> Iterator[Tuple[str, nn.Parameter]]:
        """Yield all parameters belonging to a named adapter.

        Args:
            model: The transformed model (or list of model chunks).
            name: Adapter name.
        """
        idx = self._name_to_idx[name]
        for module_name, module in self._named_multi_lora_modules(model):
            for param_name, param in module.named_parameters_for_adapter(idx):
                yield f"{module_name}.{param_name}", param

    def state_dict_for_adapter(self, model, name: str) -> Dict[str, torch.Tensor]:
        """Collect state dict for a named adapter across the model.

        Args:
            model: The transformed model (or list of model chunks).
            name: Adapter name.
        """
        idx = self._name_to_idx[name]
        sd: Dict[str, torch.Tensor] = {}
        for module_name, module in self._named_multi_lora_modules(model):
            sd.update(module.state_dict_for_adapter(idx, prefix=f"{module_name}."))
        return sd

    # ==================================================================
    # Checkpoint filtering
    # ==================================================================

    def adapter_key_filter(self, key) -> bool:
        if isinstance(key, tuple):
            return key[1].requires_grad
        return ".adapters." in key or ".weight_A." in key or ".weight_B." in key

    # ==================================================================
    # Helpers
    # ==================================================================

    _multi_lora_types = (MultiLoRALinear, SimpleMultiLoRALinear)

    def _iter_multi_lora_modules(self, model) -> Iterator[nn.Module]:
        models = model if isinstance(model, list) else [model]
        for model_chunk in models:
            for module in model_chunk.modules():
                if isinstance(module, self._multi_lora_types):
                    yield module

    def _named_multi_lora_modules(self, model) -> Iterator[Tuple[str, nn.Module]]:
        models = model if isinstance(model, list) else [model]
        for model_chunk in models:
            for name, module in model_chunk.named_modules():
                if isinstance(module, self._multi_lora_types):
                    yield name, module

    @staticmethod
    def _detect_device(model) -> torch.device:
        models = model if isinstance(model, list) else [model]
        for model_chunk in models:
            for param in model_chunk.parameters():
                return param.device
        return torch.device("cpu")
