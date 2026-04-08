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

"""Multi-adapter LoRA PEFT class.

:class:`MultiLoRA` is the multi-adapter analogue of :class:`LoRA`.  It wraps
each target module with a :class:`MultiLoRALinear` that holds *N* concurrent
:class:`ParallelLinearAdapter` instances.

Expert linears and ``TopKRouter`` modules are skipped — only dense attention
projections and non-expert MLP layers are adapted.

Example::

    from megatron.bridge.peft.multi_lora import MultiLoRA

    multi_lora = MultiLoRA(n_adapters=4, dim=32, alpha=32)
    model = multi_lora(base_model, training=True)

    # Per-adapter optimizer
    params = list(multi_lora.named_parameters_for_adapter(model, idx=0))
    optimizer = torch.optim.Adam([p for _, p in params])
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.transformer.moe.router import TopKRouter

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear, SimpleMultiLoRALinear
from megatron.bridge.peft.utils import ParallelLinearAdapter, get_adapter_attributes_from_linear, is_expert_linear

logger = logging.getLogger(__name__)


@dataclass
class MultiLoRA(PEFT, ModuleMatcher):
    """Multi-adapter LoRA supporting *N* concurrent adapters on a shared base model.

    Each target module is wrapped with :class:`MultiLoRALinear` holding *N*
    :class:`ParallelLinearAdapter` instances.  The active adapter is selected
    at forward time via the global ``lora_num_tokens`` state.

    Args:
        target_modules: Module names or wildcard patterns to apply multi-LoRA to.
        n_adapters: Number of concurrent adapter slots.
        dim: LoRA rank (bottleneck dimension).
        alpha: LoRA scaling parameter.
        dropout: Dropout probability for the adapter.
        dropout_position: Where to apply dropout (``'pre'`` or ``'post'``).
        lora_A_init_method: Initialisation method for the A matrix.
        lora_B_init_method: Initialisation method for the B matrix.
        a2a_experimental: Enable experimental all-to-all communication.
        lora_dtype: Data type for adapter weights (``None`` = use model dtype).
        use_grouped_mm: Use experimental grouped GEMM forward path.
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
    use_grouped_mm: bool = False

    # ------------------------------------------------------------------
    # transform (called by walk for every module in the model)
    # ------------------------------------------------------------------

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        # Skip already transformed modules
        if isinstance(module, (MultiLoRALinear, SimpleMultiLoRALinear)):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans

            # Skip expert linears (expert × adapter routing is not supported)
            if is_expert_linear(full_name):
                return module

            # Skip TopKRouter
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

            adapters = nn.ModuleList()
            for _ in range(self.n_adapters):
                adapter = ParallelLinearAdapter(
                    attrs.in_features,
                    attrs.out_features,
                    self.dim,
                    base_linear_name=full_name,
                    activation="identity",
                    column_init_method=self.lora_A_init_method,
                    row_init_method=self.lora_B_init_method,
                    input_is_parallel=attrs.input_is_parallel,
                    dropout=self.dropout,
                    dropout_position=self.dropout_position,
                    model_parallel_config=getattr(module, "config", None),
                    alpha=self.alpha,
                    a2a_experimental=self.a2a_experimental,
                    disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                    disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                    base_linear_is_parallel=attrs.base_linear_is_parallel,
                )
                adapters.append(adapter)

            return MultiLoRALinear(
                module,
                adapters,
                self.n_adapters,
                use_grouped_mm=self.use_grouped_mm,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
            )

        return module

    # ------------------------------------------------------------------
    # Model-level adapter lifecycle
    # ------------------------------------------------------------------

    def reset_adapter(self, model, idx: int) -> None:
        """Re-initialise adapter *idx* across all :class:`MultiLoRALinear` layers."""
        for module in self._iter_multi_lora_modules(model):
            module.reset_adapter(idx)

    def set_adapter_scaling(self, model, idx: int, alpha: float, rank: int) -> None:
        """Set per-adapter scaling for adapter *idx* across all layers."""
        for module in self._iter_multi_lora_modules(model):
            module.set_scaling(idx, alpha, rank)

    def named_parameters_for_adapter(self, model, idx: int) -> Iterator[Tuple[str, nn.Parameter]]:
        """Yield all parameters belonging to adapter *idx* across the model."""
        for module_name, module in self._named_multi_lora_modules(model):
            for param_name, param in module.named_parameters_for_adapter(idx):
                yield f"{module_name}.{param_name}", param

    def state_dict_for_adapter(self, model, idx: int) -> Dict[str, torch.Tensor]:
        """Collect state dict for adapter *idx* across the entire model."""
        sd: Dict[str, torch.Tensor] = {}
        for module_name, module in self._named_multi_lora_modules(model):
            sd.update(module.state_dict_for_adapter(idx, prefix=f"{module_name}."))
        return sd

    def load_adapter(self, model, idx: int, adapter_state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a single-adapter checkpoint into slot *idx*.

        *adapter_state_dict* keys use the single-LoRA prefix ``adapter.``
        (e.g. ``decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight``).
        This method remaps them to the target :class:`MultiLoRALinear` module.
        """
        for module_name, module in self._named_multi_lora_modules(model):
            adapter_prefix = f"{module_name}.adapter."
            local_sd = {}
            for key, value in adapter_state_dict.items():
                if key.startswith(adapter_prefix):
                    local_key = key[len(adapter_prefix) :]
                    local_sd[local_key] = value
            if local_sd:
                module.load_adapter(idx, local_sd)

    # ------------------------------------------------------------------
    # Checkpoint filtering
    # ------------------------------------------------------------------

    def adapter_key_filter(self, key) -> bool:
        if isinstance(key, tuple):
            return key[1].requires_grad
        return ".adapters." in key

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _multi_lora_types = (MultiLoRALinear, SimpleMultiLoRALinear)

    def _iter_multi_lora_modules(self, model) -> Iterator[nn.Module]:
        """Yield all multi-LoRA modules in *model*."""
        models = model if isinstance(model, list) else [model]
        for model_chunk in models:
            for module in model_chunk.modules():
                if isinstance(module, self._multi_lora_types):
                    yield module

    def _named_multi_lora_modules(self, model) -> Iterator[Tuple[str, nn.Module]]:
        """Yield ``(fqn, module)`` pairs for all multi-LoRA modules."""
        models = model if isinstance(model, list) else [model]
        for model_chunk in models:
            for name, module in model_chunk.named_modules():
                if isinstance(module, self._multi_lora_types):
                    yield name, module


