"""Debug script: apply MultiLoRA with 3 adapters to a small HF model.

Usage:
    uv run python debug_multi_lora.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear
from megatron.bridge.peft.multi_lora_state import set_lora_num_tokens, get_lora_num_tokens, reset_state

N_ADAPTERS = 3
MODEL_NAME = "Qwen/Qwen3-0.6B"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Base model params: {total_params_before:,}")

    # Apply multi-LoRA
    # Note: MultiLoRA.transform() skips nn.Linear (designed for Megatron parallel linears).
    # For this debug script with an HF model, we manually wrap the target modules.
    print(f"\nApplying MultiLoRA with {N_ADAPTERS} adapters to {TARGET_MODULES}...")

    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    class MockParallelLinearAdapter(nn.Module):
        """Lightweight LoRA adapter for debug (no Megatron dependency)."""
        def __init__(self, in_features, out_features, dim=16, alpha=32):
            super().__init__()
            self.linear_in = nn.Linear(in_features, dim, bias=False)
            self.linear_out = nn.Linear(dim, out_features, bias=False)
            self.dim = dim
            self.alpha = alpha
            nn.init.xavier_normal_(self.linear_in.weight)
            nn.init.zeros_(self.linear_out.weight)

        def _get_init_fn(self, method):
            if method == "xavier":
                return nn.init.xavier_normal_
            elif method == "zero":
                return lambda t: nn.init.constant_(t, 0.0)
            return nn.init.xavier_normal_

        def forward(self, x):
            return self.linear_out(self.linear_in(x)) * (self.alpha / self.dim)

    wrapped_count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name.split(".") for t in TARGET_MODULES):
            continue

        adapters = nn.ModuleList([
            MockParallelLinearAdapter(module.in_features, module.out_features)
            for _ in range(N_ADAPTERS)
        ])

        # Wrap with a mock base that returns tuple like Megatron
        class TupleWrapper(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear
            def forward(self, x, *args, **kwargs):
                return self.linear(x), None

        multi_lora = MultiLoRALinear(
            TupleWrapper(module), adapters, N_ADAPTERS,
        )

        # Replace in model
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], multi_lora)
        wrapped_count += 1

    total_params_after = sum(p.numel() for p in model.parameters())
    adapter_params = total_params_after - total_params_before
    print(f"Wrapped {wrapped_count} modules")
    print(f"Total params after: {total_params_after:,}")
    print(f"Adapter params: {adapter_params:,} ({N_ADAPTERS} adapters × {adapter_params // N_ADAPTERS:,} each)")

    # Freeze base model
    for name, param in model.named_parameters():
        if "adapters" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # Test forward with mixed adapters
    print("\n--- Forward test ---")
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    # Split tokens: 2 for adapter 0, rest for adapter 1, 0 for adapter 2
    n0 = 2
    n1 = seq_len - n0
    n2 = 0
    lora_num_tokens = torch.tensor([n0, n1, n2], dtype=torch.int32)
    set_lora_num_tokens(lora_num_tokens, reset_reference=True)
    print(f"Input: '{text}' ({seq_len} tokens)")
    print(f"lora_num_tokens: {lora_num_tokens.tolist()}")

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Output logits sample: {outputs.logits[0, -1, :5]}")

    # Test reset
    print("\n--- Reset adapter 1 ---")
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            module.reset_adapter(1)
            break
    print("Reset adapter 1 (B weights zeroed)")

    # Test per-adapter parameters
    print("\n--- Per-adapter parameter counts ---")
    for module in model.modules():
        if isinstance(module, MultiLoRALinear):
            for idx in range(N_ADAPTERS):
                params = list(module.named_parameters_for_adapter(idx))
                count = sum(p.numel() for _, p in params)
                print(f"  Adapter {idx}: {count:,} params, names: {[n for n, _ in params]}")
            break

    # Test state dict for adapter
    print("\n--- State dict for adapter 0 (first module) ---")
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALinear):
            sd = module.state_dict_for_adapter(0, prefix=f"{name}.")
            for k, v in sd.items():
                print(f"  {k}: {v.shape}")
            break

    reset_state()
    print("\nDone!")


if __name__ == "__main__":
    main()
