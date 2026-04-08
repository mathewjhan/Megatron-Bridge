"""Debug script: apply MultiLoRA with 3 adapters to a small HF model.

Usage:
    python debug_multi_lora.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear
from megatron.bridge.peft.multi_lora_state import set_lora_num_tokens, reset_state

N_ADAPTERS = 3
MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Base model params: {total_params_before:,}")

    # Apply multi-LoRA directly via MultiLoRA PEFT class
    multi_lora = MultiLoRA(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        n_adapters=N_ADAPTERS,
        dim=16,
        alpha=32,
        lora_dtype=torch.bfloat16,
    )
    model = multi_lora(model, training=True)

    total_params_after = sum(p.numel() for p in model.parameters())
    adapter_params = total_params_after - total_params_before
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wrapped = sum(1 for m in model.modules() if isinstance(m, MultiLoRALinear))
    print(f"Wrapped {wrapped} modules")
    print(f"Total params after: {total_params_after:,}")
    print(f"Adapter params: {adapter_params:,} ({N_ADAPTERS} adapters x {adapter_params // N_ADAPTERS:,} each)")
    print(f"Trainable params: {trainable:,}")

    # Test forward with mixed adapters
    print("\n--- Forward test (mixed adapters) ---")
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    n0, n1, n2 = 2, seq_len - 2, 0
    set_lora_num_tokens(torch.tensor([n0, n1, n2], dtype=torch.int32), reset_reference=True)
    print(f"Input: '{text}' ({seq_len} tokens)")
    print(f"lora_num_tokens: [{n0}, {n1}, {n2}]")

    with torch.no_grad():
        outputs = model(**inputs)
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Output logits sample: {outputs.logits[0, -1, :5]}")

    # Test single adapter
    print("\n--- Forward test (single adapter 2) ---")
    set_lora_num_tokens(torch.tensor([0, 0, seq_len], dtype=torch.int32))
    with torch.no_grad():
        outputs2 = model(**inputs)
    print(f"Output logits sample: {outputs2.logits[0, -1, :5]}")

    # Test reset
    print("\n--- Reset adapter 1 ---")
    multi_lora.reset_adapter(model, 1)
    print("Done")

    # Test per-adapter parameters
    print("\n--- Per-adapter parameter counts ---")
    for idx in range(N_ADAPTERS):
        params = list(multi_lora.named_parameters_for_adapter(model, idx))
        print(f"  Adapter {idx}: {sum(p.numel() for _, p in params):,} params")

    # Test state dict for adapter
    print("\n--- State dict for adapter 0 (first 5 keys) ---")
    sd = multi_lora.state_dict_for_adapter(model, 0)
    for i, (k, v) in enumerate(sd.items()):
        if i >= 5:
            print(f"  ... and {len(sd) - 5} more")
            break
        print(f"  {k}: {v.shape}")

    # Test scaling
    print("\n--- Set custom scaling for adapter 2 ---")
    multi_lora.set_adapter_scaling(model, 2, alpha=64, rank=16)
    print("Done")

    reset_state()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
