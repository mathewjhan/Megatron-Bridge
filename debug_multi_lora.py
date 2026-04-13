"""Debug script: apply MultiLoRA with 3 adapters to a small HF model.

Usage:
    python debug_multi_lora.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import SimpleMultiLoRALinear

N_ADAPTERS = 3
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda()
    model.eval()

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Base model params: {total_params_before:,}")

    # Create MultiLoRA and transform model
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
    wrapped = sum(1 for m in model.modules() if isinstance(m, SimpleMultiLoRALinear))
    print(f"Wrapped {wrapped} modules")
    print(f"Total params after: {total_params_after:,}")
    print(f"Adapter params: {adapter_params:,} ({N_ADAPTERS} adapters x {adapter_params // N_ADAPTERS:,} each)")
    print(f"Trainable params: {trainable:,}")

    # Register adapters
    multi_lora.register_adapter("math-lora", rank=16, alpha=32)
    multi_lora.register_adapter("code-lora", rank=16, alpha=32)
    multi_lora.register_adapter("chat-lora", rank=16, alpha=32)
    print(f"\nRegistered adapters: {multi_lora.registered_adapters}")

    # Test forward with mixed adapters
    print("\n--- Forward test (mixed adapters) ---")
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    multi_lora.set_batch({"math-lora": 2, "code-lora": seq_len - 2})
    print(f"Input: '{text}' ({seq_len} tokens)")
    print(f"Batch: math-lora=2, code-lora={seq_len - 2}")

    with torch.no_grad():
        outputs = model(**inputs)
    next_token = outputs.logits[0, -1].argmax()
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Next token: '{tokenizer.decode(next_token)}'")

    # Test single adapter
    print("\n--- Forward test (single adapter: chat-lora) ---")
    multi_lora.set_batch({"chat-lora": seq_len})
    with torch.no_grad():
        outputs2 = model(**inputs)
    next_token2 = outputs2.logits[0, -1].argmax()
    print(f"Next token: '{tokenizer.decode(next_token2)}'")

    # Test reset
    print("\n--- Reset math-lora ---")
    multi_lora.reset_adapter(model, "math-lora")
    print("Done")

    # Test per-adapter parameters
    print("\n--- Per-adapter parameter counts ---")
    for name in multi_lora.registered_adapters:
        params = list(multi_lora.named_parameters_for_adapter(model, name))
        print(f"  {name}: {sum(p.numel() for _, p in params):,} params")

    # Test state dict
    print("\n--- State dict for math-lora (first 5 keys) ---")
    sd = multi_lora.state_dict_for_adapter(model, "math-lora")
    for i, (k, v) in enumerate(sd.items()):
        if i >= 5:
            print(f"  ... and {len(sd) - 5} more")
            break
        print(f"  {k}: {v.shape}")

    # Test unregister
    print("\n--- Unregister code-lora ---")
    multi_lora.unregister_adapter("code-lora")
    print(f"Remaining adapters: {multi_lora.registered_adapters}")

    # Re-register in freed slot
    print("\n--- Register writing-lora in freed slot ---")
    multi_lora.register_adapter("writing-lora", rank=8, alpha=16)
    print(f"Adapters: {multi_lora.registered_adapters}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
