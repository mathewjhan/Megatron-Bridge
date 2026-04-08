"""Debug script: packed sequences with different adapters through an HF model.

Simulates slime's thd packing: two sequences from different adapters
concatenated into one tensor, with flash attention varlen masking.

Usage:
    python debug_multi_lora_packed.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import SimpleMultiLoRALinear
from megatron.bridge.peft.multi_lora_state import set_lora_num_tokens, reset_state

N_ADAPTERS = 2
MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.eval()

    # Apply multi-LoRA
    multi_lora = MultiLoRA(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        n_adapters=N_ADAPTERS,
        dim=16,
        alpha=32,
        lora_dtype=torch.bfloat16,
    )
    model = multi_lora(model, training=True)

    # Make adapter 0 produce different output than adapter 1
    for module in model.modules():
        if isinstance(module, SimpleMultiLoRALinear):
            nn.init.normal_(module.adapters[0].linear_out.weight, std=0.1)

    # Two separate prompts, one per adapter
    text_a = "The capital of France is"  # adapter 0
    text_b = "Once upon a time"          # adapter 1

    ids_a = tokenizer.encode(text_a, add_special_tokens=False)
    ids_b = tokenizer.encode(text_b, add_special_tokens=False)
    len_a, len_b = len(ids_a), len(ids_b)
    total = len_a + len_b

    print(f"\nSeq A (adapter 0): '{text_a}' ({len_a} tokens)")
    print(f"Seq B (adapter 1): '{text_b}' ({len_b} tokens)")

    # --- Autoregressive generation: packed vs solo ---
    num_steps = 20

    # Generate with each sequence solo
    print(f"\n--- Solo generation ({num_steps} steps) ---")

    gen_a = list(ids_a)
    gen_b = list(ids_b)

    for step in range(num_steps):
        # Seq A with adapter 0
        set_lora_num_tokens(torch.tensor([len(gen_a), 0], dtype=torch.int32), reset_reference=True)
        with torch.no_grad():
            out_a = model(input_ids=torch.tensor([gen_a], device="cuda"))
        next_a = out_a.logits[0, -1].argmax().item()
        gen_a.append(next_a)

        # Seq B with adapter 1
        set_lora_num_tokens(torch.tensor([0, len(gen_b)], dtype=torch.int32), reset_reference=True)
        with torch.no_grad():
            out_b = model(input_ids=torch.tensor([gen_b], device="cuda"))
        next_b = out_b.logits[0, -1].argmax().item()
        gen_b.append(next_b)

    decoded_a = tokenizer.decode(gen_a)
    decoded_b = tokenizer.decode(gen_b)
    print(f"Seq A (adapter 0): {decoded_a}")
    print(f"Seq B (adapter 1): {decoded_b}")

    # Generate with packed sequences
    print(f"\n--- Packed generation ({num_steps} steps) ---")

    packed_a = list(ids_a)
    packed_b = list(ids_b)

    for step in range(num_steps):
        total = len(packed_a) + len(packed_b)
        packed_ids = torch.tensor([packed_a + packed_b], device="cuda")
        position_ids = torch.cat([
            torch.arange(len(packed_a)),
            torch.arange(len(packed_b)),
        ]).unsqueeze(0).cuda()

        set_lora_num_tokens(torch.tensor([len(packed_a), len(packed_b)], dtype=torch.int32), reset_reference=True)

        with torch.no_grad():
            packed_out = model(
                input_ids=packed_ids,
                position_ids=position_ids,
                cache_position=torch.arange(total, device="cuda"),
            )

        logits = packed_out.logits[0]
        next_a = logits[len(packed_a) - 1].argmax().item()
        next_b = logits[total - 1].argmax().item()
        packed_a.append(next_a)
        packed_b.append(next_b)

    decoded_packed_a = tokenizer.decode(packed_a)
    decoded_packed_b = tokenizer.decode(packed_b)
    print(f"Seq A (adapter 0): {decoded_packed_a}")
    print(f"Seq B (adapter 1): {decoded_packed_b}")

    # Compare
    print(f"\n--- Comparison ---")
    match_a = decoded_a == decoded_packed_a
    match_b = decoded_b == decoded_packed_b
    print(f"Seq A: {'✓ match' if match_a else '✗ MISMATCH'}")
    if not match_a:
        print(f"  Solo:   {decoded_a}")
        print(f"  Packed: {decoded_packed_a}")
    print(f"Seq B: {'✓ match' if match_b else '✗ MISMATCH'}")
    if not match_b:
        print(f"  Solo:   {decoded_b}")
        print(f"  Packed: {decoded_packed_b}")

    reset_state()
    print("\nDone!")


if __name__ == "__main__":
    main()
