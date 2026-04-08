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

    # --- Packed forward ---
    print("\n--- Packed forward (both sequences, different adapters) ---")

    # Concatenate tokens: [seq_a..., seq_b...]
    packed_ids = torch.tensor([ids_a + ids_b], device="cuda")  # [1, total]

    # Position ids: reset for each sequence
    position_ids = torch.cat([
        torch.arange(len_a),
        torch.arange(len_b),
    ]).unsqueeze(0).cuda()  # [1, total]

    # cu_seqlens for flash attention varlen
    cu_seqlens = torch.tensor([0, len_a, total], dtype=torch.int32, device="cuda")

    # lora_num_tokens: adapter 0 gets len_a tokens, adapter 1 gets len_b
    set_lora_num_tokens(torch.tensor([len_a, len_b], dtype=torch.int32), reset_reference=True)

    with torch.no_grad():
        packed_out = model(
            input_ids=packed_ids,
            position_ids=position_ids,
            cache_position=torch.arange(total, device="cuda"),
        )

    logits = packed_out.logits[0]  # [total, vocab]
    next_a = tokenizer.decode(logits[len_a - 1].argmax())
    next_b = tokenizer.decode(logits[total - 1].argmax())
    print(f"Seq A next token (adapter 0): '{next_a}'")
    print(f"Seq B next token (adapter 1): '{next_b}'")

    # --- Unpacked forward for comparison ---
    print("\n--- Unpacked forward (each sequence separately) ---")

    # Seq A alone with adapter 0
    set_lora_num_tokens(torch.tensor([len_a, 0], dtype=torch.int32))
    with torch.no_grad():
        out_a = model(input_ids=torch.tensor([ids_a], device="cuda"))
    next_a_solo = tokenizer.decode(out_a.logits[0, -1].argmax())
    print(f"Seq A solo next token (adapter 0): '{next_a_solo}'")

    # Seq B alone with adapter 1
    set_lora_num_tokens(torch.tensor([0, len_b], dtype=torch.int32))
    with torch.no_grad():
        out_b = model(input_ids=torch.tensor([ids_b], device="cuda"))
    next_b_solo = tokenizer.decode(out_b.logits[0, -1].argmax())
    print(f"Seq B solo next token (adapter 1): '{next_b_solo}'")

    # Compare
    print("\n--- Comparison ---")
    print(f"Seq A packed vs solo: '{next_a}' vs '{next_a_solo}' {'✓ match' if next_a == next_a_solo else '✗ MISMATCH'}")
    print(f"Seq B packed vs solo: '{next_b}' vs '{next_b_solo}' {'✓ match' if next_b == next_b_solo else '✗ MISMATCH'}")

    reset_state()
    print("\nDone!")


if __name__ == "__main__":
    main()
