#!/usr/bin/env python
"""
Generate new TasR-like protein sequences using a base ProGen3 model + LoRA adapter.

This script loads the base model, attaches the LoRA adapter, then uses
ProGen3's generator to sample sequences from a simple prompt (unconditional by default).

Examples:

python tasr_fintune_lora/generate_with_lora.py \
  --base-model Profluent-Bio/progen3-339m \
  --adapter-dir outputs/tasr_lora_339m \
  --output-fasta outputs/tasr_generations.fasta \
  --num-sequences 1000 --min-new 100 --max-new 300 \
  --temperature 0.85 --top-p 0.95
"""

import argparse
import os
import sys
from typing import Dict

import torch

# Ensure local package is importable without installation
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from progen3.generator import ProGen3Generator
from progen3.tools.utils import write_fasta_sequences
from progen3.modeling import ProGen3ForCausalLM

try:
    from peft import PeftModel
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "peft is required for LoRA inference. Please install with `pip install peft`."
    ) from e


def parse_args():
    p = argparse.ArgumentParser(description="Generate protein sequences with ProGen3 + LoRA adapter")
    p.add_argument("--base-model", required=True, help="HF model id or local path of base ProGen3 model")
    p.add_argument("--adapter-dir", required=True, help="Path to saved LoRA adapter (from train_lora.py)")
    p.add_argument("--output-fasta", required=True, help="Path to write generated sequences (FASTA)")
    p.add_argument("--num-sequences", type=int, default=100, help="Number of sequences to generate")
    p.add_argument("--min-new", type=int, default=100, help="Min new tokens to generate")
    p.add_argument("--max-new", type=int, default=300, help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--batch-max-tokens", type=int, default=65536, help="Batch token budget for generator")
    p.add_argument(
        "--prompt",
        type=str,
        default="1",
        help="Prompt string. For unconditional forward generation, use '1'. For reverse, use '2'.",
    )
    return p.parse_args()


def get_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = get_dtype(args.precision)

    print(f"Loading base model: {args.base_model}")
    base = ProGen3ForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    base = base.to(device)
    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    generator = ProGen3Generator(model=model, max_batch_tokens=args.batch_max_tokens,
                                 temperature=args.temperature, top_p=args.top_p)

    results: Dict[str, str] = {}
    i = 0
    for gen in generator.generate(
        prompt=args.prompt,
        num_sequences=args.num_sequences,
        min_new_tokens=args.min_new,
        max_new_tokens=args.max_new,
    ):
        if gen.sequence:  # only keep valid compiled sequences
            results[str(i)] = gen.sequence
            i += 1

    os.makedirs(os.path.dirname(args.output_fasta) or ".", exist_ok=True)
    write_fasta_sequences(args.output_fasta, results)
    print(f"Wrote {len(results)} sequences to {args.output_fasta}")


if __name__ == "__main__":
    main()

