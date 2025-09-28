#!/usr/bin/env python
"""
LoRA finetuning script for ProGen3 on TasR sequences.

Data format: a plain text file where each line is a protein sequence (AAs only).
This script applies LoRA adapters to selected Linear layers and runs a simple
PyTorch training loop.

Example:
python tasr_fintune_lora/train_lora.py \
  --base-model Profluent-Bio/progen3-339m \
  --data-file tasRdata/protein_sequences.txt \
  --output-dir outputs/tasr_lora_339m \
  --epochs 2 --batch-size 8 --lr 5e-5 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Ensure local package is importable without installation
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from progen3.batch_preparer import ProGen3BatchPreparer
from progen3.modeling import ProGen3ForCausalLM

try:
    from peft import LoraConfig, get_peft_model
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "peft is required for LoRA finetuning. Please install with `pip install peft`."
    ) from e


AA_ALPHABET = set(list("ACDEFGHIKLMNPQRSTVWY"))


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    # Filter to standard AAs only
    return "".join([c for c in seq if c in AA_ALPHABET])


class TasRDataset(Dataset):
    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:  # type: ignore[override]
        return self.sequences[idx]


class Collator:
    def __init__(self, device: torch.device):
        self.bp = ProGen3BatchPreparer()
        self.device = device

    def __call__(self, batch: List[str]):
        # ProGen3BatchPreparer expects sequences in 1->2 orientation only
        encodings = [self.bp.prepare_singleseq(seq, reverse_sequence=False) for seq in batch]
        padded = self.bp.pad_encodings(encodings)
        return {k: v.to(device=self.device, non_blocking=True) for k, v in padded.items()}


@dataclass
class TrainArgs:
    base_model: str
    data_file: str
    output_dir: str
    epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # target module names in ProGen3
    target_modules: str = "q_proj,k_proj,v_proj,o_proj,w1,w2,w3"
    seed: int = 42
    precision: str = "bf16"  # options: bf16, fp16, fp32
    num_workers: int = 2
    max_steps: int = -1  # set >0 to override epochs
    save_every_steps: int = 0  # 0 means save only at end of each epoch


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser(description="LoRA finetuning for ProGen3 on TasR sequences")
    p.add_argument("--base-model", required=True, help="HF model id or local path of base ProGen3 model")
    p.add_argument("--data-file", default="tasRdata/protein_sequences.txt", help="Sequences txt file (one per line)")
    p.add_argument("--output-dir", required=True, help="Where to save LoRA adapter and logs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,w1,w2,w3",
        help="Comma-separated module name substrings to apply LoRA to",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-every-steps", type=int, default=0)
    args = p.parse_args()
    return TrainArgs(
        base_model=args.base_model,
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        seed=args.seed,
        precision=args.precision,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        save_every_steps=args.save_every_steps,
    )


def load_sequences(path: str) -> List[str]:
    sequences: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = clean_sequence(line)
            if s:
                sequences.append(s)
    if not sequences:
        raise RuntimeError(f"No valid sequences found in {path}")
    return sequences


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_scheduler(optimizer: torch.optim.Optimizer, num_warmup: int, num_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < num_warmup:
            return float(current_step) / float(max(1, num_warmup))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def maybe_save_adapter(peft_model, output_dir: str, step: int | None = None):
    save_dir = output_dir if step is None else os.path.join(output_dir, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)
    peft_model.save_pretrained(save_dir)


def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = get_dtype(cfg.precision)

    print(f"Loading base model: {cfg.base_model}")
    model = ProGen3ForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    model = model.to(device)
    model.config.use_cache = False

    lora_targets = [x.strip() for x in cfg.target_modules.split(",") if x.strip()]
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    sequences = load_sequences(cfg.data_file)
    dataset = TasRDataset(sequences)
    collate = Collator(device=device)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # steps per epoch
    steps_per_epoch = math.ceil(len(dataset) / (cfg.batch_size * max(1, cfg.grad_accum)))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    autocast_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16

    global_step = 0
    model.train()
    for epoch in range(cfg.epochs if cfg.max_steps < 0 else 10**9):
        running_loss = 0.0
        for step, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=(dtype in (torch.float16, torch.bfloat16)), dtype=autocast_dtype):
                out = model(**batch, return_dict=True)
                loss = out.loss / cfg.grad_accum

            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                if dtype == torch.float16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"step {global_step}/{total_steps} - loss: {out.loss.item():.4f}")

                if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                    maybe_save_adapter(model, cfg.output_dir, step=global_step)

                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    break

            running_loss += out.loss.item()

        print(f"Epoch {epoch+1} done. avg_loss: {running_loss / max(1, step+1):.4f}")
        if cfg.max_steps < 0:
            maybe_save_adapter(model, cfg.output_dir, step=None)
        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    # Save training args
    with open(os.path.join(cfg.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Finished. LoRA adapter saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()

