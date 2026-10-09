"""A plain PyTorch LoRA SFT loop (no Trainer), sized for a CPU.

Loss is computed only on the answer tokens, and only those positions go
through the LM head: Qwen's vocabulary is 151,936 tokens, so full-sequence
logits for a batch of 8 x 250 tokens would cost ~1.2 GB on their own.
"""
from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .prompts import answer_ids, prompt_ids


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lr: float = 2e-4
    epochs: float = 1.0
    batch_size: int = 4
    grad_accum: int = 2
    warmup_ratio: float = 0.06
    max_text_tokens: int = 192
    max_steps: int | None = None  # optimiser steps; overrides epochs (smoke runs)
    seed: int = 0
    threads: int | None = None
    gradient_checkpointing: bool = True


def encode_rows(tokenizer, rows: list[dict], labels: tuple[str, ...], max_text_tokens: int) -> list[tuple[list[int], list[int]]]:
    out = []
    for r in rows:
        p = prompt_ids(tokenizer, r["text"], labels, max_text_tokens)
        a = answer_ids(tokenizer, r["label"])
        out.append((p + a, [-100] * len(p) + a))
    return out


def collate(batch, pad_id: int):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    return ids, lab, att


def answer_loss(model, ids, labels, attention_mask):
    """Causal-LM loss on labelled positions only, projecting just those hidden states."""
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = lm.base_model(input_ids=ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    h = hidden[:, :-1][mask]
    logits = lm.get_output_embeddings()(h).float()
    return torch.nn.functional.cross_entropy(logits, tgt[mask])


def add_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=cfg.target_modules, task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def train(model, tokenizer, rows: list[dict], labels: tuple[str, ...], cfg: TrainConfig, log=print) -> dict:
    """Train LoRA adapters in place on ``model`` (already wrapped by ``add_lora``)."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = encode_rows(tokenizer, rows, labels, cfg.max_text_tokens)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    micro_per_epoch = math.ceil(len(data) / cfg.batch_size)
    steps = cfg.max_steps or max(1, math.ceil(micro_per_epoch * cfg.epochs / cfg.grad_accum))
    warm = max(1, int(steps * cfg.warmup_ratio))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )
    rng = random.Random(cfg.seed)
    order: list[int] = []
    history = []
    model.train()
    t0 = time.perf_counter()
    for step in range(steps):
        total = 0.0
        for _ in range(cfg.grad_accum):
            if len(order) < cfg.batch_size:
                fresh = list(range(len(data)))
                rng.shuffle(fresh)
                order += fresh
            batch = [data[i] for i in order[: cfg.batch_size]]
            del order[: cfg.batch_size]
            ids, lab, att = collate(batch, pad)
            loss = answer_loss(model, ids, lab, att) / cfg.grad_accum
            loss.backward()
            total += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        history.append({"step": step + 1, "loss": round(total, 5), "lr": round(sched.get_last_lr()[0], 8), "elapsed_s": round(time.perf_counter() - t0, 1)})
        if log and (step == 0 or (step + 1) % 5 == 0 or step + 1 == steps):
            log(f"step {step + 1}/{steps} loss {total:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    model.eval()
    trainable = sum(p.numel() for p in params)
    return {
        "config": asdict(cfg),
        "n_train": len(data),
        "optimizer_steps": steps,
        "examples_seen": steps * cfg.batch_size * cfg.grad_accum,
        "trainable_params": trainable,
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "first_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "mean_loss_last_10": round(sum(h["loss"] for h in history[-10:]) / len(history[-10:]), 5),
        "history": history,
        "hardware": hardware(),
    }


def hardware() -> dict:
    cpu = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import os

    mem = None
    try:
        mem = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "ram_gb": mem,
        "gpu": torch.cuda.is_available(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def save_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")
