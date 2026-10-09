"""Command line: prepare -> train -> predict (base, finetuned) -> report -> gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .data import LABELS, prepare, read_jsonl


def load_model(name: str, adapter: str | None = None, threads: int | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if threads:
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def _progress(prefix: str):
    t0 = time.perf_counter()

    def cb(i, n):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"{prefix} {i}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)

    return cb


def cmd_prepare(a):
    m = prepare(Path(a.out), seed=a.seed)
    print(json.dumps(m["files"], indent=2))
    print("excluded cross-split duplicates:", m["excluded_cross_split_duplicates"])


def cmd_train(a):
    from .train import TrainConfig, add_lora, save_json, train

    model, tok = load_model(a.model, threads=a.threads)
    rows = read_jsonl(Path(a.data) / "train.jsonl")
    if a.limit:
        rows = rows[: a.limit]
    cfg = TrainConfig(model=a.model, r=a.r, alpha=a.alpha, lr=a.lr, epochs=a.epochs, batch_size=a.batch_size,
                      grad_accum=a.grad_accum, max_steps=a.max_steps, seed=a.seed, threads=a.threads,
                      max_text_tokens=a.max_text_tokens)
    model = add_lora(model, cfg)
    model.print_trainable_parameters()
    rep = train(model, tok, rows, LABELS, cfg)
    model.save_pretrained(a.out)
    size = sum(f.stat().st_size for f in Path(a.out).glob("*.safetensors"))
    rep["adapter_bytes"] = size
    save_json(Path(a.report), rep)
    print(f"adapter saved to {a.out} ({size / 1e6:.1f} MB); {rep['optimizer_steps']} steps in {rep['wall_time_s']}s; "
          f"loss {rep['first_loss']:.4f} -> {rep['final_loss']:.4f}")


def cmd_predict(a):
    from .scoring import LabelScorer
    from .train import hardware

    model, tok = load_model(a.model, a.adapter, a.threads)
    rows = read_jsonl(Path(a.data) / f"{a.split}.jsonl")
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.perf_counter()
    preds = LabelScorer(model, tok, LABELS, a.max_text_tokens).predict(rows, _progress(a.name))
    out = Path(a.runs)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.name}.jsonl"
    path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in preds))
    acc = sum(p["pred"] == p["label"] for p in preds) / len(preds)
    meta = {"system": a.name, "model": a.model, "adapter": a.adapter, "split": a.split, "n": len(preds),
            "accuracy": round(acc, 4), "wall_time_s": round(time.perf_counter() - t0, 1), "hardware": hardware()}
    (out / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{a.name}: accuracy {acc:.4f} on {len(preds)} {a.split} rows -> {path}")


def cmd_report(a):
    from .gating import build_report, write_json

    rep = build_report(Path(a.data) / f"{a.split}.jsonl", Path(a.runs), [a.baseline, a.candidate], LABELS, a.baseline, a.candidate)
    write_json(Path(a.out), rep)
    for s in (a.baseline, a.candidate):
        m = rep["systems"][s]
        ci = m["ci"]
        print(f"{s:10s} accuracy {m['accuracy']:.4f} [{ci['accuracy']['low']:.4f}, {ci['accuracy']['high']:.4f}]  "
              f"macro-F1 {m['macro_f1']:.4f} [{ci['macro_f1']['low']:.4f}, {ci['macro_f1']['high']:.4f}]")
    mc = rep["comparison"]["mcnemar"]
    print(f"McNemar: {a.baseline} right & {a.candidate} wrong = {mc['b']}, reverse = {mc['c']}, exact p = {mc['p_exact']:.3g}")
    for m, r in rep["comparison"]["permutation"].items():
        print(f"permutation {m}: delta {r['delta']:+.4f} 95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p = {r['p_value']:.4f}")


def cmd_gate(a):
    from .gating import gate, load_rules

    rep = json.loads(Path(a.report).read_text())
    res = gate(rep, Path(a.runs), Path(a.data), load_rules(Path(a.rules)), a.baseline, a.candidate)
    md = res.markdown()
    print(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md)
    if a.out:
        Path(a.out).write_text(json.dumps({"passed": res.passed, "checks": res.checks, "notes": res.notes}, indent=2) + "\n")
    return 0 if res.passed else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lfe", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="download LEDGAR (pinned revision) and write data/*.jsonl")
    s.add_argument("--out", default="data")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("train", help="LoRA fine-tune")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--data", default="data")
    s.add_argument("--out", default="artifacts/adapter")
    s.add_argument("--report", default="reports/train.json")
    s.add_argument("--r", type=int, default=8)
    s.add_argument("--alpha", type=int, default=16)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--grad-accum", type=int, default=2)
    s.add_argument("--max-steps", type=int)
    s.add_argument("--max-text-tokens", type=int, default=192)
    s.add_argument("--limit", type=int, help="use only the first N training rows")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("predict", help="score the label set for every row of a split")
    s.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    s.add_argument("--adapter")
    s.add_argument("--name", required=True, help="system name, e.g. base or finetuned")
    s.add_argument("--data", default="data")
    s.add_argument("--split", default="test")
    s.add_argument("--runs", default="runs")
    s.add_argument("--limit", type=int)
    s.add_argument("--max-text-tokens", type=int, default=192)
    s.add_argument("--threads", type=int)
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("report", help="metrics + CIs + paired tests via llm-eval-lab")
    s.add_argument("--data", default="data")
    s.add_argument("--split", default="test")
    s.add_argument("--runs", default="runs")
    s.add_argument("--baseline", default="base")
    s.add_argument("--candidate", default="finetuned")
    s.add_argument("--out", default="reports/eval.json")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("gate", help="pass/fail regression gate (exit 1 on FAIL)")
    s.add_argument("--report", default="reports/eval.json")
    s.add_argument("--rules", default="gate.yaml")
    s.add_argument("--data", default="data")
    s.add_argument("--runs", default="runs")
    s.add_argument("--baseline")
    s.add_argument("--candidate")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_gate)

    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
