"""Compare base vs fine-tuned predictions with llm-eval-lab's statistics and gate.

Metrics, bootstrap CIs, McNemar, the permutation test and ``run_gate`` all come
from https://github.com/vk-ai/llm-eval-lab (installed from git, pinned commit).
This module only adapts our files to its report/baseline shapes and adds one
rule llm-eval-lab doesn't have: a fine-tune must be *significantly better*.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from llm_eval_lab.gate import GateResult, run_gate
from llm_eval_lab.metrics import bootstrap_ci, classification_report
from llm_eval_lab.stats import mcnemar, permutation_test

from .data import read_jsonl


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_preds(runs_dir: Path, system: str, test_rows: list[dict]) -> list[str]:
    preds = {r["id"]: r["pred"] for r in read_jsonl(Path(runs_dir) / f"{system}.jsonl")}
    missing = [r["id"] for r in test_rows if r["id"] not in preds]
    if missing:
        raise ValueError(f"{system}: no prediction for {len(missing)} test ids, e.g. {missing[:3]}")
    return [preds[r["id"]] for r in test_rows]


def build_report(test_path: Path, runs_dir: Path, systems: list[str], labels: tuple[str, ...], baseline: str, candidate: str,
                 n_resamples: int = 1000, seed: int = 0) -> dict:
    rows = read_jsonl(test_path)
    gold = [r["label"] for r in rows]
    report = {
        "testset": {"name": Path(test_path).name, "sha256": file_sha256(test_path), "n": len(rows)},
        "labels": list(labels),
        "systems": {},
        "comparison": {"baseline": baseline, "candidate": candidate},
        "mt": {"systems": {}},  # llm-eval-lab report shape; no MT here
    }
    preds = {}
    for s in systems:
        p = load_preds(runs_dir, s, rows)
        preds[s] = p
        rep = classification_report(gold, p, labels)
        rep["ci"] = {m: bootstrap_ci(gold, p, labels, m, n_resamples, seed=seed).to_dict() for m in ("accuracy", "macro_f1")}
        lat = [r["latency_ms"] for r in read_jsonl(Path(runs_dir) / f"{s}.jsonl") if "latency_ms" in r]
        if lat:
            lat.sort()
            rep["latency_ms"] = {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1]}
        report["systems"][s] = rep
    a, b = preds[baseline], preds[candidate]
    mc = mcnemar([x == y for x, y in zip(a, gold)], [x == y for x, y in zip(b, gold)])
    report["comparison"]["mcnemar"] = mc.to_dict()
    report["comparison"]["permutation"] = {
        m: permutation_test(gold, a, b, labels, m, n_resamples, seed=seed).to_dict() for m in ("accuracy", "macro_f1")
    }
    return report


def make_baseline(report: dict, runs_dir: Path, system: str) -> dict:
    rows = {r["id"]: r["pred"] for r in read_jsonl(Path(runs_dir) / f"{system}.jsonl")}
    m = report["systems"][system]
    return {
        "system": system,
        "testset": report["testset"],
        "metrics": {"accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4)},
        "mt": {},
        "predictions": dict(sorted(rows.items())),
    }


def gate(report: dict, runs_dir: Path, data_dir: Path, rules: dict, baseline: str | None = None, candidate: str | None = None) -> GateResult:
    """llm-eval-lab's run_gate (drop limits, floors, not-significantly-worse) plus a must-improve rule."""
    baseline = baseline or report["comparison"]["baseline"]
    candidate = candidate or report["comparison"]["candidate"]
    rep = dict(report, comparison=dict(report["comparison"], candidate=candidate))
    base = make_baseline(report, runs_dir, baseline)
    res = run_gate(rep, Path(runs_dir), base, {k: v for k, v in rules.items() if k != "require_improvement"}, Path(data_dir))
    res.notes.insert(0, f"baseline = {baseline}, candidate = {candidate}, n = {report['testset']['n']}")
    imp = rules.get("require_improvement")
    if imp:
        metric, alpha, min_delta = imp.get("metric", "macro_f1"), imp.get("alpha", 0.05), imp.get("min_delta", 0.0)
        rows = read_jsonl(Path(data_dir) / report["testset"]["name"])
        gold = [r["label"] for r in rows]
        pa = load_preds(runs_dir, baseline, rows)
        pb = load_preds(runs_dir, candidate, rows)
        labels = tuple(report["labels"])
        pt = permutation_test(gold, pa, pb, labels, metric, imp.get("n_resamples", 1000), seed=0)
        ok = pt.delta >= min_delta and pt.p_value < alpha and pt.ci_low > 0
        res.checks.append({
            "check": f"{metric} improvement",
            "baseline": round(pt.a, 4),
            "current": round(pt.b, 4),
            "rule": f"delta >= {min_delta}, permutation p < {alpha}, 95% CI low > 0 "
                    f"(delta {pt.delta:+.4f}, CI [{pt.ci_low:+.4f}, {pt.ci_high:+.4f}], p={pt.p_value:.4g})",
            "ok": ok,
        })
        res.passed &= ok
    return res


def load_rules(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def write_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")
