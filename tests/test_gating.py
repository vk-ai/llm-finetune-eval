import json
import random
from pathlib import Path

import pytest

from llm_finetune_eval.cli import main
from llm_finetune_eval.data import LABELS, write_jsonl
from llm_finetune_eval.gating import build_report, gate

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "max_drop": {"macro_f1": 0.0, "accuracy": 0.0},
    "min": {"macro_f1": 0.5},
    "significance": {"alpha": 0.05},
    "allow_testset_change": False,
    "require_improvement": {"metric": "macro_f1", "alpha": 0.05, "min_delta": 0.02},
}


def _setup(tmp_path, acc_base, acc_ft, n=200, seed=0):
    rng = random.Random(seed)
    rows = [{"id": f"t{i:03d}", "text": "x", "label": LABELS[i % len(LABELS)]} for i in range(n)]
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").mkdir()
    write_jsonl(tmp_path / "data" / "test.jsonl", rows)

    def preds(acc):
        out = []
        for r in rows:
            right = rng.random() < acc
            wrong = LABELS[(LABELS.index(r["label"]) + 1) % len(LABELS)]
            out.append({"id": r["id"], "label": r["label"], "pred": r["label"] if right else wrong, "latency_ms": 10.0})
        return out

    write_jsonl(tmp_path / "runs" / "base.jsonl", preds(acc_base))
    write_jsonl(tmp_path / "runs" / "finetuned.jsonl", preds(acc_ft))
    rep = build_report(tmp_path / "data" / "test.jsonl", tmp_path / "runs", ["base", "finetuned"], LABELS, "base", "finetuned", n_resamples=300)
    return rep


def test_report_shape_and_stats(tmp_path):
    rep = _setup(tmp_path, 0.5, 0.9)
    ft = rep["systems"]["finetuned"]
    assert ft["ci"]["accuracy"]["low"] <= ft["accuracy"] <= ft["ci"]["accuracy"]["high"]
    assert rep["comparison"]["mcnemar"]["c"] > rep["comparison"]["mcnemar"]["b"]
    assert rep["comparison"]["permutation"]["macro_f1"]["delta"] > 0.2
    assert rep["systems"]["base"]["latency_ms"]["p50"] == 10.0


def test_gate_passes_clear_improvement(tmp_path):
    rep = _setup(tmp_path, 0.5, 0.9)
    res = gate(rep, tmp_path / "runs", tmp_path / "data", RULES)
    assert res.passed, res.markdown()
    assert {c["check"] for c in res.checks} >= {"macro_f1 drop", "macro_f1 floor", "McNemar vs baseline", "macro_f1 improvement"}


def test_gate_fails_when_roles_are_swapped(tmp_path):
    rep = _setup(tmp_path, 0.5, 0.9)
    res = gate(rep, tmp_path / "runs", tmp_path / "data", RULES, baseline="finetuned", candidate="base")
    assert not res.passed
    failed = {c["check"] for c in res.checks if not c["ok"]}
    assert {"McNemar vs baseline", "macro_f1 improvement", "macro_f1 drop"} <= failed


def test_gate_fails_when_no_significant_gain(tmp_path):
    rep = _setup(tmp_path, 0.8, 0.8, seed=3)
    res = gate(rep, tmp_path / "runs", tmp_path / "data", dict(RULES, max_drop={"macro_f1": 0.1}))
    assert not res.passed
    assert [c["check"] for c in res.checks if not c["ok"]] == ["macro_f1 improvement"]


def test_gate_blocks_changed_testset(tmp_path):
    rep = _setup(tmp_path, 0.5, 0.9)
    with open(tmp_path / "data" / "test.jsonl", "a") as f:
        f.write("")
    rep["testset"]["sha256"] = "0" * 64
    # the baseline is rebuilt from the same report, so simulate a stale baseline via run_gate directly
    from llm_eval_lab.gate import run_gate
    from llm_finetune_eval.gating import make_baseline

    base = make_baseline(rep, tmp_path / "runs", "base")
    base["testset"] = dict(base["testset"], sha256="f" * 64)
    res = run_gate(rep, tmp_path / "runs", base, RULES, tmp_path / "data")
    assert not res.passed and res.checks[0]["check"] == "test set hash"


def test_missing_predictions_raise(tmp_path):
    _setup(tmp_path, 0.5, 0.9)
    lines = (tmp_path / "runs" / "finetuned.jsonl").read_text().splitlines()[:-1]
    (tmp_path / "runs" / "finetuned.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="no prediction"):
        build_report(tmp_path / "data" / "test.jsonl", tmp_path / "runs", ["base", "finetuned"], LABELS, "base", "finetuned")


def test_cli_report_and_gate_exit_codes(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, 0.5, 0.9)
    import yaml

    (tmp_path / "gate.yaml").write_text(yaml.safe_dump(RULES))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["--data", str(tmp_path / "data"), "--runs", str(tmp_path / "runs")]
    assert main(["report", *args, "--out", str(tmp_path / "eval.json")]) == 0
    assert main(["gate", *args, "--report", str(tmp_path / "eval.json"), "--rules", str(tmp_path / "gate.yaml"), "--out", str(tmp_path / "g.json")]) == 0
    assert json.loads((tmp_path / "g.json").read_text())["passed"] is True
    assert main(["gate", *args, "--report", str(tmp_path / "eval.json"), "--rules", str(tmp_path / "gate.yaml"),
                 "--baseline", "finetuned", "--candidate", "base"]) == 1
    assert "Eval gate: PASSED" in summary.read_text() and "Eval gate: FAILED" in summary.read_text()


@pytest.mark.skipif(not (ROOT / "reports" / "eval.json").exists(), reason="no committed real-run results yet")
def test_committed_results_reproduce_and_pass_gate(tmp_path):
    """The committed eval.json is recomputed from the committed per-example runs, and the gate passes."""
    from llm_finetune_eval.gating import load_rules

    committed = json.loads((ROOT / "reports" / "eval.json").read_text())
    fresh = build_report(ROOT / "data" / "test.jsonl", ROOT / "runs", ["base", "finetuned"], LABELS, "base", "finetuned")
    assert json.dumps(fresh, sort_keys=True) == json.dumps(committed, sort_keys=True)
    res = gate(fresh, ROOT / "runs", ROOT / "data", load_rules(ROOT / "gate.yaml"))
    assert res.passed, res.markdown()
    swapped = gate(fresh, ROOT / "runs", ROOT / "data", load_rules(ROOT / "gate.yaml"), baseline="finetuned", candidate="base")
    assert not swapped.passed
