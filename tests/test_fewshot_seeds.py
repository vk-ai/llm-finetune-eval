"""Few-shot baseline (stratified demos, cached shared prefix) and the multi-seed summary."""
import json
import random

import torch

from llm_finetune_eval.data import LABELS, write_jsonl
from llm_finetune_eval.gating import seeds_summary
from llm_finetune_eval.prompts import answer_ids, prompt_ids, select_shots, shared_prefix_ids
from llm_finetune_eval.scoring import LabelScorer

from conftest import TOY_TEXT, make_tokenizer

TRAIN = [{"id": f"tr{l[:3]}{i}", "text": TOY_TEXT[l], "label": l} for l in LABELS for i in range(4)]


def test_select_shots_is_stratified_seeded_and_train_only():
    s5 = select_shots(TRAIN, 5, seed=0)
    assert len(s5) == 5 and len({s["label"] for s in s5}) == 5
    assert s5 == select_shots(TRAIN, 5, seed=0) and s5 != select_shots(TRAIN, 5, seed=1)
    s10 = select_shots(TRAIN, 10, seed=0)
    assert len({s["label"] for s in s10}) == 8 and len({s["id"] for s in s10}) == 10
    assert all(s["id"].startswith("tr") for s in s10)


def test_fewshot_prompt_starts_with_shared_prefix(tok):
    shots = select_shots(TRAIN, 5)
    for chat in (True, False):
        t = tok if chat else make_tokenizer(chat=False)
        pre = shared_prefix_ids(t, LABELS, shots)
        full = prompt_ids(t, TOY_TEXT["Vacations"], LABELS, shots=shots)
        assert pre and full[: len(pre)] == pre and len(full) > len(pre)
        assert len(full) > len(prompt_ids(t, TOY_TEXT["Vacations"], LABELS)) + 5 * 8
    assert shared_prefix_ids(tok, LABELS, None) == []


def test_cached_fewshot_scores_equal_full_sequence(model, tok):
    shots = select_shots(TRAIN, 5)
    scorer = LabelScorer(model, tok, LABELS, shots=shots)
    for text in (TOY_TEXT["Benefits"], TOY_TEXT["Terminations"], TOY_TEXT["Benefits"]):  # reuse + crop path
        got = scorer.scores(text)
        p = prompt_ids(tok, text, LABELS, shots=shots)
        for label in LABELS:
            a = answer_ids(tok, label)
            with torch.no_grad():
                logp = torch.log_softmax(model(input_ids=torch.tensor([p + a])).logits[0].float(), -1)
            want = float(sum(logp[len(p) - 1 + i, t] for i, t in enumerate(a)))
            assert abs(got[label] - want) < 1e-4, label


def test_seeds_summary_mean_sd_and_gates(tmp_path):
    rng = random.Random(0)
    rows = [{"id": f"t{i:03d}", "text": "x", "label": LABELS[i % 8]} for i in range(240)]
    (tmp_path / "data").mkdir(); (tmp_path / "runs").mkdir()
    write_jsonl(tmp_path / "data" / "test.jsonl", rows)

    def preds(acc):
        return [{"id": r["id"], "label": r["label"], "latency_ms": 1.0,
                 "pred": r["label"] if rng.random() < acc else LABELS[(LABELS.index(r["label"]) + 1) % 8]} for r in rows]

    for name, acc in [("base", 0.5), ("base_5shot", 0.6), ("ft0", 0.92), ("ft1", 0.9), ("ft2", 0.94)]:
        write_jsonl(tmp_path / "runs" / f"{name}.jsonl", preds(acc))
    rules = tmp_path / "gate.yaml"
    rules.write_text(json.dumps({"max_drop": {"macro_f1": 0.0}, "min": {"macro_f1": 0.5}, "significance": {"alpha": 0.05},
                                 "require_improvement": {"metric": "macro_f1", "alpha": 0.05, "min_delta": 0.05, "n_resamples": 200}}))
    rep = seeds_summary(tmp_path / "data", tmp_path / "runs", LABELS, ["base", "base_5shot"], ["ft0", "ft1", "ft2"], rules)
    assert len(rep["gates"]) == 6 and rep["all_gates_passed"]
    agg = rep["candidates_vs"]["base_5shot"]
    f1 = [rep["systems"][c]["macro_f1"] for c in ("ft0", "ft1", "ft2")]
    assert agg["macro_f1_min"] == min(f1) and agg["macro_f1_max"] == max(f1) and 0 < agg["macro_f1_sd"] < 0.1
    # a "seed" no better than the few-shot baseline must fail its gate
    write_jsonl(tmp_path / "runs" / "ft2.jsonl", preds(0.6))
    rep = seeds_summary(tmp_path / "data", tmp_path / "runs", LABELS, ["base_5shot"], ["ft0", "ft1", "ft2"], rules)
    assert not rep["all_gates_passed"] and rep["candidates_vs"]["base_5shot"]["gates_passed"] == 2
