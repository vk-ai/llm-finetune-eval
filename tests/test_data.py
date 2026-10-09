import json
from collections import Counter
from pathlib import Path

import pytest

from llm_finetune_eval.data import LABELS, build_splits, normalise, read_jsonl, select_balanced, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _raw(n_per_label, prefix, dup_every=0):
    rows = []
    for label in LABELS:
        for i in range(n_per_label):
            text = f"{prefix} {label} clause {i}"
            if dup_every and i % dup_every == 0:
                text = f"shared   {label} CLAUSE {i}"  # differs only by case/whitespace across splits
            rows.append({"src_index": len(rows), "text": text, "label": label})
    return rows


def test_select_balanced_is_balanced_and_seeded():
    rows = _raw(10, "x")
    a = select_balanced(rows, 4, seed=1)
    b = select_balanced(rows, 4, seed=1)
    c = select_balanced(rows, 4, seed=2)
    assert a == b and a != c
    assert Counter(r["label"] for r in a) == {l: 4 for l in LABELS}


def test_select_balanced_drops_duplicates_and_excluded():
    rows = _raw(3, "x") + [{"src_index": 99, "text": "X  base salary CLAUSE 0", "label": "Base Salary"}]
    with pytest.raises(ValueError, match="Base Salary"):
        select_balanced(rows, 3, seed=0, exclude={normalise("x Base Salary clause 1")})
    assert len(select_balanced(rows, 3, seed=0)) == 3 * len(LABELS)


def test_build_splits_has_no_cross_split_duplicates():
    raw = {"test": _raw(6, "te", dup_every=2), "validation": _raw(6, "va", dup_every=2), "train": _raw(9, "tr", dup_every=2)}
    splits, manifest = build_splits(raw, {"test": 3, "validation": 2, "train": 4}, seed=0)
    seen = {}
    for split, rows in splits.items():
        for r in rows:
            k = normalise(r["text"])
            assert k not in seen, (split, seen.get(k))
            seen[k] = split
    assert manifest["excluded_cross_split_duplicates"]["test"] == 0
    assert manifest["excluded_cross_split_duplicates"]["train"] > 0
    assert len({r["id"] for rows in splits.values() for r in rows}) == sum(len(v) for v in splits.values())


def test_write_jsonl_hash_is_stable(tmp_path):
    rows = [{"id": "a", "text": "é", "label": "Vacations"}]
    assert write_jsonl(tmp_path / "a.jsonl", rows) == write_jsonl(tmp_path / "b.jsonl", rows)
    assert read_jsonl(tmp_path / "a.jsonl") == rows


def test_committed_data_matches_manifest_and_is_disjoint():
    import hashlib

    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    texts = {}
    for name, meta in manifest["files"].items():
        path = ROOT / "data" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
        rows = read_jsonl(path)
        assert len(rows) == meta["n"]
        assert set(r["label"] for r in rows) == set(LABELS)
        texts[name] = {normalise(r["text"]) for r in rows}
    assert not texts["test.jsonl"] & texts["train.jsonl"]
    assert not texts["validation.jsonl"] & texts["train.jsonl"]
    assert not texts["test.jsonl"] & texts["validation.jsonl"]
