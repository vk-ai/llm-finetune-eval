"""How much of the fine-tune's test accuracy could come from near-duplicate boilerplate?

data.py removes exact (case/whitespace-normalised) duplicates across splits. LEDGAR
provisions are often lightly edited copies of each other, so this also flags test
items whose word-set Jaccard similarity to some training item is >= 0.8 and reports
accuracy with and without them. Writes reports/near_duplicates.json.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.8


def rd(p):
    return [json.loads(l) for l in (ROOT / p).read_text().splitlines() if l.strip()]


def words(s):
    return set(re.findall(r"\w+", s.lower()))


train = [words(r["text"]) for r in rd("data/train.jsonl")]
test = rd("data/test.jsonl")
near = {r["id"] for r in test if max(len(w & t) / len(w | t) for w in [words(r["text"])] for t in train) >= THRESHOLD}
out = {"threshold": THRESHOLD, "n_test": len(test), "n_near_duplicate": len(near), "accuracy": {}}
for system in ("base", "finetuned"):
    preds = {r["id"]: r for r in rd(f"runs/{system}.jsonl")}
    for name, ids in (("near_duplicate", near), ("rest", {r["id"] for r in test} - near)):
        out["accuracy"].setdefault(system, {})[name] = round(sum(preds[i]["pred"] == preds[i]["label"] for i in ids) / len(ids), 4)
(ROOT / "reports" / "near_duplicates.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
