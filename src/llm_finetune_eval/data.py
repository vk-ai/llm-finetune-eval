"""Build a small, balanced HR/payroll/compliance subset of LEDGAR (LexGLUE).

LEDGAR (Tuggener et al., 2020) labels provisions from contracts filed with the
SEC. LexGLUE (Chalkidis et al., 2022) redistributes it under CC BY 4.0. We keep
8 of its 100 provision types that matter for employment and compliance work.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET_REPO = "coastalcph/lex_glue"
DATASET_REVISION = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"

# LEDGAR label id -> name (subset of the 100 classes in the LexGLUE card).
LEDGAR_IDS = {
    11: "Base Salary",
    12: "Benefits",
    19: "Compliance With Laws",
    20: "Confidentiality",
    64: "Non-Disparagement",
    86: "Tax Withholdings",
    88: "Terminations",
    93: "Vacations",
}
LABELS: tuple[str, ...] = tuple(sorted(LEDGAR_IDS.values()))

DEFAULT_SIZES = {"train": 100, "validation": 25, "test": 40}  # per class


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def select_balanced(rows: list[dict], per_class: int, seed: int, exclude: set[str] = frozenset()) -> list[dict]:
    """Seeded sample of ``per_class`` rows per label, skipping duplicate or excluded texts."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for r in rows:
        key = normalise(r["text"])
        if key in exclude or key in seen:
            continue
        seen.add(key)
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    out = []
    for label in sorted(by_label):
        pool = by_label[label]
        if len(pool) < per_class:
            raise ValueError(f"only {len(pool)} usable rows for {label!r}, need {per_class}")
        out.extend(rng.sample(pool, per_class))
    rng.shuffle(out)
    return out


def load_ledgar_split(split: str, revision: str = DATASET_REVISION) -> list[dict]:
    """Download one LEDGAR parquet split at a pinned dataset revision; keep our 8 labels."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_REPO, f"ledgar/{split}-00000-of-00001.parquet", repo_type="dataset", revision=revision)
    table = pq.read_table(path).to_pydict()
    return [
        {"src_index": i, "text": t, "label": LEDGAR_IDS[l]}
        for i, (t, l) in enumerate(zip(table["text"], table["label"]))
        if l in LEDGAR_IDS
    ]


def build_splits(raw: dict[str, list[dict]], sizes: dict[str, int] = DEFAULT_SIZES, seed: int = 0) -> tuple[dict[str, list[dict]], dict]:
    """Test first, then validation, then train; each later split excludes texts already used,
    so no exact (whitespace/case-normalised) duplicate crosses a split boundary."""
    used: set[str] = set()
    out: dict[str, list[dict]] = {}
    dropped = {}
    for split in ("test", "validation", "train"):
        dropped[split] = sum(1 for r in raw[split] if normalise(r["text"]) in used)
        rows = select_balanced(raw[split], sizes[split], seed, exclude=used)
        for j, r in enumerate(rows):
            r["id"] = f"{split[:2]}{j:04d}"
        out[split] = [{"id": r["id"], "text": r["text"], "label": r["label"], "src_index": r["src_index"]} for r in rows]
        used |= {normalise(r["text"]) for r in rows}
    manifest = {
        "source": f"{DATASET_REPO}@{DATASET_REVISION} (config ledgar)",
        "license": "CC BY 4.0",
        "labels": list(LABELS),
        "seed": seed,
        "per_class": sizes,
        "excluded_cross_split_duplicates": dropped,
        "files": {},
    }
    return out, manifest


def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    Path(path).write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def prepare(out_dir: Path, sizes: dict[str, int] = DEFAULT_SIZES, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = {s: load_ledgar_split(s) for s in ("train", "validation", "test")}
    splits, manifest = build_splits(raw, sizes, seed)
    for split, rows in splits.items():
        sha = write_jsonl(out_dir / f"{split}.jsonl", rows)
        manifest["files"][f"{split}.jsonl"] = {"n": len(rows), "sha256": sha, "labels": dict(Counter(r["label"] for r in rows))}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
