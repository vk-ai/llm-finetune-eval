# llm-finetune-eval

A real LoRA fine-tune of **Qwen2.5-0.5B-Instruct** on a CPU, for classifying
employment and compliance provisions in contracts (8 LEDGAR classes). The base
model and the fine-tune are scored on the same 320 held-out provisions, then
compared with the statistics and regression gate from
[llm-eval-lab](https://github.com/vk-ai/llm-eval-lab). The result is a
pass/fail decision with bootstrap CIs and paired significance tests.

> **Learning project.** I built this to practise parameter-efficient
> fine-tuning and checking whether a fine-tune is actually better. It is not
> used in production anywhere, and nothing here comes from an employer. All
> the numbers below are from real runs on a CPU box: the original seed-0 run,
> plus (since round 6) two more LoRA seeds and three 5-shot base-model prompts. See
> [What is real vs stubbed](#what-is-real-vs-stubbed).

## Fair baselines: 5-shot base model and 3 LoRA seeds (real runs, n = 320)

The first version compared the fine-tune only with a **zero-shot** base model,
from **one** training run. Both make a fine-tune look better than it is, so
these were added (`lfe predict --shots 5`, `lfe seeds`, `reports/seeds.json`):

| System | Macro-F1 | Accuracy |
|---|---|---|
| Base, zero-shot | 0.696 | 0.722 |
| Base, 5-shot, demo set 0 | 0.806 | 0.803 |
| Base, 5-shot, demo set 1 | 0.777 | 0.772 |
| Base, 5-shot, demo set 2 | 0.769 | 0.766 |
| **Base, 5-shot, mean ± sd over 3 demo sets** | **0.784 ± 0.019** | 0.781 |
| LoRA seed 0 / 1 / 2 | 0.956 / 0.956 / 0.959 | 0.956 / 0.956 / 0.959 |
| **LoRA, mean ± sd over 3 seeds** | **0.957 ± 0.002** | 0.957 ± 0.002 |

- **Few-shot helps the base model a lot**, raising macro-F1 by 0.07–0.11. Which
  demos you pick matters (0.77–0.81): each 5-shot set covers only 5 of the 8
  classes (one random training example each, so test items never appear in the prompt).
- **The fine-tune's gain shrinks but holds.** Against the best 5-shot prompt it is
  +0.150 macro-F1 (paired-bootstrap 95% CI [+0.116, +0.192]); against the
  5-shot mean it is about +0.17. Against zero-shot it was +0.26.
- **Seed spread is small** (sd 0.002, range 0.956–0.959). Seeds 0 and 1 land on
  the same macro-F1 with different per-item predictions.
- **Gate:** all 12 seed × baseline pairs **PASS** the llm-eval-lab gate
  (drop limits, the 0.80 floor, McNemar, and must-improve with Δ ≥ 0.05, p < 0.05
  and CI low > 0). For example, best 5-shot vs seed 0: McNemar b = 0, c = 49.
  The permutation p = 0.001 in every pair is the floor for 1,000 resamples.
- **Extra cost:** each extra seed trained in 1,599 s and 2,131 s on a CPU box shared
  with other jobs (seed 0 took 1,097 s on a quieter box). Seed 2 was restarted
  with 4 threads after it stalled under contention; seed 1 was trained with 8
  threads. 5-shot scoring took 450–525 s per demo set, with the shared demo
  prefix KV-cached.

## Results, original single run (test set n = 320)

| | Base (zero-shot) | LoRA fine-tune |
|---|---|---|
| Accuracy | 0.722 [0.672, 0.772] | **0.956** [0.931, 0.975] |
| Macro-F1 | 0.696 [0.655, 0.734] | **0.956** [0.932, 0.975] |
| Latency per item, p50 / p95 (CPU, 8 threads) | 1028 / 1600 ms | 1095 / 1698 ms |

The brackets are 95% percentile bootstrap CIs (1,000 resamples) from `llm_eval_lab.metrics.bootstrap_ci`.

Paired comparison on the same 320 items:
- **McNemar (exact):** base right / fine-tune wrong = 5, the reverse = 80, p = 1.8e-18.
- **Permutation test, accuracy:** Δ +0.234, 95% paired-bootstrap CI [+0.184, +0.288], p = 0.001.
- **Permutation test, macro-F1:** Δ +0.260, CI [+0.220, +0.303], p = 0.001. This p is the floor for 1,000 permutations.

**Gate: PASSED** (`reports/gate.json`). The negative control, with the base model as
the candidate against the fine-tune, **FAILS** every rule (`reports/gate_negative_control.json`).

| Check | Baseline | Candidate | Rule | Result |
|---|---|---|---|---|
| macro_f1 drop | 0.6957 | 0.9559 | >= baseline − 0.0 | ok |
| accuracy drop | 0.7219 | 0.9563 | >= baseline − 0.0 | ok |
| macro_f1 floor | | 0.9559 | >= 0.80 | ok |
| McNemar vs baseline | b=5 | c=80 | not significantly worse (α 0.05) | ok |
| macro_f1 improvement | 0.6957 | 0.9559 | Δ >= 0.05, permutation p < 0.05, CI low > 0 | ok |

Per-class F1, base → fine-tune:

| Class | Base | Fine-tune |
|---|---|---|
| Base Salary | 0.988 | 0.988 |
| Benefits | 0.610 | 0.909 |
| Compliance With Laws | 0.605 | 0.974 |
| Confidentiality | 0.660 | 0.933 |
| Non-Disparagement | 0.043 | 0.952 |
| Tax Withholdings | 0.909 | 0.976 |
| Terminations | 0.776 | 0.940 |
| Vacations | 0.974 | 0.975 |

Most of the gain comes from classes the base model confuses. It labels 28 of 40
non-disparagement clauses as "Confidentiality", and it over-predicts
"Compliance With Laws" as a catch-all. On validation (n = 200), the base model
scored 0.690 and the fine-tune 0.960 (`runs/validation/`).

### Training run

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae5576`, fp32 |
| LoRA | r 8, α 16, dropout 0.05, on `q_proj k_proj v_proj o_proj`: 1,081,344 trainable params (0.22%) |
| Data | 800 train provisions (100 per class), 1 epoch |
| Optimiser | AdamW lr 2e-4, 6% warmup then cosine, batch 4 × grad-accum 2 = 100 steps, grad-clip 1.0, seed 0 |
| Loss | answer tokens only; mean of steps 1–10 = 0.177, mean of steps 91–100 = 0.032 (first 0.220, last 0.0026) |
| Wall time | **1,097 s (18.3 min)** training; scoring 320 test items took 355 s (base) and 378 s (fine-tune) |
| Hardware | 8 vCPU "Intel(R) Xeon(R) Processor" (cloud VM), 15.6 GB RAM, no GPU; torch 2.14.1+cpu, Python 3.12 |
| Adapter | 4.4 MB safetensors, attached to the [v0.1.0 release](https://github.com/vk-ai/llm-finetune-eval/releases/tag/v0.1.0) (not committed) |

The per-token loss looks small from the start because each target is the label
name plus `<|im_end|>`, and the label names already appear in the prompt. The
full curve is in `reports/train.json`.

## How it works

```
lfe prepare  ─► data/{train,validation,test}.jsonl + MANIFEST.json  (LEDGAR @ pinned HF revision, seeded, balanced)
lfe train    ─► artifacts/adapter/ + reports/train.json            (plain PyTorch loop + peft LoRA)
lfe predict  ─► runs/base.jsonl, runs/finetuned.jsonl               (same scorer for both)
lfe report   ─► reports/eval.json                                   (llm-eval-lab metrics, CIs, McNemar, permutation)
lfe gate     ─► PASS / FAIL, exit code, $GITHUB_STEP_SUMMARY        (llm-eval-lab run_gate + "must improve" rule)
```

- **Data** (`data.py`). Comes from [LEDGAR](https://aclanthology.org/2020.lrec-1.155/)
  via [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) (CC BY 4.0),
  pinned to revision `c23fdff`. I kept 8 of the 100 provision types that matter
  for HR, payroll and compliance work: Base Salary, Benefits, Compliance With
  Laws, Confidentiality, Non-Disparagement, Tax Withholdings, Terminations and
  Vacations. Each split is sampled separately and balanced: 40 per class for
  test (from LEDGAR test), 25 for validation and 100 for train. Exact
  duplicates are removed across splits (5 train candidates were dropped). The
  small subset is committed, and its sha256 hashes are in `data/MANIFEST.json`.
- **Prompt** (`prompts.py`). Uses the Qwen chat template with a system line and
  the list of 8 categories. The provision is truncated to 192 tokens. Training
  and scoring share the same code.
- **Scoring** (`scoring.py`). This is rank classification. Each label string
  (plus the end-of-turn token) is scored by its total log-likelihood, and the
  highest wins. The prompt goes through the model once, and each label reuses
  the KV cache. Every prediction is therefore a valid label, and the base model
  gets no "unparseable output" penalty.
- **Training** (`train.py`). A short PyTorch loop rather than `Trainer`, with
  gradient checkpointing. The loss is computed only on answer positions, and
  only those hidden states go through the LM head. Full-sequence logits over
  Qwen's 151,936-token vocabulary ran an earlier attempt out of memory on this
  box.
- **Gate** (`gating.py`, `gate.yaml`). Metrics, bootstrap CIs, McNemar,
  permutation tests and `run_gate` are imported from llm-eval-lab, installed
  from git at commit `df9768b` (see `pyproject.toml`). This repo only adapts
  file shapes and adds one rule: a fine-tune must be *significantly better*,
  not just "not worse".

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"            # also installs llm-eval-lab from GitHub
pytest -q                          # 23 passed in ~5-25 s, offline, tiny random-init model

# Re-check the committed results without any model (what CI does)
lfe report && lfe gate             # exit 0
lfe gate --baseline finetuned --candidate base   # negative control, exit 1

# Full real run (~1 GB model download; ~18 min train + ~12 min scoring on 8 vCPUs)
lfe prepare                        # rebuilds data/ (hashes should match MANIFEST.json)
lfe train --threads 8              # -> artifacts/adapter, reports/train.json
lfe predict --name base --threads 8
lfe predict --name finetuned --adapter artifacts/adapter --threads 8
lfe report && lfe gate
python scripts/near_duplicates.py  # contamination check, see below
```

To skip training, download `adapter-v0.1.0.tar.gz` from the release, untar it
into `artifacts/adapter`, and run the two `predict` commands.

## Tests and CI

`pytest -q`: **27 passed** (4 added for few-shot and seeds). The tests never download a model. They build a
2-layer random-init `Qwen2ForCausalLM` and a word-level tokenizer in memory.

- `test_prompts_scoring.py`: KV-cached label scores equal a full-sequence
  log-prob computation to 1e-4. Predictions are always valid labels. Also
  covers the chat-template and plain prompt paths, and truncation.
- `test_train.py`: our answer-only loss equals HF's own masked LM loss. LoRA
  learns a toy 8-class task (>= 87.5% accuracy, from near chance) while every base weight stays bit-identical.
  Also covers an adapter save/load round trip, and gradient checkpointing with
  grad accumulation.
- `test_gating.py`: the gate passes a clear improvement and fails swapped roles
  and "no significant gain". It blocks a changed test-set hash and errors on
  missing predictions. The CLI exit codes and step summary are checked. The
  committed `reports/eval.json` is recomputed from the committed per-example
  runs, and the gate passes on it.
- `test_fewshot_seeds.py`: k-shot demos are stratified, seeded and taken from train only. Every few-shot prompt starts with the shared prefix, and cached few-shot scores equal full-sequence log-probs to 1e-4 (including cache reuse and crop). `seeds_summary` gives the right mean, sd and range, and fails a seed that's no better than the baseline.
- `test_data.py`: balanced, seeded sampling. No exact duplicate crosses splits.
  The committed files match their manifest hashes.

GitHub Actions (`.github/workflows/ci.yml`) runs on Python 3.11 and 3.12 with
CPU-only torch. Each job runs the unit tests with `HF_HUB_OFFLINE=1`,
recomputes the report from the committed runs and runs the gate plus the
negative control. An optional `smoke-finetune` job trains 3 LoRA steps on
`trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` (4.9 MB, random weights),
scores 32 items with both models and runs report and gate. That job checks
plumbing only; the random model is expected to fail the improvement rule.

## What is real vs stubbed

**Real**
- The fine-tune: real Qwen2.5-0.5B-Instruct weights, real LoRA via `peft`, real backprop on CPU, 3 seeds, with the numbers above copied from `reports/*.json`.
- The data: real contract provisions from public SEC filings (LEDGAR), with real labels.
- The evaluation: every per-example prediction for both models is committed in `runs/`, and the statistics and gate are llm-eval-lab's code.

**Simulated, simplified or worth a caveat**
- **Task size.** 8 classes, 800 training and 320 test examples. LEDGAR has 100 classes, and this is a subset I chose by hand for the HR/compliance angle.
- **No hyperparameter search.** One configuration (r 8, lr 2e-4, 1 epoch), picked before training. It was run with 3 seeds (0, 1, 2). Three seeds is a small sample for estimating variance, and the bootstrap CIs per run still cover only test-set sampling.
- **Tuning after seeing results.** No hyperparameter, prompt or threshold was changed after looking at any fine-tuned number. `gate.yaml` was written before the fine-tune finished. I looked at the base model's *validation* errors before training, but nothing was changed because of them. The test set was scored once per model.
- **The base-model baselines are zero-shot and 5-shot.** All use the same prompt and rank-classification scoring. 5-shot uses 3 random stratified demo sets covering 5 of 8 classes each, and demos are truncated to 96 tokens. I did not try 8-shot (one per class), retrieval-chosen demos, prompt tuning or a bigger base model; any of those could narrow the gap further.
- **Contamination.** Only exact duplicates are removed. `scripts/near_duplicates.py` finds 17 of 320 test items with ≥ 0.8 word-Jaccard similarity to some training item. The fine-tune scores 1.000 on those and 0.954 on the other 303 (base: 0.706 / 0.723), so they barely move the headline (`reports/near_duplicates.json`).
- **Not instruction-following quality.** The model is scored by ranking the 8 label strings, not by free generation. It was not checked for regressions on general chat ability, and the gate only covers this task.
- **Latency** is per-item scoring on a shared CPU VM with 8 forward passes per item (1 prompt + 8 short label continuations); it is not a serving benchmark. No cost comparison or llm-platform-k8s routing is included.
- **No QLoRA, TRL or GPU.** This is plain fp32 LoRA on CPU, and the `Trainer` and TRL paths are not used.
- **CI does not fine-tune Qwen.** CI re-checks the committed outputs and runs a 3-step smoke fine-tune on a random tiny model. The real run happened once, on the box described above.

## Credits

- [llm-eval-lab](https://github.com/vk-ai/llm-eval-lab) (my own learning repo): metrics, bootstrap CIs, McNemar, permutation test, `run_gate`.
- LEDGAR: Tuggener et al., LREC 2020. LexGLUE: Chalkidis et al., ACL 2022. Data is CC BY 4.0, and the subset in `data/` is redistributed under it.
- Qwen2.5-0.5B-Instruct: Qwen team, Apache-2.0.

MIT licensed (code).
