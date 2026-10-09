"""Classify by ranking the label strings by their log-likelihood under the model.

Every prediction is one of the labels (no output parsing, no "invalid" answers),
and the base and fine-tuned models are scored with exactly the same procedure.
The prompt is encoded once; each label continuation reuses the KV cache.
"""
from __future__ import annotations

import time

import torch

from .prompts import answer_ids, prompt_ids


class LabelScorer:
    def __init__(self, model, tokenizer, labels: tuple[str, ...], max_text_tokens: int = 192):
        self.model = model
        self.tok = tokenizer
        self.labels = labels
        self.max_text_tokens = max_text_tokens
        self.answers = {l: answer_ids(tokenizer, l) for l in labels}

    @torch.no_grad()
    def scores(self, text: str) -> dict[str, float]:
        self.model.eval()
        p = torch.tensor([prompt_ids(self.tok, text, self.labels, self.max_text_tokens)])
        out = self.model(input_ids=p, use_cache=True)
        cache = out.past_key_values
        first = torch.log_softmax(out.logits[0, -1].float(), -1)
        res = {}
        for label, a in self.answers.items():
            lp = float(first[a[0]])
            if len(a) > 1:
                o = self.model(input_ids=torch.tensor([a[:-1]]), past_key_values=cache, use_cache=True)
                logp = torch.log_softmax(o.logits[0].float(), -1)
                lp += float(logp.gather(1, torch.tensor(a[1:])[:, None]).sum())
                cache = o.past_key_values
                cache.crop(-(len(a) - 1))  # drop this label's tokens, keep the prompt
            res[label] = lp
        return res

    def predict(self, rows: list[dict], progress=None) -> list[dict]:
        out = []
        for i, r in enumerate(rows):
            t0 = time.perf_counter()
            s = self.scores(r["text"])
            ms = (time.perf_counter() - t0) * 1000
            ranked = sorted(s, key=s.get, reverse=True)
            out.append({
                "id": r["id"],
                "label": r["label"],
                "pred": ranked[0],
                "margin": round(s[ranked[0]] - s[ranked[1]], 4),
                "latency_ms": round(ms, 1),
            })
            if progress:
                progress(i + 1, len(rows))
        return out
