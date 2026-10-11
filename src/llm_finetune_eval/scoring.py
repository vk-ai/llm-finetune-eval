"""Classify by ranking the label strings by their log-likelihood under the model.

Every prediction is one of the labels (no output parsing, no "invalid" answers),
and the base and fine-tuned models are scored with exactly the same procedure.
The prompt is encoded once; each label continuation reuses the KV cache.
"""
from __future__ import annotations

import time

import torch

from .prompts import answer_ids, prompt_ids, shared_prefix_ids


class LabelScorer:
    def __init__(self, model, tokenizer, labels: tuple[str, ...], max_text_tokens: int = 192,
                 shots: list[dict] | None = None, shot_tokens: int = 96):
        self.model = model
        self.tok = tokenizer
        self.labels = labels
        self.max_text_tokens = max_text_tokens
        self.shots = shots or None
        self.shot_tokens = shot_tokens
        self.answers = {l: answer_ids(tokenizer, l) for l in labels}
        self.prefix = shared_prefix_ids(tokenizer, labels, self.shots, shot_tokens)
        self._prefix_cache = None

    @torch.no_grad()
    def scores(self, text: str) -> dict[str, float]:
        self.model.eval()
        ids = prompt_ids(self.tok, text, self.labels, self.max_text_tokens, self.shots, self.shot_tokens)
        n = len(self.prefix)
        if n and ids[:n] == self.prefix and len(ids) > n:
            # few-shot demos are identical for every item: encode them once, then crop back to them
            if self._prefix_cache is None:
                self._prefix_cache = self.model(input_ids=torch.tensor([self.prefix]), use_cache=True).past_key_values
            cache = self._prefix_cache
            cache.crop(n)
            out = self.model(input_ids=torch.tensor([ids[n:]]), past_key_values=cache, use_cache=True)
        else:
            out = self.model(input_ids=torch.tensor([ids]), use_cache=True)
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
