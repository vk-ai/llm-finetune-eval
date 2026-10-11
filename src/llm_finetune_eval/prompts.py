"""Prompt format shared by training and scoring, so the two can't drift apart."""
from __future__ import annotations

SYSTEM = "You label provisions from employment and commercial contracts."


def user_message(text: str, labels: tuple[str, ...]) -> str:
    return (
        "Classify this contract provision into exactly one category: "
        + ", ".join(labels)
        + ".\n\nProvision: "
        + text
        + "\n\nAnswer with the category name only."
    )


def truncate(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens]) + " ..."


def end_token_id(tokenizer) -> int | None:
    """Token that closes the assistant turn (<|im_end|> for Qwen chat), else EOS."""
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tokenizer.get_vocab() else None
    return tid if tid is not None else tokenizer.eos_token_id


def select_shots(train_rows: list[dict], k: int, seed: int = 0) -> list[dict]:
    """Stratified k-shot demos: k distinct classes (round-robin if k > #classes), one random example each.

    Drawn only from the training split, so test items can never leak into the prompt.
    """
    import random

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in train_rows:
        by_label.setdefault(r["label"], []).append(r)
    labels = sorted(by_label)
    rng.shuffle(labels)
    pools = {l: rng.sample(by_label[l], len(by_label[l])) for l in labels}
    shots = [pools[labels[i % len(labels)]][i // len(labels)] for i in range(k)]
    rng.shuffle(shots)
    return shots


def _render(tokenizer, msgs: list[dict], add_generation_prompt: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):  # BatchEncoding in newer transformers
        ids = ids["input_ids"]
    return list(ids)


def _shot_turns(tokenizer, shots, labels, shot_tokens):
    msgs = []
    for s in shots or []:
        msgs.append({"role": "user", "content": user_message(truncate(tokenizer, s["text"], shot_tokens), labels)})
        msgs.append({"role": "assistant", "content": s["label"]})
    return msgs


def prompt_ids(tokenizer, text: str, labels: tuple[str, ...], max_text_tokens: int = 192,
               shots: list[dict] | None = None, shot_tokens: int = 96) -> list[int]:
    """Prompt token ids. ``shots`` (few-shot demos) become earlier user/assistant turns."""
    text = truncate(tokenizer, text, max_text_tokens)
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": SYSTEM}, *_shot_turns(tokenizer, shots, labels, shot_tokens),
                {"role": "user", "content": user_message(text, labels)}]
        return _render(tokenizer, msgs, True)
    demos = "".join(f"{user_message(truncate(tokenizer, s['text'], shot_tokens), labels)}\nAnswer: {s['label']}\n\n"
                    for s in shots or [])
    plain = f"{SYSTEM}\n\n{demos}{user_message(text, labels)}\nAnswer: "
    return tokenizer(plain, add_special_tokens=False)["input_ids"]


def shared_prefix_ids(tokenizer, labels: tuple[str, ...], shots: list[dict] | None, shot_tokens: int = 96) -> list[int]:
    """Token ids every prompt starts with (system line + demos), so its KV cache can be reused."""
    if not shots:
        return []
    if getattr(tokenizer, "chat_template", None):
        return _render(tokenizer, [{"role": "system", "content": SYSTEM}, *_shot_turns(tokenizer, shots, labels, shot_tokens)], False)
    demos = "".join(f"{user_message(truncate(tokenizer, s['text'], shot_tokens), labels)}\nAnswer: {s['label']}\n\n" for s in shots)
    return tokenizer(f"{SYSTEM}\n\n{demos}", add_special_tokens=False)["input_ids"]


def answer_ids(tokenizer, label: str) -> list[int]:
    ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    end = end_token_id(tokenizer)
    return list(ids) + ([end] if end is not None else [])
