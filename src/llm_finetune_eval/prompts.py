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


def prompt_ids(tokenizer, text: str, labels: tuple[str, ...], max_text_tokens: int = 192) -> list[int]:
    text = truncate(tokenizer, text, max_text_tokens)
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(text, labels)}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):  # BatchEncoding in newer transformers
            ids = ids["input_ids"]
        return list(ids)
    plain = f"{SYSTEM}\n\n{user_message(text, labels)}\nAnswer: "
    return tokenizer(plain, add_special_tokens=False)["input_ids"]


def answer_ids(tokenizer, label: str) -> list[int]:
    ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    end = end_token_id(tokenizer)
    return list(ids) + ([end] if end is not None else [])
