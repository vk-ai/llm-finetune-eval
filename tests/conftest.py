"""Tiny random-init Qwen2 model + an in-memory word-level tokenizer: no downloads."""
from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from llm_finetune_eval.data import LABELS
from llm_finetune_eval.prompts import SYSTEM, user_message

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)

TOY_TEXT = {
    "Base Salary": "the executive shall receive an annual base salary payable in installments",
    "Benefits": "the employee may participate in health and welfare benefit plans",
    "Compliance With Laws": "the company shall comply with all applicable laws and regulations",
    "Confidentiality": "the recipient shall keep all confidential information secret",
    "Non-Disparagement": "neither party shall make disparaging statements about the other",
    "Tax Withholdings": "all payments are subject to required tax withholding",
    "Terminations": "this agreement may be terminated upon written notice for cause",
    "Vacations": "the executive is entitled to four weeks of paid vacation per year",
}


def _vocab() -> dict[str, int]:
    words = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "...", "system", "user", "assistant", "Answer:"]
    corpus = [SYSTEM, user_message(" ".join(TOY_TEXT.values()), LABELS), *LABELS, *TOY_TEXT.values()]
    for text in corpus:
        for w in text.split():
            if w not in words:
                words.append(w)
    return {w: i for i, w in enumerate(words)}


def make_tokenizer(chat: bool = True) -> PreTrainedTokenizerFast:
    t = Tokenizer(models.WordLevel(_vocab(), unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>")
    if chat:
        tok.chat_template = CHAT_TEMPLATE
    return tok


def make_model(vocab_size: int, seed: int = 0) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True,
                      initializer_range=0.5)  # big enough init that a frozen tiny LM head can express peaked logits
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture
def tok():
    return make_tokenizer()


@pytest.fixture
def model(tok):
    return make_model(len(tok))


@pytest.fixture
def toy_rows():
    rows = []
    for i in range(3):
        for label, text in TOY_TEXT.items():
            rows.append({"id": f"r{len(rows):03d}", "text": text, "label": label})
    return rows
