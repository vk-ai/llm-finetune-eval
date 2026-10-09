import torch

from llm_finetune_eval.data import LABELS
from llm_finetune_eval.prompts import answer_ids, end_token_id, prompt_ids, truncate
from llm_finetune_eval.scoring import LabelScorer

from conftest import TOY_TEXT, make_tokenizer


def test_answer_ends_with_turn_end(tok):
    a = answer_ids(tok, "Tax Withholdings")
    assert a[-1] == end_token_id(tok) == tok.convert_tokens_to_ids("<|im_end|>")
    assert tok.decode(a[:-1]) == "Tax Withholdings"


def test_prompt_uses_chat_template_or_plain_fallback(tok):
    chat = prompt_ids(tok, TOY_TEXT["Vacations"], LABELS)
    assert tok.convert_tokens_to_ids("<|im_start|>") in chat
    plain_tok = make_tokenizer(chat=False)
    plain = prompt_ids(plain_tok, TOY_TEXT["Vacations"], LABELS)
    assert plain_tok.convert_tokens_to_ids("<|im_start|>") not in plain
    assert plain_tok.decode(plain).endswith("Answer:")


def test_truncate(tok):
    text = " ".join(["vacation"] * 50)
    short = truncate(tok, text, 10)
    assert short.endswith(" ...") and len(tok(short, add_special_tokens=False)["input_ids"]) == 11
    assert truncate(tok, "paid vacation", 10) == "paid vacation"


def _full_sequence_logprob(model, p, a):
    ids = torch.tensor([p + a])
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
    return float(sum(logp[len(p) - 1 + i, t] for i, t in enumerate(a)))


def test_cached_scores_equal_full_sequence_logprob(model, tok):
    scorer = LabelScorer(model, tok, LABELS)
    text = TOY_TEXT["Benefits"]
    got = scorer.scores(text)
    p = prompt_ids(tok, text, LABELS)
    for label in LABELS:
        want = _full_sequence_logprob(model, p, answer_ids(tok, label))
        assert abs(got[label] - want) < 1e-4, label


def test_predict_always_returns_a_label(model, tok, toy_rows):
    out = LabelScorer(model, tok, LABELS).predict(toy_rows[:8])
    assert [o["id"] for o in out] == [r["id"] for r in toy_rows[:8]]
    assert all(o["pred"] in LABELS and o["margin"] >= 0 and o["latency_ms"] >= 0 for o in out)
