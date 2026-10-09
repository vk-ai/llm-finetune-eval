import torch

from llm_finetune_eval.data import LABELS
from llm_finetune_eval.scoring import LabelScorer
from llm_finetune_eval.train import TrainConfig, add_lora, answer_loss, collate, encode_rows, train

from conftest import make_model


def test_collate_right_pads_and_masks():
    ids, lab, att = collate([([5, 6, 7], [-100, 6, 7]), ([8], [8])], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert lab.tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert att.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_answer_loss_matches_hf_loss_on_answer_tokens(model, tok, toy_rows):
    data = encode_rows(tok, toy_rows[:4], LABELS, 64)
    ids, lab, att = collate(data, tok.pad_token_id)
    with torch.no_grad():
        ours = answer_loss(model, ids, lab, att)
        hf = model(input_ids=ids, attention_mask=att, labels=lab).loss
    assert torch.allclose(ours, hf, atol=1e-5)
    assert all(sum(1 for y in l if y != -100) >= 2 for _, l in data)  # label tokens + end token


def test_lora_training_learns_toy_task_and_freezes_base(tok, toy_rows):
    model = make_model(len(tok), seed=1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    base_acc = sum(p["pred"] == p["label"] for p in LabelScorer(model, tok, LABELS).predict(toy_rows)) / len(toy_rows)
    cfg = TrainConfig(r=8, alpha=32, lr=2e-2, batch_size=8, grad_accum=1, max_steps=40, gradient_checkpointing=False, max_text_tokens=64)
    peft_model = add_lora(model, cfg)
    rep = train(peft_model, tok, toy_rows, LABELS, cfg, log=None)
    assert rep["optimizer_steps"] == 40 and len(rep["history"]) == 40
    assert rep["mean_loss_last_10"] < rep["first_loss"] / 4
    acc = sum(p["pred"] == p["label"] for p in LabelScorer(peft_model, tok, LABELS).predict(toy_rows)) / len(toy_rows)
    assert acc == 1.0 and acc > base_acc
    trained = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trained and all("lora_" in n for n in trained)
    checked = 0
    for name, p in peft_model.get_base_model().named_parameters():  # frozen base weights are untouched
        if "lora_" not in name:
            assert torch.equal(p, before[name.replace(".base_layer", "")]), name
            checked += 1
    assert checked == len(before) - 1  # all but the tied lm_head, which shares embed_tokens
    assert rep["hardware"]["torch"] == torch.__version__


def test_adapter_roundtrip(tmp_path, tok, toy_rows):
    from peft import PeftModel

    model = make_model(len(tok), seed=2)
    cfg = TrainConfig(lr=1e-2, batch_size=8, grad_accum=1, max_steps=5, gradient_checkpointing=False, max_text_tokens=64)
    peft_model = add_lora(model, cfg)
    train(peft_model, tok, toy_rows, LABELS, cfg, log=None)
    peft_model.save_pretrained(tmp_path / "adapter")
    reloaded = PeftModel.from_pretrained(make_model(len(tok), seed=2), tmp_path / "adapter").eval()
    a = LabelScorer(peft_model, tok, LABELS).scores(toy_rows[0]["text"])
    b = LabelScorer(reloaded, tok, LABELS).scores(toy_rows[0]["text"])
    assert all(abs(a[k] - b[k]) < 1e-4 for k in LABELS)


def test_gradient_checkpointing_path_runs(tok, toy_rows):
    cfg = TrainConfig(lr=1e-3, batch_size=4, grad_accum=2, max_steps=2, gradient_checkpointing=True, max_text_tokens=64)
    m = add_lora(make_model(len(tok)), cfg)
    rep = train(m, tok, toy_rows, LABELS, cfg, log=None)
    assert rep["examples_seen"] == 16
