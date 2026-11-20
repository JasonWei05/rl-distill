"""Model directories against HF transformers, and the decoder against the teacher-forced forward."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from data import Dataset
from model import ARCH, Decoder, Qwen2, evaluate, half, hf_config, init_weights, load_model, response_mask, save_model
from tasks import EOS, PAD


@pytest.mark.parametrize("size", ["2XS", "S"])
def test_init_matches_hf_layout(size):
    c = hf_config(size, 37, 256)
    ref = Qwen2ForCausalLM(Qwen2Config(**c)).state_dict()
    w = init_weights(c, 0)
    assert {k: v.shape for k, v in w.items()} == {k: v.shape for k, v in ref.items() if k != "lm_head.weight"}
    assert w["model.embed_tokens.weight"][PAD].abs().sum() == 0
    std = torch.cat([v.flatten() for k, v in w.items() if v.ndim == 2]).std()
    assert abs(std - 0.02) < 1e-3
    assert torch.equal(init_weights(c, 0)["model.norm.weight"], w["model.norm.weight"])
    assert not torch.equal(init_weights(c, 1)["model.embed_tokens.weight"], w["model.embed_tokens.weight"])


def test_model_dir_loads_in_hf(sudoku_data, tmp_path):
    """save_model -> HF AutoModelForCausalLM / AutoTokenizer: same logits (fp32) and token ids."""
    d = Dataset(sudoku_data)
    c = hf_config("2XS", len(d.task.WORDS), d.width)
    w = {k: v.cuda() for k, v in init_weights(c, 3).items()}
    w = {k: v + 0.02 * torch.randn_like(v) for k, v in w.items()}  # break the unit norms / zero biases
    save_model(str(tmp_path / "m"), w, c, {"config": d.cfg})
    c2, synth, w2 = load_model(str(tmp_path / "m"))
    assert c2 == c and synth == {"config": d.cfg} and all(torch.equal(w[k], w2[k]) for k in w)
    hf = AutoModelForCausalLM.from_pretrained(str(tmp_path / "m"), torch_dtype=torch.float32).cuda().eval()
    ids = torch.from_numpy(d.split("val")[:8].astype("int64")).cuda()
    m = Qwen2(c)
    mine = m.logits(w, m.forward(w, ids, torch.arange(ids.shape[1], device="cuda"), m.causal()))
    with torch.no_grad():
        ref = hf(ids).logits
    assert (mine - ref).abs().max() < 1e-4
    tok = AutoTokenizer.from_pretrained(str(tmp_path / "m"))
    for row in ids[:, :d.prompt_len].tolist():
        assert tok(" ".join(d.task.WORDS[t] for t in row), add_special_tokens=False)["input_ids"] == row


@pytest.mark.parametrize("mem_gb", [4.0, 0.001])
def test_decoder_logprobs_match_forward(sudoku_data, mem_gb):
    """Sampled tokens' log-probs and the EOS / pad layout against train_logprobs on the same tokens (fp16), whether the
    prompts are decoded in one chunk or one per chunk."""
    d = Dataset(sudoku_data)
    c = hf_config("2XS", len(d.task.WORDS), d.width)
    w = half({k: v.cuda() for k, v in init_weights(c, 0).items()})
    m = Qwen2(c)
    prompts = d.prompts("dev", 12).cuda()
    torch.manual_seed(0)
    check_samples(c, w, prompts, *Decoder(m, 70, mem_gb).generate(w, prompts, 4))


def test_decoder_grows_and_parks_caches(sudoku_data, monkeypatch):
    """Response caches grown a segment at a time under a budget that holds every prompt only at the first width: the
    layout splits, its last groups wait in host memory and resume, and the samples still match train_logprobs."""
    import model
    monkeypatch.setattr(model, "GROW", model.SEG)
    d = Dataset(sudoku_data)
    c = hf_config("2XS", len(d.task.WORDS), d.width)
    w = half({k: v.cuda() for k, v in init_weights(c, 0).items()})
    prompts = d.prompts("dev", 12).cuda()
    dec = Decoder(Qwen2(c), 70, 1.0)
    dec.mem = dec.kv_bytes * 12 * (prompts.shape[1] + 4 * model.SEG)
    splits, split = [], dec._split
    monkeypatch.setattr(dec, "_split", lambda st, g: splits.append(g) or split(st, g))
    torch.manual_seed(0)
    check_samples(c, w, prompts, *dec.generate(w, prompts, 4))
    assert splits


def check_samples(c, w, prompts, tok, logp, ent):
    mask = response_mask(tok)
    assert tok.shape == (4 * len(prompts), 70) and (tok[~mask] == PAD).all()
    assert ((tok == EOS).sum(1) <= 1).all() and ((tok == EOS).any(1) | mask.all(1)).all()
    with torch.no_grad():
        ref = Qwen2(c).train_logprobs(w, prompts, tok, 4)
    assert ((logp - ref).abs() * mask).max() < 2e-2
    assert ((ent > 0) | ~mask).all()


@pytest.mark.parametrize("rows", [12, 40, 10 ** 6])
def test_evaluate_maps_samples_to_prompts(sudoku_data, rows):
    """evaluate judges each chunk's samples against their own prompts: a decoder that returns the demonstration for even
    prompts and a lone DONE for odd ones scores exactly 0.5, however the prompts are chunked."""
    d = Dataset(sudoku_data)
    dev = torch.from_numpy(d.split("dev")[:30].astype("int64")).cuda()
    prompts, demos = dev[:, :d.prompt_len], dev[:, d.prompt_len:]
    wrong = torch.full_like(demos[0], PAD)
    wrong[:2] = torch.tensor([d.task.ids["DONE"], EOS])
    which = {tuple(p.tolist()): i for i, p in enumerate(prompts)}

    class DemoDecoder:
        def generate(self, w, chunk, n):
            ids = [which[tuple(p.tolist())] for p in chunk]
            return (torch.stack([demos[i] if i % 2 == 0 else wrong for i in ids]).repeat_interleave(n, 0),)

    out = evaluate(d.task, DemoDecoder(), None, prompts, d.task.parse(prompts), 4, rows)
    assert out["score"] == 0.5 and out["valid_format"] == 1.0
