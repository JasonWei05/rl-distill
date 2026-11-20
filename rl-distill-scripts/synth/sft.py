"""Supervised training on a dataset's demonstrations: from scratch (one model size, one init seed) or from an HF model.

- Model: Qwen2 of ARCH[size], initialized from --seed; or --init, an HF model directory (chess: a released Qwen3,
  extended as their SFT does: new words get the mean embedding; YaRN to sft.context positions). Every run sees the same
  batches: epoch e is a permutation of the train split drawn from (data_seed, e); a batch is cut to its longest row
  (rounded up to 64 tokens).
- Loss: next-token cross entropy on the response tokens (everything after the prompt up to and including <eos>; chess:
  not the environment's, + ENV), averaged over the batch's response tokens. fp32 master weights, bf16 compute (the
  forward is compiled), AdamW (0.9, 0.95), weight decay 0.01, constant lr, grad-norm clip 1.0. Chess follows their SFT
  (sft/scripts/train/run_sft.py on 4 GPUs) with optional settings: sft.group (the loss is the mean over the batch's
  consecutive groups of that many rows of each group's token mean: their per-micro-batch means under gradient
  accumulation), sft.dtype float32 (compute), sft.decay_norms false (no weight decay on the norms), sft.warmup and
  sft.min_lr (linear warmup over that many steps, then cosine to min_lr at sft.decay_steps); --chunk rows per forward.
- Every sft.save_every steps: the weights to ckpt/<step>.safetensors (fp32) and the training state to state.pt (a
  relaunch resumes from it). Every sft.screen_every steps: validation loss and a screen, the judge's outcomes over
  sft.screen_k samples of each prompt of the dev split (or sft.screen_split; chess: test, the benchmark its models are
  matched on) (fp16, temperature 1), appended to screen.jsonl. Every sft.val_every steps, if set: the validation loss
  alone, appended to val.jsonl.
- Stops at sft.max_steps, or once the median screen score of the last 4 screens reaches sft.stop_at; writes DONE.

    python sft.py --data /tmp/synth/data/maze33 --size 2XS --seed 0 --out /tmp/synth/sft/maze33/2XS_s0
"""

import argparse
import json
import math
import os
import queue
import statistics
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from safetensors.torch import load_file, save_file

from data import Dataset, same_build
from files import atomic, read_json, write_json
from model import Decoder, Qwen2, evaluate, half, hf_config, init_weights
from tasks import ENV, PAD

ALIGN = 64  # batch widths are rounded up to this many response tokens (fewer distinct shapes to compile)


def joined(rows, slot, prompt_len):
    """Rows of a task with variable-length prompts in its first `slot` columns (chess), each prompt followed by its
    response: (B, width) uint8 (width rounded up to ALIGN; pad after), and the prompt lengths (B,)."""
    plen = slot - (rows[:, :slot][:, ::-1] != 0).argmax(1)
    rlen = (rows[:, prompt_len:] != 0).sum(1)
    width = -(-int((plen + rlen).max()) // ALIGN) * ALIGN
    cols = np.arange(width)
    src = np.where(cols < plen[:, None], cols, cols - plen[:, None] + prompt_len).clip(max=rows.shape[1] - 1)
    x = np.take_along_axis(rows, src, 1) * (cols < (plen + rlen)[:, None])
    return x.astype(np.uint8), plen


class Batches:
    """Training batches (step 1, 2, ...) in the fixed data order, prepared by a background thread as pinned uint8 tensors
    (and, for variable-length prompts (slot), their lengths; rows then stay in the epoch's order, as their groups)."""

    def __init__(self, rows, prompt_len, batch, data_seed, first_step, depth=4, slot=None):
        self.rows, self.Lp, self.batch, self.seed, self.slot = rows, prompt_len, batch, data_seed, slot
        self.per_epoch, self.perm = len(rows) // batch, (None, None)
        self.q = queue.Queue(depth)
        threading.Thread(target=self._fill, args=(first_step,), daemon=True).start()

    def indices(self, step, sort=True):
        """The row indices of a step's batch, sorted or in the epoch's order (the epoch's permutation is memoized)."""
        epoch, i = divmod(step - 1, self.per_epoch)
        if self.perm[0] != epoch:
            self.perm = (epoch, np.random.default_rng([self.seed, epoch]).permutation(len(self.rows)))
        idx = self.perm[1][i * self.batch:(i + 1) * self.batch]
        return np.sort(idx) if sort else idx

    def _fill(self, step):
        try:
            while True:
                if self.slot:  # the epoch's order: their loss groups consecutive rows
                    x, plen = joined(self.rows[self.indices(step, sort=False)], self.slot, self.Lp)
                    self.q.put((torch.from_numpy(x).pin_memory(), torch.from_numpy(plen).pin_memory()))
                else:
                    x = self.rows[self.indices(step)]
                    resp = x[:, self.Lp:]
                    width = self.Lp + min(resp.shape[1], -(-int((resp != PAD).sum(1).max()) // ALIGN) * ALIGN)
                    self.q.put(torch.from_numpy(np.ascontiguousarray(x[:, :width])).pin_memory())
                step += 1
        except Exception as e:  # handed to the trainer, which raises it (a dead thread would hang next() instead)
            self.q.put(e)

    def next(self):
        x = self.q.get()
        if isinstance(x, Exception):
            raise x
        return tuple(y.cuda(non_blocking=True).long() for y in x) if self.slot else x.cuda(non_blocking=True).long()


def from_hf(path, task, context):
    """An HF model directory extended as their SFT does (sft_trainer.py): the embedding rows of new words start at the
    mean embedding (transformers' mean_resizing), and a context over max_position_embeddings gets YaRN to it."""
    c = read_json(os.path.join(path, "config.json"))
    w = {k: v.float() for k, v in load_file(os.path.join(path, "model.safetensors")).items() if k != "lm_head.weight"}  # tied
    emb = w["model.embed_tokens.weight"]
    if len(emb) < len(task.WORDS):
        w["model.embed_tokens.weight"] = torch.cat([emb, emb.mean(0, keepdim=True).expand(len(task.WORDS) - len(emb), -1)])
    if context > c["max_position_embeddings"]:
        c["rope_scaling"] = {"type": "yarn", "factor": context / c["max_position_embeddings"],
                             "original_max_position_embeddings": c["max_position_embeddings"]}
        c["max_position_embeddings"] = context
    c.update(vocab_size=len(task.WORDS), bos_token_id=task.PAD, eos_token_id=task.EOS, pad_token_id=task.PAD,
             env_token_id=task.CALL, dtype="float32")  # the weights are saved in fp32
    return c, w


class SFT:
    def __init__(self, a):
        self.a, self.data = a, Dataset(a.data)
        self.cfg, self.task, self.s = self.data.cfg, self.data.task, self.data.cfg["sft"]
        self.init = a.init
        if self.init:
            self.hf, init = from_hf(self.init, self.task, self.s["context"])
        else:
            self.hf = hf_config(a.size, len(self.task.WORDS), self.data.width)
            init = init_weights(self.hf, a.seed)
        self.model = Qwen2(self.hf)
        self.run_meta = {"config": self.cfg, "size": a.size, "seed": a.seed, "hf": self.hf, **({"init": self.init} if self.init else {})}
        assert self.s["screen_every"] % self.s["save_every"] == 0 and self.s["max_steps"] % self.s["save_every"] == 0, \
            "screens and the last step must land on saves (the state keeps the screens; tuning walks the saved steps)"
        self.ckpt_dir = os.path.join(a.out, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.state_path = os.path.join(a.out, "state.pt")
        self.params = {k: v.cuda().requires_grad_() for k, v in init.items()}
        groups = self.params.values() if self.s.get("decay_norms", True) else [
            {"params": [p for k, p in self.params.items() if "norm" not in k]},
            {"params": [p for k, p in self.params.items() if "norm" in k], "weight_decay": 0.0}]
        self.opt = torch.optim.AdamW(groups, lr=self.s["lr"], betas=(0.9, 0.95), weight_decay=0.01, fused=True)
        self.step, self.screens, self.vals = 0, [], []
        if os.path.exists(self.state_path):
            st = torch.load(self.state_path)
            r = st["run"]
            assert same_build(r["config"], self.cfg) and (r["size"], r["seed"]) == (a.size, a.seed), "resuming another run"
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(st["params"][k])
            self.opt.load_state_dict(st["opt"])
            self.step, self.screens, self.vals = st["step"], st["screens"], st.get("vals", [])
        # a relaunch restores the run dir from S3, which can hold checkpoints and screens of the lost steps past the state
        for name in os.listdir(self.ckpt_dir):
            if int(name.split(".")[0]) > self.step:
                os.remove(os.path.join(self.ckpt_dir, name))
        self.write_screens()
        self.write_vals()
        write_json(os.path.join(a.out, "run.json"), self.run_meta)
        self.slot = getattr(self.task, "slot", None)
        self.loss = torch.compile(self._loss_joined if self.slot else self._loss)
        self.chunk = a.chunk or self.s["batch"]
        assert not self.slot or self.s["batch"] % self.chunk == 0 and self.chunk % self.s["group"] == 0, (self.chunk, self.s)
        self.decoder = Decoder(self.model, **self.cfg["decode"])
        self.dev = self.data.prompts(self.s.get("screen_split", "dev")).cuda()
        self.dev_parsed = self.task.parse(self.dev)
        val = np.array(self.data.split("val"))
        self.val = [tuple(torch.from_numpy(x).cuda().long() for x in joined(val[i:i + 64], self.slot, self.data.prompt_len))
                    for i in range(0, len(val), 64)] if self.slot else torch.from_numpy(val).cuda().long()
        wandb.init(project=a.wandb_project, id=a.wandb_id, name=a.wandb_id, resume="allow", dir=a.out,
                   config={**vars(a), **self.run_meta})

    def _loss(self, params, ids):
        """Mean cross entropy of the response tokens of ids (B, T) (bf16 forward, fp32 log-softmax)."""
        Lp = self.data.prompt_len
        w = {k: p.bfloat16() for k, p in params.items()}
        hidden = self.model.forward(w, ids, torch.arange(ids.shape[1], device=ids.device), self.model.causal())
        logits = self.model.logits(w, hidden[:, Lp - 1:-1]).float()
        target = ids[:, Lp:]
        nll = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
        mask = target != PAD
        return (nll * mask).sum() / mask.sum()

    def _loss_joined(self, params, ids, plen):
        """Per row: the summed cross entropy of its trained response tokens (after the prompt, not + ENV) and their count,
        for joined rows ids (B, T) (uint8 values) with prompt lengths plen (B,)."""
        w = params if self.s.get("dtype") == "float32" else {k: p.bfloat16() for k, p in params.items()}
        tok = ids % ENV
        hidden = self.model.forward(w, tok, torch.arange(ids.shape[1], device=ids.device), self.model.causal())
        logits = self.model.logits(w, hidden[:, :-1]).float()
        target = ids[:, 1:]
        mask = (torch.arange(1, ids.shape[1], device=ids.device) >= plen[:, None]) & (target != PAD) & (target < ENV)
        nll = F.cross_entropy(logits.flatten(0, 1), (target % ENV).flatten(), reduction="none").view_as(target)
        return (nll * mask).sum(1), mask.sum(1)

    def batch_loss(self, ids, plen):
        """Backward of the step's loss (their per-group token means, averaged over the batch's groups) in chunks of rows;
        returns the loss."""
        g, total = self.s["group"], 0.0
        for i in range(0, len(ids), self.chunk):
            nll, count = self.loss(self.params, ids[i:i + self.chunk], plen[i:i + self.chunk])
            loss = (nll.view(-1, g).sum(1) / count.view(-1, g).sum(1)).sum() / (self.s["batch"] // g)
            loss.backward()
            total += loss.detach()
        return total

    @torch.no_grad()
    def screen(self):
        torch.manual_seed(self.step)
        out = evaluate(self.task, self.decoder, half(self.params), self.dev, self.dev_parsed, self.s["screen_k"])
        return {"step": self.step, "val_loss": self.val_loss(), **out}

    @torch.no_grad()
    def val_loss(self):
        if self.slot:  # the token mean over the val rows
            sums = torch.stack([torch.stack([x.sum() for x in self.loss(self.params, *v)]) for v in self.val]).sum(0)
            return (sums[0] / sums[1]).item()
        chunks = [self.val[i:i + 64] for i in range(0, len(self.val), 64)]  # the token mean over the val rows
        tokens = [int((c[:, self.data.prompt_len:] != PAD).sum()) for c in chunks]
        return sum(self.loss(self.params, c).item() * k for c, k in zip(chunks, tokens)) / sum(tokens)

    def lr(self):
        """The next step's lr: constant, or linear warmup over sft.warmup steps then cosine to sft.min_lr at step
        sft.decay_steps (a frozen setting: an extended run stays at min_lr)."""
        s, k = self.s, self.step
        if "warmup" not in s:
            return s["lr"]
        if k < s["warmup"]:
            return s["lr"] * (k + 1) / s["warmup"]
        p = min(1.0, (k - s["warmup"]) / (s["decay_steps"] - s["warmup"]))
        return s["min_lr"] + (s["lr"] - s["min_lr"]) * 0.5 * (1 + math.cos(math.pi * p))

    def write_screens(self):
        def write(tmp):
            with open(tmp, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in self.screens)
        atomic(os.path.join(self.a.out, "screen.jsonl"), write)

    def write_vals(self):
        if self.s.get("val_every"):
            def write(tmp):
                with open(tmp, "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in self.vals)
            atomic(os.path.join(self.a.out, "val.jsonl"), write)

    def record_val(self):
        self.vals.append({"step": self.step, "val_loss": self.val_loss()})
        wandb.log({"val/loss": self.vals[-1]["val_loss"]}, step=self.step)
        self.write_vals()

    def record(self, screen):
        self.screens.append(screen)
        wandb.log({f"screen/{k}": v for k, v in screen.items()}, step=self.step)
        self.write_screens()

    def save(self):
        weights = {k: p.detach().cpu() for k, p in self.params.items()}
        atomic(os.path.join(self.ckpt_dir, f"{self.step}.safetensors"), lambda tmp: save_file(weights, tmp))
        atomic(self.state_path, lambda tmp: torch.save({"run": self.run_meta, "params": weights, "opt": self.opt.state_dict(),
                                                        "step": self.step, "screens": self.screens, "vals": self.vals}, tmp))

    def stopped(self):
        last = [r["score"] for r in self.screens[-4:]]
        return len(last) == 4 and statistics.median(last) >= self.s["stop_at"]

    def run(self):
        s = self.s
        if not self.screens:
            self.record(self.screen())
            if s.get("val_every"):
                self.record_val()
            self.save()
        batches = Batches(self.data.split("train"), self.data.prompt_len, s["batch"], s["data_seed"], self.step + 1, slot=self.slot)
        t0, tokens = time.perf_counter(), 0
        while self.step < s["max_steps"] and not self.stopped():
            for group in self.opt.param_groups:
                group["lr"] = self.lr()
            if self.slot:
                ids, plen = batches.next()
                loss = self.batch_loss(ids, plen)
            else:
                ids = batches.next()
                loss = self.loss(self.params, ids)
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params.values(), 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.step += 1
            tokens += ids.numel()
            if self.step % s["log_every"] == 0:
                dt = time.perf_counter() - t0
                m = {"train/loss": loss.item(), "train/grad_norm": grad_norm.item(), "train/lr": self.opt.param_groups[0]["lr"],
                     "perf/steps_per_s": s["log_every"] / dt,
                     "perf/tokens_per_s": tokens / dt, "perf/max_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
                wandb.log(m, step=self.step)
                print(f"step {self.step} " + " ".join(f"{k.split('/')[1]} {v:.4g}" for k, v in m.items()), flush=True)
                t0, tokens = time.perf_counter(), 0
            if s.get("val_every") and self.step % s["val_every"] == 0:
                self.record_val()
            if self.step % s["screen_every"] == 0:
                self.record(self.screen())
            if self.step % s["save_every"] == 0:
                self.save()
        write_json(os.path.join(self.a.out, "DONE"), {"step": self.step, "reason": "stop_at" if self.stopped() else "max_steps"})
        wandb.finish()


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset dir (data.py)")
    ap.add_argument("--size", required=True, help="model size, a key of model.ARCH (with --init: the run's name for it)")
    ap.add_argument("--seed", type=int, required=True, help="init seed (unused with --init)")
    ap.add_argument("--init", help="start from this HF model directory instead of ARCH[size]")
    ap.add_argument("--chunk", type=int, help="rows per forward (default: the whole batch)")
    ap.add_argument("--out", required=True, help="run dir")
    ap.add_argument("--wandb-project", required=True)
    ap.add_argument("--wandb-id", required=True)
    return ap


if __name__ == "__main__":
    SFT(parser().parse_args()).run()
