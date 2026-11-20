"""Fast GRPO + PPO-clip + TIS on a synth task: the policy loss of the math RL recipe (docs/RECIPES.md 1.1) on the maze t3
setup (rl-distill-scripts/maze/rl/rl_tis_run.sh).

- Prompts are drawn in verl's order: a RandomSampler seeded with --seed over the train split, drop_last batches.
- rl.n samples per prompt at temperature 1 from the fp16 policy (model.Decoder); the log-prob and entropy of every
  sampled token are kept (the behaviour policy for TIS).
- Reward: the task's judge score (maze: a shortest path to the goal; sudoku: a valid completed grid).
- GRPO advantage: (r - group mean) / (group unbiased std + 1e-6).
- --minibatches k optimizer steps per batch (the batch's prompts in k consecutive mini-batches, one pass). pi_old is
  the policy before the step's first update. Each token's loss is the PPO clip loss max(-A rho, -A clip(rho, 1 - 0.2,
  1 + 0.28)), rho = pi / pi_old, bounded by -10 A for A < 0 (dual clip), times the detached token TIS weight
  min(pi_old / pi_rollout, rl.tis_cap), and its token-mean is over the whole mini-batch: summed over micro-batches
  (whole prompts, as many as hold no more context than rl.micro[prompts per mini-batch] samples of full-length
  responses: few long responses, many short ones) and ranks and divided by the mini-batch's token count, so the update
  does not depend on the micro-batch size or the number of GPUs. With one mini-batch the ratio is 1 and the
  gradient is -A min(pi / pi_rollout, cap) grad log pi. Master weights are fp32; the forward and
  backward run in fp16 under a dynamic loss scale (2^16, growth every 400 steps). Grad-norm clip 1.0, AdamW (0.9, 0.999),
  weight decay 0.01, lr (rl.lr, or --lr) * min(1, (step - 1) / rl.warmup) for every update of a step.
- Chess (a scripted multi-turn task, tasks.Chess): prompts from rl.split, rollouts with the environment's replies fed
  after each call (model.Decoder), the loss and the advantage on the sampled tokens only; --n overrides rl.n, --micro
  rl.micro, --decode-mem-gb decode.mem_gb.
- Validation at step 0 and every rl.test_freq steps on the first rl.val_prompts prompts of rl.val_split (dev), every
  rl.val_stride-th (1), x rl.val_n samples. Writes
  best.json, best_hf/ (the best validation score's weights) and COMPLETE.json, and checkpoints at every validation
  (resumed in place).
- --gpus K > 1: data parallel. Each rank samples, judges and updates on its 1/K of every mini-batch's prompts (and of
  the validation prompts) with its own sampling seed; gradients are averaged before each optimizer step, so the updates
  equal the one-GPU updates on the same samples. Rank 0 writes files.

    python fastrl.py --base <sft run>/tune/p3 --data /tmp/synth/data/maze33 --out /tmp/synth/rl/<name> --batch 64 --seed 2
"""

import argparse
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb

from data import Dataset, same_build
from files import atomic, write_json
from model import Decoder, Qwen2, evaluate, half, load_model, response_mask, save_model
from tasks import ENV, EOS, first_true

ES_METRIC = "val/score"


def grpo_advantage(scores, n):
    """verl compute_grpo_outcome_advantage for consecutive groups of n: (r - mean) / (unbiased std + 1e-6), (B,)."""
    g = scores.view(-1, n)
    return ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)).view(-1)


CLIP_LOW, CLIP_HIGH, CLIP_C = 0.2, 0.28, 10.0  # the math RL recipe's PPO clip range and dual clip
STATS = ("actor/pg_loss", "actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower", "rollout_corr/tis_capped_frac",
         "rollout_corr/log_ratio_old_rollout")


def ppo_tis_loss(logp, old_logp, rollout_logp, adv, mask, tokens, cap):
    """The math recipe's per-token loss (module docstring; both log-ratios clamped to +-20, the TIS weight detached),
    summed over these tokens and divided by `tokens` (pass the mini-batch's token count over all ranks / ranks), and the
    STATS on the same scale (sums over the micro-batches and the mean over ranks give mini-batch token means)."""
    log_ratio = (logp - old_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    loss1, loss2 = -adv * ratio, -adv * ratio.clamp(1 - CLIP_LOW, 1 + CLIP_HIGH)
    clipped = torch.maximum(loss1, loss2)
    pg = torch.where(adv < 0, torch.minimum(-adv * CLIP_C, clipped), clipped)
    is_log_ratio = (old_logp - rollout_logp).clamp(-20.0, 20.0)
    total = lambda x: (x * mask).sum() / tokens
    loss = total(pg * is_log_ratio.exp().clamp(max=cap))
    return loss, torch.stack([loss.detach(), total((loss2 > loss1).float()), total(-log_ratio.detach()),
                              total(((clipped > -adv * CLIP_C) & (adv < 0)).float()),
                              total((is_log_ratio.exp() > cap).float()), total(is_log_ratio)])


class PromptOrder:
    """verl's train prompts per step: an epoch of RandomSampler(generator seeded with seed) is one torch.randperm, plus a
    second one it draws for the remainder slice [:0] when the epoch runs out; batches drop the remainder."""

    def __init__(self, n, batch, seed):
        self.n, self.batch, self.gen, self.perm, self.epoch = n, batch, torch.Generator().manual_seed(seed), None, -1

    def __call__(self, step):
        epoch, i = divmod(step - 1, self.n // self.batch)
        while self.epoch < epoch:
            self.perm, _ = torch.randperm(self.n, generator=self.gen), torch.randperm(self.n, generator=self.gen)
            self.epoch += 1
        return self.perm[i * self.batch:(i + 1) * self.batch].numpy()


def train_tflops(c, seqlens):
    """verl FlopsCounter._estimate_qwen2_flops (forward + backward) in TFLOPs for sequences of the given lengths."""
    H, h, kvh, nl = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["num_hidden_layers"]
    d = c.get("head_dim") or H // h  # Qwen3 sets it apart from H / h (chess 50M / 100M: 128, not 64)
    dense = (H * c["intermediate_size"] * 3 + H * (h * d + 2 * kvh * d + h * d)) * nl + c["vocab_size"] * H * 2
    s = seqlens.double()
    return float(6 * dense * s.sum() + 12 * (s * s).sum() * d * h * nl) / 1e12


def all_mean(x):
    """The mean over ranks of a tensor (every rank holds the same number of samples)."""
    dist.all_reduce(x, op=dist.ReduceOp.AVG)
    return x


def all_sum(x):
    dist.all_reduce(x)
    return x


class Trainer:
    def __init__(self, a):
        self.a, self.rank, self.world = a, dist.get_rank(), dist.get_world_size()
        assert a.batch % (self.world * a.minibatches) == 0, (a.batch, self.world, a.minibatches)
        self.hf, self.synth, weights = load_model(a.base)
        self.data = Dataset(a.data)
        assert same_build(self.synth["config"], self.data.cfg), "the base was trained on another dataset config"
        self.task, self.r = self.data.task, {**self.data.cfg["rl"], **({"patience": a.patience} if a.patience else {})}
        assert self.r["val_prompts"] % self.world == 0, (self.r["val_prompts"], self.world)
        self.n, self.micro = a.n or self.r["n"], a.micro or self.r["micro"][str(a.batch // a.minibatches)]
        self.model = Qwen2(self.hf)
        self.params = {k: v.requires_grad_() for k, v in weights.items()}
        self.lr = a.lr or self.r["lr"]
        self.opt = torch.optim.AdamW(self.params.values(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16, growth_interval=400)
        self.decoder = Decoder(self.model, **{**self.data.cfg["decode"], **({"mem_gb": a.decode_mem_gb} if a.decode_mem_gb else {})})
        self.train = self.data.split(self.r.get("split", "train"))[:, :self.data.prompt_len]
        V = self.r["val_prompts"] // self.world
        val = self.data.prompts(self.r.get("val_split", "dev"))[::self.r.get("val_stride", 1)][:self.r["val_prompts"]]
        assert len(val) == self.r["val_prompts"] and len(val) % self.world == 0, (len(val), self.r["val_prompts"], self.world)
        self.val = val[self.rank * V:(self.rank + 1) * V].cuda()
        self.val_parsed = self.task.parse(self.val)
        self.order = PromptOrder(len(self.train), a.batch, a.seed)
        self.step, self.flops = 0, {"gen": 0.0, "update_actor": 0.0, "total": 0.0}
        self.best = {"metric": ES_METRIC, "best_value": None, "best_step": None, "since_best": 0, "history": []}
        os.makedirs(a.out, exist_ok=True)
        self.ckpt = os.path.join(a.out, "ckpt.pt")
        if os.path.exists(self.ckpt):
            s = torch.load(self.ckpt, map_location="cpu")
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(s["params"][k])
            self.opt.load_state_dict(s["opt"])
            self.scaler.load_state_dict(s["scaler"])
            assert len(s["rng"]) == self.world, "resume with the same number of GPUs"
            # checkpoints from before --minibatches / --lr / --n ran one mini-batch at rl.lr with rl.n samples
            assert (s.get("minibatches", 1), s.get("lr", self.r["lr"]), s.get("n", self.r["n"])) == (a.minibatches, self.lr, self.n), \
                "resume with the same --minibatches / --lr / --n"
            torch.cuda.set_rng_state(s["rng"][self.rank])
            self.step, self.flops, self.best = s["step"], s["flops"], s["best"]
        else:
            torch.manual_seed(a.seed + 1000 * self.rank)
        wandb.init(project=a.wandb_project, id=a.wandb_id, name=a.wandb_id, resume="allow", dir=a.out,
                   mode=None if self.rank == 0 else "disabled",
                   config={**vars(a), "rl": self.r, "base_synth": {k: v for k, v in self.synth.items() if k != "config"}})

    def weights16(self):
        """The fp16 compute copy of the fp32 master weights; differentiable, so gradients reach the masters in fp32."""
        return {k: p.half() for k, p in self.params.items()}

    def validate(self):
        out = evaluate(self.task, self.decoder, half(self.params), self.val, self.val_parsed, self.r["val_n"])
        means = all_mean(torch.tensor(list(out.values()), device="cuda")).tolist()
        return {f"val/{k}": x for k, x in zip(out, means)}

    def micro_batches(self, M, lens, Lp, R):
        """Each mini-batch's (M rows) micro-batches: consecutive whole prompts (n rows each) while their rows, at the
        longest response among them, hold no more context than self.micro rows of full-length responses (Lp + R tokens
        each), the configured micro-batch's memory (train_logprobs holds every row's prompt and response)."""
        n, cap, out = self.n, self.micro * (Lp + R), []
        for j in range(0, len(lens), M):
            mb, a, top = [], j, 0
            for g in range(j, j + M, n):
                top_g = max(top, *lens[g:g + n])
                if g > a and (g + n - a) * (Lp + top_g) > cap:
                    mb.append(slice(a, g))
                    a, top_g = g, max(lens[g:g + n])
                top = top_g
            out.append(mb + [slice(a, j + M)])
        return out

    def update(self, prompts, tok, rollout_logp, adv, mask, plen=None):
        """The step's a.minibatches optimizer steps on this rank's rows (grouped n per prompt; mini-batch j = the j-th
        1/k of them; plen: the prompts' lengths, if right-padded); returns the actor metrics, means over the mini-batches."""
        n, k = self.n, self.a.minibatches
        M = len(tok) // k  # rows per mini-batch
        assert M % n == 0, (len(tok), k, n)
        lens = [max(x, 2) for x in (mask.shape[1] - first_true(mask.flip(1))).tolist()]  # through the last trained token
        micro = self.micro_batches(M, lens, prompts.shape[1] if plen is None else int(plen.max()), mask.shape[1])
        g = lambda s: slice(s.start // n, s.stop // n)
        logprobs = lambda w, s: self.model.train_logprobs(  # >= 1 response query for train_logprobs
            w, prompts[g(s)], tok[s, :max(lens[s])] % ENV, n, None if plen is None else plen[g(s)])
        with torch.no_grad():  # pi_old of the later mini-batches (the first reads its own log-probs, before any update)
            w = self.weights16()
            old = {s.start: logprobs(w, s) for mb in micro[1:] for s in mb}
        stats, norms = [], []
        for j in range(k):
            tokens = all_sum(mask[j * M:(j + 1) * M].sum().float()) / self.world
            st = 0
            for s in micro[j]:
                logp = logprobs(self.weights16(), s)
                r = logp.shape[1]
                loss, st_s = ppo_tis_loss(logp, old[s.start] if j else logp.detach(), rollout_logp[s, :r], adv[s, :r],
                                          mask[s, :r].float(), tokens, self.r["tis_cap"])
                self.scaler.scale(loss).backward()
                st = st + st_s
            for p in self.params.values():
                all_mean(p.grad)
            self.scaler.unscale_(self.opt)
            norms.append(torch.nn.utils.clip_grad_norm_(self.params.values(), 1.0))
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            stats.append(all_mean(st))
        return {**dict(zip(STATS, torch.stack(stats).mean(0).tolist())), "actor/grad_norm": torch.stack(norms).mean().item()}

    def train_step(self):
        n = self.n
        t0 = time.perf_counter()
        mine = self.order(self.step).reshape(self.a.minibatches, self.world, -1)[:, self.rank].reshape(-1)
        prompts = torch.from_numpy(self.train[mine]).cuda().long()
        parsed = self.task.parse(prompts)
        script = parsed.get("script")
        tok, logp, ent = self.decoder.generate(half(self.params), prompts, n, **({"script": script} if script else {}))
        # the trained tokens: through EOS, or (scripted) the sampled ones (environment tokens are + ENV, done rows pad)
        mask = response_mask(tok) if script is None else (tok != self.hf["pad_token_id"]) & (tok < ENV)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = self.task.judge(tok, parsed, torch.arange(len(prompts), device="cuda").repeat_interleave(n))
        adv = grpo_advantage(out["score"], n)[:, None] * mask
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        lr = self.lr * (min(1.0, (self.step - 1) / self.r["warmup"]) if self.r["warmup"] else 1.0)
        for group in self.opt.param_groups:
            group["lr"] = lr
        plen = None if script is None else script["plen"]
        metrics = self.update(prompts, tok, logp * mask, adv, mask, plen)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        seqlens = (prompts.shape[1] if plen is None else plen.repeat_interleave(n)) + mask.shape[1] - first_true(mask.flip(1))
        fwd = train_tflops(self.hf, seqlens) / 3
        old = train_tflops(self.hf, seqlens[len(seqlens) // self.a.minibatches:]) / 3  # pi_old of the later mini-batches
        gen, upd = all_sum(torch.tensor([fwd, 3 * fwd + old], device="cuda")).tolist()
        for key, x in (("gen", gen), ("update_actor", upd), ("total", gen + upd)):
            self.flops[key] += x
        maskf = mask.float()
        if script is None:  # hit the response cap
            clipped = ~(tok == EOS).any(1)
        else:  # the sampled-token budget, or the context (prompt and every response token, the environment's too)
            clipped = (mask.sum(1) >= self.decoder.budget) | (
                plen.repeat_interleave(n) + (tok != self.hf["pad_token_id"]).sum(1) >= self.decoder.max_ctx)
        stats = all_mean(torch.stack([(ent * maskf).sum(), maskf.sum(), clipped.float().mean(),
                                      *(x.mean() for x in out.values())]))
        metrics.update({
            "actor/entropy": (stats[0] / stats[1]).item(), "actor/lr": lr,
            "response_length/mean": (stats[1] / len(tok)).item(),
            "response_length/clip_ratio": stats[2].item(),
            **{f"train/{k}": x.item() for k, x in zip(out, stats[3:])},
            "timing_s/gen": t1 - t0, "timing_s/reward": t2 - t1, "timing_s/update_actor": t3 - t2,
            "tflops/gen": gen, "tflops/update_actor": upd, "tflops/total": gen + upd,
            **{f"tflops/cumulative_{k}": x for k, x in self.flops.items()},
            "training/global_step": self.step, "perf/max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
        return metrics

    def checkpoint(self, val_metrics):
        """Track the best validation (strictly greater = new best, saved to best_hf), save the training state; returns
        True when rl.patience (or --patience) validations in a row brought no new best."""
        value, b = val_metrics[ES_METRIC], self.best  # every rank keeps the same record
        b["history"].append([self.step, value])
        improved = b["best_value"] is None or value > b["best_value"]
        if improved:
            b.update(best_value=value, best_step=self.step, since_best=0)
        else:
            b["since_best"] += 1
        rng = [None] * self.world
        dist.all_gather_object(rng, torch.cuda.get_rng_state())
        if self.rank == 0:
            if improved:
                save_model(os.path.join(self.a.out, "best_hf"), self.params, self.hf,
                           {**self.synth, "rl": {"base": self.a.base, "step": self.step, "metric": ES_METRIC, "value": value}})
            write_json(os.path.join(self.a.out, "best.json"), b)
            atomic(self.ckpt, lambda tmp: torch.save({
                "params": {k: p.detach() for k, p in self.params.items()}, "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(), "rng": rng, "step": self.step, "flops": self.flops, "best": b,
                "minibatches": self.a.minibatches, "lr": self.lr, "n": self.n}, tmp))
        return b["since_best"] >= self.r["patience"]

    def run(self):
        r = self.r
        if not self.best["history"]:
            t = time.perf_counter()
            metrics = self.validate()
            metrics["timing_s/testing"] = time.perf_counter() - t
            self.checkpoint(metrics)
            wandb.log(metrics, step=0)
        stop = self.best["since_best"] >= r["patience"]
        while self.step < self.a.steps and not stop:
            self.step += 1
            t = time.perf_counter()
            metrics = self.train_step()
            if self.step % r["test_freq"] == 0 or self.step == self.a.steps:
                tv = time.perf_counter()
                metrics.update(self.validate())
                metrics["timing_s/testing"] = time.perf_counter() - tv
                stop = self.checkpoint(metrics)
                metrics.update({f"early_stop/{k}": self.best[k] for k in ("best_value", "best_step", "since_best")})
                if self.rank == 0:
                    print(f"step {self.step} val {metrics[ES_METRIC]:.4f} best {self.best['best_value']:.4f} "
                          f"@{self.best['best_step']} step_s {time.perf_counter() - t:.3f}", flush=True)
            metrics["timing_s/step"] = time.perf_counter() - t
            wandb.log(metrics, step=self.step)
        if self.rank == 0:
            write_json(os.path.join(self.a.out, "COMPLETE.json"), {
                "reason": "early_stop" if stop else "max_steps", "last_step": self.step, "best_step": self.best["best_step"],
                "best_value": self.best["best_value"], "metric": ES_METRIC, "flops_total_tflops": self.flops["total"]})
        wandb.finish()


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="model dir of the starting policy (tune.py's tune/<level>)")
    ap.add_argument("--data", required=True, help="the dataset dir it was trained on")
    ap.add_argument("--out", required=True, help="run dir: ckpt.pt, best.json, best_hf/, COMPLETE.json")
    ap.add_argument("--batch", type=int, required=True, help="prompts per step (sampled; split into --minibatches)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--gpus", type=int, default=1, help="data-parallel ranks (the visible GPUs)")
    ap.add_argument("--minibatches", type=int, default=1, help="optimizer steps per batch (it splits into this many)")
    ap.add_argument("--lr", type=float, help="instead of the config's rl.lr")
    ap.add_argument("--patience", type=int, help="instead of rl.patience: stop after this many validations without a new best")
    ap.add_argument("--n", type=int, help="instead of rl.n: samples per prompt (the GRPO group)")
    ap.add_argument("--micro", type=int, help="instead of rl.micro[...]: samples per update micro-batch")
    ap.add_argument("--decode-mem-gb", type=float, help="instead of decode.mem_gb: the sampler's KV-cache budget per chunk")
    ap.add_argument("--wandb-project", required=True)
    ap.add_argument("--wandb-id", required=True)
    return ap


def worker(rank, a, store, fn):
    """Rank `rank` of a.gpus (a process per GPU; NCCL over the file store): fn(a)."""
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", store=dist.FileStore(store, a.gpus), rank=rank, world_size=a.gpus, device_id=torch.device("cuda", rank))
    fn(a)
    dist.destroy_process_group()


def launch(a, fn):
    """fn(a) on a.gpus ranks."""
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")  # one machine: bootstrap over loopback (pods' eth0 is IPv6-only)
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # through host memory: works under MPS; the gradients are a few MB
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(worker, args=(a, os.path.join(d, "store"), fn), nprocs=a.gpus, start_method="spawn")


def train(a):
    Trainer(a).run()


if __name__ == "__main__":
    launch(parser().parse_args(), train)
