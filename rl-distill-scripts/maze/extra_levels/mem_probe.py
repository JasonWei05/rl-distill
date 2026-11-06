"""Peak GPU memory of screen_sft_ckpts.screen_one for one checkpoint at a given k / SCREEN_MAX_SEQS (A800 40GB check)."""
import os, sys, time
from types import SimpleNamespace
sys.path.insert(0, os.environ["M"])
import torch
import screen_sft_ckpts as ss
ckpt, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
judge = ss.load_judge(ss.DEFAULT_TAILRL)
prompts, gts = ss.load_prompts(os.path.join(ss.DEFAULT_TAILRL, "experiments/text_maze/data/eval1000/test.parquet"), n)
torch.manual_seed(0)
t = time.time()
r = ss.screen_one(ckpt, prompts, gts, judge, SimpleNamespace(k=k, batch_prompts=128, max_new_tokens=180), "cuda")
print(f"k={k} n={n} cap={os.environ.get('SCREEN_MAX_SEQS', 8192)} goal p@1 {r['goal_reached']['pass@1']:.4f} "
      f"peak reserved {torch.cuda.max_memory_reserved()/2**30:.1f} GiB, allocated {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, {time.time()-t:.0f}s", flush=True)
