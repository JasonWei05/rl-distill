#!/usr/bin/env python3
"""Compare the Gemma-4-26B-PT and Qwen3-30B-A3B-Base DeepScaleR difficulty bands (same 38,796-question dedup_v2 pool, n=8 each).

Prints overlap of the released 3,300-question band sets, val/train cross-contamination, the full-pool pass-count crosstab and each
model's pass rate on the other model's bands. Paths default to the local builds on EFS.

    .venv/bin/python rl-distill-scripts/analysis/compare_gemma_qwen_band_overlap.py
"""

import pandas as pd, numpy as np
import os; os.chdir('/mnt/efs/jasonwei/verl/data')
G='deepscaler_easy_medium_hard_gemma4_26b_pt_10k_20260817'; Q='deepscaler_easy_medium_hard_very_hard_qwen3_30b_a3b_base_20261010'
def load(root, bands):
    out={}
    for b in bands:
        for s in ['train','validation']:
            d=pd.read_parquet(f'{root}/{b}/{s}.parquet',columns=['extra_info'])
            out[(b,s)]=set(d['extra_info'].map(lambda e:e['source_uid']))
    return out
g=load(G,['easy','medium','hard']); q=load(Q,['easy','medium','hard','very_hard'])
gb={b:g[(b,'train')]|g[(b,'validation')] for b in ['easy','medium','hard']}
qb={b:q[(b,'train')]|q[(b,'validation')] for b in ['easy','medium','hard','very_hard']}
print('released sets sizes', {k:len(v) for k,v in gb.items()}, {k:len(v) for k,v in qb.items()})
print('\n== released 3300-q sets: |G∩Q| (Jaccard%)')
print('gemma\\qwen', *qb, 'none', sep='\t')
for a,A in gb.items():
    row=[f'{len(A&B)} ({100*len(A&B)/len(A|B):.1f})' for B in qb.values()]
    print(a,*row,len(A-set().union(*qb.values())),sep='\t')
print('\n== val-split cross contamination: gemma val in qwen train / qwen val in gemma train')
for a in gb:
    for b in qb:
        x=len(g[(a,'validation')]&q[(b,'train')]); y=len(q[(b,'validation')]&g[(a,'train')])
        if x or y: print(f' G-{a} val ∩ Q-{b} train = {x};  Q-{b} val ∩ G-{a} train = {y}')
# full pool
gg=pd.read_parquet('deepscaler_gemma4_26b_a4b_pt8_dedup_v2_20260817/questions_with_current_rl_grades.parquet')[['uid','current_rl_n_correct']].rename(columns={'current_rl_n_correct':'g'})
qq=pd.read_parquet('deepscaler_qwen3_30b_a3b_base_n8_dedup_v2_20261005/questions_with_current_rl_grades.parquet')[['uid','current_rl_n_correct']].rename(columns={'current_rl_n_correct':'q'})
m=gg.merge(qq,on='uid'); print('\n== full pool', len(m), 'questions')
print(f'mean pass rate: gemma {m.g.mean()/8:.3f}  qwen {m.q.mean()/8:.3f}; pearson {m.g.corr(m.q):.3f} spearman {m.g.corr(m.q,method="spearman"):.3f}')
print(f'qwen > gemma {np.mean(m.q>m.g):.3f}, equal {np.mean(m.q==m.g):.3f}, gemma > qwen {np.mean(m.g>m.q):.3f}')
def bucket(n): return np.select([n==0,n==1,n<=3],['0/8','hard 1/8','medium 2-3/8'],'easy 4-8/8')
m['gb']=bucket(m.g); m['qb']=bucket(m.q)
ct=pd.crosstab(m.gb,m.qb,margins=True); print('\nrows=gemma bucket, cols=qwen bucket (counts)\n',ct)
print('\nrow-normalized % (of gemma bucket, where qwen puts it)\n',(100*pd.crosstab(m.gb,m.qb,normalize='index')).round(1))
print('\ncol-normalized % (of qwen bucket, where gemma puts it)\n',(100*pd.crosstab(m.gb,m.qb,normalize='columns')).round(1))
ct8=pd.crosstab(m.g,m.q); print('\nfull 9x9 n_correct crosstab rows=gemma cols=qwen\n',ct8)
# other model's pass rate on each released band
print('\n== other model pass rate (n=8) on each released band')
gi=m.set_index('uid')
for a,A in gb.items():
    s=gi.loc[list(A)]; print(f' gemma-{a}: gemma {s.g.mean()/8:.3f}, qwen {s.q.mean()/8:.3f}, qwen 0/8 {np.mean(s.q==0):.1%}')
for b,B in qb.items():
    s=gi.loc[list(B)]; print(f' qwen-{b}: qwen {s.q.mean()/8:.3f}, gemma {s.g.mean()/8:.3f}, gemma 0/8 {np.mean(s.g==0):.1%}')
