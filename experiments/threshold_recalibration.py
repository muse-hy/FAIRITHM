# -*- coding: utf-8 -*-
"""Split-half threshold recalibration for the FAIRITHM detection signal.

Splits the labeled test set at the occ-template level into calibration and
evaluation halves (prevents Cycle1/Cycle2 near-duplicate leakage), selects a
threshold on the calibration half under several criteria (Youden J, F1-max,
max-recall-at-precision>=0.5, balanced accuracy), and reports held-out
performance of each, alongside the study operating point (topk_mean >= 0.2).
Also computes per-category Youden thresholds. See RESULTS.md for the run used
in the revision. Usage: python experiments/threshold_recalibration.py
"""
import pandas as pd, numpy as np, os, sys
from sklearn.metrics import roc_curve

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "out_detection", "features.tsv"), sep="	")
rng = np.random.default_rng(123)
occs = sorted(df.occ.unique()); rng.shuffle(occs)
half = set(occs[:len(occs)//2])
cal, ev = df[df.occ.isin(half)], df[~df.occ.isin(half)]

def metrics(sub, thr):
    f = sub.topk_mean >= thr; l = sub.label == 1
    tp=int((f&l).sum()); fp=int((f&~l).sum()); fn=int((~f&l).sum()); tn=int((~f&~l).sum())
    p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
    return dict(thr=round(float(thr),4), precision=round(p,3), recall=round(r,3),
                F1=round(2*p*r/(p+r),3) if p+r else 0,
                FPR=round(fp/(fp+tn),3) if fp+tn else 0, flag_rate=round(float(f.mean()),3))

def pick(sub, mode):
    fpr,tpr,thrs = roc_curve(sub.label, sub.topk_mean)
    if mode=="youden": return float(thrs[np.argmax(tpr-fpr)])
    grid = np.unique(np.round(sub.topk_mean,4)); best, bt = -1, None
    for t in grid:
        m = metrics(sub, t)
        score = {"f1": m["F1"],
                 "prec50": m["recall"] if m["precision"]>=0.5 else -1,
                 "balacc": (m["recall"]+(1-m["FPR"]))/2}[mode]
        if score > best: best, bt = score, t
    return float(bt if bt is not None else 0.2)

for mode,label in [("youden","Youden J"),("f1","F1-max"),
                   ("prec50","max R s.t. P>=0.5"),("balacc","balanced accuracy")]:
    print(label, metrics(ev, pick(cal, mode)))
print("study 0.2", metrics(ev, 0.2))
print()
for c in sorted(df.category.unique()):
    cc, ce = cal[cal.category==c], ev[ev.category==c]
    if cc.label.nunique()<2 or ce.label.nunique()<2: continue
    t = pick(cc,"youden")
    print(c, "thr=%.4f" % t, metrics(ce,t), "| recall@0.2 =", metrics(ce,0.2)["recall"])
