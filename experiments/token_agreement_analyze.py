# -*- coding: utf-8 -*-
"""
Token-level agreement analysis (R2-C2) between system bias highlights and
human annotations.

Inputs (in experiments/out_token_agreement/):
  - annotation_sheet_<name>.tsv : one filled sheet per annotator; the
    'biased_words(주석자 기입)' column contains the marked words separated by
    ' | ' (copied verbatim from the 'words' column).
  - system_key.tsv : system top-1 / top-5 words per item.

Outputs:
  - Human–human agreement: word-level Fleiss' kappa (each word in each
    sentence is one instance; marked = 1).
  - System–human: top-1 and top-5 hit rates (vs each annotator, vs the union
    and majority of annotators), word-level precision/recall/F1 (system top-5
    as predictions), and Cohen's kappa system-vs-each-annotator.
  - Subject-vs-predicate breakdown: hit rates counted separately for cases
    where the matched word is the sentence-initial (group-denoting) word.

Usage:
  python experiments/token_agreement_analyze.py sheetA.tsv sheetB.tsv [sheetC.tsv]
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_token_agreement")


def load_sheet(path):
    marks = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            col = [c for c in row if c.startswith("biased_words")][0]
            marked = [w.strip() for w in (row[col] or "").split("|") if w.strip()]
            marks[int(row["item"])] = set(marked)
    return marks


def fleiss_kappa(mat):
    """mat: instances x 2 (counts of raters voting 0 / 1)."""
    n = mat.sum(axis=1)[0]
    P_i = ((mat ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    p_j = mat.sum(axis=0) / mat.sum()
    P_e = (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def main():
    key = {}
    with open(os.path.join(BASE, "system_key.tsv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key[int(row["item"])] = row

    annotators = {os.path.basename(p): load_sheet(p) for p in sys.argv[1:]}
    assert len(annotators) >= 2, "need at least 2 filled sheets"
    names = list(annotators)
    items = sorted(key)

    # word-level instance table
    rows = []  # (item, word, is_first_word, votes per annotator)
    for it in items:
        words = key[it]["sentence"].split()
        for j, w in enumerate(words):
            votes = [int(w in annotators[a][it]) for a in names]
            rows.append((it, w, j == 0, votes))

    mat = np.array([[len(names) - sum(v), sum(v)] for *_, v in rows])
    print(f"instances (words): {len(rows)}, annotators: {len(names)}")
    print(f"Human-human Fleiss kappa (word level): {fleiss_kappa(mat):.3f}")

    # per-annotator and consensus sets
    union = {it: set().union(*(annotators[a][it] for a in names)) for it in items}
    majority = {it: {w for it2, w, _, v in rows if it2 == it and sum(v) * 2 > len(names)}
                for it in items}

    def hits(target_sets, sys_words_fn):
        top1 = np.mean([sys_words_fn(it)[0] in target_sets[it] if sys_words_fn(it) else False
                        for it in items])
        topk = np.mean([bool(set(sys_words_fn(it)) & target_sets[it]) for it in items])
        return top1, topk

    def sys_words(it):
        t5 = [w.strip() for w in key[it]["system_top5"].split("|") if w.strip()]
        return t5

    for label, tgt in [("union", union), ("majority", majority)] + \
                      [(f"annotator:{a}", {it: annotators[a][it] for it in items}) for a in names]:
        t1, tk = hits(tgt, sys_words)
        print(f"vs {label:20s} top-1 hit {t1:.3f} | top-5 any-hit {tk:.3f}")

    # word-level P/R/F1: system top-5 as predicted positives, majority as truth
    tp = fp = fn = 0
    subj_hit = pred_hit = 0
    for it in items:
        pred = set(sys_words(it))
        truth = majority[it]
        words = key[it]["sentence"].split()
        tp += len(pred & truth); fp += len(pred - truth); fn += len(truth - pred)
        for w in pred & truth:
            if words and w == words[0]:
                subj_hit += 1
            else:
                pred_hit += 1
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    print(f"\nword-level vs majority: precision {p:.3f} recall {r:.3f} "
          f"F1 {2*p*r/(p+r):.3f}" if p + r else "no overlap")
    print(f"matched words that are the sentence-initial (subject) word: {subj_hit}, "
          f"other positions: {pred_hit}")


if __name__ == "__main__":
    main()
