# -*- coding: utf-8 -*-
"""
Evaluate GPT-4o ablation rewrites (experiments/out_ablation/rewrites.jsonl).

Automatic measures per condition (full / sentence_flag / none):
  - residual detector signal on the rewrite (PCGU PLL-difference score01 and
    deployed flag level), i.e., does the system's own detector see less bias
    after rewriting;
  - surface preservation vs. the original biased sentence: character-bigram
    Jaccard overlap and length ratio (proxies for structure preservation at
    the 약하게/Standard level);
  - failure count (missing rewrites).

Also emits a blinded human-rating sheet (rating_sheet.tsv) with the three
conditions shuffled per item (key in rating_key.tsv) for 5-10 external raters:
columns for fairness / meaning preservation / completeness on 7-point scales.
"""
import json
import os
import sys
import csv
import random

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("BASELINE_MODEL_PATH", os.path.join(ROOT, "models", "kcbert_baseline"))
os.environ.setdefault("RETRAINED_MODEL_PATH", os.path.join(ROOT, "models", "best_model", "best_model"))


def char_bigrams(s):
    s = "".join(s.split())
    return set(s[i:i + 2] for i in range(len(s) - 1))


def jaccard(a, b):
    A, B = char_bigrams(a), char_bigrams(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def main():
    import run_model

    out_dir_name = sys.argv[1] if len(sys.argv) > 1 else "out_ablation"

    rows = []
    with open(os.path.join(BASE, out_dir_name, "rewrites.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    results = []
    for r in rows:
        rw = r.get("rewrite")
        rec = {"sentence": r["sentence"], "category": r["category"], "condition": r["condition"]}
        if not rw:
            rec.update({"ok": 0})
            results.append(rec)
            continue
        analysis = run_model.compute_bias_pll_0to10(rw)
        rec.update({
            "ok": 1,
            "rewrite": rw,
            "residual_score01": analysis["편향 점수"],
            "residual_level": analysis["편향 단계"],
            "jaccard_overlap": round(jaccard(r["sentence"], rw), 4),
            "len_ratio": round(len(rw) / len(r["sentence"]), 3),
        })
        results.append(rec)

    out_tsv = os.path.join(BASE, out_dir_name, "rewrite_metrics.tsv")
    cols = ["sentence", "category", "condition", "ok", "rewrite",
            "residual_score01", "residual_level", "jaccard_overlap", "len_ratio"]
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # summary by condition
    import pandas as pd
    df = pd.DataFrame([r for r in results if r.get("ok")])
    summary = df.groupby("condition").agg(
        n=("rewrite", "size"),
        residual_score01_mean=("residual_score01", "mean"),
        residual_score01_sd=("residual_score01", "std"),
        flag_none=("residual_level", lambda s: (s == "편향 없음").mean()),
        flag_caution=("residual_level", lambda s: (s == "편향 주의").mean()),
        flag_strong=("residual_level", lambda s: (s == "편향 강함").mean()),
        jaccard_mean=("jaccard_overlap", "mean"),
        len_ratio_mean=("len_ratio", "mean"),
    ).round(4)
    summary.to_csv(os.path.join(BASE, out_dir_name, "summary_by_condition.tsv"), sep="\t")
    print(summary.to_string())

    # blinded rating sheet — identical rewrites across conditions are shown once;
    # the key maps each slot back to ALL conditions that produced that text,
    # so per-condition scores are recovered after rating.
    rng = random.Random(7)
    by_item = {}
    for r in results:
        if r.get("ok"):
            by_item.setdefault(r["sentence"], {})[r["condition"]] = r["rewrite"]
    sheet, key = [], []
    for i, (sent, conds) in enumerate(sorted(by_item.items()), 1):
        uniq = {}  # rewrite text -> [conditions]
        for cname in ("full", "sentence_flag", "none"):
            if cname in conds:
                uniq.setdefault(conds[cname], []).append(cname)
        texts = list(uniq.keys())
        rng.shuffle(texts)
        row = {"item": i, "original": sent}
        krow = {"item": i}
        for slot, text in zip("ABC", texts):
            row[f"rewrite_{slot}"] = text
            for crit in ("fairness", "meaning", "completeness"):
                row[f"{slot}_{crit}_1to7"] = ""
            krow[slot] = "+".join(uniq[text])
        sheet.append(row)
        key.append(krow)
    with open(os.path.join(BASE, out_dir_name, "rating_sheet.tsv"), "w", encoding="utf-8", newline="") as f:
        cols = ["item", "original"]
        for slot in "ABC":
            cols += [f"rewrite_{slot}"] + [f"{slot}_{c}_1to7" for c in ("fairness", "meaning", "completeness")]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(sheet)
    with open(os.path.join(BASE, out_dir_name, "rating_key.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "A", "B", "C"], delimiter="\t")
        w.writeheader()
        w.writerows(key)

    # rater instructions accompany every regenerated sheet
    instructions_src = os.path.join(BASE, "out_ablation", "README_raters.md")
    instructions_dst = os.path.join(BASE, out_dir_name, "README_raters.md")
    if os.path.exists(instructions_src) and instructions_src != instructions_dst:
        import shutil
        shutil.copyfile(instructions_src, instructions_dst)
    print("\nwrote rewrite_metrics.tsv, summary_by_condition.tsv, rating_sheet.tsv (+key, README_raters.md)")


if __name__ == "__main__":
    main()
