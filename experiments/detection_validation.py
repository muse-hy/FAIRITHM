# -*- coding: utf-8 -*-
"""
Detection-validation experiment for FAIRITHM revision (R1-C3, R2-C2, Q1, and the
KcBERT-without-PCGU detection ablation part of R1-C4/R2-C1).

Computes, for every sentence in the labeled test set:
  - token-level PLL under the pre-unlearning (baseline) and post-unlearning
    (retrained) models,
  - sentence-level detection features derived from the PLL difference,
then evaluates each feature as a binary bias detector against the triplet
labels (bias = positive; counter-stereotype and neutral = negative):
  precision / recall / F1 / accuracy / FPR at a threshold sweep,
  ROC-AUC, F1-optimal and Youden-J-optimal thresholds,
  per-category AUC, and false-positive breakdown by sentence role.

Features evaluated:
  max_delta   : max over tokens of (logP_baseline - logP_retrained)  [Table 5 rule]
  topk_mean   : mean of top-k positive per-token deltas, k = max(3, ceil(0.2*n_pos))
                (the deployed run_model.py scoring rule; score01 = topk_mean / 0.3)
  pll_absdiff : |sentence-mean PLL baseline - retrained|
  pll_bl_neg  : -(sentence-mean PLL under baseline only)   [single-model ablation]
  pll_rt_neg  : -(sentence-mean PLL under retrained only)  [single-model ablation]

Usage:
  python experiments/detection_validation.py \
      --baseline models/kcbert_baseline \
      --retrained models/best_model/best_model \
      --test data/test_set.tsv \
      --out experiments/out_detection
Smoke test (no models needed): add --dry-run
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROLES = ["bias", "counter", "neutral"]


def load_triplets(tsv_path):
    """Group rows by 'occ' preserving file order, chunk into triplets.
    Within each triplet the order is [bias, counter, neutral] (cf. src/eval.py)."""
    groups = defaultdict(list)
    with open(tsv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            groups[row["occ"]].append(row)
    records = []
    for occ, rows in groups.items():
        assert len(rows) % 3 == 0, f"group {occ} has {len(rows)} rows (not divisible by 3)"
        for i in range(0, len(rows), 3):
            for k in range(3):
                r = rows[i + k]
                records.append({
                    "sentence": r["sentence"],
                    "role": ROLES[k],
                    "label": 1 if k == 0 else 0,
                    "occ": occ,
                    "category": re.sub(r"_[0-9a-f]{6}$", "", occ),
                    "set_id": r.get("set_id", ""),
                    "cycle": r.get("cycle", ""),
                })
    return records


# ---------------- PLL computation (batched: all masked variants in one forward) ----


def token_logps(model, tok, sentence, device, max_batch=64):
    """Return per-token log-probs of the gold token, masking one position at a time.
    Batches all masked variants of the sentence for speed."""
    import torch
    enc = tok(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"][0]
    n = input_ids.size(0)
    positions = list(range(1, n - 1))  # exclude CLS/SEP
    if not positions:
        return [], []
    mask_id = tok.mask_token_id
    logps = []
    with torch.no_grad():
        for start in range(0, len(positions), max_batch):
            chunk = positions[start:start + max_batch]
            batch = input_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
            for j, pos in enumerate(chunk):
                batch[j, pos] = mask_id
            out = model(input_ids=batch.to(device)).logits
            for j, pos in enumerate(chunk):
                lp = torch.log_softmax(out[j, pos], dim=-1)[input_ids[pos]].item()
                logps.append(lp)
    tokens = tok.convert_ids_to_tokens(input_ids)
    return [tokens[p] for p in positions], logps


def sentence_features(logps_bl, logps_rt):
    """Sentence-level detection features from aligned per-token log-probs."""
    deltas = [b - r for b, r in zip(logps_bl, logps_rt)]
    positives = [d for d in deltas if d > 0]
    if positives:
        k = max(3, int(math.ceil(0.2 * len(positives))))
        topk_mean = float(np.mean(sorted(positives, reverse=True)[:k]))
    else:
        topk_mean = 0.0
    pll_bl = float(np.mean(logps_bl)) if logps_bl else -1e9
    pll_rt = float(np.mean(logps_rt)) if logps_rt else -1e9
    return {
        "pll_bl": pll_bl,
        "pll_rt": pll_rt,
        "max_delta": float(max(deltas)) if deltas else 0.0,
        "topk_mean": topk_mean,
        "score01": min(topk_mean / 0.3, 1.0),
        "pll_absdiff": abs(pll_bl - pll_rt),
    }


# ---------------- Metrics ----------------


def confusion_at(scores, labels, thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "threshold": round(float(thr), 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4), "recall": round(rec, 4), "F1": round(f1, 4),
        "accuracy": round((tp + tn) / len(labels), 4),
        "FPR": round(fp / (fp + tn), 4) if fp + tn else 0.0,
    }


def evaluate_feature(name, scores, labels, fixed_thresholds, out_dir, roles, categories):
    from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    auc = roc_auc_score(labels, scores)
    fpr, tpr, roc_thr = roc_curve(labels, scores)
    youden_thr = float(roc_thr[int(np.argmax(tpr - fpr))])
    prec_c, rec_c, pr_thr = precision_recall_curve(labels, scores)
    f1_c = 2 * prec_c[:-1] * rec_c[:-1] / np.clip(prec_c[:-1] + rec_c[:-1], 1e-12, None)
    f1_thr = float(pr_thr[int(np.argmax(f1_c))])

    # sweep grid: fixed values + negative-distribution percentiles + optimal points
    neg = scores[labels == 0]
    pct_grid = [float(np.percentile(neg, p)) for p in (50, 75, 90, 95, 99)]
    grid = sorted(set(round(t, 4) for t in list(fixed_thresholds) + pct_grid + [youden_thr, f1_thr]))
    sweep = [confusion_at(scores, labels, t) for t in grid]

    # false positives by role (counter vs neutral) at each fixed threshold
    roles = np.asarray(roles)
    fp_by_role = {}
    for t in fixed_thresholds:
        pred = (scores >= t).astype(int)
        d = {}
        for role in ("counter", "neutral"):
            m = roles == role
            d[role] = round(float(pred[m & (labels == 0)].mean()), 4) if m.any() else None
        fp_by_role[str(t)] = d

    # per-category AUC
    categories = np.asarray(categories)
    cat_auc = {}
    for c in sorted(set(categories)):
        m = categories == c
        if len(set(labels[m])) == 2:
            cat_auc[c] = round(float(roc_auc_score(labels[m], scores[m])), 4)

    with open(os.path.join(out_dir, f"sweep_{name}.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(sweep)

    return {
        "feature": name,
        "ROC_AUC": round(float(auc), 4),
        "youden_threshold": round(youden_thr, 4),
        "f1_optimal_threshold": round(f1_thr, 4),
        "best_F1": round(float(np.max(f1_c)), 4),
        "fp_rate_by_role_at_fixed_thresholds": fp_by_role,
        "per_category_AUC": cat_auc,
        "sweep_table": f"sweep_{name}.tsv",
    }


# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="beomi/kcbert-base")
    ap.add_argument("--retrained", default=os.path.join("models", "best_model", "best_model"))
    ap.add_argument("--test", default=os.path.join("data", "test_set.tsv"))
    ap.add_argument("--out", default=os.path.join("experiments", "out_detection"))
    ap.add_argument("--limit", type=int, default=0, help="limit #triplets for smoke test")
    ap.add_argument("--dry-run", action="store_true", help="random features; no models")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    records = load_triplets(args.test)
    if args.limit:
        records = records[: args.limit * 3]
    print(f"Loaded {len(records)} sentences ({len(records)//3} triplets)")

    if args.dry_run:
        rng = np.random.default_rng(0)
        for r in records:
            base = 0.25 if r["label"] else 0.12
            feats = {
                "pll_bl": -5.0, "pll_rt": -5.2,
                "max_delta": max(rng.normal(base + 0.15, 0.1), 0),
                "topk_mean": max(rng.normal(base, 0.08), 0),
                "pll_absdiff": max(rng.normal(base * 0.6, 0.05), 0),
            }
            feats["score01"] = min(feats["topk_mean"] / 0.3, 1.0)
            r.update(feats)
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        models = {}
        for tag, path in (("bl", args.baseline), ("rt", args.retrained)):
            print(f"Loading {tag}: {path}")
            tok = AutoTokenizer.from_pretrained(path)
            mdl = AutoModelForMaskedLM.from_pretrained(path).eval().to(device)
            models[tag] = (mdl, tok)
        from tqdm import tqdm
        for r in tqdm(records, desc="PLL"):
            (bl_m, bl_t), (rt_m, rt_t) = models["bl"], models["rt"]
            toks_bl, lp_bl = token_logps(bl_m, bl_t, r["sentence"], device)
            toks_rt, lp_rt = token_logps(rt_m, rt_t, r["sentence"], device)
            n = min(len(lp_bl), len(lp_rt))  # same tokenizer family; guard anyway
            r.update(sentence_features(lp_bl[:n], lp_rt[:n]))

    # persist per-sentence features
    feat_cols = ["pll_bl", "pll_rt", "max_delta", "topk_mean", "score01", "pll_absdiff"]
    with open(os.path.join(args.out, "features.tsv"), "w", encoding="utf-8", newline="") as f:
        cols = ["sentence", "role", "label", "occ", "category", "set_id", "cycle"] + feat_cols
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    labels = [r["label"] for r in records]
    roles = [r["role"] for r in records]
    cats = [r["category"] for r in records]
    fixed = [0.1, 0.15, 0.2, 0.25, 0.3]
    summary = {"n_sentences": len(records), "n_triplets": len(records) // 3, "features": []}
    # difference-based detectors + single-model ablation detectors
    feature_defs = {
        "max_delta": [r["max_delta"] for r in records],
        "topk_mean": [r["topk_mean"] for r in records],
        "pll_absdiff": [r["pll_absdiff"] for r in records],
        "pll_bl_neg": [-r["pll_bl"] for r in records],
        "pll_rt_neg": [-r["pll_rt"] for r in records],
    }
    for name, scores in feature_defs.items():
        res = evaluate_feature(name, scores, labels, fixed, args.out, roles, cats)
        summary["features"].append(res)
        print(f"{name}: AUC={res['ROC_AUC']}  bestF1={res['best_F1']} "
              f"(thr={res['f1_optimal_threshold']})  youden_thr={res['youden_threshold']}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}/features.tsv, summary.json, sweep_*.tsv")


if __name__ == "__main__":
    main()
