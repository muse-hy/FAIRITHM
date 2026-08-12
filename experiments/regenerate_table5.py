# -*- coding: utf-8 -*-
"""
Regenerate manuscript Table 5 with the final PCGU checkpoint.

For each input sentence, computes word-level (subword-summed) log-probabilities
under the baseline and retrained models — the aggregation the original Table 5
used (verified: '부족해' = 부족 + ##해 subword sum reproduces the published
baseline value) — and reports:

  - the word with the largest positive PLL difference (baseline - retrained),
  - its baseline / retrained logP and difference,
  - the flag under the manuscript rule (word-level difference >= 0.2), and
  - the deployed run_model rule (top-k mean / 0.3 -> score01, levels 0.2/0.6)

so the revised table and the deployed system can be described consistently.

Usage:
  python experiments/regenerate_table5.py            # built-in candidates
  python experiments/regenerate_table5.py file.txt   # one sentence per line
Output: experiments/out_detection/table5_regenerated.tsv (+ console table)
"""
import math
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("BASELINE_MODEL_PATH", os.path.join(ROOT, "models", "kcbert_baseline"))
os.environ.setdefault("RETRAINED_MODEL_PATH", os.path.join(ROOT, "models", "best_model", "best_model"))

CANDIDATES = [
    "저소득층은 기초 지식이 없어 생활이 어렵다.",
    "청년층은 맡은 일을 끝까지 하지 않는다.",
    "북한주민은 정보 사회에서 소외되어 있다.",
    "인간은 숨을 쉰다.",
]


def word_logps(tokens, logps):
    """Merge subwords to words, summing logP per word (matches Table 5's scheme)."""
    words, cur_w, cur_s = [], "", 0.0

    def flush():
        nonlocal cur_w, cur_s
        w = cur_w.strip()
        if w and any(ch.isalnum() for ch in w):
            words.append((w, cur_s))
        cur_w, cur_s = "", 0.0

    for tok, s in zip(tokens, logps):
        if tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"):
            flush()
            continue
        if tok.startswith("##"):
            cur_w += tok[2:]
            cur_s += float(s)
            continue
        flush()
        cur_w, cur_s = tok, float(s)
    flush()
    return words


def main():
    import run_model
    from run_model import ensure_models, compute_token_logps_with_tokens

    sentences = CANDIDATES
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8-sig") as f:
            sentences = [ln.strip() for ln in f if ln.strip()]

    bl_m, bl_t, rt_m, rt_t, dev = ensure_models()
    rows = []
    for sent in sentences:
        toks_bl, idx_bl, lp_bl = compute_token_logps_with_tokens(sent, bl_m, bl_t, dev)
        toks_rt, idx_rt, lp_rt = compute_token_logps_with_tokens(sent, rt_m, rt_t, dev)
        toks = [toks_bl[i] for i in idx_bl]
        w_bl = word_logps(toks, lp_bl)
        w_rt = word_logps(toks, lp_rt)
        assert [w for w, _ in w_bl] == [w for w, _ in w_rt], f"word alignment failed: {sent}"

        # word with largest positive (baseline - retrained) difference
        best = max(
            ((w, b, r, b - r) for (w, b), (_, r) in zip(w_bl, w_rt)),
            key=lambda x: x[3],
        )
        word, b, r, d = best
        flag_manuscript = "Bias(red)" if d >= 0.2 else "Neutral(green)"

        # deployed rule (token-level top-k mean, run_model.py)
        deltas = [max(b_ - r_, 0.0) for b_, r_ in zip(lp_bl, lp_rt)]
        positives = [x for x in deltas if x > 0]
        if positives:
            k = max(3, int(math.ceil(0.2 * len(positives))))
            topk = sum(sorted(positives, reverse=True)[:k]) / k
        else:
            topk = 0.0
        score01 = min(topk / 0.3, 1.0)
        level = "편향 강함" if score01 >= 0.6 else ("편향 주의" if score01 >= 0.2 else "편향 없음")

        rows.append({
            "sentence": sent, "target_word": word,
            "baseline_logP": round(b, 4), "retrained_logP": round(r, 4),
            "difference_logP": round(d, 4), "flag_manuscript_rule": flag_manuscript,
            "deployed_score01": round(score01, 3), "deployed_level": level,
        })

    out = os.path.join(BASE, "out_detection", "table5_regenerated.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"{r['sentence']}\n  word={r['target_word']}  BL={r['baseline_logP']}  "
              f"RT={r['retrained_logP']}  diff={r['difference_logP']}  "
              f"-> {r['flag_manuscript_rule']} | deployed: {r['deployed_level']} ({r['deployed_score01']})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
