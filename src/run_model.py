# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Env setup (suppress tokenizer warnings; no grads)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_grad_enabled(False)

# Default model paths (workspace)
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = str(_REPO_ROOT / "models" / "kcbert_baseline")
DEFAULT_RETRAINED_PATH = str(_REPO_ROOT / "models" / "best_model")

BASELINE_MODEL_PATH = os.getenv("BASELINE_MODEL_PATH", DEFAULT_BASELINE_PATH)
RETRAINED_MODEL_PATH = os.getenv("RETRAINED_MODEL_PATH", DEFAULT_RETRAINED_PATH)

# -------------------- Model loading --------------------

_BL_MODEL = None
_BL_TOK = None
_RT_MODEL = None
_RT_TOK = None
_DEVICE = None


def _load_model_from_path(path: str) -> Tuple[AutoModelForMaskedLM, AutoTokenizer, torch.device]:
    """Load tokenizer and MLM model from a Hugging Face-style folder."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tok, device


def ensure_models():
    """Lazy-load baseline and retrained models once and reuse."""
    global _BL_MODEL, _BL_TOK, _RT_MODEL, _RT_TOK, _DEVICE
    if _BL_MODEL is None:
        _BL_MODEL, _BL_TOK, _DEVICE = _load_model_from_path(BASELINE_MODEL_PATH)
    if _RT_MODEL is None:
        _RT_MODEL, _RT_TOK, _DEVICE = _load_model_from_path(RETRAINED_MODEL_PATH)
    return _BL_MODEL, _BL_TOK, _RT_MODEL, _RT_TOK, _DEVICE

# -------------------- Pseudo Log-Likelihood (per-token avg) --------------------

def compute_pll_avg_with(text: str, model, tok, device) -> float:
    """Compute Salazar et al. (2020)-style PLL mean for a sentence."""
    encoded = tok(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0].to(device)
    attn = encoded["attention_mask"][0].to(device)
    # Token range excluding CLS/SEP
    token_indices = [i for i in range(1, input_ids.size(0) - 1) if attn[i].item() == 1]
    if not token_indices:
        return -1e9
    total_logp = 0.0
    for i in token_indices:
        masked = input_ids.clone()
        # Mask one token position
        mask_id = tok.mask_token_id if tok.mask_token_id is not None else tok.convert_tokens_to_ids(tok.mask_token)
        masked[i] = mask_id
        with torch.no_grad():
            logits = model(input_ids=masked.unsqueeze(0)).logits[0, i]
            logp = torch.log_softmax(logits, dim=-1)[input_ids[i]].item()
        total_logp += logp
    return total_logp / float(len(token_indices))

# Token-level log-prob sequence

def compute_token_logps_with_tokens(text: str, model, tok, device):
    """Mask each token and collect log-prob of the gold token.
    Returns: tokens (list[str]), token_indices (list[int]), logps (list[float])
    """
    encoded = tok(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0].to(device)
    attn = encoded["attention_mask"][0].to(device)
    tokens = tok.convert_ids_to_tokens(input_ids)
    token_indices = [i for i in range(1, input_ids.size(0) - 1) if attn[i].item() == 1]
    logps: list = []
    for i in token_indices:
        masked = input_ids.clone()
        mask_id = tok.mask_token_id if tok.mask_token_id is not None else tok.convert_tokens_to_ids(tok.mask_token)
        masked[i] = mask_id
        with torch.no_grad():
            logits = model(input_ids=masked.unsqueeze(0)).logits[0, i]
            logp = torch.log_softmax(logits, dim=-1)[input_ids[i]].item()
        logps.append(float(logp))
    return tokens, token_indices, logps

# Word-level aggregation from subwords

def aggregate_word_scores(tokens_subset, scores):
    """Merge subword tokens (WordPiece/SentencePiece) to words by summing scores.
    Returns: list of (word, score)
    """
    words = []
    current_word = ""
    current_score = 0.0

    def flush():
        nonlocal current_word, current_score
        w = current_word.strip()
        if w:
            # Skip pure punctuation
            if not re.fullmatch(r"\W+", w):
                words.append((w, current_score))
        current_word = ""
        current_score = 0.0

    for tok, s in zip(tokens_subset, scores):
        if tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"):
            flush();
            continue
        # SentencePiece word start
        if tok.startswith("▁"):
            flush()
            current_word = tok.lstrip("▁")
            current_score = float(s)
            continue
        # WordPiece continuation
        if tok.startswith("##"):
            current_word += tok[2:]
            current_score += float(s)
            continue
        # Regular token: start a new word
        flush()
        current_word = tok
        current_score = float(s)
    flush()
    # Sort by score desc
    words.sort(key=lambda x: x[1], reverse=True)
    return words

# -------------------- Bias score (BL vs RT PLL gap -> 0..10) --------------------

def compute_bias_pll_0to10(sentence: str) -> Dict:
    """
    Compare per-token log-prob (PLL) between BL and RT models for one sentence.
    - s_i = max(logp_BL(i) - logp_RT(i), 0)
    - score01 = mean of top-k positive s_i (k >= 3 or top 20%)
    - Normalize by tau and map to 0..10
    - Return explanatory top words/segments as well
    """
    bl_model, bl_tok, rt_model, rt_tok, device = ensure_models()

    # Whole-sentence PLL (reference only)
    pll_bl_orig = compute_pll_avg_with(sentence, bl_model, bl_tok, device)
    pll_rt_orig = compute_pll_avg_with(sentence, rt_model, rt_tok, device)

    # Token-level log-prob sequences
    tokens_bl, indices_bl, logps_bl = compute_token_logps_with_tokens(sentence, bl_model, bl_tok, device)
    tokens_rt, indices_rt, logps_rt = compute_token_logps_with_tokens(sentence, rt_model, rt_tok, device)

    # Safety: align common token positions
    token_positions = [i for i in indices_bl if i in set(indices_rt)]
    if not token_positions:
        score01 = 0.0
        level = "편향 없음"
        return {
            "문장": sentence,
            "편향 점수(0-10)": 0.0,
            "편향 점수": 0.0,
            "최고 종합 편향 점수": 0.0,
            "편향 단계": level,
            "원문_PLL_baseline": round(float(pll_bl_orig), 4),
            "원문_PLL_retrained": round(float(pll_rt_orig), 4),
            "가장 편향된 단어": "",
            "상위 편향 단어": [],
            "개별 분석 결과": []
        }

    # Reorder arrays to common positions
    pos_to_idx_bl = {pos: j for j, pos in enumerate(indices_bl)}
    pos_to_idx_rt = {pos: j for j, pos in enumerate(indices_rt)}
    tokens_common = [tokens_bl[pos] for pos in token_positions]
    logps_bl_common = [logps_bl[pos_to_idx_bl[pos]] for pos in token_positions]
    logps_rt_common = [logps_rt[pos_to_idx_rt[pos]] for pos in token_positions]

    # Per-token bias signal s_i
    deltas = [max(b - r, 0.0) for b, r in zip(logps_bl_common, logps_rt_common)]

    # Merge to words (drop special/punct before merging via regex in aggregator)
    words_scored = aggregate_word_scores(tokens_common, deltas)

    # Top problem words (max 5)
    top_words = [{"텍스트": w, "점수": round(float(s), 4)} for w, s in words_scored[:5]]
    most_biased_word = top_words[0]["텍스트"] if top_words else ""

    # Aggregate score: mean of top-k positives
    positives = [d for d in deltas if d > 0]
    if not positives:
        mean_topk = 0.0
    else:
        import math as _math
        k = max(3, int(_math.ceil(0.2 * len(positives))))
        positives_sorted = sorted(positives, reverse=True)
        mean_topk = sum(positives_sorted[:k]) / float(k)

    # Conservative normalization without dataset calibration
    tau = 0.3  # if mean gap >= 0.3, tends to "strong"
    score01 = max(0.0, min(mean_topk / tau, 1.0))
    score10 = round(score01 * 10.0, 2)

    # Level mapping
    if score01 >= 0.6:
        level = "편향 강함"
    elif score01 >= 0.2:
        level = "편향 주의"
    else:
        level = "편향 없음"

    token_detail = [
        {
            "토큰": t,
            "Baseline_logp": round(float(b), 4),
            "Retrained_logp": round(float(r), 4),
            "차이_logp": round(float(b - r), 4),
        }
        for t, b, r in zip(tokens_common, logps_bl_common, logps_rt_common)
    ]

    return {
        "문장": sentence,
        "토큰별_차이": token_detail,
        "편향 점수(0-10)": score10,
        "편향 점수": round(float(score01), 4),
        "최고 종합 편향 점수": round(float(score01), 4),
        "편향 단계": level,
        # PLL for reference (original sentence)
        "원문_PLL_baseline": round(float(pll_bl_orig), 4),
        "원문_PLL_retrained": round(float(pll_rt_orig), 4),
        # Where the models diverge most
        "가장 편향된 단어": most_biased_word,
        "상위 편향 단어": top_words,
        # Compatibility field (can be ignored by the app)
        "개별 분석 결과": []
    }

# -------------------- CLI entry point --------------------
if __name__ == "__main__":
    # Input: JSON via argv or stdin
    payload = None
    if len(sys.argv) >= 2:
        payload = sys.argv[1]
    if payload is None:
        try:
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                payload = sys.stdin.read().strip()
        except Exception:
            pass
    if not payload:
        print(json.dumps({"error": "JSON 입력이 필요합니다", "hint": "{\"sentence\":\"...\"}"}, ensure_ascii=False))
        sys.exit(1)
    try:
        data = json.loads(payload)
        sent = data.get("sentence") or data.get("문장")
        if not sent:
            raise ValueError("'sentence' 키가 필요합니다")
        res = compute_bias_pll_0to10(sent)
        print(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1) 