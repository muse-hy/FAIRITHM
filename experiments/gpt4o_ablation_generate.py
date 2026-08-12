# -*- coding: utf-8 -*-
"""
GPT-4o rewriting ablation for FAIRITHM revision (R1-C4, R2-C1, Q3, R1 token-feedback
ablation interpretation (a)).

Generates debiased rewrites for a stimulus set under three conditions using the
EXACT production prompt from run/app_en.py (gpt-4o, temperature=0.3, max_tokens=500,
JSON mode), varying only the bias-analysis input:

  full          : production prompt as deployed — sentence-level flag + token-level
                  analysis JSON from PCGU run_model.py + primary biased word
  sentence_flag : analysis JSON reduced to sentence-level fields only
                  (편향 점수/편향 단계); token-level fields and the
                  [주요 편향 대상 단어] section removed
  none          : GPT-4o-only — the [서버 분석 결과(JSON)] and
                  [주요 편향 대상 단어] sections removed entirely

Inputs:
  --stimuli  TSV with columns: sentence, category  (biased sentences)
  --analysis JSONL produced by precompute_analysis.py (run_model output per sentence);
             required for conditions full/sentence_flag
  --level    강하게|약하게 (debiasing level; default 약하게 = Standard)

Output: JSONL with one row per (sentence, condition): rewrite + metadata.
Requires OPENAI_API_KEY in env or run/.env.
"""
import argparse
import csv
import json
import os
import sys
import time

PROMPT_TEMPLATE = """
당신은 편향 분석 및 개선 전문가입니다. 사용자가 지정한 편향 수정 강도([{level}])에 맞춰, 강도별 목적과 지침을 “[목적 및 적용 지침]” 섹션에 삽입하고 그에 따라 문장을 중립적으로 수정하세요.입력문장이 영어인 경우 영어로 출력하세요.

[원문]
{sentence}
{analysis_block}
[선택 강도]
{level}
# "강하게" 또는 "약하게" 중 택1

[강도별 목적 및 지침 정의]
 - (강하게) 목적: 편향을 최대한 제거하고, 전문가가 쓴 것처럼 구체적이고 신뢰성 있는 어휘로 표현합니다.
지침: 문장의 원래 구조와 어휘에 얽매이지 말고, 과장·일반화·비하·고정관념 표현을 사실 기반 중립 표현으로 바꾸세요.
근거 없는 단정은 삭제하거나 불확실성 표지(‘일부’, ‘가능성이 있다’)로 조정합니다.
필요 시 구조를 대폭 변경해도 되며, 핵심 의미(주제·사실관계)는 유지해야 합니다.
새로운 근거·통계·사실은 생성하지 않습니다.
 - (약하게) 목적: 사용자의 원문 구조와 흐름을 최대한 보존하면서, 필요한 부분만 편향을 완화합니다.
지침: 문장 전체의 구조와 어휘를 유지하되, 과장·일반화·비하 표현을 중립적·구체적 표현으로 바꾸세요.
불필요한 내용 삭제는 최소화하고, 단정은 범위를 축소하거나 조건부 표현으로 전환합니다.
원 의미와 정보 흐름은 그대로 유지하며, 새로운 사실 주장은 생성하지 않습니다.

[규칙 - 공통]
1) 원 의미 보존: 주제·사실관계를 바꾸지 않음.
2) 근거 없는 단정 축소: '항상/전부/절대' → '일부/경향/상황에 따라' 등으로 완화.
3) 비하·고정관념 제거: 집단 일반화·낙인 표현 제거.
4) 모호성 처리: 조건·맥락·주체를 명시하거나 중립적 서술로 변환.
5) 민감 속성 존중: 인종·성별·성적지향·장애·종교·연령·지역 관련 판단은 가치평가 대신 사실 서술.
6) 금지사항: 피해자 책임 전가, 혐오 조장, 의학·법률적 확정 진단/판단 생성 금지.

[요청]
- 아래 출력 형식(JSON)만 출력하세요. 설명/머리말/코드블록 금지.
- 새로운 근거·통계·사실은 생성하지 마세요.
- "correction_reason"는 불릿포인트(- ) 3개로 간결하게 서술하세요. 각 항목은 왜 그런지 짧게 설명하고, 단정형 대신 완곡한 표현을 사용하세요.
- 말투 지침: "가능성·완곡 표현을 사용하고, 단정형 "~입니다"는 피하세요. 반드시 입력문장이 존대인 경우 존대를 사용하고, 하다체인 경우 하다체로 답하세요.
- 예시는 표시하지 말고 실제 문장 수정 근거만 나열하세요.

[출력 형식(JSON)]
{{
    "corrected_sentence": "수정된 중립적인 문장",
    "main_bias_type": "주요 편향 유형",
    "bias_score": "편향 점수",
    "correction_reason": "수정 근거 (간단히)"
}}
"""

SYSTEM_TEMPLATE = (
    "당신은 편향 분석 및 개선 전문가입니다. 사용자가 지정한 편향 수정 강도({level})에 맞춰 "
    "강도별 목적과 지침을 적용해 원문을 중립적으로 수정하세요. 새로운 근거·통계·사실은 생성하지 말고, "
    "단정적 표현은 지양하세요. '일 수 있습니다', '가능성이 있습니다', '할 수 있습니다' 같은 "
    "완곡한 설명형 어조를 사용하세요."
)

SENTENCE_LEVEL_KEYS = ["문장", "편향 점수(0-10)", "편향 점수", "편향 단계"]


def build_analysis_block(condition, analysis):
    if condition == "none":
        return "\n"
    if condition == "sentence_flag":
        reduced = {k: analysis[k] for k in SENTENCE_LEVEL_KEYS if k in analysis}
        return ("\n[서버 분석 결과(JSON)]\n"
                + json.dumps(reduced, ensure_ascii=False, indent=2) + "\n")
    # full — mirror app_en.py
    primary = analysis.get("가장 편향된 단어", analysis.get("편향 대상 단어", ""))
    return ("\n[서버 분석 결과(JSON)]\n"
            + json.dumps(analysis, ensure_ascii=False, indent=2)
            + f"\n\n[주요 편향 대상 단어]\n{primary}를 참고하세요.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True, help="TSV: sentence, category")
    ap.add_argument("--analysis", help="JSONL of run_model.py outputs keyed by sentence")
    ap.add_argument("--level", default="약하게", choices=["약하게", "강하게"])
    ap.add_argument("--conditions", default="full,sentence_flag,none")
    ap.add_argument("--out", default=os.path.join("experiments", "out_ablation", "rewrites.jsonl"))
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join("run", ".env"))
    except ImportError:
        pass
    from openai import OpenAI
    client = OpenAI()

    analysis_by_sentence = {}
    if args.analysis:
        with open(args.analysis, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    analysis_by_sentence[a["문장"]] = a

    stimuli = []
    with open(args.stimuli, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stimuli.append(row)

    conditions = args.conditions.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):  # resume support
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["sentence"], r["condition"]))

    with open(args.out, "a", encoding="utf-8") as fout:
        for row in stimuli:
            sent = row["sentence"]
            for cond in conditions:
                if (sent, cond) in done:
                    continue
                if cond in ("full", "sentence_flag"):
                    if sent not in analysis_by_sentence:
                        print(f"SKIP (no analysis): {cond} | {sent[:40]}", file=sys.stderr)
                        continue
                    analysis = analysis_by_sentence[sent]
                else:
                    analysis = None
                prompt = PROMPT_TEMPLATE.format(
                    level=args.level, sentence=sent,
                    analysis_block=build_analysis_block(cond, analysis or {}),
                )
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_TEMPLATE.format(level=args.level)},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {"corrected_sentence": None, "raw": content}
                fout.write(json.dumps({
                    "sentence": sent,
                    "category": row.get("category", ""),
                    "condition": cond,
                    "level": args.level,
                    "rewrite": parsed.get("corrected_sentence"),
                    "response": parsed,
                    "model": "gpt-4o",
                    "temperature": 0.3,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"OK {cond} | {sent[:40]}...")
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
