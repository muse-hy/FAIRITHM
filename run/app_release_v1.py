import os
import json
import re
import streamlit as st, json, os, base64
try:
    import paramiko  # only needed when SSH_HOST is configured (remote mode)
except ImportError:
    paramiko = None
from dotenv import load_dotenv
from streamlit.components.v1 import html
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

# Load .env from the current directory
load_dotenv(str((Path(__file__).resolve().parent / ".env")))

import pandas as pd
from kiwipiepy import Kiwi
import openai
from openai import OpenAI
kiwi = Kiwi()

# Initialize OpenAI API client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY is not set. Please check the .env file.")
    st.stop()
client = OpenAI(api_key=api_key)

# Persistent history file path (same directory as this script)
HISTORY_FILE = str((Path(__file__).resolve().parent / "history.jsonl"))

# Utilities
def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def svg_to_base64(path):
    with open(path, "rb") as svg_file:
        return base64.b64encode(svg_file.read()).decode()

# History utilities
def append_history(entry: Dict):
    try:
        dirpath = os.path.dirname(HISTORY_FILE)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        # Deduplicate: skip if signature matches within the recent tail of the file
        new_sig = build_history_signature(entry)
        is_dup = False
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "rb") as rf:
                    rf.seek(0, os.SEEK_END)
                    size = rf.tell()
                    rf.seek(max(0, size - 8192))  # check only the recent part
                    tail = rf.read().decode("utf-8", errors="ignore").splitlines()
                    for line in reversed(tail):
                        try:
                            rec = json.loads(line)
                            if build_history_signature(rec) == new_sig:
                                is_dup = True
                                break
                        except Exception:
                            continue
            except Exception:
                pass
        if is_dup:
            return
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        st.warning(f"Failed to write history: {e} ({HISTORY_FILE})")

def load_history(limit: Optional[int] = None) -> List[Dict]:
    records: List[Dict] = []
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        if limit:
            records = records[-limit:]
    except Exception as e:
        st.warning(f"Failed to load history: {e}")
    return records

# Clipboard copy button
def copy_button(label: str, text: str, key: str, width: str = '100%', bg: str = '#eaf2ff', fg: str = '#0b4da8', border: str = '#bfdbfe'):
    html(f"""
    <style>
      #{key} {{
        width:{width};
        padding:12px 16px;
        border:1px solid {border};
        border-radius:10px;
        background:{bg};
        color:{fg};
        cursor:pointer;
        display:inline-flex;align-items:center;justify-content:center;gap:8px;
        font-weight:700;
        box-shadow:0 1px 2px rgba(16,24,40,0.04);
        transition:background .15s ease, transform .05s ease;
      }}
      #{key}:hover {{ background:#eef5ff; }}
      #{key}:active {{ transform: translateY(1px); }}
    </style>
    <button id="{key}">{label}</button>
    <script>
      const btn_{key} = document.getElementById("{key}");
      if (btn_{key}) {{
        btn_{key}.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText({json.dumps(text)});
            const old = btn_{key}.innerText;
            btn_{key}.innerText = "✅ Copied";
            btn_{key}.style.opacity = '0.9';
            setTimeout(() => {{ btn_{key}.innerText = old; btn_{key}.style.opacity = '1'; }}, 1200);
          }} catch (e) {{
            console.error(e);
            alert("Failed to copy to clipboard");
          }}
        }});
      }}
    </script>
    """, height=56)

st.set_page_config(page_title="FAIRITHM", layout="wide")

# Current page
page = st.query_params.get("page", "FAIRITHM")

REMOTE_BIAS_SCRIPT = os.getenv(
    "REMOTE_BIAS_SCRIPT",
    "run_model.py",
)
REMOTE_CONDA_ENV = os.getenv("REMOTE_CONDA_ENV", "bias_39")
REMOTE_CONDA_INIT = os.getenv(
    "REMOTE_CONDA_INIT", "~/miniconda3/etc/profile.d/conda.sh"
)


def _run_local_bias_analysis(sentence: str):
    """Local mode: call src/run_model.py in-process (no SSH required).
    Models are loaded once per Streamlit process and reused."""
    import sys as _sys
    _root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("BASELINE_MODEL_PATH", str(_root / "models" / "kcbert_baseline"))
    os.environ.setdefault("RETRAINED_MODEL_PATH", str(_root / "models" / "best_model" / "best_model"))
    src_dir = str(_root / "src")
    if src_dir not in _sys.path:
        _sys.path.insert(0, src_dir)
    import run_model as _run_model
    return _run_model.compute_bias_pll_0to10(sentence)


def run_remote_bias_analysis(sentence: str):
    # Local mode when no SSH host is configured
    host = os.getenv("SSH_HOST")
    if not host:
        return _run_local_bias_analysis(sentence)
    if paramiko is None:
        raise RuntimeError("SSH_HOST is set but paramiko is not installed (pip install paramiko)")

    # Extract nouns via morphological analysis
    kiwi = Kiwi()
    tokens = kiwi.tokenize(sentence)
    nouns = [tok.form for tok in tokens if tok.tag in ("NNG", "NNP", "NP")]

    # Remote SSH settings (read from environment variables)
    user = os.getenv("SSH_USER")
    pw = os.getenv("SSH_PASSWORD")
    script = "run_model.py"

    # JSON payload composition
    payload_data = {"sentence": sentence}
    # Token list for legacy script compatibility, ignored in new scripts
    payload_data["tokens"] = nouns
    payload = json.dumps(payload_data, ensure_ascii=False)

    # Compose command using echo and pipe
    cmd = (
        f"echo '{payload}' | bash -c 'source {REMOTE_CONDA_INIT} && "
        f"conda activate {REMOTE_CONDA_ENV} && python {script}'"
    )



    # Connect via SSH and execute
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=pw)
    _, stdout, stderr = ssh.exec_command(cmd)
    out, err = stdout.read().decode(), stderr.read().decode()
    ssh.close()

    # Validate server response
    if not out.strip():
        raise RuntimeError(f"No server response\nstderr: {err}")

    # Parse JSON
    try:
        return json.loads(out.strip())
    except Exception as e:
        raise RuntimeError(f"Failed to parse response\nout: {out}\nerr: {err}")

# Decimal formatting function (to prevent None, NaN)
def safe_float(val, digits=2):
    try:
        return round(float(val), digits)
    except:
        return 0.0


def get_kcbert_section(resp: Dict) -> Dict:
    if isinstance(resp, dict):
        for key in ("kcbert상대점수", "kcbert_original"):
            section = resp.get(key)
            if isinstance(section, Dict):
                return section
        return resp if "편향 점수" in resp else {}
    return {}


def get_independent_section(resp: Dict) -> Dict:
    if isinstance(resp, dict):
        for key in ("독립평가점수", "independent_original"):
            section = resp.get(key)
            if isinstance(section, Dict):
                return section
    return {}

# Normalize text for comparison
def normalize_text_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# Build signature for history dedup
def build_history_signature(rec: Dict) -> tuple:
    mode = (rec.get("mode") or "").strip()
    level = (rec.get("level") or "").strip()
    inp = normalize_text_for_compare(rec.get("input", ""))
    corr = normalize_text_for_compare(rec.get("corrected", ""))
    return (mode, level, inp, corr)

# Normalize level display label (ko/en legacy → Standard/high)
def map_level_display(level: str) -> str:
    val = str(level or "").strip().lower()
    if val in ("기본", "약하게", "basic", "standard", "standart"):
        return "Standard"
    if val in ("강하게", "aggressive", "high"):
        return "high"
    return str(level or "")

# Styles
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    padding: 1rem;
}
.sidebar-logo {
    display: block;
    margin: 0 auto 5px auto;
    width: 80px;
}
.sidebar-title {
    text-align: center;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 12px;
}
.sidebar-sub {
    font-size: 13px;
    color: #9CA3AF;
    display: flex;
    justify-content: space-around;
    margin-bottom: 16px;
}
.sidebar-section {
    font-size: 15px;
    font-weight: 600;
    margin: 12px 0 6px 0;
}
.sidebar-divider {
    height: 1px;
    background: #ddd;
    margin: 16px 0;
}

/* 채팅 입력창 스타일링 */
div[data-testid="stChatInput"] {
    position: sticky !important;
    bottom: 0 !important;
    background: white !important;
    border-top: 1px solid #e0e0e0 !important;
    padding: 10px !important;
    z-index: 100 !important;
}

/* 채팅 메시지 컨테이너 */
div[data-testid="stChatMessage"] {
    margin-bottom: 10px !important;
}

/* 메인 컨테이너에 하단 패딩 */
.main .block-container {
    padding-bottom: 120px !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    logo_svg = svg_to_base64(str(Path(__file__).resolve().parent.parent / "icon" / "reshot-icon-flickr-PTXY7M2H6V.svg"))
    st.markdown(f'<img src="data:image/svg+xml;base64,{logo_svg}" class="sidebar-logo">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">FAIRITHM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">• 🔔 Notifications • 🌐 Language • 🧑‍💼 Profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Advanced Settings</div>', unsafe_allow_html=True)
    
    # Debiasing level
    correction_level = st.radio(
        "Debiasing Level",
        ["기본", "강하게"],
        index=0,
        horizontal=True,
        help="Standard: Preserve sentence structure while removing bias.\High: Remove as much bias as possible.",
        format_func=lambda x: "Standard" if x == "기본" else ("High" if x == "강하게" else x)
    )


    # Analysis options
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    
    detailed_analysis = st.checkbox("Detailed analysis", value=True, help="Run per-token bias analysis")
    
    highlight_changes = st.checkbox("Highlight changes", value=True, help="Highlight modified segments")
    
    show_alternatives = st.checkbox("Show alternatives", value=False, help="Provide alternative revisions")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">History Preview</div>', unsafe_allow_html=True)

    try:
        file_records = load_history(limit=50)
    except Exception:
        file_records = []
    session_records = st.session_state.get("history_records", [])
    # Merge file and session records, deduplicate, and sort by latest
    merged_raw = file_records + [r for r in session_records if r not in file_records]
    seen = set()
    merged = []
    for r in reversed(merged_raw):  # filter from latest to oldest
        sig = build_history_signature(r)
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(r)
    merged = list(reversed(merged))
    # Ensure latest-first by sorting by timestamp desc
    try:
        merged.sort(key=lambda r: r.get("ts", ""), reverse=True)
    except Exception:
        pass

    if merged:
        preview_records = merged[:5]
        for r in preview_records:
            ts = r.get("ts", "")
            mode = r.get("mode", "")
            level = r.get("level", "")
            display_level = map_level_display(level)
            inp = r.get("input", "")
            corr = r.get("corrected", "")
            inp_trim = inp if len(inp) <= 60 else inp[:59] + "…"
            corr_trim = corr if len(corr) <= 60 else corr[:59] + "…"
            st.markdown(
                f"""
                <div style='border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; margin: 6px 0; background: #fafafa;'>
                    <div style='font-size: 11px; color: #6b7280; margin-bottom: 4px;'>
                        {ts} · {mode} · {display_level}
                    </div>
                    <div style='font-size: 12px; color: #374151;'>
                        <strong>Original:</strong> {inp_trim}
                    </div>
                    <div style='font-size: 12px; color: #374151; margin-top: 2px;'>
                        <strong>Revised:</strong> {corr_trim}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        st.caption("See details in the History tab of Main.")
    else:
        st.info("No records to display.")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.caption("© 2025 Bias Assessment Toolkit v2 / Unlearners - KoBBQ based Bias Research Team – All rights reserved.")

# Main content
if page in ("FAIRITHM", "BIAS CHECKER"):
    # ──────────────────────────────────────────────────────────────
    #  BIAS CHECKER & CORRECTOR  (Header → Tabs → Two Panels)
    # ──────────────────────────────────────────────────────────────
    
    # Title
    st.title("FAIRITHM : Bias Checker & Corrector")
    
    # App description
    st.markdown("""
    You can check and correct bias in your text!
    
    """)
    
    # Tabs
    tab1, tab2 = st.tabs(["Main", "History"])
    
    with tab1:
        # Initialize session keys
        for k, v in {"bias_resp": None, "inference_ms": None}.items():
            st.session_state.setdefault(k, v)
        # Initialize history in session
        if "history_records" not in st.session_state:
            st.session_state.history_records = load_history(limit=500)

        # Default values
        threshold = 30
        gpt_model = "gpt-4o"

        # Two-column layout
        col_left, col_right = st.columns([1, 1], gap="large")

        # Left: Sentence input
        with col_left:
            st.subheader("📝  Input")
            sentence = st.text_area(
                label="",
                height=150,
                placeholder="Enter a sentence or paragraph to check for bias…\nExample: 저소득층은 기초 지식이 없어 생활이 어렵다.(Low-income people lack basic knowledge, making daily life difficult.)")
            
            run = st.button("🔍  Check bias & correct", type="primary", use_container_width=True)

            with st.expander("Show example sentences"):
                st.caption("Use the copy button to copy an example.")
                examples = [   
                    "Women lack the ability to think mathematically and often make mistakes in problem solving.",
                    "High school graduates lack basic intellectual ability and culture, so they can only handle simple tasks and will stay in unskilled jobs for life.",
                    "Older people do not understand new technology and find it difficult to adapt to digital environments.",
                    "Low-income people lack diligence and self-discipline, making it difficult to succeed.",
                ]
                for i, ex in enumerate(examples, 1):
                    st.markdown(f"{i}. {ex}")
                    copy_button("Copy", ex, key=f"ex_copy_{i}", bg="#ede9fe", fg="#6d28d9", border="#ddd6fe")

            if run:
                if not sentence.strip():
                    st.warning("⚠️ Please enter a sentence first."); st.stop()

                # Mark run in this session
                st.session_state.just_ran = True

                # If input is identical to a previously neutralized sentence, skip server call
                try:
                    last_corrected = None
                    if "history_records" in st.session_state and st.session_state.history_records:
                        last_corrected = st.session_state.history_records[-1].get("corrected", "")

                    # Collect recent corrected sentences and alternatives to block duplicate inputs
                    def _norm_dup(s: str) -> str:
                        try:
                            t = str(s)
                        except Exception:
                            t = ""
                        t = t.strip()
                        t = re.sub(r"^대안\s*\d+\s*:\s*", "", t)
                        t = re.sub(r"\s+", " ", t)
                        t = re.sub(r"[\-–—*•●◦○·∙⋅]+", "", t)
                        t = re.sub(r"[\.,!\?:;\"'“”‘’()\[\]{}]", "", t)
                        return t.lower()

                    recent_texts = set()
                    if "history_records" in st.session_state:
                        for rec in st.session_state.history_records[-20:]:
                            c = rec.get("corrected", "")
                            if c:
                                recent_texts.add(_norm_dup(c))
                            for alt in rec.get("alternatives", []) or []:
                                alt_text = alt.get("text", "") if isinstance(alt, dict) else str(alt)
                                if ":" in alt_text:
                                    alt_text = alt_text.split(":", 1)[1].strip()
                                recent_texts.add(_norm_dup(alt_text))

                    if recent_texts and _norm_dup(sentence) in recent_texts:
                        st.info("This sentence has already been neutralized.")
                        neutral_resp = {
                            "종합_편향점수": 0.0,
                            "편향 점수(0-100)": 0.0,
                            "상위 높은 확률 단어": [],
                            "상위 낮은 확률 단어": [],
                            "PLL": 0.0
                        }
                        st.session_state.bias_resp = {
                            "kcbert_original": neutral_resp,
                            "independent_original": neutral_resp
                        }
                        st.session_state.inference_ms = 0

                        cache_key = (sentence, correction_level, bool(show_alternatives))
                        st.session_state["main_cache"] = {
                            "key": cache_key,
                            "data": {
                                "corrected_sentence": sentence,
                                "alternatives": [],
                                "main_bias_type": "없음",
                                "bias_score": "0.0",
                                "correction_reason": "This sentence has already been neutralized."
                            }
                        }
                    else:
                        st.info("This demo may take ~10–20 seconds to run.")
                        with st.spinner("Running model inference…"):
                            import time
                            t0 = time.perf_counter()
                            try:
                                resp = run_remote_bias_analysis(sentence)
                                st.session_state.bias_resp = resp
                                st.session_state.inference_ms = int((time.perf_counter()-t0)*1000)
                                st.success("✅ Bias analysis completed!")
                            except Exception as e:
                                st.error(f"🚨 Server error: {e}"); st.stop()
                except Exception as e:
                    st.error(f"Error during pre-check: {e}")



        # Right: Result panel
        with col_right:
            st.subheader("🛠  Result")
            resp = st.session_state.bias_resp
            kcbert_section = get_kcbert_section(resp) if resp else {}
            independent_section = get_independent_section(resp) if resp else {}
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            
            if resp:
                # Determine correction instruction based on intensity
                def get_correction_instruction(level: str) -> str:
                    """편향 강도(기본/강하게)에 따른 수정 지침 반환"""
                    if level == "강하게":
                        return (
                            "편향 요소를 최대한 제거하세요. 과장/일반화/비하 표현을 중립적 사실 중심 표현으로 바꾸고,"
                            " 모호한 단정은 근거 가능한 범위로 일반화 수준을 낮추세요."
                        )
                    return (
                        "문장의 구조와 의미를 어느 정도 유지하면서 편향을 제거하세요."
                        " 과장/일반화/비하 표현은 중립적으로 완화하고, 불필요한 추가/삭제는 최소화하세요."
                    )
                
                correction_instruction = get_correction_instruction(correction_level)

                # Detailed rules and examples per level
                base_rules = (
                    "[수정 규칙 - 기본]\n"
                    "1. 문장의 구조와 의미를 어느 정도 유지합니다.\n"
                    "2. 과장/일반화/비하/고정관념 표현을 중립적이고 구체적인 표현으로 완화합니다.\n"
                    "3. 사실관계는 보존하고 불필요한 추가/삭제를 최소화합니다.\n"
                )
                base_examples = (
                    "[기본 모드 예시]\n"
                    "- 원문 : 충청도 출신은 우물쭈물하며 선택을 미루는 경향이 있다.\n"
                    "- 수정문 : 충청도 출신이 우물쭈물하며 선택을 미룬다는 인식은 지역에 대한 비하일 수 있다.\n\n"
                    "- 원문 : 어린이는 공공장소에서 예의를 지키는 법을 잘 배우지 못했다.\n"
                    "- 수정문 : 일부 어린이는 아직 공공장소에서 예절을 배우지 못할 수 있으므로 지도와 교육이 필요하다.\n\n"
                    "- 원문 : 기독교인은 종교적 믿음을 타인에게 강요한다.\n"
                    "- 수정문 : 일부 기독교인들은 기독교의 기본 교리가 포교이므로 주위 사람들에게 적극적으로 전하려고 한다.\n\n"
                    "- 원문 : 저소득층은 자녀 교육에 필요한 기본적인 관심이 부족하다.\n"
                    "- 수정문 : 저소득층은 경제적 여건이 어려워서 자녀 교육에 필요한 지원과 관심이 부족할 수 있다.\n"
                )

                strong_rules = (
                    "[수정 규칙 - 강하게]\n"
                    "1. 편향 요소를 최대한 제거합니다.\n"
                    "2. 과장/일반화/비하/고정관념 표현을 사실 중심의 중립 표현으로 전환합니다.\n"
                    "3. 문장 구조 보존은 우선순위가 아니며, 편향 제거가 최우선입니다.\n"
                )
                strong_examples = (
                    "[강하게 예시]\n"
                    "- 원문 : 여성은 논리적 사고가 약해 수학을 어려워한다.\n"
                    "- 수정문 : 성별에 따른 논리적 사고 능력 차이는 과학적 근거가 없다.\n\n"
                    "- 원문 : 고소득층은 사회적 통념에 대한 이해가 적다.\n"
                    "- 수정문 : 다양한 소득 계층은 사회적 통념에 대한 여러 가지 이해를 가질 수 있다.\n\n"
                    "- 원문 : 동성애자들은 정신적으로 문제가 있는 비정상적인 사람들이며 자신들의 왜곡된 성향을 사회에 강요하고 있어 건전한 사회 발전에 해가 된다.\n"
                    "- 수정문 : 동성애자들은 다양한 성적 지향을 가진 사람들로, 사회의 다양한 구성원 중 하나이다.\n"
                    "- 원문 : 자폐성 장애인은 정서적 반응을 표현하는 데 어려움이 있다.\n"
                    "- 수정문 : 자폐 스펙트럼을 가진 사람들은 정서적 반응을 표현하는 데 다양한 방식이 있을 수 있다.\n"
                )

                level_rules = base_rules + "\n" + base_examples if correction_level == "기본" else strong_rules + "\n" + strong_examples

                # Adjust prompt for alternatives option
                alternatives_instruction = ""
                response_format = """{
    "corrected_sentence": "수정된 중립적인 문장",
    "main_bias_type": "주요 편향 유형",
    "bias_score": "편향 점수",
    "correction_reason": "수정 근거 (간단히)"
}"""
                
                if show_alternatives:
                    alternatives_instruction = "메인 수정안 외에 2-3개의 다른 대안도 함께 제시해주세요."
                    response_format = """{
    "corrected_sentence": "수정된 중립적인 문장",
    "alternatives": [
        "대안 1: 다른 방식으로 수정된 문장",
        "대안 2: 또 다른 방식으로 수정된 문장",
        "대안 3: 세 번째 방식으로 수정된 문장"
    ],
    "main_bias_type": "주요 편향 유형",
    "bias_score": "편향 점수",
    "correction_reason": "수정 이유 (간단히)"
}"""

                # Cache key
                cache_key = (sentence, correction_level, bool(show_alternatives))
                cached = st.session_state.get("main_cache")
                if cached and cached.get("key") == cache_key:
                    gpt_analysis = cached.get("data", {})
                else:
                    # Use GPT-4o to produce JSON analysis and revised sentence
                    with st.spinner("The AI is analyzing results and revising the sentence…"):
                        try:
                            analysis_prompt = f"""
당신은 편향 분석 및 개선 전문가입니다. 사용자가 지정한 편향 수정 강도([{ '강하게' if correction_level == '강하게' else '약하게' }])에 맞춰, 강도별 목적과 지침을 “[목적 및 적용 지침]” 섹션에 삽입하고 그에 따라 문장을 중립적으로 수정하세요.입력문장이 영어인 경우 영어로 출력하세요.

[원문]
{sentence}

[서버 분석 결과(JSON)]
{json.dumps(resp, ensure_ascii=False, indent=2)}

[선택 강도]
{ '강하게' if correction_level == '강하게' else '약하게' }
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

{alternatives_instruction}

[출력 형식(JSON)]
{response_format}
"""
                            
                            gpt_response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": f"당신은 편향 분석 및 개선 전문가입니다. 사용자가 지정한 편향 수정 강도({correction_level})에 맞춰 강도별 목적과 지침을 적용해 원문을 중립적으로 수정하세요. 새로운 근거·통계·사실은 생성하지 말고, 단정적 표현은 지양하세요. '일 수 있습니다', '가능성이 있습니다', '할 수 있습니다' 같은 완곡한 설명형 어조를 사용하세요."},
                                    {"role": "user", "content": analysis_prompt}
                                ],
                                temperature=0.3,
                                max_tokens=500,
                                response_format={"type": "json_object"}
                            )
                            
                            # Parse JSON from GPT response
                            content = gpt_response.choices[0].message.content
                            try:
                                gpt_analysis = json.loads(content)
                            except Exception:
                                # Even in JSON mode, guard against malformed content
                                json_start = content.find('{')
                                json_end = content.rfind('}') + 1
                                if json_start != -1 and json_end != 0:
                                    json_str = content[json_start:json_end]
                                    gpt_analysis = json.loads(json_str)
                                else:
                                    gpt_analysis = {
                                        "corrected_sentence": sentence,
                                        "main_bias_type": "분석 실패",
                                        "bias_score": "0.0",
                                        "correction_reason": "GPT 분석 실패"
                                    }

                            # History save is handled in the common section below

                        except Exception as e:
                            st.error(f"Error during AI analysis: {e}")
                            gpt_analysis = {
                                "corrected_sentence": sentence,
                                "main_bias_type": "오류",
                                "bias_score": "0.0",
                                "correction_reason": "분석 오류 발생"
                            }
                    # Cache result
                    st.session_state["main_cache"] = {"key": cache_key, "data": gpt_analysis}

                # Save to history (main, common)
                try:
                    entry = {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "mode": "main",
                        "input": sentence,
                        "corrected": gpt_analysis.get("corrected_sentence", ""),
                        "alternatives": gpt_analysis.get("alternatives", []),
                        "level": correction_level,
                        "reason": gpt_analysis.get("correction_reason", ""),
                        "bias_type": gpt_analysis.get("main_bias_type", ""),
                        "bias_score": gpt_analysis.get("bias_score", ""),
                    }
                    append_history(entry)
                    # Reflect into session immediately (signature-based dedup)
                    if "history_records" not in st.session_state:
                        st.session_state.history_records = []
                    new_sig = build_history_signature(entry)
                    existing_sigs = {build_history_signature(r) for r in st.session_state.history_records[-50:]}
                    if new_sig not in existing_sigs:
                        st.session_state.history_records.append(entry)
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to save history: {e}")

                # Corrected sentence box
                # st.subheader("✨ 수정된 문장")
                corrected_sentence = gpt_analysis.get("corrected_sentence", sentence)
                
                # Highlight changes between original and corrected
                def highlight_changes(original, corrected):
                    """원문과 수정문을 비교하여 변경된 부분을 하이라이트"""
                    original_words = original.split()
                    corrected_words = corrected.split()
                    
                    highlighted_text = []
                    max_len = max(len(original_words), len(corrected_words))
                    
                    for i in range(max_len):
                        if i < len(corrected_words):
                            word = corrected_words[i]
                            # Highlight words not in original or at different positions
                            if i >= len(original_words) or word not in original_words or (i < len(original_words) and word != original_words[i]):
                                highlighted_text.append(f'<span style="background-color: #fef3c7; padding: 2px 4px; border-radius: 3px; font-weight: 600;">{word}</span>')
                            else:
                                highlighted_text.append(word)
                    
                    return ' '.join(highlighted_text)
                
                highlighted_sentence = highlight_changes(sentence, corrected_sentence)
                
                # Render output styled like the input box
                st.markdown(
                    f"""
                    <div style='
                        border: 1px solid #d1d5db;
                        border-radius: 0.375rem;
                        padding: 0.75rem;
                        background-color: #f9fafb;
                        min-height: 150px;
                        font-family: inherit;
                        font-size: 20px;
                        line-height: 1.6;
                        color: #374151;
                        margin-bottom: 1rem;
                    '>
                        {highlighted_sentence}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Clipboard copy button
                copy_button("📋 Copy revised sentence", corrected_sentence, key="copy_corrected_main")
                
                # Alternative revisions
                if show_alternatives and "alternatives" in gpt_analysis:
                    st.markdown("---")
                    st.subheader("🔄 Alternative revisions")
                    alternatives = gpt_analysis.get("alternatives", [])
                    
                    for i, alt in enumerate(alternatives, 1):
                        # Split optional prefix "N:" if present
                        if ":" in alt:
                            alt_content = alt.split(":", 1)[1].strip()
                        else:
                            alt_content = alt
                        
                        # Render each alternative in a small box
                        st.markdown(
                            f"""
                            <div style='
                                border: 1px solid #e0e0e0;
                                border-radius: 6px;
                                padding: 10px;
                                margin: 8px 0;
                                background-color: #f8f9fa;
                                font-size: 16px;
                                line-height: 1.5;
                            '>
                                <strong>Alternative {i}:</strong> {alt_content}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Copy button per alternative
                        copy_button(f"📋 Copy alternative {i}", alt_content, key=f"copy_alt_{i}")

                st.markdown("---")

                # Analysis summary
                st.subheader("📊 Analysis summary")
                
                # Extract token-level difference information (paper rule: use only positive PLL drop)
                token_diffs = kcbert_section.get("토큰별_차이", [])
                
                # Token-level positive PLL drop s_i = max(Baseline logP - Retrained logP, 0)
                pos_differences = []
                bias_tokens_list = []
                
                if token_diffs:
                    for td in token_diffs:
                        # Difference_logp = Baseline logP - Retrained logP
                        diff_logp = td.get("차이_logp", 0.0)
                        pos_diff = max(diff_logp, 0.0)
                        pos_differences.append(pos_diff)
                        
                        # Extract biased words (prioritize tokens with large differences, exclude subwords)
                        token_text = td.get("토큰", "")
                        # Exclude subwords starting with #
                        if token_text and not token_text.startswith("#") and token_text not in [bt["토큰"] for bt in bias_tokens_list]:
                            bias_tokens_list.append({
                                "토큰": token_text,
                                "차이": pos_diff
                            })
                    
                    # Sort in descending order of difference
                    bias_tokens_list.sort(key=lambda x: x["차이"], reverse=True)
                
                # Extract biased words (minimum 1, maximum 3) based on token differences
                bias_words_list = []
                if bias_tokens_list:
                    for bt in bias_tokens_list[:3]:  # Maximum 3 words
                        token_text = bt.get("토큰", "")
                        # sRemove and clean subwords
                        if token_text:
                            # Remove # 
                            cleaned_text = token_text.replace("#", "").strip()
                            if cleaned_text and cleaned_text not in bias_words_list:
                                bias_words_list.append(cleaned_text)
                
                # If no token differences exist, fallback to the existing method
                if not bias_words_list:
                    top_words_high = kcbert_section.get("상위 높은 확률 단어") or []
                    top_words_low = kcbert_section.get("상위 낮은 확률 단어") or kcbert_section.get("상위 편향 단어") or []
                    for word in (top_words_high + top_words_low)[:3]:
                        if isinstance(word, dict):
                            word_text = word.get("텍스트") or word.get("word") or ""
                        else:
                            word_text = str(word)
                        if word_text and word_text not in bias_words_list:
                            bias_words_list.append(word_text)
                
                # Binary bias determination (paper rule: top-k aggregated signal ≥ 0.2)
                # Sentence signal = average of top-k positive drops, k = max(3, ceil(0.2 * positive token count))
                import math as _math
                positive_only = [d for d in pos_differences if d > 0]
                if positive_only:
                    _k = max(3, _math.ceil(0.2 * len(positive_only)))
                    _topk = sorted(positive_only, reverse=True)[:_k]
                    aggregated_signal = sum(_topk) / len(_topk)
                else:
                    aggregated_signal = 0.0
                
                # Paper rule: aggregation signal threshold = 0.2
                BIAS_THRESHOLD = 0.2
                has_bias = aggregated_signal >= BIAS_THRESHOLD
                
                # Prepare biased word message
                if has_bias and bias_words_list:
                    flag_words = ", ".join(bias_words_list[:2])  # Maximum 3 words
                    bias_message = f"Potential bias words: {flag_words}"
                else:
                    bias_message = "No strong signal detected — this does not guarantee the sentence is bias-free."
                
                # Traffic light style: place Red and Green side by side
                # Set Red Flag
                red_active = has_bias
                red_bg = "#fee2e2" if red_active else "#f3f4f6"  # red-50 if active, gray-100 if inactive
                red_text = "#dc2626" if red_active else "#6b7280"  # red-600 if active, gray-500 if inactive
                red_border = "#b91c1c" if red_active else "#d1d5db"  # red-800 if active, gray-300 if inactive
                red_dot = "#dc2626" if red_active else "#9CA3AF"  # red-600 if active, gray-400 if inactive
                
                # Set Green Flag
                green_active = not has_bias
                green_bg = "#d1fae5" if green_active else "#f3f4f6"  # green-50 if active, gray-100 if inactive
                green_text = "#059669" if green_active else "#6b7280"  # green-600 if active, gray-500 if inactive
                green_border = "#047857" if green_active else "#d1d5db"  # green-800 if active, gray-300 if inactive
                green_dot = "#059669" if green_active else "#9CA3AF"  # green-600 if active, gray-400 if inactive
                
                # Place traffic lights side by side
                col_red, col_green = st.columns(2)
                
                with col_red:
                    st.markdown(
                        f"""
                        <div style='
                            background-color: {red_bg};
                            color: {red_text};
                            padding: 16px 20px;
                            border-radius: 8px;
                            border-left: 5px solid {red_border};
                            margin-bottom: 20px;
                            display: flex;
                            align-items: center;
                            gap: 12px;
                            font-weight: 600;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            min-height: 80px;
                        '>
                            <span style='display:inline-block;width:16px;height:16px;border-radius:50%;background:{red_dot};box-shadow:0 0 0 2px rgba(0,0,0,0.05)'></span>
                            <div style='flex: 1;'>
                                <div style='font-size: 16px; font-weight: 700; color: {red_text}; margin-bottom: 4px;'>
                                    Bias detected
                                </div>
                                <div style='font-size: 13px; color: {red_text}; opacity: 0.9;'>
                                    {bias_message if has_bias else "No strong signal detected"}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col_green:
                    st.markdown(
                        f"""
                        <div style='
                            background-color: {green_bg};
                            color: {green_text};
                            padding: 16px 20px;
                            border-radius: 8px;
                            border-left: 5px solid {green_border};
                            margin-bottom: 20px;
                            display: flex;
                            align-items: center;
                            gap: 12px;
                            font-weight: 600;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            min-height: 80px;
                        '>
                            <span style='display:inline-block;width:16px;height:16px;border-radius:50%;background:{green_dot};box-shadow:0 0 0 2px rgba(0,0,0,0.05)'></span>
                            <div style='flex: 1;'>
                                <div style='font-size: 16px; font-weight: 700; color: {green_text}; margin-bottom: 4px;'>
                                    No strong signal detected
                                </div>
                                <div style='font-size: 13px; color: {green_text}; opacity: 0.9;'>
                                    {bias_message if not has_bias else "Bias may be present"}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Disclaimer 
                st.caption(
                    "⚠️ **Note**: This indicator reflects changes in the model's internal probability distribution "
                    "rather than a direct measure of social harm. It represents the *possibility* of bias and may not "
                    "match the level of semantic or social bias perceived by human users. "
                    "A green result means no strong signal was detected — it does **not** certify that the sentence is free of bias."
                )
                
                # Additional details (collapsible)
                with st.expander("🔍 Detailed analysis"):
                    # Do not show per-token analysis; display GPT rationale only
                    correction_reason = gpt_analysis.get("correction_reason", "")
                    if correction_reason:
                        st.markdown("**Rationale for revision:**")
                        # Handle both string and list; exclude guide line
                        if isinstance(correction_reason, list):
                            lines = [str(x).strip() for x in correction_reason]
                        else:
                            lines = [ln.strip() for ln in str(correction_reason).splitlines()]
                        lines = [ln for ln in lines if ln and not ln.startswith("[목적 및 적용 지침]")]
                        # Strip existing bullets/indices and render as single bullets
                        cleaned = []
                        for ln in lines:
                            ln = re.sub(r"<[^>]+>", "", ln)
                            ln = re.sub(r"^\s*(?:[-–—*•●◦○·∙⋅]+|\d{1,2}[\.)]|[A-Za-z가-힣]\))\s*", "", ln)
                            cleaned.append(ln)
                        cleaned = [ln for ln in cleaned if ln]
                        bullet_md = "\n".join([f"- {ln}" for ln in cleaned])
                        st.markdown(bullet_md)
                
            else:
                st.info("👉 Enter a sentence on the left and click **Check Bias**.")

    
    with tab2:
        st.subheader("🗂️ History")
        st.caption("Review previous input/output records in chronological order.")

        # Filter/search options (simple)
        mode_filter = st.multiselect("Mode", ["main", "chat"], default=["main", "chat"], help="Filter by mode to display")
        level_filter = st.multiselect(
            "Intensity",
            ["기본", "강하게"],
            default=["기본", "강하게"],
            help="Select intensity levels to display",
            format_func=lambda x: "Standard" if x == "기본" else ("high" if x == "강하게" else x)
        )

        # Prioritize session state records, merge with file
        file_records = load_history(limit=500)
        session_records = st.session_state.get("history_records", [])
        merged = file_records + [r for r in session_records if r not in file_records]

        if merged:
            # Filtering
            filtered = [r for r in merged if r.get("mode") in mode_filter and r.get("level") in level_filter]
            if not filtered:
                st.info("No records to show. Try adjusting the filters.")
            else:
                for r in reversed(filtered):  # Latest on top
                    with st.container(border=True):
                        st.markdown(f"**Time**: {r.get('ts','')}  |  **Mode**: {r.get('mode','')}  |  **Intensity**: {map_level_display(r.get('level',''))}")
                        st.write(f"**Original:** {r.get('input','')}")
                        st.write(f"**Revised:** {r.get('corrected','')}")
                        alts = r.get("alternatives", [])
                        if alts:
                            with st.expander("Show alternatives"):
                                for i, alt in enumerate(alts, 1):
                                    st.write(f"Alternative {i}: {alt}")

                        cols = st.columns(2)
                        with cols[0]:
                            copy_button("📋 Copy original", r.get('input',''), key=f"hist_copy_in_{r.get('ts','')}")
                        with cols[1]:
                            copy_button("📋 Copy revised", r.get('corrected',''), key=f"hist_copy_out_{r.get('ts','')}")

            st.markdown("---")
            cols = st.columns(3)
            with cols[0]:
                if st.button("♻️ Refresh"):
                    st.rerun()
            with cols[1]:
                if st.button("🧹 Delete all"):
                    try:
                        if os.path.exists(HISTORY_FILE):
                            os.remove(HISTORY_FILE)
                            st.session_state.history_records = []
                            st.success("History deleted.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
            with cols[2]:
                try:
                    content = "".join(open(HISTORY_FILE, "r", encoding="utf-8").readlines()) if os.path.exists(HISTORY_FILE) else ""
                except Exception:
                    content = ""
                st.download_button("⬇️ Export JSONL", data=content, file_name="history.jsonl", mime="application/json")
        else:
            st.info("No records yet. Run something from Main or Chat first.")

        with st.expander("Debug: history file path"):
            st.code(HISTORY_FILE) 