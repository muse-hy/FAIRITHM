# FAIRITHM

Korean text bias detection and correction system using PCGU machine unlearning.

FAIRITHM separates bias handling into two inspectable stages: a PCGU-unlearned
KcBERT assessment component that scores potentially biased tokens and sentences,
and a GPT-4o rewriting module that generates editable debiased alternatives from
that analysis.

## Repository layout

```
data/         KoBBQ-derived triplet dataset (train/test splits, occupation stats)
user_study/   De-identified questionnaire data from the N = 34 user study (with codebook)
src/          Training, evaluation, and scoring code
run/          Streamlit demo apps (v1 = study version, v2 = 3-tier post-study UI)
experiments/  Validation and ablation scripts reported in the paper
models/       Model weights directory (see models/README.md — weights distributed separately)
icon/         App logo asset
```

## Detection rule (as studied)

For each token, the signed positive PLL drop is computed as
`s_i = max(logP_baseline − logP_retrained, 0)`. The sentence-level signal is the
mean of the top-k positive drops with `k = max(3, ceil(0.2 × n_positive))`.
A red flag fires when this aggregated signal reaches **0.2**. The `τ = 0.3`
constant in `run_model.py` only normalizes the 0–1 display score and does not
affect the flag decision.

## Dataset

KoBBQ-derived sentence triplets (biased / counter-stereotypical / unrelated),
grouped by 12 bias subcategories:

- `data/train_set.tsv` — 10,176 training sentences (3,392 triplets)
- `data/test_set.tsv`  — 2,544 test sentences (848 triplets; templates disjoint from training)
- `data/kobbq_stats.tsv` — per-occupation bias statistics (occ → pct_disadv/pct_adv)

Note: the triplet labels denote roles in the contrastive schema inherited from
BBQ/KoBBQ (stereotype-congruent / counter-stereotypical / unrelated), not
human annotations of perceived bias. See the paper for the implications when
these role labels are repurposed for detector-style classification metrics.

## User-study data

`user_study/FAIRITHM_questionnaire_score.csv` contains the de-identified
questionnaire responses from the N = 34 user study (ABPS, manual-rewriting
workload measures, CABRS, post-use perceptions, and SUS; one row per
participant, 179 columns). Demographic variables and interview transcripts are
not included. The accompanying codebook
(`user_study/README_FAIRITHM_questionnaire_score.md`) documents every variable,
its coding, and the derived composite scores; all aggregate statistics reported
in the paper (e.g., SUS = 72.7, RTLX composite = 4.76, CABRS Part 3 choice
distributions) are reproducible from this file.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model weights are not included in this repository due to size; follow
`models/README.md` to place the baseline and the PCGU-retrained checkpoint.

## Quick start

### 1) Train (PCGU)
```bash
python src/general_similarity_retrain_ko.py \
  -m models/kcbert_baseline \
  -d data/train_set.tsv \
  -s data/kobbq_stats.tsv \
  -b 16 -l 1e-5 -k 10000 --use-dynamic-k --pairwise_symmetric
```

### 2) Evaluate (StereoSet-style)
```bash
python src/eval.py -m models/best_model -d data/test_set.tsv
```
Outputs SS/LMS/ICAT and an optional detail TSV. Values may vary by roughly
±0.005 across library versions; see `requirements.txt` for the pinned versions.

### 3) Score a sentence (PLL-based)
```bash
python src/run_model.py '{"sentence":"여기에 한국어 문장을 입력합니다."}'
```
Compares baseline vs. retrained PLL per token and returns the aggregated
signal, a 0–10 display score, top words, and per-token differences.

### 4) Run the demo apps
```bash
cp .env.example run/.env   # then put your OpenAI API key in run/.env
streamlit run run/app_release_v1.py          # study version (binary red/green flag)
streamlit run run/app_release_v2_tiered.py   # post-study 3-tier UI (red/yellow/green)
```
Both apps load the models in-process (no remote server needed). `app_release_v1.py`
reproduces the interface rule used in the user study (binary flag at 0.2).
`app_release_v2_tiered.py` is a post-study improvement that adds an intermediate
caution tier for aggregated signals between 0.1 and 0.2 — the red threshold and
the detection rule are identical to the studied system; see the header comment
in that file for the rationale.

## Reproducing the validation and ablation experiments

See `experiments/README.md` for the scripts that reproduce the detection
validation, threshold recalibration, generation ablation, Table 5, and the
token-agreement analysis reported in the paper.

## Environment variables (optional)

- `BASELINE_MODEL_PATH`, `RETRAINED_MODEL_PATH` — override model locations for `src/run_model.py`
- `CHECKPOINT_BASE_DIR` — where PCGU training checkpoints are written (default `models/sim_checkpoints`)
- `SSH_HOST`, `SSH_USER`, `SSH_PASSWORD`, `REMOTE_BIAS_SCRIPT`, `REMOTE_CONDA_ENV`, `REMOTE_CONDA_INIT`
  — only needed if you run the analysis on a remote host instead of in-process

## Requirements

See `requirements.txt`, which pins the original experiment environment.
The pipeline has also been verified end-to-end on Python 3.11 with
torch 2.3.1 / transformers 4.55 (version differences are handled in the code;
metric values may vary by roughly ±0.005 across library versions).
An OpenAI API key is required only for the rewriting module (demo apps) and
the generation-ablation script.

## Attribution

`src/utils.py`, `src/model_utils.py`, `src/consts.py`, and
`src/partition_params.py` are redistributed from the original PCGU
implementation (https://github.com/CharlesYu2000/PCGU-UnlearningBias,
CC BY-SA 4.0), and `src/general_similarity_retrain_ko.py` adapts that
repository's retraining script to Korean subject-slot masking.
