# FAIRITHM Questionnaire Score Data Codebook

## 1. Public files

- `FAIRITHM_questionnaire_score.csv`: de-identified wide-format questionnaire data and manual bias-rewriting task completion time
- `README_FAIRITHM_questionnaire_score.md`: Description of variable names, coding, item codes, and composite scores

## 2. Data scope and structure

- Number of participants: **34** (`P01`–`P34`)
- Rows: one row per participant
- Columns: **179**
  - 1 de-identified participant ID
  - 175 raw item responses
  - 1 manual bias-rewriting task completion-time variable
  - 2 derived scores used in the analysis
- Included data: ABPS, manual bias-rewriting task completion time, RTLX/SEQ, CABRS, post-use perception items (T1–T4), and SUS
- Not included: demographic variables and interview transcripts
- No directly identifying personal information is included
- No missing values are present

## 3. General coding

| Variable group | Coding |
|---|---|
| `ParticipantID` | De-identified participant ID |
| `ABPS_EXP_PriorBiasExperience` | `1 = No prior experience`, `2 = Prior experience` |
| ABPS BP items | 1–7 scale; higher scores indicate stronger agreement |
| `Task_TimeOnTaskSeconds` | Completion time for the manual bias-rewriting task, measured in seconds |
| RTLX items | 1–7 scale; higher scores indicate a higher level of the workload dimension. The performance item is reverse-coded when calculating the RTLX composite score |
| SEQ | 1–7 scale; higher scores indicate greater perceived task ease |
| CABRS O/R items | 1–7 scale; higher scores indicate stronger agreement with the statement |
| CABRS Part 3 choices | `1 = AI revision`, `2 = No difference`, `3 = Single expert-authored revision` |
| Post-use perception items T1–T4 | 1–7 scale; higher scores indicate stronger agreement |
| SUS items | 1–5 scale; total score is calculated using the standard SUS scoring procedure |

## 4. ABPS variables

| Public variable | Manuscript code | Meaning |
|---|---|---|
| `ABPS_EXP_PriorBiasExperience` | EXP | Prior experience encountering biased or unfair expressions while using generative AI services |
| `ABPS_Pre_BP1_NegativeSocialImpact` / `ABPS_Post_BP1_NegativeSocialImpact` | BP1 | Perceived negative societal impact of biased expressions |
| `ABPS_Pre_BP2_NeedDetectionMitigation` / `ABPS_Post_BP2_NeedDetectionMitigation` | BP2 | Perceived need for bias detection and mitigation functions |
| `ABPS_Pre_BP3_HumanDetectionDifficulty` / `ABPS_Post_BP3_HumanDetectionDifficulty` | BP3 | Perceived difficulty of humans detecting all bias unaided |
| `ABPS_Pre_BP4_HumanRevisionEffortTime` / `ABPS_Post_BP4_HumanRevisionEffortTime` | BP4 | Human effort and time required for fair sentence revision |
| `ABPS_Post_BP5_AIRevisionEaseComparedHumans` | BP5 | Perceived ease of AI-assisted revision compared with unaided human revision |

## 5. Manual bias-rewriting task, RTLX, and SEQ variables

| Public variable | Meaning |
|---|---|
| `Task_TimeOnTaskSeconds` | Completion time for the manual bias-rewriting task, measured in seconds |
| `RTLX_MentalDemand` | Mental demand |
| `RTLX_PhysicalDemand` | Physical demand |
| `RTLX_TemporalDemand` | Temporal demand |
| `RTLX_PerformanceRaw` | Raw performance rating. Higher values indicate higher self-rated performance |
| `RTLX_Effort` | Effort |
| `RTLX_Frustration` | Frustration |
| `SEQ_TaskEase` | Overall perceived ease of the manual bias-rewriting task |
| `Derived_RTLX_Mean` | Mean of the six RTLX dimensions after reverse-coding performance as `8 − RTLX_PerformanceRaw`, rounded to one decimal place for each participant |

`Derived_RTLX_Mean` is the final RTLX composite score calculated using the same procedure as the manuscript analysis file. The raw RTLX items are also included so researchers can recalculate the composite if needed.

## 6. CABRS variable-name rules

### 6.1 Bias categories

| Variable label | Manuscript category |
|---|---|
| `Family` | Family and domestic background |
| `Religion` | Cultural and religious identity |
| `SES` | Socioeconomic background |
| `Physical` | Biological and physical characteristics / physical appearance |
| `Gender` | Gender and sexual identity |
| `Political` | Political orientation |

### 6.2 Column-name patterns

| Pattern | Meaning |
|---|---|
| `CABRS_<Domain>_Original_O1` … `O5` | Part 1 evaluation of the original biased sentence |
| `CABRS_<Domain>_SingleExpert_R1` … `R8` | Part 2 evaluation of the single expert-authored revision |
| `CABRS_<Domain>_AI_R1` … `R8` | Part 2 evaluation of the AI revision |
| `CABRS_<Domain>_C1_Choice` … `C3_Choice` | Part 3 direct comparative choice |

### 6.3 CABRS item codes

| Code | Evaluation content |
|---|---|
| O1 / R1 | Grammatical accuracy |
| O2 / R2 | Naturalness and fluency of expression |
| O3 / R3 | Whether the sentence is unbiased and fair |
| O4 / R4 | Whether the sentence does not contain stereotypes about a particular group |
| O5 / R5 | Whether the content or expression does not cause discomfort |
| R6 | Preservation of the core meaning of the original text |
| R7 | Avoidance of excessive dilution or weakening of the original meaning |
| R8 | Perceived bias improvement relative to the original text |
| C1 | Bias mitigation and fairness |
| C2 | Appropriateness for use in actual media or services |
| C3 | Overall trustworthiness of content and expression |

### 6.4 CABRS composite scores used in the manuscript

- Sentence completeness: mean of O1–O2 for the original sentence and mean of R1–R2 for each revision
- Sentence fairness: mean of O3–O5 for the original sentence and mean of R3–R5 for each revision
- Meaning preservation: mean of R6–R7
- Perceived bias improvement: R8, analyzed as a separate single item

## 7. Post-use perception items and SUS

| Public variable | Manuscript code | Meaning |
|---|---|---|
| `PostUse_T1_BiasJudgmentAppropriateness` | T1 | Perceived appropriateness of the system’s bias judgment |
| `PostUse_T2_RevisionRationaleClarity` | T2 | Perceived clarity of the revision rationale and explanation |
| `PostUse_T3_ExpectedConsistency` | T3 | Expectation that the system would consistently improve other biased sentences |
| `PostUse_T4_FutureTrustUseIntention` | T4 | Future trust in and intention to use the system |
| `SUS_01` … `SUS_10` | SUS1–SUS10 | Responses to the ten SUS items on a 1–5 scale |
| `Derived_SUS_Total` | SUS total score | Standard 0–100 SUS score, calculated by transforming odd-numbered items as `response − 1`, even-numbered items as `5 − response`, summing the transformed scores, and multiplying by 2.5 |
