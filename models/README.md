# Model weights

Weights are not stored in this repository because of their size (~2.1 GB total).
Place them as follows before running the scoring code or the demo apps:

```
models/
  kcbert_baseline/     # baseline KcBERT-base (MLM head), safetensors format
  best_model/
    best_model/        # PCGU-retrained checkpoint (final model used in the paper)
```

- **Baseline**: KcBERT-base (`beomi/kcbert-base` on Hugging Face). Download and
  save it locally, e.g.:

  ```python
  from transformers import AutoTokenizer, AutoModelForMaskedLM
  tok = AutoTokenizer.from_pretrained("beomi/kcbert-base")
  mdl = AutoModelForMaskedLM.from_pretrained("beomi/kcbert-base")
  tok.save_pretrained("models/kcbert_baseline")
  mdl.save_pretrained("models/kcbert_baseline")   # writes safetensors
  ```

- **Retrained (PCGU) checkpoint**: distributed as a separate release artifact
  (see the release page of this repository). Alternatively, retrain from
  scratch with `src/general_similarity_retrain_ko.py` using the configuration
  in the paper (batch size 16, lr 1e-5, k = 10,000, dynamic-k, symmetric
  pairwise gradients).

Different locations can be set via the `BASELINE_MODEL_PATH` and
`RETRAINED_MODEL_PATH` environment variables.

Note: recent `transformers` versions refuse to load `.bin` checkpoints
(CVE-related policy). Use safetensors — `save_pretrained` after a manual
`torch.load(..., weights_only=True)` converts old checkpoints safely.
