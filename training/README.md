# MuRIL Fine-Tuning

Fine-tunes `google/muril-base-cased` on English / Hinglish / Hindi labeled data for a 3-class sequence classifier:

| Label | Class | Description |
|-------|-------|-------------|
| 0 | safe | benign user message |
| 1 | toxic | abusive / hateful language |
| 2 | jailbreak | prompt-injection / role-play evasion |

Once trained, both `toxicity` and `jailbreak` guardrails load the same checkpoint and read the right class probability from `id2label`.

---

## Step 1 — Prepare data

```bash
# Full dataset (~900 synthetic rows + optional HuggingFace rows)
python training/prepare_data.py

# Quick smoke-test subset (~63 rows)
python training/prepare_data.py --smoke-test
```

Outputs: `data/training/train.csv`, `val.csv`, `test.csv`

---

## Step 2 — Fine-tune

### M3 Mac (smoke test / development)

Expected runtime: ~2–4 min per epoch on 63 rows (MPS)

```bash
python training/prepare_data.py --smoke-test

python training/finetune_muril.py \
    --train data/training/train.csv \
    --val   data/training/val.csv   \
    --test  data/training/test.csv  \
    --epochs 1 --batch-size 8 --lr 2e-5 --max-length 128
```

On M3 the MPS backend is used automatically. fp16 is disabled (MPS doesn't support it); the script handles this automatically.

### DGX A100 (full training)

Expected runtime: ~15–30 min for 5 epochs on ~900 rows (fp16, CUDA)

```bash
python training/prepare_data.py   # full dataset

python training/finetune_muril.py \
    --train data/training/train.csv \
    --val   data/training/val.csv   \
    --test  data/training/test.csv  \
    --epochs 5 --batch-size 32 --lr 2e-5 --max-length 256
```

fp16 is enabled automatically on CUDA.

---

## Step 3 — Point the pipeline at the checkpoint

`config.py` auto-detects `models/muril-guardrail/` if it exists:

```python
_LOCAL_MURIL = "models/muril-guardrail"
MODELS = {
    "toxicity":  _LOCAL_MURIL if os.path.isdir(_LOCAL_MURIL) else "google/muril-base-cased",
    "jailbreak": _LOCAL_MURIL if os.path.isdir(_LOCAL_MURIL) else "google/muril-base-cased",
    ...
}
```

No code change needed after training — the pipeline picks up the checkpoint automatically as long as it's saved to `models/muril-guardrail/`.

To override the path at runtime:

```bash
MURIL_CHECKPOINT=models/my-other-checkpoint streamlit run app.py
```

---

## Expected metrics (full training, 5 epochs)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| safe | ~0.95 | ~0.97 | ~0.96 |
| toxic | ~0.89 | ~0.86 | ~0.87 |
| jailbreak | ~0.91 | ~0.90 | ~0.90 |

Smoke-test (1 epoch, 63 rows) will overfit but should reach ~0.80 macro-F1 on the tiny test set.

---

## Dataset sources and licenses

| Source | License | Used for |
|--------|---------|---------|
| Synthetic (this project) | MIT (same as project) | All three languages, all three classes |
| Davidson et al. 2017 `hate_speech_offensive` (optional HF pull) | CC0 | English toxic |
| SemEval 2019 `tweet_eval/hate` (optional HF pull) | Apache 2.0 | English toxic |
| Bohra et al. 2018 Hindi-English code-mixed hate speech | research use | Not auto-downloaded; add manually to `data/training/` if available |

The optional HuggingFace datasets are pulled at runtime inside `prepare_data.py`. If the download fails (no internet, gated dataset), the script falls back to the synthetic data silently.
