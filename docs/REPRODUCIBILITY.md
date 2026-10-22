# Reproducibility

## Environments
| Purpose | Needs | Notes |
|---|---|---|
| Rebuild figures, tables, `docs/RESULTS.md` | Python >= 3.9, `requirements.txt` (no GPU) | seconds; uses the saved outputs in `results/` |
| Smoke test of the whole pipeline | same, CPU | `SMOKE_TINY=1 RUN=date_full python src/date_pipeline.py`; uses a tiny BERT and a data subset, a few minutes |
| Full experiments | one or two NVIDIA GPUs with >= 16 GB (tested: Tesla T4), internet for model and dataset downloads | see below |

The reported results were produced on Kaggle (Python 3.12, PyTorch 2.9.0, transformers 5.2.0, scikit-learn 1.6.1, Tesla T4 GPUs). `results/tables/latency_env.json` records the exact versions and the SHA-256 of the data file used.

## Run times (one Tesla T4, 10 folds, seed 42)
| Configuration | GPU time |
|---|---|
| `date_full` | about 30 min |
| `no_aug` | about 12 min |
| other BERTweet-base ablations | 20-30 min each |
| `bertweet_large` | about 65 min |
| other base-size encoders | 10-30 min each |
| whole grid (`RUN=all`) | about 7 GPU-hours, about 4 h wall-clock on two GPUs |

Each additional seed of the core grid costs about 2.3 GPU-hours.

## Determinism
Seeds are fixed (`set_seed`), fold assignment is deterministic (`StratifiedKFold`, shuffle, seed 42), augmentation is seeded per fold and cached. GPU kernels and fp16 mixed precision are not bit-exact across hardware or library versions, so re-running on different hardware reproduces the results up to small numerical differences (typically a few tenths of a percentage point in accuracy).

## Reproducing step by step
```bash
# 0. environment and data
pip install -r requirements.txt
python scripts/verify_data.py

# 1. (GPU) run the experiments; results go to results/
RUN=core SEEDS=42 python src/date_pipeline.py            # or use notebooks/kaggle_runner.ipynb
ANALYZE_ONLY=1 RUN=all SEEDS=42 python src/date_pipeline.py   # rebuild all tables from saved folds

# 2. rebuild figures, LaTeX tables and the results summary
python analysis/run_all.py
python analysis/make_results_summary.py
```
To reuse the shipped outputs instead of training, skip step 1.

## Tests
```bash
python tests/test_repository.py     # or: python -m pytest tests
```
The tests check the data file against its recorded hash and split sizes, that the main result tables exist and are consistent, that the annotation agreement can be recomputed, and that the analysis scripts run.

## Known limitations of the shipped results
- One seed (42) for the full grid; the multi-seed tables need further seeds (`SEEDS=42,43,44,45,46`).
- The DeBERTa-v3 configuration is not part of the shipped results (its fp16 training run failed with a gradient error; the pipeline now keeps fp32 master weights).
- Language identification (`langdetect`) is approximate on short, lower-cased posts.
- External transfer needs internet access; if a dataset cannot be downloaded, that table is skipped with a log warning.
