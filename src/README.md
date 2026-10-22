# src

`date_pipeline.py` is the complete experiment pipeline: data loading and audit, augmentation, fold-wise training, evaluation, calibration and robustness studies, baselines, figures, tables and error exports. It is written in notebook style (`# %%` cells), so it can be run as a script or cell by cell in an IDE or on Kaggle.

## Cell groups

| Cells | What they do |
|---|---|
| Setup, grid | environment, seeds, logger, the configuration grid (see `docs/CONFIGURATIONS.md`) |
| 1. Data | loads `dataset.csv`, applies the official row split, dataset statistics, SHA-256, language check, vocabulary log-odds |
| 2. Leakage audit | TF-IDF near-duplicate audit and duplicate-cluster groups for grouped cross-validation |
| 3. Augmentation | masked-language-model word substitution plus random insertion, swap and deletion; applied to the inner training portion only; cached |
| 4. Probe sets | perturbed test sets (typos, word drop, identity-term swaps) and external hate-speech sets |
| 5. Training engine | one model per fold, inner validation split, early stopping, true copy of the best weights, per-fold prediction files |
| 6. Grid | runs the selected configurations and seeds; resumable; a failing configuration is logged and skipped |
| 7-17. Analyses | metrics, single-vs-ensemble calibration, temperature scaling, ensemble-size curve, comparisons and significance tests, multi-seed aggregation, selective prediction, prior shift, slices, robustness, external transfer, label-version analysis, classical baselines, confusion tables |
| 18-21. Outputs | figures, error exports, LaTeX tables, latency and environment record, zip bundle |

## Environment variables

| Variable | Meaning |
|---|---|
| `RUN` | configuration names or groups, comma separated (`core`, `models`, `curve`, `augops`, `all`); default `core` |
| `SEEDS` | comma-separated seeds; the first is the primary seed for single-seed tables; default `42` |
| `ANALYZE_ONLY=1` | skip training and rebuild all analyses from the saved folds |
| `DATA_CSV` | path to `dataset.csv` (default: `data/dataset.csv`, or any `dataset.csv` under `/kaggle/input`) |
| `OUT` | output directory (default `results` locally, `/kaggle/working/outputs` on Kaggle) |
| `COMP_TRAIN` | optional competition-time `train.csv` for the label-version analysis |
| `SMOKE=1` / `SMOKE_TINY=1` | tiny CPU test; `SMOKE_TINY` swaps every model for a small BERT |

## Outputs

Everything is written below `OUT`: `runs/` (per-fold predictions), `tables/`, `exports/`, `audit/`, `figures/`, `aug/` (augmentation cache), `run.log`, and a zip bundle. See `results/README.md` for a description of each file.

## Notes

- Interrupted runs resume: a fold that already has its `.npz` file is skipped.
- Model weights are not saved (they are not needed for the analyses); the per-fold probabilities are.
- Training uses fp16 autocast with fp32 master weights; models are cast to fp32 after loading.
