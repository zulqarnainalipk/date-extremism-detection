# notebooks

`kaggle_runner.ipynb` runs `src/date_pipeline.py` on Kaggle with two Tesla T4 GPUs: one process per GPU, with the configurations balanced by estimated run time.

## Stages

| Stage | Contents | Time (2 x T4) |
|---|---|---|
| A (`STAGE = "A"`) | seed 42: core ablation grid, other encoders, learning curves, augmentation operators | about 4 hours |
| B (`STAGE = "B"`) | seeds 43-46 for the core grid, plus the DeBERTa-v3 configuration at seed 42; then a final analysis over all seeds | about 5 hours |

## Setup

1. Create a Kaggle dataset containing `src/date_pipeline.py` and `data/dataset.csv`.
2. New notebook, import `kaggle_runner.ipynb`; settings: accelerator *GPU T4 x2*, Internet *On* (model downloads, external hate-speech sets).
3. Add the dataset as an input. For Stage B, also add the Stage A output (notebook output, or a dataset containing its `runs/` folder).
4. Set `STAGE` in the second code cell, then *Save Version -> Save & Run All*; it keeps running if the browser closes.
5. Download `outputs_bundle.zip` from the Output tab and copy its `tables/`, `exports/`, `audit/`, `runs/` and `run.log` into `results/`.

The run is resumable: finished folds are skipped when a cell is re-run with the same inputs.
