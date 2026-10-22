# DATE: Domain-Aware Transformer Ensembles for Extremism Detection

Code, data, results and analysis scripts for the paper **"DATE: Domain-Aware Transformer Ensembles for Reliable Extremism Detection in Social Media"**.

DATE fine-tunes BERTweet on short social-media posts, applies augmentation only inside each training fold, and averages ten fold models (soft voting). The study evaluates accuracy and, in particular, the *calibration* of single models versus the ensemble on held-out posts, under a leakage-controlled protocol.

- **One script runs everything:** `src/date_pipeline.py` trains and evaluates the whole experiment grid (resumable, GPU recommended).
- **Everything in the paper is regenerable:** `analysis/` rebuilds all figures, LaTeX tables and the results summary from the saved outputs in `results/` (CPU, seconds).
- **Documentation:** each folder has its own README, and `docs/` describes the method, the configurations, the results and how to reproduce them.

## Repository layout

```
data/            dataset.csv (3,000 posts) and its provenance, licence and caveats
src/             date_pipeline.py - the complete experiment pipeline ("# %%" cells)
notebooks/       kaggle_runner.ipynb - runs the pipeline on two T4 GPUs in two stages
analysis/        scripts that rebuild the paper's figures, tables and docs/RESULTS.md from results/
results/         outputs of the experiment run: tables, error exports, per-fold predictions, log
annotation/      protocol, codebook, sheets and scripts of the manual error categorisation
docs/            METHODS.md, CONFIGURATIONS.md, REPRODUCIBILITY.md, RESULTS.md
scripts/         verify_data.py - checks the data file against its recorded hash and split sizes
tests/           test_repository.py - sanity tests for data, results and analysis scripts
paper_assets/    (generated) figures and LaTeX tables written by analysis/run_all.py
```

## Headline results (seed 42 unless stated; 749 held-out test posts)

| | Single fold models | 10-model ensemble |
|---|---|---|
| Accuracy | 0.823 +- 0.036 | 0.843 (95% CI 0.814-0.868) |
| Macro-F1 / ROC-AUC | - | 0.744 / 0.863 |
| ECE (15 bins) | 0.064 | 0.030 (paired difference -0.029, 95% CI -0.046 to -0.010) |

10-fold cross-validation (single models): accuracy 0.822 +- 0.027. The six core configurations were repeated with seeds 42 to 46: the full pipeline reaches a test accuracy of 0.847 +- 0.004 (ensemble) and an ensemble ECE of 0.037 against 0.061 for single models, and the ensemble is better calibrated than its members in all 30 runs (`results/tables/multiseed_summary.csv`). The test part is 16.4% extremist against 44.7% of the training part, so accuracy alone is a weak summary (always predicting "non-extremist" scores 0.836). The complete results - encoder comparison, ablations, calibration, robustness, external transfer - are in [`docs/RESULTS.md`](docs/RESULTS.md) and `results/tables/`.

## Quick start

```bash
git clone https://github.com/zulqarnainalipk/date-extremism-detection.git && cd date-extremism-detection
pip install -r requirements.txt

python scripts/verify_data.py            # check the data file
python analysis/run_all.py               # rebuild figures and tables into paper_assets/
SMOKE_TINY=1 RUN=date_full python src/date_pipeline.py   # end-to-end test on a tiny model (CPU, minutes)
```

Full experiments need a GPU. The recommended way is the Kaggle notebook (`notebooks/README.md`); a local run is `RUN=core SEEDS=42 python src/date_pipeline.py`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for environments, run times and environment variables.

## Method in brief

The first 2,250 rows of `data/dataset.csv` are the training part and the last 750 rows the official test part (one test label is missing, leaving 749). The training part is split into 10 stratified folds. Within each fold, 10% of the outer training part is held out for early stopping and temperature scaling, and **augmentation is applied only to the remaining inner training portion**, so no augmented copy of a validation or test post can reach training. Each fold model predicts its own validation fold (out-of-fold predictions) and the test part; the ensemble is the mean of the ten models' probabilities. Cross-validation, out-of-fold, held-out test and competition numbers are always reported separately. Details: [`docs/METHODS.md`](docs/METHODS.md).

## Data

`data/dataset.csv` is the *Social Media Extremism Detection Dataset* (Suresh, Lu and Iyer), post-competition release, version DOI [10.34740/kaggle/dsv/17368479](https://doi.org/10.34740/kaggle/dsv/17368479), licence CDLA-Permissive-1.0, SHA-256 `ea585a13cbdf93c0398be75a2149c0744c7cbec598f3d55597473c8924e36744`. See [`data/README.md`](data/README.md) for provenance and caveats: annotator information is not reported by the data providers, and 20.8% of the training labels differ from the competition-time release.

## Responsible use

The data contain hateful and violent language. The models are research artefacts: they are not validated for deployment, they rely partly on lexical shortcuts (profanity, identity terms), and they must not be used for automated sanctions on individuals. They are meant to support, not replace, human review.

## Citation

See `CITATION.cff`. Please also cite the dataset and the Kaggle challenge (Suresh, Lu and Iyer).

## Licence

Code: MIT (see `LICENSE`). Data: CDLA-Permissive-1.0 (see `data/README.md`).
