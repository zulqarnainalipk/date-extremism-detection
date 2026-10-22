# analysis

Scripts that rebuild every derived artefact of the paper from `results/` (the output of `src/date_pipeline.py`) and `data/dataset.csv`. No GPU is needed, and they run in seconds.

```bash
python analysis/run_all.py          # runs the four scripts below, writes paper_assets/
python analysis/make_results_summary.py   # rewrites docs/RESULTS.md
```

| Script | Produces |
|---|---|
| `make_fig1.py` | the pipeline figure (`Fig1_pipeline.pdf/png`) |
| `make_figures.py` | data exploration, augmentation size, per-fold, reliability diagrams and the remaining figures |
| `make_tables.py` | all LaTeX tables (`tab_*.tex`) from `results/tables/*.csv` |
| `wrap_landscape.py` | puts the wide tables on landscape pages |
| `make_results_summary.py` | `docs/RESULTS.md`, a Markdown summary of the main results |
| `run_all.py` | runs the first four in order |

Outputs go to `paper_assets/` (git-ignored). Paths are relative to the repository root, so the scripts can be started from any directory.
