# Manual error categorisation

To understand why the classifier fails, 60 randomly sampled out-of-fold errors were categorised by two annotators with a nine-category codebook. This folder documents the sampling and annotation procedure and contains all inputs and outputs.

## Sample
`sample60_key.csv` holds 60 posts drawn with a fixed seed (42) from the 400 out-of-fold errors of the DATE configuration (single fold models, seed 42): 36 false positives and 24 false negatives. Selection is random and independent of the model's confidence.

## Procedure
1. Two of the authors (ZA and MIQ) categorised the posts **independently**, using `annotator1.csv` and `annotator2.csv` (text, gold label and model prediction; the model's confidence was not shown) and the nine-category `codebook.md`. Results: `annotator1_completed.csv`, `annotator2_complete.csv`.
2. Agreement: Cohen's kappa = 0.44 (31 of 60 posts in agreement; `compute_kappa.py`).
3. The 29 disagreements were resolved by discussion between the two annotators (`adjudication_sheet_completed.csv`, with a note per post).
4. `finalize_human_categories.py` merges the agreed and adjudicated categories into `adjudicated.csv` and `final_categories.csv` and writes the table and text used in the paper.

## Files
| File | Content |
|---|---|
| `codebook.md` | the nine categories and guidance |
| `sample60_key.csv` | the 60 sampled errors with gold label, prediction, confidence, error type |
| `annotator1.csv`, `annotator2.csv` | blank annotation sheets |
| `annotator1_completed.csv`, `annotator2_complete.csv` | the two annotators' independent categorisations |
| `adjudication_sheet_completed.csv` | the 29 disagreements with the final decision and a discussion note |
| `adjudicated.csv`, `final_categories.csv` | final categories (all 60 posts) |
| `compute_kappa.py`, `finalize_human_categories.py` | agreement and table scripts |

## Welfare and ethics
The posts contain hateful, violent and sexual language. Annotators worked in short sessions and could skip a post (category 9). Do not copy post text into public places.
