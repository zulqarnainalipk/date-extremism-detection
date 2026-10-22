# Configurations

Every configuration is trained with the same folds, augmentation, early stopping and ensembling unless stated otherwise. Select them with `RUN=<name or group>[,...]`.

## Group `core`: ablation and control grid
| Name | Description |
|---|---|
| `date_full` | BERTweet-base, augmentation, class weights, cosine schedule (the DATE configuration) |
| `no_aug` | without augmentation (also probes label preservation of the augmentation) |
| `no_cw` | without class weights |
| `no_cosine` | constant learning rate after warm-up |
| `roberta_base` | RoBERTa-base instead of BERTweet (isolates domain pre-training) |
| `date_grouped` | DATE trained and validated on duplicate-grouped folds (leakage check) |

## Group `models`: other encoders under the same protocol
`bert_base`, `distilbert`, `deberta_v3`, `xlm_roberta`, `twitter_roberta`, `twitter_roberta_hate` (hate-speech-tuned), `twhin_bert`, `toxic_bert` (toxicity-tuned), `bertweet_large` (batch 32 x 4 accumulation).

## Group `curve`: learning curves
`date_full_f25`, `date_full_f50`, `no_aug_f25`, `no_aug_f50` - random 25% / 50% of each outer training part, with and without augmentation.

## Group `augops`: augmentation operators
`aug_ctx_only` (contextual substitution only), `aug_eda_only` (random insertion, swap and deletion only).

## Group `all`
All of the above.

## Choosing seeds
`SEEDS=42` runs one seed; `SEEDS=42,43,44,45,46` repeats every selected configuration. Per-seed results are in `results/tables/multiseed_*.csv` once at least two seeds exist.
