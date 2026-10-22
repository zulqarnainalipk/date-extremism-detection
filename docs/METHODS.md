# Methods

This document describes what the code in `src/date_pipeline.py` does. The paper contains the full discussion.

## Task and data
Binary, post-level, English text classification: `EXTREMIST` (1) versus `NON_EXTREMIST` (0). The data are the *Social Media Extremism Detection Dataset* (`data/dataset.csv`, 3,000 posts, post-competition labels). The **official split is by row position**: rows 1-2,250 are the training part, rows 2,251-3,000 the test part. One test post has no label, so the test part has 749 labelled posts (123 extremist, 626 non-extremist); the training part has 2,250 posts (1,006 extremist, 1,244 non-extremist).

## Evaluation protocol (what is computed on which data)
1. **Cross-validation (CV).** The training part is split into 10 stratified outer folds (seed 42). Each fold model is evaluated once on its own outer validation fold; results are mean +- standard deviation over folds. These are single-model results.
2. **Pooled out-of-fold (OOF) predictions.** The ten validation-fold predictions are pooled (n = 2,250) for ECE, Brier score and NLL. Also single-model results: no ensemble prediction exists for these posts.
3. **Held-out test.** The 749 test posts are scored by each of the ten single models and by the soft-vote ensemble. Confidence intervals are percentile bootstrap intervals (2,000 resamples of the test posts); differences between systems reuse the same resamples (paired).
4. **Competition score.** Reported by the Kaggle platform for the submitted ensemble; not recomputed here (different trained models, different label version).

## Inside one fold
- The outer training part (about 2,025 posts) is split again: 90% *inner train*, 10% *inner validation* (stratified).
- **Augmentation is applied to the inner-train posts only**, after the split. Each post gets one or two rounds (drawn at random); each round applies masked-LM contextual substitution (p = 0.6; 1-3 words replaced, sampled from the top-50 whole-word predictions of `bert-base-uncased`), random insertion (0.3), random swap (0.3) and random deletion (0.2; each word dropped with probability 0.1). Expected variants per post: 1.5 x 1.4 = 2.1, so the training set grows to about 3.1x.
- The model is BERTweet-base with the standard sequence-classification head, trained with class-weighted cross-entropy (inverse class frequency, normalised by the weight sum), AdamW (lr 2e-5, weight decay 0.01), 10% linear warm-up then cosine decay over at most 8 epochs, effective batch 128 (64 x 2), fp16 autocast, gradient clipping 1.0.
- **Early stopping** uses the inner-validation loss (patience 3); the weights of the best epoch are restored from a true copy. The outer validation fold is never used for selection.
- **Temperature scaling** (analysis only) is fitted on the inner-validation predictions.
- The fold model then predicts its outer validation fold, the test part, the inner-validation posts, the perturbed test sets and the external sets; all probabilities are saved.

## Ensemble
Soft voting: the mean of the ten fold models' class probabilities. An ensemble prediction therefore exists only for posts that none of the ten models saw in training, i.e. the test part.

## Metrics
Accuracy, weighted and macro F1, balanced accuracy, MCC, ROC-AUC, PR-AUC; calibration: ECE (15 equal-width bins; also 10 and 20 bins and an equal-mass variant), Brier score (sum over classes, range 0-2, and the standard binary form), negative log-likelihood; selective prediction: accuracy at fixed coverage and area under the risk-coverage curve.

## Statistics
- Paired comparisons of configurations: McNemar's exact test on paired predictions and a paired bootstrap of the accuracy difference.
- CV comparison: Nadeau-Bengio corrected resampled t-test over the 10 fold accuracies (paired t-tests over CV folds are not used because training sets overlap).
- Calibration comparison: paired bootstrap of (ensemble ECE - mean single-model ECE) on the test posts.
- Multi-seed runs: mean and standard deviation over seeds; a seed changes the fold shuffle, the initialisation and the augmentation.
- No correction for multiple comparisons is applied; results are exploratory beyond the pre-specified calibration comparison.

## Additional analyses
Near-duplicate audit and duplicate-grouped cross-validation (leakage); learning curves with 25% and 50% of the training data; augmentation-operator ablation and a label-preservation probe; robustness to character noise, word dropout and identity-term swaps; lexical slices; class-prior shift (oracle and EM prior adjustment); external transfer to two hate-speech sets; cost-sensitive thresholds; TF-IDF baselines; rule-based error tags; label-version analysis (competition-time versus post-competition labels).
