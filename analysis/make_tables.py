"""Builds every results table of the paper (LaTeX, booktabs) directly from results/tables/*.csv."""
import ast, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "results", "tables"); OUT = os.path.join(ROOT, "paper_assets"); os.makedirs(OUT, exist_ok=True)
rd = lambda f: pd.read_csv(f"{T}/{f}.csv")
S = rd("summary_all_configs").set_index("config"); CAL = rd("calibration_single_vs_ensemble").set_index("config"); CMP = rd("config_comparisons")
CF = rd("confusion_and_perclass"); PF = rd("per_fold_metrics"); BL = rd("classical_baselines").set_index("model")
st = json.load(open(f"{T}/dataset_stats.json")); lat = json.load(open(f"{T}/latency_env.json"))
ci = lambda s: ast.literal_eval(s)
f4 = lambda x: f"{x:.4f}"; f3 = lambda x: f"{x:.3f}"; f2 = lambda x: f"{x:.2f}"
def pval(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def write(name, body, caption, label, cols, size="\\small", pos="h!", extra=""):
    tcs = 2 if size == "\\scriptsize" else 5
    open(f"{OUT}/{name}.tex", "w", encoding="utf8").write(
        f"\\begin{{table}}[{pos}]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\setlength{{\\tabcolsep}}{{{tcs}pt}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}\n{extra}\\end{{table}}\n")
def rows(lst): return " \\\\\n".join(" & ".join(map(str, r)) for r in lst) + " \\\\"

NICE = {"date_full": "BERTweet-base (DATE)", "bertweet_large": "BERTweet-large", "roberta_base": "RoBERTa-base", "bert_base": "BERT-base", "distilbert": "DistilBERT-base",
        "xlm_roberta": "XLM-RoBERTa-base", "twitter_roberta": "Twitter-RoBERTa-base", "twitter_roberta_hate": "Twitter-RoBERTa-hate", "twhin_bert": "TwHIN-BERT-base", "toxic_bert": "Toxic-BERT"}

# ---- dataset
ex = rd("augmentation_sizes")
rows_ = [["Posts", f"{st['train_n']:,}", f"{st['test_n']:,}"],
         ["EXTREMIST", f"{st['train_ext']:,} ({st['train_ext_pct']:.1f}\\%)", f"{st['test_ext']:,} ({st['test_ext_pct']:.1f}\\%)"],
         ["NON\\_EXTREMIST", f"{st['train_non']:,} ({100-st['train_ext_pct']:.1f}\\%)", f"{st['test_non']:,} ({100-st['test_ext_pct']:.1f}\\%)"],
         ["Mean / median / max length (BERTweet tokens, train)", f"{st['bertweet_tokens_mean']:.1f} / {st['bertweet_tokens_median']:.0f} / {st['bertweet_tokens_max']}", "--"],
         ["Mean / median / max length (words, train)", f"{st['words_mean']:.1f} / {st['words_median']:.0f} / {st['words_max']}", "--"],
         ["Truncated at $L{=}128$", f"{st['train_truncated_at_128']} ({st['train_truncated_at_128']/st['train_n']*100:.2f}\\%)", f"{st['test_truncated_at_128']} (0.00\\%)"],
         ["Exact duplicate texts / test texts also in train", f"{st['exact_duplicate_texts_train']}", f"{st['test_texts_also_in_train']}"],
         ["Detected as English (approximate, \\texttt{langdetect})", f"\\multicolumn{{2}}{{c}}{{{st['pct_detected_english']:.1f}\\% of all posts}}"]]
write("tab_dataset", "Statistic & Train part & Test part \\\\\n\\midrule\n" + rows(rows_), "Statistics of the exact file used (\\texttt{dataset.csv}, SHA-256 \\texttt{ea585a13\\ldots e36744}). One post in the test part has a missing label and is excluded.", "tab:dataset", "lcc")

# ---- hyper-parameters
hp = [["Backbone", "BERTweet-base (135M parameters)"], ["Max sequence length", "128 (dynamic padding)"], ["Batch size", "64 $\\times$ 2 gradient-accumulation steps (effective 128)"],
      ["Learning rate / weight decay", "$2\\times10^{-5}$ / 0.01 (AdamW, $\\epsilon{=}10^{-8}$, gradient clipping 1.0)"], ["Scheduler", "Cosine decay with 10\\% linear warm-up over a maximum of 8 epochs"],
      ["Early stopping", "Patience 3 epochs on the \\emph{inner-validation} loss; best epoch restored (deep copy of weights)"],
      ["Inner validation split", "10\\% of each outer training part (stratified); never used for reporting"], ["Loss", "Cross-entropy with inverse-frequency class weights"],
      ["Precision", "FP16 mixed precision"], ["Cross-validation", "10-fold stratified, seed 42; per-fold seeds $42+100k$"], ["Ensemble", "Soft voting over the 10 fold models"]]
write("tab_hparams", "Parameter & Value \\\\\n\\midrule\n" + rows(hp), "Training configuration.", "tab:params", "ll")

# ---- CV summary
g = PF[PF.config == "date_full"]
def cvrow(name, col, fmt=f4): return [name, fmt(g[col].mean()), fmt(g[col].std(ddof=1)), fmt(g[col].min()), fmt(g[col].max())]
cv = [cvrow("Accuracy", "acc"), cvrow("F1 (weighted)", "f1_weighted"), cvrow("F1 (macro)", "f1_macro"), cvrow("ECE (15 bins, single model)", "ece15"), cvrow("Brier score", "brier_sum"), cvrow("NLL", "nll")]
s0 = S.loc["date_full"]
extra = f"\\par\\smallskip\\footnotesize Pooled out-of-fold predictions (n=2,250): accuracy {s0.oof_pooled_acc:.4f} (95\\% bootstrap CI {ci(s0.oof_pooled_acc_ci)[0]:.4f}--{ci(s0.oof_pooled_acc_ci)[1]:.4f}), ECE {s0.oof_pooled_ece15:.4f}, Brier {s0.oof_pooled_brier_sum:.4f}, NLL {s0.oof_pooled_nll:.4f}, AUC {s0.oof_pooled_auc:.4f}.\n"
write("tab_cv", "Metric & Mean & Std & Min & Max \\\\\n\\midrule\n" + rows(cv), "10-fold cross-validation, BERTweet-base (DATE configuration). Each fold model is evaluated once on its own outer validation fold with the checkpoint chosen on a separate inner validation split; every value is a \\emph{single-model} result.", "tab:results", "lcccc", extra=extra)

# ---- test results (single vs ensemble)
ens_ci = ci(s0.test_ens_acc_ci); f1ci = ci(s0.test_ens_f1_macro_ci); eceCI = ci(s0.test_ens_ece_ci)
tt = [["Accuracy", f"{s0.test_single_acc_mean:.4f} $\\pm$ {s0.test_single_acc_std:.4f}", f"{s0.test_ens_acc:.4f} [{ens_ci[0]:.4f}, {ens_ci[1]:.4f}]"],
      ["F1 (weighted)", "--", f4(s0.test_ens_f1)], ["F1 (macro)", "--", f"{s0.test_ens_f1_macro:.4f} [{f1ci[0]:.4f}, {f1ci[1]:.4f}]"], ["Balanced accuracy", "--", f4(s0.test_ens_bal_acc)],
      ["Matthews corr. coef.", "--", f4(s0.test_ens_mcc)], ["ROC-AUC", "--", f4(s0.test_ens_auc)], ["PR-AUC (extremist class)", "--", f4(s0.test_ens_pr_auc)],
      ["ECE (15 bins)", f4(s0.test_single_ece_mean), f"{s0.test_ens_ece15:.4f} [{eceCI[0]:.4f}, {eceCI[1]:.4f}]"], ["Brier score", f4(s0.test_single_brier_mean), f4(s0.test_ens_brier_sum)],
      ["NLL", f4(s0.test_single_nll_mean), f4(s0.test_ens_nll)]]
write("tab_test", "Metric & Single fold model (mean $\\pm$ sd of 10) & 10-model soft-vote ensemble [95\\% CI] \\\\\n\\midrule\n" + rows(tt),
      "Results on the 749 labelled test posts (123 extremist, 626 non-extremist). Intervals are percentile bootstrap intervals (2,000 resamples of the test posts).", "tab:test", "lcc")

# ---- calibration table
sel = ["date_full", "no_aug", "no_cw", "no_cosine", "roberta_base", "twitter_roberta", "twhin_bert", "bertweet_large"]
cr = []
for k in sel:
    c = CAL.loc[k]; d = ci(c.delta_ece_ci)
    cr.append([NICE.get(k, k) if k != "no_aug" else "DATE w/o augmentation", k.replace("_", "\\_") if False else "", f4(c.ece_single_mean), f4(c.ece_single_tempscaled), f4(c.ece_ensemble), f4(c.ece_ensemble_of_tempscaled), f"{c.delta_ece:+.4f} [{d[0]:+.4f}, {d[1]:+.4f}]"])
cr = [[r[0], r[2], r[3], r[4], r[5], r[6]] for r in cr]
cr[2][0] = "DATE w/o class weights"; cr[3][0] = "DATE w/o cosine schedule"
write("tab_calibration", "Configuration & Single & Single + TS & Ensemble & Ens. of TS & $\\Delta$ECE (ens.\\ $-$ single) [95\\% CI] \\\\\n\\midrule\n" + rows(cr),
      "Test-set ECE (15 bins, 749 posts) of single fold models and of the 10-model soft-vote ensemble. ``TS'' = temperature scaling fitted on each model's inner-validation split. $\\Delta$ECE is the paired-bootstrap difference between the ensemble and the mean of the single models.", "tab:calibration", "lccccc", size="\\scriptsize")

# ---- encoders under identical protocol
order = ["distilbert", "bert_base", "toxic_bert", "xlm_roberta", "twhin_bert", "roberta_base", "twitter_roberta", "twitter_roberta_hate", "date_full", "bertweet_large"]
er = []
for k in order:
    r = S.loc[k]; a = ci(r.test_ens_acc_ci)
    er.append([NICE[k], f"{r.params_millions_single:.0f}", f"{r.cv_acc_mean:.4f} $\\pm$ {r.cv_acc_std:.4f}", f"{r.test_ens_acc:.4f} [{a[0]:.3f}, {a[1]:.3f}]", f4(r.test_ens_f1_macro), f4(r.test_ens_auc), f4(r.test_ens_ece15)])
er.append(["TF-IDF + logistic regression", "--", f"{BL.loc['tfidf_logreg'].cv_acc_mean:.4f} $\\pm$ {BL.loc['tfidf_logreg'].cv_acc_std:.4f}", f4(BL.loc['tfidf_logreg'].test_acc), f4(BL.loc['tfidf_logreg'].test_f1_macro), f4(BL.loc['tfidf_logreg'].test_auc), f4(BL.loc['tfidf_logreg'].test_ece15)])
er.append(["TF-IDF + linear SVM", "--", f"{BL.loc['tfidf_linsvm'].cv_acc_mean:.4f} $\\pm$ {BL.loc['tfidf_linsvm'].cv_acc_std:.4f}", f4(BL.loc['tfidf_linsvm'].test_acc), f4(BL.loc['tfidf_linsvm'].test_f1_macro), "--", "--"])
write("tab_encoders", "Encoder & Params (M) & CV acc.\\ (mean $\\pm$ sd) & Test acc.\\ (ens.) [95\\% CI] & Test macro-F1 & Test AUC & Test ECE \\\\\n\\midrule\n" + rows(er),
      "Controlled comparison of encoders under one protocol (same folds, same augmentation, same early stopping, 10-fold soft-vote ensemble). Test columns use the 749 labelled test posts. Single seed (42).", "tab:comparison", "lcccccc", size="\\scriptsize", pos="t")

# ---- ablation
ab_keys = [("date_full", "Full DATE"), ("no_aug", "w/o augmentation"), ("no_cw", "w/o class weights"), ("no_cosine", "w/o cosine schedule (constant LR)"), ("aug_ctx_only", "contextual substitution only"),
           ("aug_eda_only", "random insert/swap/delete only"), ("date_grouped", "Full DATE, duplicate-grouped folds")]
ar = []
for k, nm in ab_keys:
    r = S.loc[k]
    if k == "date_full": ar.append([nm, f4(r.oof_pooled_acc), f4(r.test_ens_acc), f4(r.test_ens_f1_macro), f4(r.test_ens_ece15), "--", "--", "--"]); continue
    ct = CMP[(CMP.vs == k) & (CMP.on == "test_ensemble")].iloc[0]; co = CMP[(CMP.vs == k) & (CMP.on == "oof_pooled")].iloc[0]
    nb = pval(co.nadeau_bengio_p) if "nadeau_bengio_p" in co and not pd.isna(co.nadeau_bengio_p) else "--"
    ar.append([nm, f4(r.oof_pooled_acc), f4(r.test_ens_acc), f4(r.test_ens_f1_macro), f4(r.test_ens_ece15), pval(co.mcnemar_p), nb, pval(ct.mcnemar_p)])
write("tab_ablation", "Configuration & CV acc.\\ (pooled OOF) & Test acc.\\ & Test macro-F1 & Test ECE & $p$ OOF (McNemar) & $p$ OOF (Nadeau--Bengio) & $p$ test (McNemar) \\\\\n\\midrule\n" + rows(ar),
      "Ablation of the DATE pipeline (single seed 42, 10 folds each). The $p$-values compare each variant with the full pipeline: McNemar's test on the paired predictions, and the Nadeau--Bengio corrected resampled $t$-test over the 10 fold accuracies, which accounts for the overlap between training sets. None of the OOF differences is significant.", "tab:ablation", "lccccccc", size="\\scriptsize", pos="t")

# ---- learning curve
lc = rd("augmentation_studies"); lcr = []
for fr, lab in [(0.25, "25\\%"), (0.5, "50\\%"), (1.0, "100\\%")]:
    row = [lab]
    for base in ("date_full", "no_aug"):
        r = lc[(lc.study == "learning_curve") & (lc.frac == fr) & (lc.config.str.startswith(base))].iloc[0]; row += [f4(r.oof_acc), f4(r.test_ens_acc), f4(r.test_ens_ece)]
    lcr.append(row)
pr = lc[lc.study == "label_preservation_probe"].iloc[0]
extra = f"\\par\\smallskip\\footnotesize Label-preservation probe: a model trained \\emph{{without}} augmentation changes its prediction for {pr.flip_rate*100:.1f}\\% of {int(pr.n_augmented):,} augmented validation posts; accuracy on the augmented variants is {pr.acc_on_augmented:.3f} against {pr.acc_on_same_originals:.3f} on the same original posts.\n"
write("tab_learning", "Training data & \\multicolumn{3}{c}{With augmentation} & \\multicolumn{3}{c}{Without augmentation} \\\\\n & OOF acc. & Test acc. & Test ECE & OOF acc. & Test acc. & Test ECE \\\\\n\\midrule\n" + rows(lcr),
      "Learning curves: effect of augmentation as the amount of training data shrinks (random subsets of each outer training part; same folds).", "tab:learning", "lcccccc", extra=extra)

# ---- per-class + confusion
pc = []
for on, lab in [("oof_pooled", "OOF, single models (n=2,250)"), ("test_ensemble", "Test, ensemble (n=749)")]:
    r = CF[(CF.config == "date_full") & (CF.on == on)].iloc[0]
    pc.append([lab, int(r.TN), int(r.FP), int(r.FN), int(r.TP), f2(r.prec_ext), f2(r.rec_ext), f2(r.f1_ext), f2(r.prec_non), f2(r.rec_non), f2(r.f1_non)])
write("tab_perclass", "Evaluation & TN & FP & FN & TP & P$_\\text{ext}$ & R$_\\text{ext}$ & F1$_\\text{ext}$ & P$_\\text{non}$ & R$_\\text{non}$ & F1$_\\text{non}$ \\\\\n\\midrule\n" + rows(pc),
      "Absolute confusion-matrix counts and per-class precision (P), recall (R) and F1 for the extremist (ext) and non-extremist (non) classes; positive class = EXTREMIST.", "tab:perclass", "lcccccccccc", size="\\scriptsize")

# ---- threshold + cost
th = rd("threshold_sweep_date_full"); th = th[th.threshold.isin([0.3, 0.5, 0.7, 0.9])]
tr_ = [[f"{r.threshold:.1f}", int(r.TP), int(r.FP), int(r.FN), int(r.TN), f3(r.precision_ext), f3(r.recall_ext), f"{r.flagged_pct:.1f}\\%"] for r in th.itertuples()]
cs = rd("cost_sensitive_thresholds"); cr2 = [[int(r.cost_FN_over_FP), f"{r.threshold_chosen_on_oof:.2f}", int(r.test_FP), int(r.test_FN), f3(r.test_cost_per_post), int(r.FP_at_0_5), int(r.FN_at_0_5), f3(r.test_cost_at_0_5)] for r in cs.itertuples()]
write("tab_threshold", "Threshold on $P(\\text{ext})$ & TP & FP & FN & TN & Precision & Recall & Flagged \\\\\n\\midrule\n" + rows(tr_),
      "Operating points of the test-set ensemble (749 posts, 123 extremist). Lowering the threshold trades false negatives for false positives.", "tab:threshold", "ccccccccc" if False else "cccccccc")
write("tab_cost", "Cost ratio FN:FP & Threshold (chosen on OOF) & FP & FN & Cost/post & FP @0.5 & FN @0.5 & Cost/post @0.5 \\\\\n\\midrule\n" + rows(cr2),
      "Cost-sensitive thresholds. The threshold minimising the expected cost $c_\\text{FN}\\cdot\\text{FN}+c_\\text{FP}\\cdot\\text{FP}$ is chosen on the out-of-fold predictions only and then applied to the test ensemble. Which cost ratio is appropriate is a policy decision, not a modelling one.", "tab:cost", "cccccccc", size="\\scriptsize")

# ---- risk-coverage
rc = rd("risk_coverage"); rc = rc[(rc.config == "date_full")]
rcr = []
for tag, lab in [("test_ensemble", "Test, ensemble"), ("test_single_mean", "Test, single model (mean)"), ("oof_single", "OOF, single models")]:
    r = rc[rc.on == tag].iloc[0]; rcr.append([lab] + [f3(r[f"acc@cov{c}"]) for c in (50, 70, 80, 90, 100)] + [f3(r.aurc)])
write("tab_riskcov", "Evaluation & 50\\% & 70\\% & 80\\% & 90\\% & 100\\% & AURC \\\\\n\\midrule\n" + rows(rcr),
      "Selective prediction: accuracy on the most confident $x$\\% of posts (coverage); the remaining posts would be escalated to human reviewers. AURC = area under the risk--coverage curve (lower is better).", "tab:riskcov", "lcccccc")

# ---- robustness
rb = rd("robustness_and_fairness_probes"); rb = rb[rb.config.isin(["date_full", "no_aug", "roberta_base"])]
nm = {"typo10": "Character swaps in 10\\% of words", "typo20": "Character swaps in 20\\% of words", "worddrop15": "Random word dropout (15\\%)", "ident_swap": "Identity terms swapped (muslims$\\to$christians, \\ldots)", "ident_neutral": "Identity terms $\\to$ ``people''"}
rr = []
for p in ["typo10", "typo20", "worddrop15", "ident_swap", "ident_neutral"]:
    row = [nm[p], int(rb[(rb.config == "date_full") & (rb.perturbation == p)].n.iloc[0])]
    for k in ["date_full", "no_aug", "roberta_base"]:
        r = rb[(rb.config == k) & (rb.perturbation == p)].iloc[0]; row += [f"{r.ens_acc:.3f} ({r.ens_acc_clean:.3f})", f"{r.ens_flip_rate*100:.1f}\\%"]
    rr.append(row)
write("tab_robust", "Perturbation of the test posts & $n$ & \\multicolumn{2}{c}{DATE} & \\multicolumn{2}{c}{DATE w/o aug.} & \\multicolumn{2}{c}{RoBERTa-base} \\\\\n & & acc.\\ (clean) & flips & acc.\\ (clean) & flips & acc.\\ (clean) & flips \\\\\n\\midrule\n" + rows(rr),
      "Robustness and identity-term probes on the test ensemble. ``flips'' = share of predictions that change relative to the unperturbed post. The identity probes only use the 134 test posts that contain one of 24 identity terms.", "tab:robust", "lccccccc", size="\\scriptsize")

# ---- external transfer
ext = rd("external_transfer"); ek = ["date_full", "no_aug", "roberta_base", "bertweet_large", "twitter_roberta_hate"]; er2 = []
for k in ek:
    d1 = ext[(ext.config == k) & (ext.external == "davidson_hate")].iloc[0]; d2 = ext[(ext.config == k) & (ext.external == "tweets_hate")].iloc[0]
    er2.append([NICE.get(k, "DATE w/o augmentation") if k != "no_aug" else "DATE w/o augmentation", f3(d1.ens_auc), f3(d1.ens_pr_auc), f3(d2.ens_auc), f3(d2.ens_pr_auc)])
write("tab_external", "Model (trained on our data only) & \\multicolumn{2}{c}{Davidson et al.\\ tweets} & \\multicolumn{2}{c}{Hate-tweets benchmark} \\\\\n & AUC & PR-AUC & AUC & PR-AUC \\\\\n\\midrule\n" + rows(er2),
      "Zero-shot transfer of the 10-model ensembles to two external hate-speech sets (3,000 posts each, 30\\% positive; positive = hate speech, so labels are related but not identical to ``extremist'').", "tab:external", "lcccc")

# ---- prior shift
ps = rd("prior_shift_analysis"); ps = ps[ps.config == "date_full"]; lab = {"raw": "Raw ensemble", "prior_adjusted_oracle": "Prior-adjusted to true test prior (oracle)", "prior_adjusted_EM_unlabelled": "Prior-adjusted with EM-estimated prior"}
prr = [[lab[r.variant], f3(r.acc), f3(r.bal_acc), f3(r.f1_macro), f3(r.mcc), f3(r.ece15)] for r in ps.itertuples()]
est = ps.em_estimated_test_prior.iloc[0]
write("tab_prior", "Variant & Accuracy & Balanced acc. & Macro-F1 & MCC & ECE \\\\\n\\midrule\n" + rows(prr),
      f"Class-prior shift: the training part is 44.7\\% extremist, the test part 16.4\\%. Models trained with class weights behave as if the prior were 50\\%. Prior adjustment is an analysis only (the oracle variant uses test labels); the EM estimate of the test prior was {est:.3f}, so unsupervised prior estimation was not reliable here.", "tab:prior", "lccccc")

# ---- slices
sl = rd("slice_analysis"); sl = sl[(sl.on == "test_ensemble") & sl.slice.isin(["len<=10", "len>40", "has_profanity", "no_profanity", "has_violence_word", "has_identity_term", "no_identity_term", "sim>=0.9(near-dup)"])]
sn = {"len<=10": "$\\le$10 words", "len>40": "$>$40 words", "has_profanity": "Contains profanity", "no_profanity": "No profanity", "has_violence_word": "Contains a violence term",
      "has_identity_term": "Contains an identity term", "no_identity_term": "No identity term", "sim>=0.9(near-dup)": "Near-duplicate of a train post"}
slr = [[sn[r.slice], int(r.n), f"{r.pct_extremist:.0f}\\%", f3(r.acc), f3(r.fpr), f3(r.fnr)] for r in sl.itertuples()]
write("tab_slices", "Test slice & $n$ & \\% extremist & Accuracy & FPR & FNR \\\\\n\\midrule\n" + rows(slr),
      "Test-set ensemble accuracy and error rates on lexical slices (keyword lexicons listed in the supplementary code). FPR/FNR = false-positive/false-negative rate.", "tab:slices", "lccccc", size="\\scriptsize")

# ---- rule-based error tags
tg = rd("error_rate_by_rule_tag"); tn = {"short(<6 words)": "Short ($<$6 words)", "long(>40 words)": "Long ($>$40 words)", "profanity": "Profanity", "violence_word": "Violence term", "identity_term": "Identity term", "negation": "Negation"}
tgr = []
for t_ in tn:
    a = tg[(tg.on == "oof") & (tg.tag == t_)].iloc[0]; b = tg[(tg.on == "test_ensemble") & (tg.tag == t_)].iloc[0]
    tgr.append([tn[t_], int(a.n), f"{a.error_rate*100:.1f}\\% / {a.error_rate_without_tag*100:.1f}\\%", int(b.n), f"{b.error_rate*100:.1f}\\% / {b.error_rate_without_tag*100:.1f}\\%"])
write("tab_tags", "Rule-based tag & OOF $n$ & OOF error: with / without tag & Test $n$ & Test error: with / without tag \\\\\n\\midrule\n" + rows(tgr),
      "Objective error analysis with rule-based tags (OOF: 400 errors of 2,250; test: 118 errors of 749). The tags are keyword and length rules, not a manual taxonomy.", "tab:tags", "lcccc", size="\\scriptsize")

# ---- numbers used in text
json.dump(dict(latency_ms=lat["ms_per_post_single_model"], ens_latency_ms=lat["ms_per_post_10_model_ensemble_sequential"]), open(f"{OUT}/numbers.json", "w"))
print("tables written")

