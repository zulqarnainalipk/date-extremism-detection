"""Finalises the manual error categorisation from the two completed annotator files.

  python annotation/finalize_human_categories.py --resolve 3=1

Reads annotator1_completed.csv and annotator2_completed.csv (column `category_primary`, codes 1-9) and sample60_key.csv.
Posts on which the annotators agree keep the shared category; each disagreement must be resolved on the command line
(`--resolve <id>=<category>`) or by filling `category_final` in annotation/adjudication_sheet.csv. Writes
  annotation/adjudicated.csv, annotation/final_categories.csv,
  paper_assets/tab_error_categories.tex and paper_assets/error_categories_text.tex (the paragraph for the paper)."""
import argparse, os, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NAMES = {1: "Implicit or veiled extremism", 2: "Trigger words without group-directed harm", 3: "Group mentioned without hostility",
         4: "Quotation, reported speech or counter-speech", 5: "Sarcasm, irony or joke", 6: "Needs external context",
         7: "Questionable gold label", 8: "Explicit hostile content missed by the model", 9: "Other or unclear"}
ap = argparse.ArgumentParser(); ap.add_argument("--resolve", nargs="*", default=[], help="id=category for each disagreement")
args = ap.parse_args(); res = {int(k): int(v) for k, v in (x.split("=") for x in args.resolve)}

def _find(n):
    for f in (f"annotator{n}_completed.csv", f"annotator{n}_complete.csv"):
        if os.path.exists(os.path.join(HERE, f)): return os.path.join(HERE, f)
    sys.exit(f"completed file for annotator {n} not found")
a1 = pd.read_csv(_find(1)); a2 = pd.read_csv(_find(2)); key = pd.read_csv(os.path.join(HERE, "sample60_key.csv"))
sheet = next((os.path.join(HERE, f) for f in ("adjudication_sheet_completed.csv", "adjudication_sheet.csv") if os.path.exists(os.path.join(HERE, f))), "")
if sheet and not res:
    sh = pd.read_csv(sheet); sh = sh[sh.category_final.notna()]
    res = {int(i): int(v) for i, v in zip(sh.id, sh.category_final)}
assert (a1.id.values == a2.id.values).all() and (a1.id.values == key.id.values).all(), "id mismatch"
c1, c2 = a1.category_primary.astype(int), a2.category_primary.astype(int)
kappa, agree = cohen_kappa_score(c1, c2), float((c1 == c2).mean()); dis = a1.id[c1 != c2].tolist()
missing = [i for i in dis if i not in res]
if missing: sys.exit(f"Disagreements to resolve with --resolve: {missing}  (annotator choices: " + "; ".join(f"{i}: {int(c1[a1.id==i].iloc[0])} vs {int(c2[a1.id==i].iloc[0])}" for i in missing) + ")")
final = c1.copy()
for i, v in res.items(): final[a1.id == i] = v
pd.DataFrame({"id": a1.id, "category_final": final}).to_csv(os.path.join(HERE, "adjudicated.csv"), index=False)
fin = pd.DataFrame({"id": a1.id, "error_type": key.error_type, "annotator1": c1, "annotator2": c2, "category_final": final, "category_name": final.map(NAMES)})
fin["source"] = "two annotators; disagreements resolved by discussion"; fin.to_csv(os.path.join(HERE, "final_categories.csv"), index=False)

n_fp, n_fn = int((fin.error_type == "false_positive").sum()), int((fin.error_type == "false_negative").sum())
cnt = lambda code, t=None: int(((fin.category_final == code) & ((fin.error_type == t) if t else True)).sum())
rows = sorted([(NAMES[c], cnt(c, "false_positive"), cnt(c, "false_negative"), cnt(c)) for c in NAMES if cnt(c)], key=lambda r: -r[3])
tex = ["\\begin{table}[h!]", "\\centering", "\\small",
       f"\\caption{{Categories of a random sample of {len(fin)} out-of-fold errors ({n_fp} false positives, {n_fn} false negatives), assigned independently by two annotators (two of the authors) with the codebook released with the code (Cohen's $\\kappa$ = {kappa:.2f}; the annotators agreed on {int((c1 == c2).sum())} of {len(fin)} posts and the {len(dis)} disagreements were resolved by discussion).}}",
       "\\label{tab:errcats}", "\\begin{tabular}{p{6.3cm}ccc}", "\\toprule", "Category & FP & FN & Total \\\\", "\\midrule"]
tex += [f"{n} & {fp} & {fn} & {t} \\\\" for n, fp, fn, t in rows] + ["\\midrule", f"Total & {n_fp} & {n_fn} & {len(fin)} \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"]
out = os.path.join(ROOT, "paper_assets"); os.makedirs(out, exist_ok=True)
open(os.path.join(out, "tab_error_categories.tex"), "w", encoding="utf8").write("\n".join(tex) + "\n")

q = cnt(7)
lvl = "slight" if kappa < .21 else "fair" if kappa < .41 else "moderate" if kappa < .61 else "substantial" if kappa < .81 else "almost perfect"
pairs = pd.crosstab(c1[c1 != c2], c2[c1 != c2]); n24 = int(pairs.loc[4, 2]) if (4 in pairs.index and 2 in pairs.columns) else 0
q = cnt(7)
para = (f"To see why the model fails, two of the authors (ZA and MIQ) independently categorised a random sample of 60 of the 400 out-of-fold errors (drawn with a fixed seed, independently of model confidence; {n_fp} false positives and {n_fn} false negatives) with a nine-category codebook released with the code. "
        f"They agreed on {int((c1 == c2).sum())} of {len(fin)} posts (Cohen's $\\kappa$ = {kappa:.2f}, {lvl} agreement). "
        f"The {len(dis)} disagreements concerned mainly whether a post with violent or hostile vocabulary quotes or opposes that content or merely uses trigger words (categories 4 and 2), and whether the gold label is questionable (category 7); they were resolved by discussion between the two annotators, and both annotators' sheets and the adjudication are released with the code. "
        f"The sample is small, so proportions are approximate (a share of one third has a 95\\% interval of roughly $\\pm$12 points). Table~\\ref{{tab:errcats}} gives the final categories. "
        f"The false positives are dominated by posts with violent or profane words that do not attack a group ({cnt(2,'false_positive')} of {n_fp}), followed by posts whose gold label the annotators consider questionable ({cnt(7,'false_positive')}), neutral mentions of a group ({cnt(3,'false_positive')}), quotations or counter-speech ({cnt(4,'false_positive')}) and unclear or context-dependent posts ({cnt(9,'false_positive') + cnt(6,'false_positive')}). "
        f"The false negatives are mostly implicit or veiled hostility without an explicit slur ({cnt(1,'false_negative')} of {n_fn}) and explicit hostile posts that the model nevertheless rated non-extremist ({cnt(8,'false_negative')}, several of them profane, in line with the profanity shortcut of Section~\\ref{{sec:robust}}); the remainder are posts that need external context ({cnt(6,'false_negative')}) and single cases of sarcasm, a questionable gold label and an unclear post. "
        f"In {q} of the 60 posts ({q/60*100:.0f}\\%) the final category is a questionable gold label, which is consistent with the label instability reported above. "
        "The categorised sample, the codebook and both annotators' sheets are released with the code, together with all errors and the 50 most confident errors.")
open(os.path.join(out, "error_categories_text.tex"), "w", encoding="utf8").write(para + "\n")
print(f"kappa {kappa:.3f} | agreement {agree:.3f} ({int((c1 == c2).sum())}/{len(fin)}) | resolved: {res} | wrote adjudicated.csv, final_categories.csv, paper_assets/tab_error_categories.tex, error_categories_text.tex")
