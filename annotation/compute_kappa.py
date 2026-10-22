"""Agreement and summary for the manual error categorisation.

Reads annotation/annotator1.csv, annotation/annotator2.csv (column `category_primary`, integers 1-9) and
annotation/sample60_key.csv and, after adjudication, annotation/adjudicated.csv (columns id, category_final).
Prints Cohen's kappa, raw agreement and a confusion table, and writes paper_assets/tab_error_categories.tex."""
import os, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NAMES = {1: "Implicit or veiled extremism", 2: "Trigger words without group-directed harm", 3: "Group mentioned without hostility",
         4: "Quotation, reported speech or counter-speech", 5: "Sarcasm, irony or joke", 6: "Needs external context",
         7: "Questionable gold label", 8: "Explicit hostile content missed by the model", 9: "Other or unclear"}

def _find(n):
    for f in (f"annotator{n}_completed.csv", f"annotator{n}_complete.csv", f"annotator{n}.csv"):
        if os.path.exists(os.path.join(HERE, f)): return os.path.join(HERE, f)
a1 = pd.read_csv(_find(1)); a2 = pd.read_csv(_find(2)); key = pd.read_csv(os.path.join(HERE, "sample60_key.csv"))
c1, c2 = a1["category_primary"], a2["category_primary"]
if c1.isna().any() or c2.isna().any():
    sys.exit(f"Incomplete: {int(c1.isna().sum())} unlabelled posts for annotator 1, {int(c2.isna().sum())} for annotator 2.")
c1, c2 = c1.astype(int), c2.astype(int)
kappa = cohen_kappa_score(c1, c2); agree = float((c1 == c2).mean())
print(f"posts: {len(c1)} | raw agreement: {agree:.3f} | Cohen's kappa: {kappa:.3f} | disagreements to adjudicate: {int((c1 != c2).sum())}")
labels = sorted(set(c1) | set(c2))
print(pd.DataFrame(confusion_matrix(c1, c2, labels=labels), index=[f"A1:{l}" for l in labels], columns=[f"A2:{l}" for l in labels]))

adj_path = os.path.join(HERE, "adjudicated.csv")
if os.path.exists(adj_path):
    adj = pd.read_csv(adj_path).set_index("id")["category_final"]
    fin = key.id.map(adj)
else:
    print("adjudicated.csv not found: using the shared category where both annotators agree, the rest 'unresolved'.")
    fin = c1.where(c1 == c2)
key = key.assign(final=fin)
rows = []
for code, name in NAMES.items():
    sub = key[key.final == code]
    rows.append((name, int((sub.error_type == "false_positive").sum()), int((sub.error_type == "false_negative").sum()), len(sub)))
unres = int(key.final.isna().sum())
if unres: rows.append(("Unresolved (disagreement)", int(key[key.final.isna()].error_type.eq("false_positive").sum()), int(key[key.final.isna()].error_type.eq("false_negative").sum()), unres))
tex = ["\\begin{table}[h!]", "\\centering", "\\small",
       f"\\caption{{Manual categorisation of a random sample of {len(key)} out-of-fold errors by two annotators (Cohen's $\\kappa$ = {kappa:.2f}, raw agreement {agree:.2f}; disagreements adjudicated by discussion). FP = false positive, FN = false negative.}}",
       "\\label{tab:errcats}", "\\begin{tabular}{lccc}", "\\toprule", "Category & FP & FN & Total \\\\", "\\midrule"]
tex += [f"{n} & {fp} & {fn} & {t} \\\\" for n, fp, fn, t in rows] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
out = os.path.join(ROOT, "paper_assets"); os.makedirs(out, exist_ok=True)
open(os.path.join(out, "tab_error_categories.tex"), "w", encoding="utf8").write("\n".join(tex) + "\n")
print("wrote paper_assets/tab_error_categories.tex")
