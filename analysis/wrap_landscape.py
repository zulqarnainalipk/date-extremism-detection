"""Kept for compatibility: wide tables now fit the text width through shortened headers in make_tables.py, so nothing is wrapped."""
import os
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_assets"))
for nm in []:
    t = open(nm + ".tex", encoding="utf8").read()
    if "\\resizebox" not in t:
        t = t.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
        open(nm + ".tex", "w", encoding="utf8").write(t)
print("wide tables fitted to text width")
