"""Puts wide tables on landscape pages (the journal class text width is narrow). Run after make_tables.py."""
import os
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_assets"))
for nm in ["tab_encoders", "tab_ablation", "tab_robust", "tab_perclass", "tab_cost"]:
    t = open(nm + ".tex", encoding="utf8").read()
    if "\\begin{landscape}" not in t:
        open(nm + ".tex", "w", encoding="utf8").write("\\begin{landscape}\n" + t + "\\end{landscape}\n")
print("landscape wrapped")
