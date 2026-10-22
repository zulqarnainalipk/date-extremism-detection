"""Regenerates every figure and LaTeX table used in the paper from results/ (no GPU needed).
Usage:  python analysis/run_all.py       ->  files appear in paper_assets/"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for script in ["make_figures.py", "make_fig1.py", "make_tables.py", "wrap_landscape.py"]:
    print("running", script)
    subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)
print("done: see paper_assets/")
