"""Sanity tests for the repository. Run with `python tests/test_repository.py` or `python -m pytest tests`."""
import os, subprocess, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "results", "tables")
run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, capture_output=True, text=True)

def test_data_file_matches_recorded_version():
    r = run(os.path.join("scripts", "verify_data.py")); assert r.returncode == 0, r.stdout + r.stderr

def test_main_result_tables_exist_and_are_consistent():
    s = pd.read_csv(os.path.join(T, "summary_all_configs.csv")).set_index("config")
    assert "date_full" in s.index and s.loc["date_full", "test_ens_acc"] > 0.8
    cal = pd.read_csv(os.path.join(T, "calibration_single_vs_ensemble.csv")).set_index("config")
    assert cal.loc["date_full", "ece_ensemble"] < cal.loc["date_full", "ece_single_mean"]
    for f in ["config_comparisons", "confusion_and_perclass", "external_transfer", "robustness_and_fairness_probes", "classical_baselines"]:
        assert len(pd.read_csv(os.path.join(T, f + ".csv"))) > 0, f

def test_every_complete_run_has_all_folds():
    runs = os.path.join(ROOT, "results", "runs")
    for d in os.listdir(runs):
        assert len([f for f in os.listdir(os.path.join(runs, d)) if f.startswith("fold") and f.endswith(".npz")]) == 10, d

def test_annotation_agreement_is_recomputable():
    r = run(os.path.join("annotation", "compute_kappa.py")); assert "Cohen's kappa" in r.stdout, r.stdout + r.stderr

def test_analysis_scripts_run():
    r = run(os.path.join("analysis", "run_all.py")); assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert os.path.exists(os.path.join(ROOT, "paper_assets", "tab_encoders.tex"))
    r = run(os.path.join("analysis", "make_results_summary.py")); assert r.returncode == 0, r.stderr[-500:]

if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: fn(); print("PASS", name)
            except AssertionError as ex: failed += 1; print("FAIL", name, str(ex)[:200])
    sys.exit(1 if failed else 0)
