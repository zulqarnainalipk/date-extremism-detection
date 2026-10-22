"""Checks data/dataset.csv against its recorded SHA-256, size and official split.  Usage: python scripts/verify_data.py"""
import hashlib, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "dataset.csv")
EXPECTED = dict(sha256="ea585a13cbdf93c0398be75a2149c0744c7cbec598f3d55597473c8924e36744", rows=3000, train_rows=2250, missing_label_row=2938,
                train_extremist=1006, train_non=1244, test_extremist=123, test_non=626)

def main():
    ok = True
    def check(name, got, want):
        nonlocal ok
        good = got == want; ok &= good; print(f"{'OK  ' if good else 'FAIL'} {name}: {got}" + ("" if good else f" (expected {want})"))
    check("sha256", hashlib.sha256(open(PATH, "rb").read()).hexdigest(), EXPECTED["sha256"])
    df = pd.read_csv(PATH); check("rows", len(df), EXPECTED["rows"]); check("columns", list(df.columns), ["Original_Message", "Extremism_Label"])
    check("missing-label row (0-based)", df.index[df.Extremism_Label.isna()].tolist(), [EXPECTED["missing_label_row"]])
    tr, te = df.iloc[:EXPECTED["train_rows"]], df.iloc[EXPECTED["train_rows"]:].dropna(subset=["Extremism_Label"])
    check("train EXTREMIST", int((tr.Extremism_Label == "EXTREMIST").sum()), EXPECTED["train_extremist"]); check("train NON_EXTREMIST", int((tr.Extremism_Label == "NON_EXTREMIST").sum()), EXPECTED["train_non"])
    check("test EXTREMIST", int((te.Extremism_Label == "EXTREMIST").sum()), EXPECTED["test_extremist"]); check("test NON_EXTREMIST", int((te.Extremism_Label == "NON_EXTREMIST").sum()), EXPECTED["test_non"])
    print("data file verified" if ok else "data file DOES NOT match the recorded version")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
