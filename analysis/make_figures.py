"""Regenerates the figures of the paper from results/ (output of src/date_pipeline.py) and data/dataset.csv."""
import os, json, shutil
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "results"); OUT = os.path.join(ROOT, "paper_assets"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

# ---------- data
raw = pd.read_csv(f"{ROOT}/data/dataset.csv"); raw["row"] = np.arange(len(raw))
df = raw.dropna(subset=["Extremism_Label", "Original_Message"]).reset_index(drop=True); df["y"] = (df.Extremism_Label == "EXTREMIST").astype(int)
tr, te = df[df.row < 2250], df[df.row >= 2250]
y_tr, y_te = tr.y.values, te.y.values
oof = np.zeros((len(tr), 2)); tests = []
for k in range(10):
    z = np.load(f"{B}/runs/date_full_s42/fold{k}.npz", allow_pickle=True); oof[z["va_idx"]] = z["oof"]; tests.append(z["test"])
tests = np.stack(tests); ens = tests.mean(0)

# ---------- Fig 2: EDA from the exact file used
from transformers import AutoTokenizer
try:
    tk = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True); ntok = lambda T: [len(x) for x in tk(list(T))["input_ids"]]
except Exception:
    ntok = lambda T: [len(t.split()) + 2 for t in T]
ltr, lte = ntok(tr.Original_Message), ntok(te.Original_Message)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
w = 0.38; xs = np.arange(2)
ax[0].bar(xs - w / 2, [(y_tr == 0).sum(), (y_tr == 1).sum()], w, label="Train (2,250)"); ax[0].bar(xs + w / 2, [(y_te == 0).sum(), (y_te == 1).sum()], w, label="Test (749)")
ax[0].set_xticks(xs); ax[0].set_xticklabels(["NON_EXTREMIST", "EXTREMIST"], fontsize=8); ax[0].set_ylabel("Posts"); ax[0].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.78)); ax[0].set_ylim(0, 1500); ax[0].set_title("(a) Class distribution", fontsize=10)
for i, (n0, n1) in enumerate([((y_tr == 0).sum(), (y_tr == 1).sum()), ((y_te == 0).sum(), (y_te == 1).sum())]):
    ax[0].text(0 + (i - .5) * w, n0, f"{n0}\n({n0/(n0+n1):.0%})", ha="center", va="bottom", fontsize=7); ax[0].text(1 + (i - .5) * w, n1, f"{n1}\n({n1/(n0+n1):.0%})", ha="center", va="bottom", fontsize=7)
bins = np.arange(0, 135, 5)
ax[1].hist(np.clip(ltr, 0, 130), bins=bins, alpha=.7, label="Train", density=True); ax[1].hist(np.clip(lte, 0, 130), bins=bins, alpha=.7, label="Test", density=True)
ax[1].axvline(128, color="r", ls="--", lw=1); ax[1].set(xlabel="BERTweet tokens (clipped at 130)", ylabel="Density", title="(b) Token length, L=128"); ax[1].title.set_fontsize(10); ax[1].legend(fontsize=8)
ltr = np.array(ltr)
ax[2].hist(np.clip(ltr[y_tr == 0], 0, 130), bins=bins, alpha=.7, label="NON_EXTREMIST", density=True); ax[2].hist(np.clip(ltr[y_tr == 1], 0, 130), bins=bins, alpha=.7, label="EXTREMIST", density=True)
ax[2].set(xlabel="BERTweet tokens (train)", ylabel="Density", title="(c) Token length by class"); ax[2].title.set_fontsize(10); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/Fig2_eda.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: augmentation sizes
a = pd.read_csv(f"{B}/tables/augmentation_sizes.csv")
fig, ax = plt.subplots(figsize=(6, 3.2)); x = np.arange(len(a)); ax.bar(x - .2, a.before, .4, label="Before (inner-train)"); ax.bar(x + .2, a.after, .4, label="After augmentation")
ax.set(xlabel="Outer fold", ylabel="Training posts", xticks=x, xticklabels=x + 1); ax.legend(fontsize=8)
for i, r in a.iterrows(): ax.text(i + .2, r.after, f"{r.factor:.2f}x", ha="center", va="bottom", fontsize=6.5)
fig.savefig(f"{OUT}/Fig4_augmentation.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: loss curves (shared copy) ; Fig 6 per-fold F1 / ECE
shutil.copy(f"{B}/figures/loss_curves_date_full.png", f"{OUT}/Fig5_loss.png")
pf = pd.read_csv(f"{B}/tables/per_fold_metrics.csv"); g = pf[pf.config == "date_full"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for a_, col, lab in [(ax[0], "f1_weighted", "Weighted F1 (outer fold)"), (ax[1], "ece15", "ECE, 15 bins (single model, outer fold)")]:
    m, s = g[col].mean(), g[col].std(ddof=1); a_.plot(g.fold, g[col], "o-", ms=4); a_.axhline(m, color="k", lw=1); a_.fill_between([1, 10], m - s, m + s, alpha=.15)
    a_.set(xlabel="Fold", ylabel=lab, xticks=range(1, 11)); a_.set_title(f"mean {m:.4f} $\\pm$ {s:.4f}", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/Fig6_f1.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 7: reliability with per-bin counts, three panels (OOF single / test single / test ensemble)
def rel(ax, p, y, title, bins=10):
    conf = p.max(1); corr = (p.argmax(1) == y).astype(float); edges = np.linspace(0.5, 1.0, bins + 1); which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    acc = np.array([corr[which == b].mean() if (which == b).any() else np.nan for b in range(bins)]); cnt = np.array([(which == b).sum() for b in range(bins)])
    mc = np.array([conf[which == b].mean() if (which == b).any() else np.nan for b in range(bins)])
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1); ax.bar((edges[:-1] + edges[1:]) / 2, np.nan_to_num(acc), width=0.045, alpha=.75)
    ax.plot(mc, acc, "o", color="C3", ms=3)
    for b in range(bins):
        if cnt[b]: ax.text((edges[b] + edges[b + 1]) / 2, 0.02, str(cnt[b]), ha="center", fontsize=6.5, rotation=90, color="w" if acc[b] > .15 else "k")
    ax.set(xlim=(.5, 1), ylim=(0, 1), xlabel="Predicted confidence (bin)", ylabel="Accuracy"); ax.set_title(title, fontsize=9)
def ece(p, y, bins=15):
    conf = p.max(1); corr = (p.argmax(1) == y); edges = np.linspace(0, 1, bins + 1); w_ = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return sum((w_ == b).sum() / len(y) * abs(corr[w_ == b].mean() - conf[w_ == b].mean()) for b in range(bins) if (w_ == b).any())
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
rel(ax[0], oof, y_tr, f"(a) Out-of-fold, single models\n(n=2,250, ECE={ece(oof, y_tr):.3f})")
rel(ax[1], tests[0], y_te, f"(b) Test, single model (fold 1)\n(n=749, ECE={ece(tests[0], y_te):.3f})")
rel(ax[2], ens, y_te, f"(c) Test, 10-model ensemble\n(n=749, ECE={ece(ens, y_te):.3f})")
fig.tight_layout(); fig.savefig(f"{OUT}/Fig7_reliability.png", bbox_inches="tight"); plt.close(fig)

# ---------- straight copies
for src, dst in [("confusion_oof_date_full", "Fig8a_cm_oof"), ("confusion_test_date_full", "Fig8b_cm_test"), ("confidence_test_date_full", "Fig9_confidence"),
                 ("ensemble_size_curve", "Fig11_ensemble_size"), ("risk_coverage_date_full", "Fig12_risk_coverage"), ("learning_curve_augmentation", "Fig13_learning_curve"),
                 ("robustness_flip_rate", "Fig14_robustness"), ("calibration_single_vs_ensemble", "Fig15_calibration_all")]:
    shutil.copy(f"{B}/figures/{src}.png", f"{OUT}/{dst}.png")
print("figures written to", OUT)
