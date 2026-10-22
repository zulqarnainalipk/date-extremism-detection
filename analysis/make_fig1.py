"""Fig. 1 - DATE evaluation pipeline, publication style (vector PDF + PNG).
Augmentation is applied strictly inside the training part of each fold; the outer validation fold and the test part are held out.
Palette: Okabe-Ito (colour-blind safe)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_assets"); os.makedirs(OUT, exist_ok=True)
fams = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": "Arial" if "Arial" in fams else "DejaVu Sans", "pdf.fonttype": 42, "savefig.dpi": 400})
GREEN, AMBER, RED, PURPLE, BLUE, GREY, INK = "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#0072B2", "#8a8f98", "#1f2933"
LGREEN, LAMBER, LRED, LPURPLE, LBLUE, LGREY = "#d9f2ea", "#fbeccb", "#f8dccd", "#f3dce9", "#d6e8f5", "#eef0f3"

fig, ax = plt.subplots(figsize=(7.2, 5.7)); ax.set_xlim(0, 74); ax.set_ylim(-8.2, 56); ax.axis("off")

def shadow_box(x, y, w, h, fc, ec, lw=1.0, r=0.9, ls="-"):
    ax.add_patch(FancyBboxPatch((x + .25, y - .25), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="#00000014", ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))

def label(x, y, s, fs=6.6, w="normal", c=INK, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=fs, fontweight=w, color=c, ha=ha, va=va, zorder=5, linespacing=1.25, **kw)

def arrow(p, q, c=INK, rad=0.0, lw=1.1, ls="-", ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0))

def badge(x, y, n):
    ax.add_patch(Circle((x, y), 1.35, fc=INK, ec="none", zorder=5)); label(x, y - .05, str(n), 7.5, "bold", "white")

# ------------------------------------------------------------------ (1) data
badge(1.8, 53.6, 1); label(4.0, 53.6, "Data and official split", 8.2, "bold", ha="left")
label(72.0, 53.6, "dataset.csv  |  3,000 posts  |  post-competition labels", 6.2, c="#52606d", ha="right")
x0, wtot, ybar, hbar = 1.2, 70.8, 45.0, 6.2
wtr = wtot * 2250 / 3000
shadow_box(x0, ybar, wtr - .3, hbar, LGREEN, GREEN, 1.3); shadow_box(x0 + wtr + .3, ybar, wtot - wtr - .3, hbar, LRED, RED, 1.3)
label(x0 + wtr / 2, ybar + hbar / 2 + .9, "Train part", 8, "bold", GREEN); label(x0 + wtr / 2, ybar + hbar / 2 - 1.2, "rows 1-2,250  |  2,250 posts  |  44.7% extremist", 6.4)
tx = x0 + wtr + (wtot - wtr) / 2 + .15
label(tx, ybar + hbar / 2 + .9, "Test part", 8, "bold", RED); label(tx, ybar + hbar / 2 - 1.2, "749 posts | 16.4%", 6.0)
label(tx, ybar - 1.9, "never used for training,\nearly stopping or tuning", 5.8, c=RED)
label(x0 + wtr / 2, ybar - 1.5, "10-fold stratified cross-validation (seed 42)", 6.2, c="#52606d")

# ------------------------------------------------------------------ (2) fold-wise training
badge(1.8, 38.6, 2); label(4.0, 38.6, "Fold-wise training: augmentation happens inside the fold", 8.2, "bold", ha="left")
label(1.2, 34.6, "10 outer folds", 6.4, "bold", ha="left")
for i in range(10):
    c, e = (LRED, RED) if i == 3 else (LGREEN, GREEN)
    shadow_box(1.2 + i * 2.15, 29.8, 1.8, 3.3, c, e, 0.9, 0.35)
ax.add_patch(Rectangle((1.2 + 3 * 2.15, 29.8), 1.8, 3.3, fill=False, ec=RED, lw=1.5, hatch="////", zorder=3))
vx = 1.2 + 3 * 2.15 + .9
label(vx, 27.9, "outer validation fold (~225 posts)", 5.8, c=RED, ha="left")
xs, ws = 27.5, 21.0
label(xs, 34.6, "Inner split (outer training part)", 6.4, "bold", ha="left")
shadow_box(xs, 29.8, ws * .9 - .2, 3.3, LGREEN, GREEN, 1.1, 0.4); shadow_box(xs + ws * .9 + .2, 29.8, ws * .1 - .2, 3.3, LAMBER, AMBER, 1.1, 0.4)
label(xs + ws * .45, 31.45, "inner train (90%)", 6.2, c=GREEN, w="bold"); label(xs + ws * .95, 27.9, "inner val (10%)", 5.6, c="#9a6a00")
arrow((1.2 + 10 * 2.15 + .2, 31.45), (xs - .4, 31.45), c="#52606d")
shadow_box(xs, 21.0, ws * .9 - .2, 5.6, LPURPLE, PURPLE, 1.3)
label(xs + ws * .45, 25.0, "Augmentation, ~3.1x size", 6.6, "bold", "#8e3c73"); label(xs + ws * .45, 22.55, "contextual substitution | insertion\nswap | deletion (inner train only)", 5.6)
arrow((xs + ws * .45, 29.8), (xs + ws * .45, 26.8), c=PURPLE, lw=1.3)
shadow_box(54.0, 21.0, 17.6, 12.1, LBLUE, BLUE, 1.4)
label(62.8, 30.8, "BERTweet-base", 7.2, "bold", BLUE); label(62.8, 26.0, "weighted cross-entropy\nAdamW | cosine schedule\nfp16 | max 8 epochs\nbest epoch by\ninner-val loss", 5.8)
arrow((xs + ws * .9 - .2, 23.8), (53.9, 25.0), c=PURPLE, lw=1.3)
arrow((xs + ws * .95, 33.2), (54.0, 32.4), c=AMBER, rad=-.45, ls=(0, (3, 2)), lw=1.3)
label(51.6, 36.6, "early stopping,\ntemperature scaling", 5.4, c="#9a6a00", ha="center")

# ------------------------------------------------------------------ (3) prediction and evaluation
badge(13.2, 17.4, 3); label(15.4, 17.4, "Prediction, soft voting and evaluation", 8.2, "bold", ha="left")
gx = 55.0
for i in range(10):
    shadow_box(gx + i * 1.55, 10.6, 1.3, 3.0, LBLUE, BLUE, 0.8, 0.3)
label(gx + 7.2, 15.2, "one model per fold (M1 ... M10)", 5.6, c=BLUE)
arrow((62.8, 20.9), (62.8, 16.4), c=BLUE)
ax.add_patch(Circle((49.5, 12.1), 2.5, fc=LGREY, ec=INK, lw=1.2, zorder=3)); label(49.5, 12.2, "$\\bar{P}$", 9, "bold")
label(46.6, 12.1, "mean of the 10\nsoftmax outputs", 5.6, ha="right")
arrow((gx - .3, 12.1), (52.1, 12.1), c=BLUE)

def card(x, y, w, h, fc, ec, title, body, ls="-"):
    shadow_box(x, y, w, h, fc, ec, 1.3, 0.8, ls); label(x + w / 2, y + h - 1.5, title, 6.8, "bold", ec if ec != GREY else "#52606d"); label(x + w / 2, y + h / 2 - 1.0, body, 5.8)
card(1.2, -3.0, 22.0, 8.2, LRED, RED, "CV and pooled out-of-fold", "single models, outer validation\nfolds (n = 2,250): mean +- sd,\nECE, Brier, NLL")
card(25.2, -3.0, 22.0, 8.2, LGREY, GREY, "Competition (Kaggle)", "reported separately: earlier\nmodels, other label version,\n~375 private posts", ls=(0, (4, 2)))
card(49.2, -3.0, 22.8, 8.2, LRED, RED, "Held-out test (n = 749)", "single models vs the 10-model\nensemble, 95% bootstrap CIs,\npaired differences")
arrow((vx, 26.9), (vx, 5.3), c=RED, lw=1.2)                                  # validation fold -> CV / OOF card
arrow((49.5, 9.5), (53.0, 5.3), c=RED, lw=1.2)                               # ensemble -> test card
arrow((gx + 7.2, 10.4), (65.0, 5.3), c=RED, lw=1.1, ls=(0, (3, 2)))          # single models -> test card
ax.plot([x0 + wtot - .2, 73.2, 73.2], [ybar + hbar / 2, ybar + hbar / 2, 1.1], color=RED, lw=1.2, ls=(0, (3, 2)), zorder=4)   # test part -> test card
arrow((73.2, 1.1), (72.2, 1.1), c=RED, lw=1.2)
# legend
lx = 1.2
for c, e, t, w in [(LGREEN, GREEN, "trains the model", 13.5), (LAMBER, AMBER, "selection only", 12), (LRED, RED, "held out for reporting", 16.5), (LPURPLE, PURPLE, "augmentation", 11)]:
    ax.add_patch(FancyBboxPatch((lx, -6.9), 1.6, 1.0, boxstyle="round,pad=0,rounding_size=0.25", fc=c, ec=e, lw=1.0)); label(lx + 2.3, -6.4, t, 5.8, ha="left"); lx += w + 3.0
fig.savefig(os.path.join(OUT, "Fig1_pipeline.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "Fig1_pipeline.png"), bbox_inches="tight"); plt.close(fig)
print("Fig1 written", "Arial" if "Arial" in fams else "DejaVu")
