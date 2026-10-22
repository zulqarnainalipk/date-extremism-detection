# %% [markdown]
# # DATE experiment pipeline (single file, notebook style)
# Training, evaluation and analysis for the DATE paper: robustness / generalisation / fairness / calibration studies.
#
# Data: `dataset.csv` (post-competition labels). Official split by row: rows 0-2249 train, rows 2250+ test.
# Resumable: every fold saves `OUT/runs/<config>_s<seed>/fold<k>.npz`; finished folds are skipped.
#
# Env vars
#   RUN=core,models            config names and/or groups (core, models, curve, augops, all)
#   SEEDS=42,43,44             first seed is the "primary" seed for single-seed tables
#   ANALYZE_ONLY=1             skip training; rebuild all analyses from saved folds
#   SMOKE=1 / SMOKE_TINY=1     tiny CPU test (SMOKE_TINY swaps every model for BERT-tiny)
#   DATA_CSV, OUT, COMP_TRAIN  paths (COMP_TRAIN = competition-time train.csv, optional label-version analysis)

# %% Setup
import os, sys, json, time, copy, random, hashlib, subprocess, glob, math, re, gc, shutil, warnings, logging
warnings.filterwarnings("ignore")

def _ensure(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name], check=False)

for _imp, _pip in [("transformers", None), ("sklearn", "scikit-learn"), ("scipy", None), ("datasets", None), ("sentencepiece", None), ("google.protobuf", "protobuf")]:
    _ensure(_imp, _pip)

import numpy as np, pandas as pd
from tqdm.auto import tqdm
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (average_precision_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_fscore_support, log_loss, balanced_accuracy_score, matthews_corrcoef)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.csgraph import connected_components
from scipy.stats import binomtest, ttest_1samp, t as t_dist

SMOKE_TINY = os.environ.get("SMOKE_TINY", "0") == "1"
SMOKE = os.environ.get("SMOKE", "0") == "1" or SMOKE_TINY
IN_KAGGLE = os.path.exists("/kaggle/input")
OUT = os.environ.get("OUT", "/kaggle/working/outputs" if IN_KAGGLE else ("outputs_smoke" if SMOKE else "results"))
for sub in ["runs", "aug", "tables", "figures", "exports", "audit"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

log = logging.getLogger("date")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for _h in (logging.StreamHandler(sys.stdout), logging.FileHandler(f"{OUT}/run.log", mode="a", encoding="utf-8")):
        _h.setFormatter(_fmt); log.addHandler(_h)
    log.propagate = False

SEED = 42
N_FOLDS = 3 if SMOKE else 10
MAX_LEN = 128
BATCH, ACCUM = (8, 1) if SMOKE else (64, 2)
MAX_EPOCHS = 1 if SMOKE else 8        # cosine schedule is defined over MAX_EPOCHS so it really decays
PATIENCE = 3
LR, WD, WARMUP = 2e-5, 0.01, 0.10
INNER_VAL_FRAC = 0.10                 # held out of each training part for early stopping + temperature scaling
N_BOOT = 100 if SMOKE else 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
N_TRAIN_OFFICIAL = 2250               # rows 0..2249 = train, rows 2250.. = test (official competition split)
TINY = "google/bert_uncased_L-2_H-128_A-2"
def resolve(name): return TINY if SMOKE_TINY else name

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(SEED)
log.info("device=%s | smoke=%s tiny=%s | out=%s | folds=%d | max_epochs=%d | batch=%dx%d", DEVICE, SMOKE, SMOKE_TINY, OUT, N_FOLDS, MAX_EPOCHS, BATCH, ACCUM)

# %% Experiment grid
BT = "vinai/bertweet-base"
def C(model=BT, aug=True, cw=True, sched="cosine", split="strat", **kw):
    return dict(model=model, aug=aug, cw=cw, sched=sched, split=split, **kw)
CONFIGS = {
    # --- core: ablation and control grid
    "date_full":     C(),
    "no_aug":        C(aug=False, probe_aug=True),
    "no_cw":         C(cw=False),
    "no_cosine":     C(sched="constant"),
    "roberta_base":  C(model="roberta-base"),
    "date_grouped":  C(split="group"),
    # --- other encoders under the identical protocol (is BERTweet the right choice?)
    "bert_base":           C(model="bert-base-uncased"),
    "distilbert":          C(model="distilbert-base-uncased"),
    "deberta_v3":          C(model="microsoft/deberta-v3-base"),
    "xlm_roberta":         C(model="xlm-roberta-base"),
    "twitter_roberta":     C(model="cardiffnlp/twitter-roberta-base"),
    "twitter_roberta_hate": C(model="cardiffnlp/twitter-roberta-base-hate-latest"),
    "twhin_bert":          C(model="Twitter/twhin-bert-base"),
    "toxic_bert":          C(model="unitary/toxic-bert"),
    "bertweet_large":      C(model="vinai/bertweet-large", bs=32, accum=4),
    # --- learning curves (does augmentation help when data is scarce?)
    "date_full_f25": C(frac=0.25), "date_full_f50": C(frac=0.50),
    "no_aug_f25":    C(aug=False, frac=0.25), "no_aug_f50": C(aug=False, frac=0.50),
    # --- which augmentation operators matter
    "aug_ctx_only":  C(aug_ops="ctx"), "aug_eda_only": C(aug_ops="eda"),
}
GROUPS = {
    "core":   ["date_full", "no_aug", "no_cw", "no_cosine", "roberta_base", "date_grouped"],
    "models": ["bert_base", "distilbert", "deberta_v3", "xlm_roberta", "twitter_roberta", "twitter_roberta_hate", "twhin_bert", "toxic_bert", "bertweet_large"],
    "curve":  ["date_full_f25", "date_full_f50", "no_aug_f25", "no_aug_f50"],
    "augops": ["aug_ctx_only", "aug_eda_only"],
}
GROUPS["all"] = [c for g in ("core", "models", "curve", "augops") for c in GROUPS[g]]
_run = os.environ.get("RUN", "date_full" if SMOKE else "core").split(",")
RUN = [c for r in _run if r for c in (GROUPS[r] if r in GROUPS else [r])]
SEEDS = [int(s) for s in os.environ.get("SEEDS", str(SEED)).split(",")]
PRIMARY = SEEDS[0]
ANALYZE_ONLY = os.environ.get("ANALYZE_ONLY", "0") == "1"
log.info("configs to run: %s | seeds: %s | analyze_only=%s", RUN, SEEDS, ANALYZE_ONLY)

# %% 1. Data: load, official split, statistics, datasheet extras
def find_csv():
    for c in [os.environ.get("DATA_CSV"), "data/dataset.csv", "../data/dataset.csv", "dataset.csv"] + glob.glob("/kaggle/input/**/dataset.csv", recursive=True):
        if c and os.path.exists(c): return c
    raise FileNotFoundError("dataset.csv not found; set DATA_CSV")

CSV = find_csv()
sha256 = hashlib.sha256(open(CSV, "rb").read()).hexdigest()
raw = pd.read_csv(CSV); raw["row"] = np.arange(len(raw))
missing_rows = raw.loc[raw.Extremism_Label.isna(), "row"].tolist()
df = raw.dropna(subset=["Extremism_Label", "Original_Message"]).reset_index(drop=True)
df["y"] = (df.Extremism_Label == "EXTREMIST").astype(int)         # 1 = EXTREMIST
tr_df = df[df.row < N_TRAIN_OFFICIAL].reset_index(drop=True)
te_df = df[df.row >= N_TRAIN_OFFICIAL].reset_index(drop=True)
if SMOKE:
    tr_df = tr_df.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), 60), random_state=0)).reset_index(drop=True)
    te_df = te_df.groupby("y", group_keys=False).apply(lambda g: g.sample(min(len(g), 20), random_state=0)).reset_index(drop=True)
X_train, y_train = tr_df.Original_Message.tolist(), tr_df.y.values
X_test, y_test = te_df.Original_Message.tolist(), te_df.y.values
N_TR, N_TE = len(X_train), len(X_test)
ROW_TRAIN = tr_df.row.values

TOKS = {}
def get_tok(name):
    name = resolve(name)
    if name not in TOKS:
        from transformers import AutoTokenizer
        try: TOKS[name] = AutoTokenizer.from_pretrained(name, use_fast=True)
        except Exception: TOKS[name] = AutoTokenizer.from_pretrained(name, use_fast=False)
    return TOKS[name]
tok0 = get_tok(BT)
def ntok(texts): return [len(x) for x in tok0(texts, add_special_tokens=True)["input_ids"]]
wl_tr = [len(t.split()) for t in X_train]; tl_tr = ntok(X_train); tl_te = ntok(X_test)

stats = {
    "csv": CSV, "sha256": sha256, "rows_in_file": len(raw), "missing_label_rows": missing_rows,
    "train_n": N_TR, "test_n": N_TE, "train_ext": int(y_train.sum()), "train_non": int(N_TR - y_train.sum()),
    "test_ext": int(y_test.sum()), "test_non": int(N_TE - y_test.sum()),
    "train_ext_pct": float(y_train.mean() * 100), "test_ext_pct": float(y_test.mean() * 100),
    "words_mean": float(np.mean(wl_tr)), "words_median": float(np.median(wl_tr)), "words_min": int(min(wl_tr)), "words_max": int(max(wl_tr)),
    "bertweet_tokens_mean": float(np.mean(tl_tr)), "bertweet_tokens_median": float(np.median(tl_tr)), "bertweet_tokens_max": int(max(tl_tr)),
    "train_truncated_at_128": int(sum(t > MAX_LEN for t in tl_tr)), "test_truncated_at_128": int(sum(t > MAX_LEN for t in tl_te)),
    "exact_duplicate_texts_train": int(pd.Series(X_train).duplicated().sum()), "test_texts_also_in_train": int(len(set(X_test) & set(X_train))),
    "urls_pct": float(np.mean(["http" in t for t in X_train]) * 100),
}
try:                                                              # approximate language composition (R1/R2: language and platform distribution)
    _ensure("langdetect")
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    def _lang(t):
        try: return detect_langs(t)[0].lang
        except Exception: return "unk"
    _langs = pd.Series([_lang(t) for t in (X_train + X_test)])
    _lt = _langs.value_counts(normalize=True) * 100; _lt.to_csv(f"{OUT}/audit/language_distribution_pct.csv")
    stats["pct_detected_english"] = float(_lt.get("en", 0.0)); stats["top_detected_languages_pct"] = {k: round(float(v), 2) for k, v in _lt.head(6).items()}
except Exception as _ex:
    log.warning("language detection skipped: %s", _ex)
stats["pct_posts_with_hashtag_or_mention_markers"] = float(np.mean([("#" in t) or ("@" in t) for t in X_train]) * 100)
stats["pct_posts_with_emoji_description_phrases"] = float(np.mean([any(k in t for k in ("face with", "tears of joy", "red heart", "thumbs up", "pensive face", "loudly crying")) for t in X_train]) * 100)
json.dump(stats, open(f"{OUT}/tables/dataset_stats.json", "w"), indent=2)
log.info("dataset stats:\n%s", json.dumps(stats, indent=2))

# datasheet extras: class-conditional vocabulary (log-odds) -- shows what lexical cues the data contains
from sklearn.feature_extraction.text import CountVectorizer
_cv = CountVectorizer(min_df=5, binary=True).fit(X_train)
_Xb = _cv.transform(X_train); _voc = np.array(_cv.get_feature_names_out())
_a = np.asarray(_Xb[y_train == 1].sum(0)).ravel() + 1; _b = np.asarray(_Xb[y_train == 0].sum(0)).ravel() + 1
_lo = np.log(_a / _a.sum()) - np.log(_b / _b.sum())
pd.DataFrame(dict(word=_voc, log_odds_ext_vs_non=_lo, n_ext=_a - 1, n_non=_b - 1)).sort_values("log_odds_ext_vs_non").to_csv(f"{OUT}/exports/vocab_log_odds.csv", index=False)

# lexicons used for slices, error tags and fairness probes
PROFANITY = set("fuck fucking fucked shit bitch bitches ass asshole bastard cunt dick pussy whore slut hoe skank damn crap piss idiot stupid".split())
VIOLENCE = set("kill killed killing die dies dead death burn shoot shot rape raped murder hang lynch bomb stab destroy exterminate genocide attack".split())
IDENT_P = ["muslims", "christians", "jews", "hindus", "blacks", "whites", "asians", "arabs", "mexicans", "immigrants", "gays", "women"]
IDENT_S = ["muslim", "christian", "jewish", "hindu", "black", "white", "asian", "arab", "mexican", "immigrant", "gay", "woman"]
_cyc = {}
for _l in (IDENT_P, IDENT_S):
    for _i, _w in enumerate(_l): _cyc[_w] = _l[(_i + 1) % len(_l)]
def has_any(t, S): return any(w in S for w in t.split())
def ident_swap(t): return " ".join(_cyc.get(w, w) for w in t.split())
def ident_neutral(t): return " ".join("people" if w in _cyc else w for w in t.split())
ID_MASK_TEST = np.array([has_any(t, _cyc) for t in X_test]); ID_MASK_TRAIN = np.array([has_any(t, _cyc) for t in X_train])
log.info("posts mentioning an identity term: train %.1f%% test %.1f%%", ID_MASK_TRAIN.mean() * 100, ID_MASK_TEST.mean() * 100)

# optional: label-version analysis (competition-time train.csv vs post-competition dataset.csv)
COMP_TRAIN = [p for p in [os.environ.get("COMP_TRAIN")] + glob.glob("/kaggle/input/**/train.csv", recursive=True) if p and os.path.exists(p)]
CHANGED = None
if COMP_TRAIN and not SMOKE:
    try:
        old = pd.read_csv(COMP_TRAIN[0]).dropna(subset=["Original_Message", "Extremism_Label"])
        old_map = dict(zip(old.Original_Message, (old.Extremism_Label == "EXTREMIST").astype(int)))
        OLD_LABEL = np.array([old_map.get(t, -1) for t in X_train])
        CHANGED = (OLD_LABEL >= 0) & (OLD_LABEL != y_train)
        log.info("label-version analysis: competition train.csv=%s | matched=%d | relabelled=%d (%.1f%%)", COMP_TRAIN[0], (OLD_LABEL >= 0).sum(), CHANGED.sum(), CHANGED.mean() * 100)
    except Exception as ex:
        log.warning("label-version analysis skipped: %s", ex)

# %% 2. Leakage audit: near-duplicates, duplicate-cluster groups
SIM_THR = 0.90
vec = TfidfVectorizer().fit(X_train + X_test)
A, B = vec.transform(X_train), vec.transform(X_test)
S_tt = (A @ A.T).toarray(); np.fill_diagonal(S_tt, 0)
nn_train = S_tt.max(1); SIM_TEST = (B @ A.T).toarray().max(1)
_, GROUPS_ID = connected_components(S_tt > SIM_THR, directed=False)
log.info("near-dup clusters (cos>%.2f): %d clusters over %d posts", SIM_THR, GROUPS_ID.max() + 1, N_TR)

def splits_for(kind, seed=SEED):
    if kind == "group":
        return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_train, y_train, GROUPS_ID))
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_train, y_train))

audit_rows = []; VAL_SIM = np.zeros(N_TR)           # max similarity of each train post to the training part of its own (random-split) fold
for kind in ["strat", "group"]:
    for k, (a, b) in enumerate(tqdm(splits_for(kind, PRIMARY), desc=f"audit {kind}", leave=False)):
        sim = (A[b] @ A[a].T).toarray().max(1)
        if kind == "strat": VAL_SIM[b] = sim
        audit_rows.append(dict(split=kind, fold=k + 1, n_val=len(b), val_with_neardup_in_train_pct=float((sim > SIM_THR).mean() * 100), val_max_sim_mean=float(sim.mean())))
audit = pd.DataFrame(audit_rows); audit.to_csv(f"{OUT}/audit/neardup_per_fold.csv", index=False)
summary_audit = audit.groupby("split")[["val_with_neardup_in_train_pct", "val_max_sim_mean"]].mean()
summary_audit.loc["test_vs_train"] = [float((SIM_TEST > SIM_THR).mean() * 100), float(SIM_TEST.mean())]
summary_audit.loc["train_vs_train"] = [float((nn_train > SIM_THR).mean() * 100), float(nn_train.mean())]
summary_audit.to_csv(f"{OUT}/audit/neardup_summary.csv")
log.info("near-duplicate audit:\n%s", summary_audit.round(3))

# %% 3. Augmentation (training part of each fold only; cached; three operator sets)
class ContextualSubstituter:
    """BERT masked-LM word substitution (replaces nlpaug.ContextualWordEmbsAug, which breaks on recent transformers).
    Replaces 1-3 words, one at a time, each conditioned on the already-edited sentence; the new word is sampled from the
    top-k whole-word predictions (alphabetic tokens only, original word excluded)."""
    def __init__(self, name="bert-base-uncased", top_k=50, aug_min=1, aug_max=3):
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        name = resolve(name)
        self.tk = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForMaskedLM.from_pretrained(name).to(DEVICE).eval()
        self.top_k, self.aug_min, self.aug_max = top_k, aug_min, aug_max
        vocab = sorted(self.tk.get_vocab().items(), key=lambda kv: kv[1])
        self.ok = torch.tensor([t.isalpha() for t, _ in vocab], device=DEVICE)

    @torch.no_grad()
    def augment(self, text):
        words = text.split()
        if not words: return text
        cand = [i for i, w in enumerate(words) if w.isalpha()] or list(range(len(words)))
        n = min(np.random.randint(self.aug_min, self.aug_max + 1), len(cand))
        for pos in np.random.choice(cand, n, replace=False):
            masked = list(words); orig = masked[pos].lower(); masked[pos] = self.tk.mask_token
            enc = self.tk(" ".join(masked), return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
            mpos = (enc["input_ids"][0] == self.tk.mask_token_id).nonzero()
            if len(mpos) == 0: continue
            logits = self.m(**enc).logits[0, mpos[0, 0]].float().masked_fill(~self.ok, float("-inf"))
            top = logits.topk(min(self.top_k + 1, int(self.ok.sum())))
            cands = [(self.tk.convert_ids_to_tokens(int(i)), float(v)) for i, v in zip(top.indices, top.values)]
            cands = [(t, v) for t, v in cands if t != orig][: self.top_k]
            if not cands: continue
            p = np.exp(np.array([v for _, v in cands]) - max(v for _, v in cands)); p /= p.sum()
            words[pos] = cands[np.random.choice(len(cands), p=p)][0]
        return " ".join(words)

_aug = {"obj": None}
def _ctx_aug():
    if _aug["obj"] is None: _aug["obj"] = ContextualSubstituter()
    return _aug["obj"]

def random_insertion(text):
    w = text.split()
    if not w: return text
    w.insert(np.random.randint(len(w)), w[np.random.randint(len(w))]); return " ".join(w)
def random_swap(text):
    w = text.split()
    if len(w) < 2: return text
    i, j = np.random.choice(len(w), 2, replace=False); w[i], w[j] = w[j], w[i]; return " ".join(w)
def random_deletion(text, p=0.1):
    w = text.split()
    if not w: return text
    kept = [x for x in w if np.random.random() > p]
    return " ".join(kept) if kept else w[0]

def augment_text(text, ops="all"):
    out = []
    if ops in ("all", "ctx") and np.random.random() < 0.6:
        r = _ctx_aug().augment(text); out.extend(r if isinstance(r, list) else [r])
    if ops in ("all", "eda"):
        if np.random.random() < 0.3: out.append(random_insertion(text))
        if np.random.random() < 0.3: out.append(random_swap(text))
        if np.random.random() < 0.2: out.append(random_deletion(text))
    return out

def augment_set(texts, labels, seed, ops="all"):
    set_seed(seed); T, L, per = [], [], []
    for t, l in tqdm(list(zip(texts, labels)), desc="augment", leave=False):
        T.append(t); L.append(int(l)); n_new = 0
        for _ in range(np.random.randint(1, 3)):          # 1 or 2 augmentation rounds per post
            for a in augment_text(t, ops):
                if a and a.strip(): T.append(a); L.append(int(l)); n_new += 1
        per.append(n_new)
    return T, L, per

def augment_variants(texts, seed):
    """Augmented variants only (no originals), with the index of the source post -- used to probe label preservation."""
    set_seed(seed); idx, T = [], []
    for i, t in enumerate(texts):
        for a in augment_text(t, "all"):
            if a and a.strip(): idx.append(i); T.append(a)
    return idx, T

def get_augmented(split_name, ops, frac, seed, fold, idx):
    path = f"{OUT}/aug/{split_name}_{ops}_f{frac}_s{seed}_fold{fold}.json"
    if os.path.exists(path):
        try:
            d = json.load(open(path)); return d["texts"], d["labels"]
        except Exception:
            pass
    T, L, per = augment_set([X_train[i] for i in idx], y_train[idx], seed * 1000 + 100 * fold, ops)
    tmp = f"{path}.{os.getpid()}.tmp"                      # atomic write: safe when two GPU processes share the cache
    json.dump(dict(texts=T, labels=L, before=len(idx), after=len(T), mean_new_per_post=float(np.mean(per))), open(tmp, "w"))
    os.replace(tmp, path)
    return T, L

# %% 4. Probe sets: robustness perturbations and external (out-of-distribution) data
_rng = np.random.default_rng(123)
def typo(t, p):
    ws = t.split()
    for i, w in enumerate(ws):
        if len(w) > 3 and _rng.random() < p:
            j = int(_rng.integers(1, len(w) - 2)); ws[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(ws)
def worddrop(t, p):
    ws = [w for w in t.split() if _rng.random() > p]
    return " ".join(ws) if ws else t
ROB_SETS = {"typo10": [typo(t, .10) for t in X_test], "typo20": [typo(t, .20) for t in X_test], "worddrop15": [worddrop(t, .15) for t in X_test],
            "ident_swap": [ident_swap(t) for t in X_test], "ident_neutral": [ident_neutral(t) for t in X_test]}

def clean_ext(t):
    t = re.sub(r"http\S+|@\w+|\brt\b|&\w+;", " ", str(t).lower()); t = re.sub(r"[^a-z0-9\s]", " ", t); return re.sub(r"\s+", " ", t).strip()
EXT_SETS = {}
def _read_external(repo):
    """HF hub first; plain-CSV fallback for Davidson (recent `datasets` versions refuse script-based repos)."""
    try:
        from datasets import load_dataset
        return load_dataset(repo, split="train").to_pandas()
    except Exception as ex:
        if repo.startswith("tdavidson"):
            return pd.read_csv("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv")
        raise ex
def load_external():
    n_tot = 120 if SMOKE else 3000
    for key, repo, col, pos in [("davidson_hate", "tdavidson/hate_speech_offensive", "tweet", lambda d: d["class"] == 0),
                                ("tweets_hate", "tweets_hate_speech_detection", "tweet", lambda d: d["label"] == 1)]:
        try:
            d = _read_external(repo); d["pos"] = pos(d).astype(int); d["text"] = d[col].map(clean_ext)
            d = d[d.text.str.len() > 3]
            ps = d[d.pos == 1].sample(min(int(n_tot * 0.3), (d.pos == 1).sum()), random_state=0); ns = d[d.pos == 0].sample(min(n_tot - len(ps), (d.pos == 0).sum()), random_state=0)
            s_ = pd.concat([ps, ns]).sample(frac=1, random_state=0)
            EXT_SETS[key] = (s_.text.tolist(), s_.pos.values); log.info("external set %s: n=%d pos=%.1f%%", key, len(s_), s_.pos.mean() * 100)
        except Exception as ex:
            log.warning("external set %s unavailable: %s", key, str(ex)[:150])
load_external()            # deterministic sample (random_state=0) so analysis-only passes recover the same labels

# %% 5. Training engine
from transformers import (AutoModelForSequenceClassification, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup)

class DS(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i
def make_loader(ids, labels, pad_id, shuffle, bs):
    def collate(items):
        m = max(len(ids[i]) for i in items)
        x = torch.full((len(items), m), pad_id, dtype=torch.long); a = torch.zeros((len(items), m), dtype=torch.long)
        for r, i in enumerate(items):
            x[r, :len(ids[i])] = torch.tensor(ids[i]); a[r, :len(ids[i])] = 1
        y = torch.tensor([labels[i] for i in items]) if labels is not None else torch.zeros(len(items), dtype=torch.long)
        return x, a, y
    return DataLoader(DS(len(ids)), batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=0)
def encode(tk, texts): return tk(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]

@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Ls = [], 0.0
    for x, a, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            lg = model(input_ids=x.to(DEVICE), attention_mask=a.to(DEVICE)).logits
        P.append(F.softmax(lg.float(), 1).cpu()); Ls += F.cross_entropy(lg.float(), y.to(DEVICE), reduction="sum").item()
    return torch.cat(P).numpy(), Ls
def predict_texts(model, tk, texts):
    return predict(model, make_loader(encode(tk, texts), None, tk.pad_token_id, False, 128))[0].astype(np.float32)

def inner_split(tr_idx, fold, kind, seed):
    ytr = y_train[tr_idx]
    if kind == "group":
        a, b = next(StratifiedGroupKFold(10, shuffle=True, random_state=seed + fold).split(tr_idx, ytr, GROUPS_ID[tr_idx]))
    else:
        a, b = train_test_split(np.arange(len(tr_idx)), test_size=INNER_VAL_FRAC, stratify=ytr, random_state=seed + fold)
    return tr_idx[a], tr_idx[b]

def run_fold(name, cfg, fold, tr_idx, va_idx, seed):
    path = f"{OUT}/runs/{name}_s{seed}/fold{fold}.npz"
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time(); fseed = seed * 1000 + 100 * fold; set_seed(fseed)
    tk = get_tok(cfg["model"]); bs = BATCH if SMOKE else cfg.get("bs", BATCH); accum = ACCUM if SMOKE else cfg.get("accum", ACCUM)
    itr, iva = inner_split(tr_idx, fold, cfg["split"], seed)
    frac = cfg.get("frac", 1.0)
    if frac < 1.0: itr = train_test_split(itr, train_size=frac, stratify=y_train[itr], random_state=fseed)[0]
    if cfg["aug"]: T, L = get_augmented(cfg["split"], cfg.get("aug_ops", "all"), frac, seed, fold, itr)
    else: T, L = [X_train[i] for i in itr], [int(y_train[i]) for i in itr]
    ids_tr, ids_iva = encode(tk, T), encode(tk, [X_train[i] for i in iva]); pad = tk.pad_token_id
    dl_tr = make_loader(ids_tr, L, pad, True, bs)
    dl_iva = make_loader(ids_iva, [int(y_train[i]) for i in iva], pad, False, 128)

    model = AutoModelForSequenceClassification.from_pretrained(resolve(cfg["model"]), num_labels=2, ignore_mismatched_sizes=True).float().to(DEVICE)   # fp32 master weights: GradScaler cannot unscale fp16 parameters
    cnt = np.bincount(L, minlength=2)
    w = torch.tensor(len(L) / (2 * cnt), dtype=torch.float, device=DEVICE) if cfg["cw"] else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=WD)
    steps = math.ceil(len(dl_tr) / accum) * MAX_EPOCHS
    sch = get_cosine_schedule_with_warmup(opt, int(WARMUP * steps), steps) if cfg["sched"] == "cosine" else get_constant_schedule_with_warmup(opt, int(WARMUP * steps))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best = dict(loss=1e9, epoch=0, state=None); bad = 0; hist = []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); tl = 0.0; opt.zero_grad()
        pbar = tqdm(dl_tr, desc=f"{name} s{seed} f{fold+1} ep{ep}/{MAX_EPOCHS}", leave=False)
        for s, (x, a, y) in enumerate(pbar):
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                lg = model(input_ids=x.to(DEVICE), attention_mask=a.to(DEVICE)).logits
            loss = loss_fn(lg.float(), y.to(DEVICE)); tl += loss.item(); pbar.set_postfix(loss=f"{tl/(s+1):.4f}")
            scaler.scale(loss / accum).backward()
            if (s + 1) % accum == 0 or s + 1 == len(dl_tr):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        if not math.isfinite(tl): raise RuntimeError(f"non-finite training loss for {name} (fp16 instability?)")
        p, ls = predict(model, dl_iva); yv = y_train[iva]; vl = ls / len(iva); va = float((p.argmax(1) == yv).mean())
        hist.append(dict(epoch=ep, train_loss=tl / len(dl_tr), inner_val_loss=vl, inner_val_acc=va, lr=opt.param_groups[0]["lr"]))
        log.info("[%s s%d] fold %d ep %d | train_loss=%.4f inner_val_loss=%.4f inner_val_acc=%.4f lr=%.2e", name, seed, fold + 1, ep, tl / len(dl_tr), vl, va, opt.param_groups[0]["lr"])
        if vl < best["loss"]:
            best = dict(loss=vl, epoch=ep, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); bad = 0   # true copy
        else:
            bad += 1
            if bad >= PATIENCE:
                log.info("[%s s%d] fold %d early stop at epoch %d (best=%d)", name, seed, fold + 1, ep, best["epoch"]); break
    model.load_state_dict(best["state"])
    ex = dict(oof=predict_texts(model, tk, [X_train[i] for i in va_idx]), test=predict_texts(model, tk, X_test),
              ival=predict_texts(model, tk, [X_train[i] for i in iva]), ival_idx=iva)
    for nm, texts in ROB_SETS.items(): ex["rob_" + nm] = predict_texts(model, tk, texts)
    for nm, (texts, _) in EXT_SETS.items(): ex["ext_" + nm] = predict_texts(model, tk, texts)
    if cfg.get("probe_aug"):                                        # label-preservation probe: model never trained on augmented text
        pi, pt = augment_variants([X_train[i] for i in va_idx], fseed + 7)
        ex["probe_idx"] = np.array(pi); ex["probe_probs"] = predict_texts(model, tk, pt)
    tmp = path + ".tmp.npz"
    np.savez(tmp, va_idx=va_idx, best_epoch=best["epoch"], n_params=sum(q.numel() for q in model.parameters()), hist=json.dumps(hist), n_train=len(L), n_inner_val=len(iva), seconds=time.time() - t0, **ex)
    os.replace(tmp, path)
    acc = (ex["oof"].argmax(1) == y_train[va_idx]).mean()
    log.info("[%s s%d] DONE fold %d/%d best_ep=%d outer_acc=%.4f n_train=%d (%.0fs)", name, seed, fold + 1, N_FOLDS, best["epoch"], acc, len(L), time.time() - t0)
    del model, opt, best; gc.collect(); torch.cuda.empty_cache()

# %% 6. Run the grid (resumable; a failing config is logged and skipped, not fatal)
_t_start = time.time(); FAILED = set()
if not ANALYZE_ONLY:
    _jobs = [(s, n, f, a, b) for s in SEEDS for n in RUN for f, (a, b) in enumerate(splits_for(CONFIGS[n]["split"], s))]
    for seed, name, fold, a, b in tqdm(_jobs, desc="grid (fold jobs)"):
        if (name, seed) in FAILED: continue
        try: run_fold(name, CONFIGS[name], fold, a, b, seed)
        except Exception as ex:
            FAILED.add((name, seed)); log.error("config %s seed %d FAILED at fold %d: %r", name, seed, fold + 1, ex); torch.cuda.empty_cache()
log.info("grid finished in %.1f min | failed: %s", (time.time() - _t_start) / 60, sorted(FAILED))
json.dump(dict(CONFIGS=CONFIGS, N_FOLDS=N_FOLDS, MAX_EPOCHS=MAX_EPOCHS, PATIENCE=PATIENCE, BATCH=BATCH, ACCUM=ACCUM, LR=LR, WD=WD, WARMUP=WARMUP,
               INNER_VAL_FRAC=INNER_VAL_FRAC, MAX_LEN=MAX_LEN, SEEDS=SEEDS, failed=[list(f) for f in FAILED]), open(f"{OUT}/tables/run_config.json", "w"), indent=2)

# %% 7. Metrics library
def ece_score(p, y, bins=15, adaptive=False):
    conf = p.max(1); correct = (p.argmax(1) == y).astype(float)
    if adaptive: groups = np.array_split(np.argsort(conf), bins)
    else:
        edges = np.linspace(0, 1, bins + 1); which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
        groups = [np.where(which == b)[0] for b in range(bins)]
    return float(sum(len(g) / len(y) * abs(correct[g].mean() - conf[g].mean()) for g in groups if len(g)))
def brier_sum(p, y): return float(((p - np.eye(2)[y]) ** 2).sum(1).mean())     # paper definition (0..2)
def brier_pos(p, y): return float(((p[:, 1] - y) ** 2).mean())                 # standard binary (0..1)
def metrics(p, y):
    pred = p.argmax(1); two = len(set(y)) > 1
    return dict(acc=float(accuracy_score(y, pred)), f1_weighted=float(f1_score(y, pred, average="weighted")), f1_macro=float(f1_score(y, pred, average="macro")),
                bal_acc=float(balanced_accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
                auc=float(roc_auc_score(y, p[:, 1])) if two else float("nan"), pr_auc=float(average_precision_score(y, p[:, 1])) if two else float("nan"),
                ece15=ece_score(p, y, 15), ece10=ece_score(p, y, 10), ece20=ece_score(p, y, 20), ece_adaptive15=ece_score(p, y, 15, True),
                brier_sum=brier_sum(p, y), brier_pos=brier_pos(p, y), nll=float(log_loss(y, np.clip(p, 1e-7, 1), labels=[0, 1])))
def boot_ci(p, y, key, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed); f = {"acc": lambda p, y: (p.argmax(1) == y).mean(), "ece15": ece_score, "brier_sum": brier_sum,
                                            "f1_macro": lambda p, y: f1_score(y, p.argmax(1), average="macro")}[key]
    v = [f(p[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def fit_temp(p, y):
    from scipy.optimize import minimize_scalar
    z = np.log(np.clip(p[:, 1], 1e-7, 1)) - np.log(np.clip(p[:, 0], 1e-7, 1))
    def nll(T):
        q = np.clip(1 / (1 + np.exp(-z / T)), 1e-7, 1 - 1e-7); return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)
def apply_temp(p, T):
    z = np.log(np.clip(p[:, 1], 1e-7, 1)) - np.log(np.clip(p[:, 0], 1e-7, 1)); q = 1 / (1 + np.exp(-z / T)); return np.stack([1 - q, q], 1)
def prior_adjust(p, from_prior, to_prior):
    odds = p[:, 1] / np.clip(p[:, 0], 1e-9, None) * (to_prior / (1 - to_prior)) / (from_prior / (1 - from_prior)); q = odds / (1 + odds); return np.stack([1 - q, q], 1)
def em_prior(p, prior0, iters=200):
    pi = prior0
    for _ in range(iters): pi = prior_adjust(p, prior0, pi)[:, 1].mean()
    return float(pi)
def risk_coverage(p, y, covs=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    corr = (p.argmax(1) == y)[np.argsort(-p.max(1))]
    out = {f"acc@cov{int(c*100)}": float(corr[: max(1, int(round(c * len(y))))].mean()) for c in covs}
    out["aurc"] = float(np.mean(np.cumsum(~corr) / np.arange(1, len(corr) + 1))); return out
def mcnemar(pa, pb, y):
    ca, cb = pa.argmax(1) == y, pb.argmax(1) == y; b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)
def nadeau_bengio(da, k=N_FOLDS):                                     # corrected resampled t-test over CV folds
    da = np.asarray(da); v = da.var(ddof=1) * (1 / k + 1 / (k - 1))
    t = da.mean() / math.sqrt(v) if v > 0 else 0.0; return float(t), float(2 * t_dist.sf(abs(t), k - 1))

def load_run(name, seed):
    d = f"{OUT}/runs/{name}_s{seed}"; fs = [f"{d}/fold{k}.npz" for k in range(N_FOLDS)]
    if not all(os.path.exists(f) for f in fs): return None
    if "ival_idx" not in np.load(fs[0], allow_pickle=True).files:
        log.warning("%s seed %d was produced by an older script version (no ival/probe arrays); ignored", name, seed); return None
    r = dict(oof=np.zeros((N_TR, 2)), fold_of=np.zeros(N_TR, int), tests=[], best_epoch=[], hist=[], seconds=0.0, ival=[], rob={}, ext={}, probe=[])
    for k, f in enumerate(fs):
        z = np.load(f, allow_pickle=True); r["oof"][z["va_idx"]] = z["oof"]; r["fold_of"][z["va_idx"]] = k
        r["tests"].append(z["test"]); r["best_epoch"].append(int(z["best_epoch"])); r["hist"].append(json.loads(str(z["hist"]))); r["seconds"] += float(z["seconds"])
        r["ival"].append((z["ival_idx"], z["ival"]))
        for key in z.files:
            if key.startswith("rob_"): r["rob"].setdefault(key[4:], []).append(z[key])
            if key.startswith("ext_"): r["ext"].setdefault(key[4:], []).append(z[key])
        if "probe_idx" in z.files: r["probe"].append((z["va_idx"], z["probe_idx"], z["probe_probs"]))
    r["n_params"] = int(np.load(fs[0], allow_pickle=True)["n_params"])
    r["tests"] = np.stack(r["tests"]); r["rob"] = {k: np.stack(v) for k, v in r["rob"].items()}; r["ext"] = {k: np.stack(v) for k, v in r["ext"].items()}
    return r

RES, MS = {}, {}
for _n in CONFIGS:
    for _s in sorted(set(SEEDS) | {PRIMARY}):
        _r = load_run(_n, _s)
        if _r:
            MS[(_n, _s)] = _r
            if _s == PRIMARY: RES[_n] = _r
log.info("complete configs (primary seed %d): %s", PRIMARY, list(RES))
if not RES: log.warning("nothing to analyse yet")
def ens_of(r): return r["tests"].mean(0)
def eff_prior(name): return 0.5 if CONFIGS[name]["cw"] else float(y_train.mean())

# %% 8. Per-config results (CV with selected checkpoint + test single vs ensemble)
rows, per_fold_rows = [], []
for name, r in tqdm(RES.items(), desc="per-config metrics"):
    fm = [metrics(r["oof"][r["fold_of"] == k], y_train[r["fold_of"] == k]) for k in range(N_FOLDS)]
    for k, m in enumerate(fm): per_fold_rows.append(dict(config=name, fold=k + 1, **m))
    po = metrics(r["oof"], y_train); ens = ens_of(r); me = metrics(ens, y_test); sing = [metrics(t, y_test) for t in r["tests"]]
    rows.append(dict(config=name, model=CONFIGS[name]["model"],
        cv_acc_mean=np.mean([m["acc"] for m in fm]), cv_acc_std=np.std([m["acc"] for m in fm], ddof=1),
        cv_f1_mean=np.mean([m["f1_weighted"] for m in fm]), cv_f1_std=np.std([m["f1_weighted"] for m in fm], ddof=1), cv_ece_perfold_mean=np.mean([m["ece15"] for m in fm]),
        oof_pooled_acc=po["acc"], oof_pooled_acc_ci=boot_ci(r["oof"], y_train, "acc"), oof_pooled_f1=po["f1_weighted"], oof_pooled_ece15=po["ece15"],
        oof_pooled_brier_sum=po["brier_sum"], oof_pooled_nll=po["nll"], oof_pooled_auc=po["auc"],
        test_single_acc_mean=np.mean([m["acc"] for m in sing]), test_single_acc_std=np.std([m["acc"] for m in sing], ddof=1), test_single_ece_mean=np.mean([m["ece15"] for m in sing]),
        test_single_brier_mean=np.mean([m["brier_sum"] for m in sing]), test_single_nll_mean=np.mean([m["nll"] for m in sing]),
        test_ens_acc=me["acc"], test_ens_acc_ci=boot_ci(ens, y_test, "acc"), test_ens_f1=me["f1_weighted"], test_ens_f1_macro=me["f1_macro"], test_ens_f1_macro_ci=boot_ci(ens, y_test, "f1_macro"),
        test_ens_bal_acc=me["bal_acc"], test_ens_mcc=me["mcc"], test_ens_auc=me["auc"], test_ens_pr_auc=me["pr_auc"], test_ens_ece15=me["ece15"], test_ens_ece_ci=boot_ci(ens, y_test, "ece15"),
        test_ens_brier_sum=me["brier_sum"], test_ens_nll=me["nll"], mean_best_epoch=float(np.mean(r["best_epoch"])),
        lr_at_best_epoch_over_peak=float(np.mean([h[b - 1]["lr"] for h, b in zip(r["hist"], r["best_epoch"])]) / LR),
        params_millions_single=r["n_params"] / 1e6, params_millions_ensemble=r["n_params"] * N_FOLDS / 1e6, gpu_hours=r["seconds"] / 3600))
summary = pd.DataFrame(rows); summary.to_csv(f"{OUT}/tables/summary_all_configs.csv", index=False)
pd.DataFrame(per_fold_rows).to_csv(f"{OUT}/tables/per_fold_metrics.csv", index=False)
log.info("summary:\n%s", summary.set_index("config")[["cv_acc_mean", "oof_pooled_ece15", "test_single_acc_mean", "test_ens_acc", "test_ens_f1_macro", "test_ens_auc", "test_ens_ece15"]].round(4).to_string() if len(summary) else "none")

# %% 9. Calibration study: single vs ensemble, temperature scaling, ensemble-size curve, voting rule, cross-config ensemble
cal_rows, kcurve_rows = [], []
for name, r in tqdm(RES.items(), desc="calibration"):
    T = r["tests"]; ens = T.mean(0); rng = np.random.default_rng(1); d_ece, d_brier = [], []
    for i in (rng.integers(0, N_TE, N_TE) for _ in range(N_BOOT)):
        d_ece.append(ece_score(ens[i], y_test[i]) - np.mean([ece_score(t[i], y_test[i]) for t in T]))
        d_brier.append(brier_sum(ens[i], y_test[i]) - np.mean([brier_sum(t[i], y_test[i]) for t in T]))
    rng2 = np.random.default_rng(2); sub = int(round(N_TR / N_FOLDS))
    sub_e = [ece_score(r["oof"][ix], y_train[ix]) for ix in (rng2.choice(N_TR, sub, replace=False) for _ in range(300))]
    # temperature scaling fitted on each model's inner validation split (never on test / outer fold)
    Ts = [fit_temp(p, y_train[ix]) for ix, p in r["ival"]]
    oof_ts = np.zeros_like(r["oof"]); tests_ts = []
    for k in range(N_FOLDS):
        m = r["fold_of"] == k; oof_ts[m] = apply_temp(r["oof"][m], Ts[k]); tests_ts.append(apply_temp(T[k], Ts[k]))
    tests_ts = np.stack(tests_ts); ens_ts = tests_ts.mean(0)
    hard = np.stack([t.argmax(1) for t in T]).mean(0); hard_acc = float(((hard > 0.5).astype(int) == y_test).mean())
    cal_rows.append(dict(config=name, ece_single_mean=np.mean([ece_score(t, y_test) for t in T]), ece_ensemble=ece_score(ens, y_test),
        delta_ece=np.mean(d_ece), delta_ece_ci=[float(np.percentile(d_ece, 2.5)), float(np.percentile(d_ece, 97.5))],
        delta_brier=np.mean(d_brier), delta_brier_ci=[float(np.percentile(d_brier, 2.5)), float(np.percentile(d_brier, 97.5))],
        ece_single_tempscaled=np.mean([ece_score(t, y_test) for t in tests_ts]), ece_ensemble_of_tempscaled=ece_score(ens_ts, y_test),
        ece_gain_from_tempscaling_of_ensemble=ece_score(ens, y_test) - ece_score(ens_ts, y_test), mean_temperature=float(np.mean(Ts)),
        oof_pooled_ece=ece_score(r["oof"], y_train), oof_pooled_ece_tempscaled=ece_score(oof_ts, y_train),
        oof_ece_random_subsets_of_fold_size_mean=float(np.mean(sub_e)), oof_perfold_ece_mean=float(np.mean([ece_score(r["oof"][r["fold_of"] == k], y_train[r["fold_of"] == k]) for k in range(N_FOLDS)])),
        test_acc_soft_vote=float((ens.argmax(1) == y_test).mean()), test_acc_hard_vote=hard_acc))
    rngk = np.random.default_rng(5)
    for K in range(1, N_FOLDS + 1):                                  # ensemble-size curve: random subsets of K fold models
        accs, eces, nlls = [], [], []
        for _ in range(30 if K < N_FOLDS else 1):
            p = T[rngk.choice(N_FOLDS, K, replace=False)].mean(0); accs.append((p.argmax(1) == y_test).mean()); eces.append(ece_score(p, y_test)); nlls.append(metrics(p, y_test)["nll"])
        kcurve_rows.append(dict(config=name, K=K, acc=np.mean(accs), ece=np.mean(eces), nll=np.mean(nlls)))
cal = pd.DataFrame(cal_rows); cal.to_csv(f"{OUT}/tables/calibration_single_vs_ensemble.csv", index=False)
pd.DataFrame(kcurve_rows).to_csv(f"{OUT}/tables/ensemble_size_curve.csv", index=False)
log.info("calibration:\n%s", cal.set_index("config")[["ece_single_mean", "ece_ensemble", "delta_ece", "ece_single_tempscaled", "ece_ensemble_of_tempscaled", "oof_pooled_ece", "oof_ece_random_subsets_of_fold_size_mean"]].round(4).to_string() if len(cal) else "n/a")

xrows = []                                                            # cross-config ensembles (architecture diversity)
for tag, members in [("all_bertweet_core", [n for n in GROUPS["core"] if n in RES and CONFIGS[n]["model"] == BT]),
                     ("all_configs", list(RES)), ("all_encoders_no_ablation", [n for n in RES if n in ("date_full", "roberta_base") + tuple(GROUPS["models"])])]:
    if len(members) >= 2:
        p = np.mean([ens_of(RES[n]) for n in members], 0); m = metrics(p, y_test)
        xrows.append(dict(ensemble=tag, n_configs=len(members), members=",".join(members), test_acc=m["acc"], test_f1_macro=m["f1_macro"], test_auc=m["auc"], test_ece=m["ece15"], test_brier=m["brier_sum"]))
pd.DataFrame(xrows).to_csv(f"{OUT}/tables/cross_config_ensembles.csv", index=False)

# %% 10. Config comparisons: McNemar + paired bootstrap + Nadeau-Bengio corrected t-test (vs date_full)
cmp_rows = []
if "date_full" in RES:
    base = RES["date_full"]; accf = lambda r: np.array([(r["oof"][r["fold_of"] == k].argmax(1) == y_train[r["fold_of"] == k]).mean() for k in range(N_FOLDS)])
    for name, r in RES.items():
        if name == "date_full": continue
        for tag, pa, pb, y in [("test_ensemble", ens_of(base), ens_of(r), y_test), ("oof_pooled", base["oof"], r["oof"], y_train)]:
            b, c, p = mcnemar(pa, pb, y); rng = np.random.default_rng(3); d = []
            for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)): d.append((pa[i].argmax(1) == y[i]).mean() - (pb[i].argmax(1) == y[i]).mean())
            row = dict(vs=name, on=tag, acc_date_full=float((pa.argmax(1) == y).mean()), acc_other=float((pb.argmax(1) == y).mean()), date_better_only=b, other_better_only=c,
                       mcnemar_p=p, diff_acc=float(np.mean(d)), diff_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
            if tag == "oof_pooled" and CONFIGS[name]["split"] == "strat" and CONFIGS[name].get("frac", 1) == 1:
                row["nadeau_bengio_t"], row["nadeau_bengio_p"] = nadeau_bengio(accf(base) - accf(r))
            cmp_rows.append(row)
cmp_df = pd.DataFrame(cmp_rows); cmp_df.to_csv(f"{OUT}/tables/config_comparisons.csv", index=False)
log.info("config comparisons:\n%s", cmp_df[["vs", "on", "acc_date_full", "acc_other", "mcnemar_p"]].round(4).to_string() if len(cmp_df) else "n/a")

# %% 11. Multi-seed aggregation (seed = new fold shuffle + init + augmentation); seed-ensembles
ms_rows, pw = [], []
for name in CONFIGS:
    ss = [s for s in SEEDS if (name, s) in MS]
    if len(ss) < 2: continue
    per = []
    for s in ss:
        r = MS[(name, s)]; ens = ens_of(r); me = metrics(ens, y_test); po = metrics(r["oof"], y_train)
        per.append(dict(seed=s, oof_acc=po["acc"], oof_f1=po["f1_weighted"], oof_ece=po["ece15"], test_ens_acc=me["acc"], test_ens_f1=me["f1_weighted"], test_ens_f1_macro=me["f1_macro"],
                        test_ens_auc=me["auc"], test_ens_pr_auc=me["pr_auc"], test_ens_ece=me["ece15"], test_ens_brier=me["brier_sum"], test_ens_nll=me["nll"],
                        test_single_acc=float(np.mean([accuracy_score(y_test, t.argmax(1)) for t in r["tests"]])), test_single_ece=float(np.mean([ece_score(t, y_test) for t in r["tests"]]))))
    pdf = pd.DataFrame(per); pdf.insert(0, "config", name); pdf.to_csv(f"{OUT}/tables/multiseed_{name}_per_seed.csv", index=False)
    row = dict(config=name, n_seeds=len(ss))
    for c in pdf.columns[2:]: row[c + "_mean"] = pdf[c].mean(); row[c + "_std"] = pdf[c].std(ddof=1)
    allp = np.concatenate([MS[(name, s)]["tests"] for s in ss]).mean(0); row["seed_ensemble_models"] = len(ss) * N_FOLDS
    row["seed_ensemble_test_acc"] = float((allp.argmax(1) == y_test).mean()); row["seed_ensemble_test_ece"] = ece_score(allp, y_test)
    ms_rows.append(row)
for name in CONFIGS:
    if name == "date_full": continue
    ss = [s for s in SEEDS if (name, s) in MS and ("date_full", s) in MS]
    if len(ss) < 2: continue
    d_acc = [accuracy_score(y_test, ens_of(MS[("date_full", s)]).argmax(1)) - accuracy_score(y_test, ens_of(MS[(name, s)]).argmax(1)) for s in ss]
    d_oof = [(MS[("date_full", s)]["oof"].argmax(1) == y_train).mean() - (MS[(name, s)]["oof"].argmax(1) == y_train).mean() for s in ss]
    d_ece = [ece_score(ens_of(MS[("date_full", s)]), y_test) - ece_score(ens_of(MS[(name, s)]), y_test) for s in ss]
    pw.append(dict(vs=name, n_seeds=len(ss), test_acc_diff_mean=np.mean(d_acc), test_acc_diff_std=np.std(d_acc, ddof=1), test_acc_paired_t_p=float(ttest_1samp(d_acc, 0).pvalue) if len(ss) >= 3 else np.nan,
                   oof_acc_diff_mean=np.mean(d_oof), oof_acc_diff_std=np.std(d_oof, ddof=1), test_ece_diff_mean=np.mean(d_ece),
                   seeds_date_full_better_test=int(sum(x > 0 for x in d_acc)), seeds_date_full_better_oof=int(sum(x > 0 for x in d_oof))))
ms = pd.DataFrame(ms_rows); ms.to_csv(f"{OUT}/tables/multiseed_summary.csv", index=False)
pd.DataFrame(pw).to_csv(f"{OUT}/tables/multiseed_pairwise_vs_date_full.csv", index=False)
log.info("multi-seed summary:\n%s", ms[["config", "n_seeds", "test_ens_acc_mean", "test_ens_acc_std", "oof_acc_mean", "oof_acc_std", "test_ens_ece_mean"]].round(4).to_string() if len(ms) else "(need >=2 seeds)")

# %% 12. Selective prediction / human-in-the-loop (risk-coverage) and prior-shift analysis
rc_rows, ps_rows = [], []
for name, r in RES.items():
    ens = ens_of(r)
    for tag, p in [("test_ensemble", ens), ("test_single_mean", None), ("oof_single", r["oof"])]:
        if tag == "test_single_mean":
            rcs = [risk_coverage(t, y_test) for t in r["tests"]]; rc = {k: float(np.mean([x[k] for x in rcs])) for k in rcs[0]}
        else: rc = risk_coverage(p, y_test if tag.startswith("test") else y_train)
        rc_rows.append(dict(config=name, on=tag, **rc))
    fp = eff_prior(name); true_prior = float(y_test.mean()); est = em_prior(ens, fp)
    for tag, p in [("raw", ens), ("prior_adjusted_oracle", prior_adjust(ens, fp, true_prior)), ("prior_adjusted_EM_unlabelled", prior_adjust(ens, fp, est))]:
        m = metrics(p, y_test); ps_rows.append(dict(config=name, variant=tag, effective_train_prior=fp, true_test_prior=true_prior, em_estimated_test_prior=est,
                                                    acc=m["acc"], bal_acc=m["bal_acc"], f1_macro=m["f1_macro"], mcc=m["mcc"], ece15=m["ece15"], brier_sum=m["brier_sum"]))
pd.DataFrame(rc_rows).to_csv(f"{OUT}/tables/risk_coverage.csv", index=False); pd.DataFrame(ps_rows).to_csv(f"{OUT}/tables/prior_shift_analysis.csv", index=False)

# false-positive vs false-negative trade-off made explicit (R1): threshold sweep + cost-ratio optimal thresholds chosen on OOF only
cost_rows, sweep_rows = [], []
if "date_full" in RES:
    r = RES["date_full"]; ens = ens_of(r); ts = np.linspace(0.05, 0.95, 91)
    def _cm(p, y, t):
        pred = p[:, 1] >= t; return int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()), int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    for t in np.arange(0.1, 0.91, 0.1):
        fp, fn, tp, tn = _cm(ens, y_test, t); sweep_rows.append(dict(threshold=round(float(t), 2), TP=tp, FP=fp, FN=fn, TN=tn, precision_ext=tp / max(tp + fp, 1), recall_ext=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1), flagged_pct=(tp + fp) / N_TE * 100))
    for ratio in (1, 2, 5, 10):
        cost = lambda p, y, t: (ratio * _cm(p, y, t)[1] + _cm(p, y, t)[0]) / len(y)
        bt = float(ts[int(np.argmin([cost(r["oof"], y_train, t) for t in ts]))]); fp, fn, tp, tn = _cm(ens, y_test, bt); fp0, fn0, _, _ = _cm(ens, y_test, 0.5)
        cost_rows.append(dict(cost_FN_over_FP=ratio, threshold_chosen_on_oof=bt, test_cost_per_post=(ratio * fn + fp) / N_TE, test_FP=fp, test_FN=fn, test_cost_at_0_5=(ratio * fn0 + fp0) / N_TE, FP_at_0_5=fp0, FN_at_0_5=fn0))
pd.DataFrame(sweep_rows).to_csv(f"{OUT}/tables/threshold_sweep_date_full.csv", index=False); pd.DataFrame(cost_rows).to_csv(f"{OUT}/tables/cost_sensitive_thresholds.csv", index=False)

# augmentation size bookkeeping (R2: reconcile probabilities with the "~2.1 variants per post")
_ar = []
for _f in sorted(glob.glob(f"{OUT}/aug/strat_all_f1.0_s{PRIMARY}_fold*.json")):
    _d = json.load(open(_f)); _ar.append(dict(file=os.path.basename(_f), before=_d["before"], after=_d["after"], factor=_d["after"] / _d["before"], mean_new_variants_per_post=_d["mean_new_per_post"]))
pd.DataFrame(_ar).to_csv(f"{OUT}/tables/augmentation_sizes.csv", index=False)

# %% 13. Slice analysis (length, similarity to train, lexical categories) for date_full
slice_rows = []
if "date_full" in RES:
    r = RES["date_full"]
    for tag, X, y, p, sim in [("test_ensemble", X_test, y_test, ens_of(r), SIM_TEST), ("oof", X_train, y_train, r["oof"], VAL_SIM)]:
        wl = np.array([len(t.split()) for t in X]); pred = p.argmax(1)
        slices = {"len<=10": wl <= 10, "len11-20": (wl > 10) & (wl <= 20), "len21-40": (wl > 20) & (wl <= 40), "len>40": wl > 40,
                  "sim<0.3": sim < .3, "sim0.3-0.6": (sim >= .3) & (sim < .6), "sim0.6-0.9": (sim >= .6) & (sim < .9), "sim>=0.9(near-dup)": sim >= .9,
                  "has_profanity": np.array([has_any(t, PROFANITY) for t in X]), "no_profanity": ~np.array([has_any(t, PROFANITY) for t in X]),
                  "has_violence_word": np.array([has_any(t, VIOLENCE) for t in X]), "has_identity_term": np.array([has_any(t, _cyc) for t in X]),
                  "no_identity_term": ~np.array([has_any(t, _cyc) for t in X])}
        for sname, m in slices.items():
            if m.sum() == 0: continue
            slice_rows.append(dict(on=tag, slice=sname, n=int(m.sum()), pct_extremist=float(y[m].mean() * 100), acc=float((pred[m] == y[m]).mean()),
                                   fpr=float(((pred[m] == 1) & (y[m] == 0)).sum() / max((y[m] == 0).sum(), 1)), fnr=float(((pred[m] == 0) & (y[m] == 1)).sum() / max((y[m] == 1).sum(), 1)),
                                   mean_conf=float(p[m].max(1).mean())))
pd.DataFrame(slice_rows).to_csv(f"{OUT}/tables/slice_analysis.csv", index=False)

# %% 14. Robustness (noise), fairness probes (identity swap / neutralisation), label-preservation probe, augmentation studies
rob_rows, aug_rows = [], []
for name, r in RES.items():
    if not r["rob"]: continue
    clean_e = ens_of(r); clean_pred = clean_e.argmax(1)
    for rn, arr in r["rob"].items():
        pe = arr.mean(0); mask = ID_MASK_TEST if rn.startswith("ident") else np.ones(N_TE, bool)
        if mask.sum() == 0: continue
        rob_rows.append(dict(config=name, perturbation=rn, n=int(mask.sum()), ens_acc=float((pe.argmax(1) == y_test)[mask].mean()), ens_acc_clean=float((clean_pred == y_test)[mask].mean()),
                             ens_flip_rate=float((pe.argmax(1) != clean_pred)[mask].mean()), ens_mean_abs_prob_shift=float(np.abs(pe[:, 1] - clean_e[:, 1])[mask].mean()),
                             single_acc_mean=float(np.mean([(a.argmax(1) == y_test)[mask].mean() for a in arr])),
                             single_flip_rate_mean=float(np.mean([(a.argmax(1) != t.argmax(1))[mask].mean() for a, t in zip(arr, r["tests"])]))))
pd.DataFrame(rob_rows).to_csv(f"{OUT}/tables/robustness_and_fairness_probes.csv", index=False)

if "no_aug" in RES and RES["no_aug"]["probe"]:                          # model trained WITHOUT augmentation, probed on augmented validation posts
    r = RES["no_aug"]; flips, accs_o, accs_a, shift = [], [], [], []
    for va_idx, pidx, pprobs in r["probe"]:
        po = r["oof"][va_idx]; yv = y_train[va_idx]
        flips += list(pprobs.argmax(1) != po.argmax(1)[pidx]); accs_a += list(pprobs.argmax(1) == yv[pidx]); accs_o += list((po.argmax(1) == yv)[pidx]); shift += list(np.abs(pprobs[:, 1] - po[pidx, 1]))
    aug_rows.append(dict(study="label_preservation_probe", n_augmented=len(flips), flip_rate=float(np.mean(flips)), acc_on_augmented=float(np.mean(accs_a)),
                         acc_on_same_originals=float(np.mean(accs_o)), mean_abs_prob_shift=float(np.mean(shift))))
for fr in (25, 50):                                                      # learning curve
    for base_name in ("date_full", "no_aug"):
        n_ = f"{base_name}_f{fr}"
        if n_ in RES:
            m = metrics(RES[n_]["oof"], y_train); aug_rows.append(dict(study="learning_curve", config=n_, frac=fr / 100, oof_acc=m["acc"], oof_f1_macro=m["f1_macro"], test_ens_acc=float((ens_of(RES[n_]).argmax(1) == y_test).mean()), test_ens_ece=ece_score(ens_of(RES[n_]), y_test)))
for base_name in ("date_full", "no_aug"):
    if base_name in RES:
        m = metrics(RES[base_name]["oof"], y_train); aug_rows.append(dict(study="learning_curve", config=base_name, frac=1.0, oof_acc=m["acc"], oof_f1_macro=m["f1_macro"], test_ens_acc=float((ens_of(RES[base_name]).argmax(1) == y_test).mean()), test_ens_ece=ece_score(ens_of(RES[base_name]), y_test)))
for n_ in ("aug_ctx_only", "aug_eda_only"):
    if n_ in RES: aug_rows.append(dict(study="augmentation_operators", config=n_, oof_acc=metrics(RES[n_]["oof"], y_train)["acc"], test_ens_acc=float((ens_of(RES[n_]).argmax(1) == y_test).mean()), test_ens_ece=ece_score(ens_of(RES[n_]), y_test)))
aug_tab = pd.DataFrame(aug_rows); aug_tab.to_csv(f"{OUT}/tables/augmentation_studies.csv", index=False)
for js in sorted(glob.glob(f"{OUT}/aug/strat_all_f1.0_s{PRIMARY}_fold*.json"))[:1]:
    d = json.load(open(js)); log.info("augmentation size (one fold): %d -> %d (mean %.2f new variants per post)", d["before"], d["after"], d["mean_new_per_post"])

# %% 15. External (out-of-distribution) transfer + label-version analysis
ext_rows = []
for name, r in RES.items():
    for en, arr in r["ext"].items():
        if en not in EXT_SETS or len(EXT_SETS[en][1]) != arr.shape[1]: continue
        ey = EXT_SETS[en][1]; pe = arr.mean(0); m = metrics(pe, ey)
        ext_rows.append(dict(config=name, external=en, n=len(ey), pos_pct=float(ey.mean() * 100), ens_auc=m["auc"], ens_pr_auc=m["pr_auc"], ens_acc_at_0_5=m["acc"], ens_f1_macro=m["f1_macro"],
                             single_auc_mean=float(np.mean([roc_auc_score(ey, a[:, 1]) for a in arr]))))
pd.DataFrame(ext_rows).to_csv(f"{OUT}/tables/external_transfer.csv", index=False)

lab_rows = []
if CHANGED is not None and "date_full" in RES:
    r = RES["date_full"]; pred = r["oof"].argmax(1); known = OLD_LABEL >= 0
    for tag, m in [("unchanged_labels", known & ~CHANGED), ("relabelled_after_competition", CHANGED)]:
        lab_rows.append(dict(subset=tag, n=int(m.sum()), acc_vs_new_label=float((pred[m] == y_train[m]).mean()), agreement_with_old_label=float((pred[m] == OLD_LABEL[m]).mean())))
pd.DataFrame(lab_rows).to_csv(f"{OUT}/tables/label_version_analysis.csv", index=False)

# %% 16. Classical baselines (identical splits, CPU) + top lexical features (shortcut check)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
base_rows = []; lr_probs = None
for nm, mk in [("tfidf_logreg", lambda: LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")), ("tfidf_linsvm", lambda: LinearSVC(C=0.5, class_weight="balanced"))]:
    accs = []; oofp = np.zeros(N_TR)
    for a, b in tqdm(splits_for("strat", PRIMARY), desc=nm, leave=False):
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([X_train[i] for i in a]); m = mk().fit(v.transform([X_train[i] for i in a]), y_train[a])
        pr = m.predict(v.transform([X_train[i] for i in b])); oofp[b] = pr; accs.append(accuracy_score(y_train[b], pr))
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(X_train); m = mk().fit(v.transform(X_train), y_train); pt = m.predict(v.transform(X_test))
    row = dict(model=nm, cv_acc_mean=np.mean(accs), cv_acc_std=np.std(accs, ddof=1), test_acc=accuracy_score(y_test, pt), test_f1=f1_score(y_test, pt, average="weighted"), test_f1_macro=f1_score(y_test, pt, average="macro"))
    if hasattr(m, "predict_proba"):
        pp = m.predict_proba(v.transform(X_test)); lr_probs = pp
        row.update(test_auc=roc_auc_score(y_test, pp[:, 1]), test_ece15=ece_score(pp, y_test), test_brier_sum=brier_sum(pp, y_test))
        coef = m.coef_[0]; fn = np.array(v.get_feature_names_out()); o = np.argsort(coef)
        pd.DataFrame(dict(feature=list(fn[o[:40]]) + list(fn[o[-40:]]), coef=list(coef[o[:40]]) + list(coef[o[-40:]]), pushes_toward=["NON_EXTREMIST"] * 40 + ["EXTREMIST"] * 40)).to_csv(f"{OUT}/exports/tfidf_top_features.csv", index=False)
        if "date_full" in RES:
            b_, c_, p_ = mcnemar(ens_of(RES["date_full"]), pp, y_test); row.update(mcnemar_vs_date_full_p=p_, date_full_better_only=b_, tfidf_better_only=c_)
    base_rows.append(row)
pd.DataFrame(base_rows).to_csv(f"{OUT}/tables/classical_baselines.csv", index=False); log.info("classical baselines:\n%s", pd.DataFrame(base_rows).round(4).to_string())

# %% 17. Confusion counts, per-class metrics
conf_rows = []
for name, r in RES.items():
    for tag, p, y in [("oof_pooled", r["oof"], y_train), ("test_ensemble", ens_of(r), y_test)]:
        cm = confusion_matrix(y, p.argmax(1), labels=[0, 1]); pr, rc, f, _ = precision_recall_fscore_support(y, p.argmax(1), labels=[0, 1])
        conf_rows.append(dict(config=name, on=tag, TN=cm[0, 0], FP=cm[0, 1], FN=cm[1, 0], TP=cm[1, 1], prec_non=pr[0], rec_non=rc[0], f1_non=f[0], prec_ext=pr[1], rec_ext=rc[1], f1_ext=f[1],
                              low_conf_lt_0_7_pct=float((p.max(1) < 0.7).mean() * 100), mean_conf=float(p.max(1).mean())))
pd.DataFrame(conf_rows).to_csv(f"{OUT}/tables/confusion_and_perclass.csv", index=False)

# %% 18. Figures
def savefig(fig, name): fig.savefig(f"{OUT}/figures/{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/figures/{name}.pdf", bbox_inches="tight"); plt.close(fig)
def reliability(p, y, title, name, bins=10):
    conf = p.max(1); corr = (p.argmax(1) == y).astype(float); edges = np.linspace(0.5, 1.0, bins + 1); which = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    acc = [corr[which == b].mean() if (which == b).any() else np.nan for b in range(bins)]; cnt = [(which == b).sum() for b in range(bins)]; mid = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax[0].plot([0.5, 1], [0.5, 1], "k--", lw=1); ax[0].bar(mid, np.nan_to_num(acc), width=0.045, alpha=0.7)
    ax[0].set(xlabel="Mean predicted confidence (bin)", ylabel="Accuracy in bin", title=title, xlim=(0.5, 1), ylim=(0, 1))
    ax[1].bar(mid, cnt, width=0.045); ax[1].set(xlabel="Confidence", ylabel="Samples per bin", title="Per-bin counts")
    for x, c in zip(mid, cnt): ax[1].text(x, c, str(c), ha="center", va="bottom", fontsize=7)
    savefig(fig, name)
for name, r in tqdm(RES.items(), desc="figures"):
    reliability(r["oof"], y_train, f"{name}: out-of-fold (single models)", f"reliability_oof_{name}")
    reliability(ens_of(r), y_test, f"{name}: test, 10-model ensemble", f"reliability_test_ensemble_{name}")
    reliability(r["tests"][0], y_test, f"{name}: test, single model (fold 1)", f"reliability_test_single_{name}")
    for tag, p, y in [("oof", r["oof"], y_train), ("test", ens_of(r), y_test)]:
        cm = confusion_matrix(y, p.argmax(1), labels=[0, 1]); fig, ax = plt.subplots(figsize=(3.8, 3.4)); ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/cm[i].sum():.1%})", ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NON", "EXT"], yticklabels=["NON", "EXT"], xlabel="Predicted", ylabel="True", title=f"{name} ({tag})"); savefig(fig, f"confusion_{tag}_{name}")
    fig, ax = plt.subplots(figsize=(5, 3.4)); ax.hist(ens_of(r).max(1), bins=20, alpha=.8); ax.axvline(0.7, color="r", ls="--"); ax.set(xlabel="Ensemble confidence", ylabel="Test posts", title=f"{name}: test confidence"); savefig(fig, f"confidence_test_{name}")
    fig, axs = plt.subplots(2, math.ceil(N_FOLDS / 2), figsize=(2.8 * math.ceil(N_FOLDS / 2), 5), sharey=True)
    for k, ax in enumerate(np.array(axs).ravel()[:N_FOLDS]):
        h = pd.DataFrame(r["hist"][k]); ax.plot(h.epoch, h.train_loss, label="train"); ax.plot(h.epoch, h.inner_val_loss, label="inner val"); ax.axvline(r["best_epoch"][k], color="k", ls=":"); ax.set_title(f"fold {k+1}", fontsize=9)
    np.array(axs).ravel()[0].legend(fontsize=7); savefig(fig, f"loss_curves_{name}")
if len(summary):
    s = summary.set_index("config"); fig, ax = plt.subplots(figsize=(max(7, .55 * len(s)), 3.8))
    ax.bar(np.arange(len(s)) - 0.2, s.oof_pooled_acc, 0.4, label="pooled OOF acc"); ax.bar(np.arange(len(s)) + 0.2, s.test_ens_acc, 0.4, label="test ensemble acc")
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s.index, rotation=40, ha="right", fontsize=7); ax.set_ylim(0.6, 1); ax.legend(); savefig(fig, "config_comparison")
    fig, ax = plt.subplots(figsize=(max(7, .55 * len(s)), 3.8)); ax.bar(np.arange(len(cal)) - 0.2, cal.ece_single_mean, 0.4, label="single models"); ax.bar(np.arange(len(cal)) + 0.2, cal.ece_ensemble, 0.4, label="10-model ensemble")
    ax.set_xticks(range(len(cal))); ax.set_xticklabels(cal.config, rotation=40, ha="right", fontsize=7); ax.set_ylabel("Test ECE (15 bins)"); ax.legend(); savefig(fig, "calibration_single_vs_ensemble")
    pf = pd.DataFrame(per_fold_rows); fig, ax = plt.subplots(figsize=(6, 3.4))
    for nm, g in pf[pf.config.isin(GROUPS["core"])].groupby("config"): ax.plot(g.fold, g.f1_weighted, "o-", label=nm, ms=3)
    ax.set(xlabel="Fold", ylabel="Weighted F1"); ax.legend(fontsize=6); savefig(fig, "per_fold_f1")
kc = pd.DataFrame(kcurve_rows)
if len(kc):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for nm, g in kc[kc.config.isin(["date_full", "no_aug", "roberta_base"])].groupby("config"): ax[0].plot(g.K, g.acc, "o-", ms=3, label=nm); ax[1].plot(g.K, g.ece, "o-", ms=3, label=nm)
    ax[0].set(xlabel="Ensemble size K", ylabel="Test accuracy"); ax[1].set(xlabel="Ensemble size K", ylabel="Test ECE"); ax[0].legend(fontsize=7); savefig(fig, "ensemble_size_curve")
rcdf = pd.DataFrame(rc_rows)
if len(rcdf) and "date_full" in RES:
    fig, ax = plt.subplots(figsize=(5, 3.4)); covs = [50, 60, 70, 80, 90, 100]
    for tag in ["test_ensemble", "test_single_mean"]:
        row = rcdf[(rcdf.config == "date_full") & (rcdf.on == tag)].iloc[0]; ax.plot(covs, [row[f"acc@cov{c}"] for c in covs], "o-", label=tag)
    ax.set(xlabel="Coverage (% of posts auto-decided, most confident first)", ylabel="Accuracy"); ax.legend(fontsize=7); savefig(fig, "risk_coverage_date_full")
if rob_rows:
    rb = pd.DataFrame(rob_rows); rb = rb[rb.config.isin(["date_full", "no_aug", "roberta_base"])]
    if len(rb):
        pv = rb.pivot(index="perturbation", columns="config", values="ens_flip_rate"); ax = pv.plot.bar(figsize=(6, 3.4)); ax.set_ylabel("Ensemble prediction flip rate"); savefig(ax.get_figure(), "robustness_flip_rate")
lc = aug_tab[aug_tab.study == "learning_curve"] if len(aug_tab) and "study" in aug_tab else pd.DataFrame()
if len(lc):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for nm in ("date_full", "no_aug"):
        g = lc[lc.config.str.startswith(nm)].sort_values("frac")
        if len(g): ax.plot(g.frac * 100, g.oof_acc, "o-", label=nm)
    ax.set(xlabel="% of training data", ylabel="Pooled OOF accuracy"); ax.legend(); savefig(fig, "learning_curve_augmentation")
if len(ms):
    fig, ax = plt.subplots(figsize=(max(5, .8 * len(ms)), 3.4)); ax.bar(range(len(ms)), ms.test_ens_acc_mean, yerr=ms.test_ens_acc_std, capsize=3)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms.config, rotation=30, ha="right", fontsize=7); ax.set_ylim(0.75, 0.9); ax.set_ylabel("Test ensemble acc (mean ± sd over seeds)"); savefig(fig, "multiseed_test_acc")
log.info("figures written: %d files", len(os.listdir(f"{OUT}/figures")))

# %% 19. Error-analysis exports: all errors, seeded sample of 60, rule-based tags, high-confidence errors
if "date_full" in RES:
    r = RES["date_full"]; ens = ens_of(r); tag_rows = []
    tag_fns = {"short(<6 words)": lambda t: len(t.split()) < 6, "long(>40 words)": lambda t: len(t.split()) > 40, "profanity": lambda t: has_any(t, PROFANITY), "violence_word": lambda t: has_any(t, VIOLENCE),
               "identity_term": lambda t: has_any(t, _cyc), "negation": lambda t: has_any(t, {"not", "no", "never", "cannot", "dont", "don", "nothing"})}
    for tag, p, y, X in [("oof", r["oof"], y_train, X_train), ("test_ensemble", ens, y_test, X_test)]:
        e = pd.DataFrame(dict(text=X, true=y, pred=p.argmax(1), conf=p.max(1))); err = e[e.true != e.pred].copy()
        err["error_type"] = np.where(err.true == 1, "false_negative", "false_positive"); err["category_annotator1"] = ""; err["category_annotator2"] = ""
        err.to_csv(f"{OUT}/exports/errors_{tag}.csv", index=False)
        err.sample(min(60, len(err)), random_state=SEED).to_csv(f"{OUT}/exports/errors_{tag}_sample60_seed{SEED}.csv", index=False)
        err[err.conf >= 0.9].sort_values("conf", ascending=False).head(50).to_csv(f"{OUT}/exports/errors_{tag}_high_confidence_top50.csv", index=False)
        for tn, fn in tag_fns.items():
            m = np.array([fn(t) for t in X])
            if m.sum(): tag_rows.append(dict(on=tag, tag=tn, n=int(m.sum()), error_rate=float((e.true != e.pred)[m].mean()), error_rate_without_tag=float((e.true != e.pred)[~m].mean())))
        log.info("%s errors exported: %d (of %d)", tag, len(err), len(e))
    pd.DataFrame(tag_rows).to_csv(f"{OUT}/tables/error_rate_by_rule_tag.csv", index=False)

# %% 20. LaTeX tables
def to_tex(d, name, **kw): open(f"{OUT}/tables/{name}.tex", "w").write(d.to_latex(index=False, float_format="%.4f", **kw))
if len(summary):
    to_tex(summary[["config", "cv_acc_mean", "cv_acc_std", "cv_f1_mean", "oof_pooled_ece15", "test_ens_acc", "test_ens_f1", "test_ens_f1_macro", "test_ens_auc", "test_ens_ece15"]], "tab_main")
    to_tex(cal[["config", "ece_single_mean", "ece_ensemble", "delta_ece", "ece_single_tempscaled", "ece_ensemble_of_tempscaled", "oof_pooled_ece", "oof_ece_random_subsets_of_fold_size_mean"]], "tab_calibration")
if len(ms): to_tex(ms[["config", "n_seeds", "test_ens_acc_mean", "test_ens_acc_std", "oof_acc_mean", "oof_acc_std", "test_ens_ece_mean"]], "tab_multiseed")

# %% 21. Latency + environment + zip
lat = {}
try:
    m = AutoModelForSequenceClassification.from_pretrained(resolve(BT), num_labels=2).to(DEVICE).eval()
    if USE_AMP: m.half()
    enc = [tok0(t, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE) for t in X_test[:50 if SMOKE else 300]]
    with torch.no_grad():
        for e in enc[:10]: m(**e)
        if USE_AMP: torch.cuda.synchronize()
        t0 = time.time()
        for e in enc: m(**e)
        if USE_AMP: torch.cuda.synchronize()
    ms1 = (time.time() - t0) / len(enc) * 1000
    lat = dict(ms_per_post_single_model=ms1, ms_per_post_10_model_ensemble_sequential=ms1 * 10, device=str(DEVICE), gpu=torch.cuda.get_device_name(0) if USE_AMP else "cpu", n_posts=len(enc)); del m
except Exception as ex:
    lat = dict(error=str(ex))
import transformers, sklearn
lat.update(torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__, python=sys.version.split()[0], dataset_sha256=sha256)
json.dump(lat, open(f"{OUT}/tables/latency_env.json", "w"), indent=2); log.info("latency/env: %s", lat)
shutil.make_archive(f"{OUT}_bundle", "zip", OUT)
log.info("DONE -> %s_bundle.zip", OUT)
