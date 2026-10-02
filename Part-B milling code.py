# -*- coding: utf-8 -*-
import os, re, glob, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from scipy import stats

ROOT = "/content/drive/MyDrive/Research/MILLING/mtc"
OUT  = os.path.join(ROOT, "_rerun"); os.makedirs(OUT, exist_ok=True)
FIG  = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

RNG     = 42
REF     = (100, 4)            # reference RF setting (trees, depth)
GRID    = [(50, 3), (100, 4), (200, 4), (300, 4), (300, 6), (500, 8)]
PERM_B  = 200                 # permutations -> p floor = 1/(B+1)
ALPHA   = 0.20                # 80 % intervals
GEOMS   = ["Box", "Cover", "Plate"]

# ---- label configuration (PRIMARY = what the paper text says) ----
LABEL_CFG = dict(exclude_form=True, screen_invariant=True, norm="halfband")
# norm: "halfband" = Eq.6 ; "sided" = divide by (max-target) or (target-min)

# ---- feature bookkeeping ----
KEYS  = {"part", "instance", "machine", "source_file"}
NEVER = {"severity", "was_truncated"}             # must never enter X
COUNT_PATTERNS = [r"_n$", r"^n_", r"block_max", r"n_block_events",
                  r"n_timestamped_events", r"tool_seq_len", r"Part_Count",
                  r"runtime_max", r"spindle_time_max",
                  r"Program_Runtime_Seconds", r"Spindle_Time"]
TIME_PATTERNS  = [r"Part_Count", r"Spindle_Time", r"Program_Runtime_Seconds",
                  r"runtime_max", r"spindle_time_max"]
def matches(name, pats): return any(re.search(p, name) for p in pats)

def ckpt(name):  return os.path.join(OUT, name)
def done(name):  return os.path.exists(ckpt(name))

def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

# ---- load ----
F = pd.read_csv(os.path.join(ROOT, "_mtc_features.csv"))
L = pd.read_csv(os.path.join(ROOT, "_qif_labels_final.csv"))
L["is_admin"] = as_bool(L["is_admin"]); L["is_form"] = as_bool(L["is_form"])
F = F.replace([np.inf, -np.inf], np.nan)          # NO global imputation

FEATS = [c for c in F.columns
         if c not in KEYS | NEVER and pd.api.types.is_numeric_dtype(F[c])]
assert "severity" not in FEATS and "was_truncated" not in FEATS
NOCOUNT = [c for c in FEATS if not matches(c, COUNT_PATTERNS)]
NOTIME  = [c for c in FEATS if not matches(c, TIME_PATTERNS)]
print(f"features: full={len(FEATS)}  no-count={len(NOCOUNT)}  no-time={len(NOTIME)}")

# ---- labels -> severity ----
def build_labels(L, exclude_form=True, screen_invariant=True, norm="halfband"):
    X = L[~L["is_admin"]].copy()
    X["lab"] = X["label_norm"]
    if norm == "sided":
        size = ~X["is_form"] & X[["measured", "target", "min", "max"]].notna().all(1)
        up, lo = X["max"] - X["target"], X["target"] - X["min"]
        dev = X["measured"] - X["target"]
        sided = np.where(dev >= 0, dev / up.replace(0, np.nan),
                         dev / lo.replace(0, np.nan))
        X.loc[size, "lab"] = sided[size.values]
    X = X[X["lab"].notna()]
    if exclude_form:
        X = X[~X["is_form"]]
    if screen_invariant:
        sd = X.groupby(["part", "char_name"])["lab"].transform("std").fillna(0)
        X = X[sd > 1e-9]
    X["abs_label"] = X["lab"].abs()
    sev = (X.groupby(["part", "instance"])["abs_label"].mean()
             .rename("severity").reset_index())
    return X, sev

def make_D(sev):
    D = F.merge(sev, on=["part", "instance"], how="inner")
    return D.sort_values(["part", "instance"]).reset_index(drop=True)

Lok, SEV = build_labels(L, **LABEL_CFG)
D = make_D(SEV)
print("label rows kept:", len(Lok), "| parts per geometry:",
      D.groupby("part").size().to_dict())

# ---- models / CV helpers ----
def rf(nt=REF[0], md=REF[1]):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestRegressor(n_estimators=nt, max_depth=md,
                                                  random_state=RNG, n_jobs=-1))])
def enet():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("en", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000))])

def loo_splits(n):
    return [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
def block_splits(n, k=5):          # contiguous blocks (data sorted by instance)
    edges = np.array_split(np.arange(n), k)
    return [(np.setdiff1d(np.arange(n), te), te) for te in edges]

def cv_pred(X, y, ctor, splits):
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        m = ctor(); m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
    return p
def r2(y, p):
    st = np.sum((y - y.mean())**2)
    return 1 - np.sum((y - p)**2) / st if st > 0 else np.nan
def mean_baseline(y, splits):
    p = np.full(len(y), np.nan)
    for tr, te in splits: p[te] = y[tr].mean()
    return p

def geo(g, cols, D=None):
    D = D if D is not None else globals()["D"]
    s = D[D.part == g].sort_values("instance")
    return s[cols].values.astype(float), s["severity"].values, s["instance"].values

print(f"LOO mean-baseline R^2 at n=20 is {1-(20/19)**2:+.3f} (not 0)")
print(f"permutation p floor with B={PERM_B}: {1/(PERM_B+1):.4f}")


# %% CELL 1 — leakage demonstration (C1) --------------------------------------
if not done("c1_leakage.csv"):
    rows = []
    leak_cols = FEATS + ["severity", "was_truncated"]
    Dl = D.copy(); Dl["was_truncated"] = (Dl["n_timestamped_events"] >= 131072).astype(int)
    for g in GEOMS:
        s = Dl[Dl.part == g].sort_values("instance"); y = s["severity"].values
        sp = loo_splits(len(y))
        r_leak  = r2(y, cv_pred(s[leak_cols].values.astype(float), y, rf, sp))
        r_clean = r2(y, cv_pred(s[FEATS].values.astype(float), y, rf, sp))
        rows.append(dict(part=g, cfg=f"{REF[0]}/{REF[1]}",
                         r2_with_target_in_X=round(r_leak, 3),
                         r2_clean=round(r_clean, 3)))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(ckpt("c1_leakage.csv"), index=False)
print(pd.read_csv(ckpt("c1_leakage.csv")).to_string(index=False))
# If r2_with_target_in_X reproduces ~0.43/0.40/0.82, Section 3.7 is a leakage
# artefact and must be rewritten.


# %% CELL 2 — clean stability sweep + baselines (C1, C2, C15, C16) ------------
if not done("c2_sweep.csv"):
    rows = []
    for fs_name, fs in [("full", FEATS), ("no_count", NOCOUNT)]:
        for g in GEOMS:
            X, y, _ = geo(g, fs); sp = loo_splits(len(y))
            rows.append(dict(features=fs_name, part=g, model="mean_baseline",
                             trees=None, depth=None, r2=r2(y, mean_baseline(y, sp)),
                             mae=np.mean(np.abs(y - mean_baseline(y, sp)))))
            pe = cv_pred(X, y, enet, sp)
            rows.append(dict(features=fs_name, part=g, model="elasticnet",
                             trees=None, depth=None, r2=r2(y, pe),
                             mae=np.mean(np.abs(y - pe))))
            for nt, md in GRID:
                p = cv_pred(X, y, lambda nt=nt, md=md: rf(nt, md), sp)
                rows.append(dict(features=fs_name, part=g, model="rf",
                                 trees=nt, depth=md, r2=r2(y, p),
                                 mae=np.mean(np.abs(y - p))))
            print(f"  {fs_name:8s} {g:6s} done")
    pd.DataFrame(rows).round(4).to_csv(ckpt("c2_sweep.csv"), index=False)
S2 = pd.read_csv(ckpt("c2_sweep.csv"))
print(S2[S2.model == "rf"].pivot_table(index=["features", "part"],
      columns=["trees", "depth"], values="r2").round(3).to_string())
print(S2[S2.model != "rf"][["features", "part", "model", "r2", "mae"]]
      .round(3).to_string(index=False))


# %% CELL 3 — permutation tests at REF, one geometry per run (C15) ------------
def perm_test(X, y, ctor, B=PERM_B, seed=RNG):
    sp = loo_splits(len(y)); rng = np.random.default_rng(seed)
    obs = r2(y, cv_pred(X, y, ctor, sp))
    null = np.array([r2(yp, cv_pred(X, yp, ctor, sp))
                     for yp in (rng.permutation(y) for _ in range(B))])
    return obs, (1 + np.sum(null >= obs)) / (B + 1), null

def run_perm(g, fs_name):
    name = f"c3_perm_{g}_{fs_name}.json"
    if done(name): return json.load(open(ckpt(name)))
    fs = FEATS if fs_name == "full" else NOCOUNT
    X, y, _ = geo(g, fs)
    obs, p, null = perm_test(X, y, rf)
    res = dict(part=g, features=fs_name, cfg=f"{REF[0]}/{REF[1]}", B=PERM_B,
               r2=round(obs, 3), p=round(p, 4), p_floor=round(1/(PERM_B+1), 4),
               null_mean=round(float(np.mean(null)), 3),
               null_95=round(float(np.quantile(null, 0.95)), 3))
    json.dump(res, open(ckpt(name), "w")); return res

# Plate first (the headline). Box/Cover only matter if their R^2 > 0.
for g in ["Plate", "Box", "Cover"]:
    for fs in ["full", "no_count"]:
        print(run_perm(g, fs))


# %% CELL 4 — time-order confound (C5) ----------------------------------------
if not done("c4_time.csv"):
    rows = []
    for g in GEOMS:
        s = D[D.part == g].sort_values("instance"); y = s["severity"].values
        n = len(y); loo, blk = loo_splits(n), block_splits(n, 5)
        Xi = s[["instance"]].values.astype(float)
        Xf, Xt = s[FEATS].values.astype(float), s[NOTIME].values.astype(float)
        Xn = s[NOCOUNT].values.astype(float)
        rows.append(dict(part=g,
            r2_instance_only_LOO = r2(y, cv_pred(Xi, y, rf, loo)),
            r2_full_LOO          = r2(y, cv_pred(Xf, y, rf, loo)),
            r2_no_time_LOO       = r2(y, cv_pred(Xt, y, rf, loo)),
            r2_full_BLOCK5       = r2(y, cv_pred(Xf, y, rf, blk)),
            r2_no_count_BLOCK5   = r2(y, cv_pred(Xn, y, rf, blk)),
            r2_baseline_BLOCK5   = r2(y, mean_baseline(y, blk))))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    pd.DataFrame(rows).round(3).to_csv(ckpt("c4_time.csv"), index=False)

    # which features are basically "instance order"?
    tp = []
    for g in GEOMS:
        s = D[D.part == g]
        for c in FEATS:
            v = s[[c, "instance"]].dropna()
            if len(v) > 5 and v[c].std() > 0:
                rho = stats.spearmanr(v[c], v["instance"]).correlation
                tp.append(dict(part=g, feature=c, rho_with_instance=rho))
    TP = pd.DataFrame(tp); TP["abs"] = TP.rho_with_instance.abs()
    TP.sort_values(["part", "abs"], ascending=[True, False]) \
      .drop(columns="abs").round(3).to_csv(ckpt("c4_time_proxies.csv"), index=False)
print(pd.read_csv(ckpt("c4_time.csv")).to_string(index=False))
TPr = pd.read_csv(ckpt("c4_time_proxies.csv"))
print("\nfeatures with |rho(instance)| > 0.7:")
print(TPr[TPr.rho_with_instance.abs() > 0.7].to_string(index=False))


# %% CELL 5 — Plate truncation / program coverage + regime check (C4, C11) ----
cov_cols = [c for c in ["n_timestamped_events", "block_max", "Block_Number_max",
                        "n_block_events", "n_distinct_tools", "n_tool_changes",
                        "tool_seq_len", "runtime_max", "spindle_time_max",
                        "Part_Count_median"] if c in D.columns]
P = D[D.part == "Plate"].sort_values("instance")[["instance", "severity"] + cov_cols]
P.round(3).to_csv(ckpt("c5_plate_coverage.csv"), index=False)
print(P.round(3).to_string(index=False))

mode_tools = P["n_distinct_tools"].mode().iloc[0]
P["reduced_tooling"] = (P["n_distinct_tools"] < mode_tools).astype(int)
print("\nreduced-tooling Plate instances:",
      P.loc[P.reduced_tooling == 1, "instance"].tolist())

y = P["severity"].values; sp = loo_splits(len(y))
r_flag = r2(y, cv_pred(P[["reduced_tooling"]].values.astype(float), y, rf, sp))
s_pl = D[D.part == "Plate"].sort_values("instance")
r_full = r2(y, cv_pred(s_pl[FEATS].values.astype(float), y, rf, sp))
norm = P.reduced_tooling.values == 0
yn = y[norm]; Xn = s_pl[FEATS].values.astype(float)[norm]
r_within = r2(yn, cv_pred(Xn, yn, rf, loo_splits(len(yn)))) if len(yn) >= 6 else np.nan
reg = dict(r2_regime_flag_only=r_flag, r2_full_features=r_full,
           r2_within_normal_regime=r_within, n_normal=int(norm.sum()),
           sev_mean_normal=float(y[norm].mean()), sev_mean_reduced=float(y[~norm].mean())
           if (~norm).any() else np.nan)
pd.DataFrame([reg]).round(3).to_csv(ckpt("c5_regime.csv"), index=False)
print(pd.DataFrame([reg]).round(3).to_string(index=False))
# If r2_regime_flag_only ~ r2_full_features, the Plate result is regime detection.


# %% CELL 6 — SHAP for full AND no-count models (C3) --------------------------
try:
    import shap
    for fs_name, fs in [("full", FEATS), ("no_count", NOCOUNT)]:
        for g in GEOMS:
            name = f"c6_shap_{g}_{fs_name}.csv"
            if done(name): continue
            X, y, _ = geo(g, fs)
            m = rf(); m.fit(X, y)
            Xi = m.named_steps["imp"].transform(X)
            sv = shap.TreeExplainer(m.named_steps["rf"]).shap_values(Xi)
            kept = [c for c, keep in zip(fs, ~np.all(np.isnan(X), axis=0)) if keep]
            pd.DataFrame({"feature": kept, "mean_abs_shap": np.abs(sv).mean(0)}) \
              .sort_values("mean_abs_shap", ascending=False) \
              .to_csv(ckpt(name), index=False)
    for fs_name in ["full", "no_count"]:
        top = pd.read_csv(ckpt(f"c6_shap_Plate_{fs_name}.csv")).head(8)
        print(f"\nPlate SHAP top-8 ({fs_name}):\n", top.to_string(index=False))
except ImportError:
    print("pip install shap   -> then re-run this cell")


# %% CELL 7 — nested jackknife+ conformal (C6, C7) ----------------------------
def jackknife_plus(Xtr, ytr, xte, ctor, alpha=ALPHA):
    m = len(ytr); lo, hi = np.empty(m), np.empty(m)
    for j in range(m):
        tr = np.delete(np.arange(m), j)
        mod = ctor(); mod.fit(Xtr[tr], ytr[tr])
        rj = abs(ytr[j] - mod.predict(Xtr[j:j+1])[0])
        mu = mod.predict(xte)[0]
        lo[j], hi[j] = mu - rj, mu + rj
    k_lo = max(int(np.floor(alpha * (m + 1))), 1)
    k_hi = min(int(np.ceil((1 - alpha) * (m + 1))), m)
    return np.sort(lo)[k_lo - 1], np.sort(hi)[k_hi - 1]

CONF_FEATS, CONF_NAME = NOCOUNT, "no_count"    # one model for width AND coverage
for g in GEOMS:
    name = f"c7_conformal_{g}.csv"
    if done(name): continue
    X, y, inst = geo(g, CONF_FEATS); n = len(y); rows = []
    for i in range(n):                          # outer: i never used to calibrate
        tr = np.delete(np.arange(n), i)
        lo, hi = jackknife_plus(X[tr], y[tr], X[i:i+1], rf)
        full = rf(); full.fit(X[tr], y[tr])
        rows.append(dict(part=g, instance=int(inst[i]), y_true=y[i],
                         y_pred=full.predict(X[i:i+1])[0], lo=lo, hi=hi,
                         covered=int(lo <= y[i] <= hi)))
    pd.DataFrame(rows).to_csv(ckpt(name), index=False)
    print(f"  {g} done")

CI = pd.concat([pd.read_csv(ckpt(f"c7_conformal_{g}.csv")) for g in GEOMS])
CI["width"] = CI.hi - CI.lo
summ = CI.groupby("part").agg(coverage=("covered", "mean"), mean_width=("width", "mean"),
                              n=("covered", "size")).reset_index()
rngs = D.groupby("part")["severity"].agg(lambda s: s.max() - s.min()).rename("sev_range")
summ = summ.merge(rngs, on="part"); summ["width_over_range"] = summ.mean_width / summ.sev_range
summ["coverage_95CI"] = [
    "{:.2f}-{:.2f}".format(*stats.binomtest(int(round(c*n)), n).proportion_ci(0.95))
    for c, n in zip(summ.coverage, summ.n)]
summ.round(3).to_csv(ckpt("c7_conformal_summary.csv"), index=False)
print(f"model: RF {REF} on {CONF_NAME} features")
print(summ.round(3).to_string(index=False))


# %% CELL 8 — cross-geometry transfer (C12) -----------------------------------
rows = []
for held in GEOMS:
    tr, te = D[D.part != held], D[D.part == held]
    for fs_name, fs in [("full", FEATS), ("no_count", NOCOUNT)]:
        m = rf(300, 4); m.fit(tr[fs].values.astype(float), tr["severity"].values)
        p = m.predict(te[fs].values.astype(float)); y = te["severity"].values
        rows.append(dict(held_out=held, features=fs_name, n=len(y),
                         r2=r2(y, p), mae=np.mean(np.abs(y - p))))
pd.DataFrame(rows).round(3).to_csv(ckpt("c8_transfer.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))


# %% CELL 9 — characteristic level with honest baseline (C13) -----------------
if not done("c9_charlevel.csv"):
    rows = []
    M = Lok.merge(F, on=["part", "instance"], how="inner")
    for g, s in M.groupby("part"):
        X = s[NOCOUNT].values.astype(float); y = s["abs_label"].values
        grp = s["instance"].values
        p = np.full(len(y), np.nan); b = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
            m = rf(150, 5); m.fit(X[tr], y[tr]); p[te] = m.predict(X[te])
            b[te] = y[tr].mean()                      # baseline from TRAIN folds
        rows.append(dict(part=g, n_rows=len(y), rf_r2=r2(y, p),
                         rf_mae=np.mean(np.abs(y - p)), baseline_mae=np.mean(np.abs(y - b)),
                         frac_meas_within_tol=np.mean(y <= 1.0)))
    pd.DataFrame(rows).round(3).to_csv(ckpt("c9_charlevel.csv"), index=False)
print(pd.read_csv(ckpt("c9_charlevel.csv")).to_string(index=False))


# %% CELL 10 — instance-join verification (C9) --------------------------------
def ln(t): return t.split('}')[-1] if '}' in t else t
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "FAIR", "*.qif"))):
    part = next((p for p in GEOMS if p.lower() in f.lower()), "?")
    root = __import__("xml.etree.ElementTree", fromlist=["x"]).parse(f).getroot()
    blocks = [e for e in root.iter() if ln(e.tag) == "MeasurementResults"]
    for k, b in enumerate(blocks, 1):
        ids = [ (e.text or "").strip() for e in b.iter()
                if ln(e.tag) in ("Id",) and (e.text or "").strip() ][:3]
        serial = [ (e.text or "").strip() for e in b.iter()
                   if "serial" in ln(e.tag).lower() and (e.text or "").strip() ][:2]
        times = [ (e.text or "").strip() for e in b.iter()
                  if re.search("time|date", ln(e.tag), re.I) and (e.text or "").strip() ][:2]
        rows.append(dict(part=part, qif_block=k, component_ids=";".join(ids),
                         serial=";".join(serial), qif_time=";".join(times)))
Q = pd.DataFrame(rows)

def first_ts(path):
    import xml.etree.ElementTree as ET
    for _, el in ET.iterparse(path, events=("end",)):
        ts = el.get("timestamp")
        if ts: return ts
    return None
mrows = []
for p in sorted(glob.glob(os.path.join(ROOT, "MTConnect", "*of20.xml"))):
    m = re.search(r"([A-Za-z]+)-([A-Za-z0-9]+)-(\d+)of(\d+)\.xml", os.path.basename(p))
    if m:
        mrows.append(dict(part=m.group(1).capitalize(), mtc_instance=int(m.group(3)),
                          mtc_first_ts=first_ts(p), file=os.path.basename(p)))
MT = pd.DataFrame(mrows)
J = Q.merge(MT, left_on=["part", "qif_block"], right_on=["part", "mtc_instance"], how="outer")
J.sort_values(["part", "qif_block"]).to_csv(ckpt("c10_join_check.csv"), index=False)
for g in GEOMS:
    q = set(Q[Q.part == g].qif_block); mt = set(MT[MT.part == g].mtc_instance)
    print(f"{g:6s}: QIF blocks={len(q)}  MTC files={len(mt)}  "
          f"QIF-only={sorted(q-mt)}  MTC-only={sorted(mt-q)}")
print("\nInspect c10_join_check.csv: QIF ids/serials/times must line up with "
      "the MTConnect file order. If Box is missing one, confirm WHICH one.")


# %% CELL 11 — tolerance symmetry (C14) + label sensitivity (C8) --------------
Ls = L[~L.is_admin & ~L.is_form & L[["target", "min", "max"]].notna().all(1)].copy()
Ls["band"] = Ls["max"] - Ls["min"]
Ls["asym"] = (Ls["target"] - (Ls["min"] + Ls["max"]) / 2).abs() > 1e-6 * Ls["band"].abs().clip(lower=1e-12)
asym = Ls.groupby(["part", "char_type"])["asym"].mean().reset_index()
asym.round(3).to_csv(ckpt("c11_tolerance_asymmetry.csv"), index=False)
print("fraction of size characteristics with asymmetric tolerance:")
print(asym.round(3).to_string(index=False))

variants = {"PRIMARY (paper text)": LABEL_CFG,
            "as_in_old_code": dict(exclude_form=False, screen_invariant=False, norm="halfband"),
            "sided_norm": dict(exclude_form=True, screen_invariant=True, norm="sided")}
rows = []
for vname, cfg in variants.items():
    Lv, Sv = build_labels(L, **cfg); Dv = make_D(Sv)
    for g in GEOMS:
        X, y, _ = geo(g, NOCOUNT, Dv); sp = loo_splits(len(y))
        rows.append(dict(variant=vname, part=g, n_label_rows=len(Lv[Lv.part == g]),
                         sev_sd=y.std(), r2_rf_nocount=r2(y, cv_pred(X, y, rf, sp))))
pd.DataFrame(rows).round(3).to_csv(ckpt("c11_label_sensitivity.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))


# %% CELL 12 — figure (clean sweep) + report ----------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "legend.frameon": False})
C = {"Box": "#0072B2", "Cover": "#E69F00", "Plate": "#009E73"}
S2 = pd.read_csv(ckpt("c2_sweep.csv")); full = S2[(S2.features == "full") & (S2.model == "rf")]
fig, ax = plt.subplots(figsize=(8, 4.6))
for g in GEOMS:
    s = full[full.part == g]
    ax.plot(range(len(GRID)), [s[(s.trees == t) & (s.depth == d)].r2.values[0] for t, d in GRID],
            marker="o", color=C[g], label=g)
ax.axhline(0, ls="--", lw=1, color="#222")
ax.axhline(1 - (20/19)**2, ls=":", lw=1, color="#777")
ax.text(len(GRID) - 1, 1 - (20/19)**2 - 0.04, "LOO mean baseline", ha="right",
        va="top", fontsize=11, color="#777")
ax.set_xticks(range(len(GRID))); ax.set_xticklabels([f"{t}/{d}" for t, d in GRID])
ax.set_xlabel("Random-forest setting (trees / depth)"); ax.set_ylabel("Held-out $R^2$")
ax.set_ylim(-0.9, 0.9)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))   # outside the plot
for ext in ("pdf", "png"): fig.savefig(os.path.join(FIG, f"F5_stability_sweep_clean.{ext}"))
plt.close(fig)

def md(name):
    p = ckpt(name)
    if not os.path.exists(p): return f"_{name} missing_"
    return pd.read_csv(p).round(3).to_markdown(index=False)
perm = [json.load(open(p)) for p in sorted(glob.glob(ckpt("c3_perm_*.json")))]
rep = ["# Clean rerun report\n",
       f"Label config: `{LABEL_CFG}` | RF ref {REF} | B={PERM_B} | alpha={ALPHA}\n",
       "## C1 leakage\n", md("c1_leakage.csv"),
       "\n## C2 sweep + baselines\n", md("c2_sweep.csv"),
       "\n## C3 permutation\n", pd.DataFrame(perm).to_markdown(index=False) if perm else "_none_",
       "\n## C4 time confound\n", md("c4_time.csv"),
       "\n## C5 Plate coverage / regime\n", md("c5_plate_coverage.csv"), "\n", md("c5_regime.csv"),
       "\n## C7 conformal (nested jackknife+)\n", md("c7_conformal_summary.csv"),
       "\n## C8 transfer\n", md("c8_transfer.csv"),
       "\n## C9 characteristic level\n", md("c9_charlevel.csv"),
       "\n## C11 tolerance symmetry / label sensitivity\n",
       md("c11_tolerance_asymmetry.csv"), "\n", md("c11_label_sensitivity.csv")]
open(ckpt("_rerun_report.md"), "w").write("\n".join(rep))
print("report ->", ckpt("_rerun_report.md"))