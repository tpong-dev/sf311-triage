"""Resolution-time prediction for SF 311 requests.

Two targets, both built so the labels are fully observed (no right-censoring) on every row
we train or score on:

* **late7d** (classification): 1 if the request was NOT closed within 7 days (168 h).
  A request that is still open 7 days after it was filed is late whatever happens next,
  so the label is settled for every request at least 7 days older than the data snapshot.
* **log_hours_cap30** (regression): log1p(min(resolution_hours, 720)). Requests still open
  after 30 days get the cap. Settled for every request at least 30 days old.

Administrative closures (duplicates, invalid, transferred...) are excluded from both targets.
Splits follow `sf311.dataset` (train < 2026-08, val = Aug 2026, test = Sep 2026); Oct 2026 is
not settled for the 30-day target and is used only as an extra 7-day check (Oct 1-26).

    python -m sf311.resolution train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sf311.dataset import TEST_START, VAL_START, load

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "resolution"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

LATE_H = 168.0  # 7 days
CAP_H = 720.0  # 30 days
WARMUP_END = pd.Timestamp("2025-12-01")  # first month only warms up the backlog counters
TEST_END = pd.Timestamp("2026-10-01")  # Sep 2026 is >= 33 days old at snapshot
OCT_END = pd.Timestamp("2026-10-27")  # last request date with a settled 7-day label

CAT_COLS = [
    "service_name",
    "service_subtype",
    "service_details",
    "agency_responsible",
    "source",
    "supervisor_district",
    "analysis_neighborhood",
    "police_district",
]
NUM_COLS = [
    "lat",
    "long",
    "hour",
    "weekday",
    "is_weekend",
    "has_photo",
    "n_subtype_prev_1d",
    "n_subtype_prev_7d",
    "n_name_prev_7d",
    "n_city_prev_1d",
    "open_subtype_at_t",
    "open_name_at_t",
    "open_agency_at_t",
]
FEATURES = CAT_COLS + NUM_COLS
TOP_DETAILS = 200
TOP_CATS = 10


# ----------------------------------------------------------------------------- features


def _snapshot(df: pd.DataFrame) -> pd.Timestamp:
    """Data cut-off = last update seen. closed_date has stray future values, so it is not used."""
    return df["updated_datetime"].max()


def _prior_count(df: pd.DataFrame, key: str, window: str) -> np.ndarray:
    """Requests in the same `key` group filed in [t - window, t) (strictly before t)."""
    out = np.zeros(len(df), dtype=np.int32)
    w = pd.Timedelta(window).to_timedelta64()
    for _, idx in df.groupby(key, sort=False, dropna=False).indices.items():
        t = df["requested_datetime"].to_numpy()[idx]
        order = np.argsort(t, kind="stable")
        ts = t[order]
        lo = np.searchsorted(ts, ts - w, side="left")
        hi = np.searchsorted(ts, ts, side="left")  # excludes ties at exactly t
        res = np.empty(len(idx), dtype=np.int32)
        res[order] = hi - lo
        out[idx] = res
    return out


def _open_at_t(df: pd.DataFrame, key: str) -> np.ndarray:
    """Requests in the same group filed before t and not yet closed at t.

    Uses only closures that happened before t, so it is knowable at filing time.
    Requests filed before the first downloaded month are invisible (hence the warm-up month).
    """
    out = np.zeros(len(df), dtype=np.int32)
    req_all = df["requested_datetime"].to_numpy()
    clo_all = df["closed_date"].to_numpy()
    for _, idx in df.groupby(key, sort=False, dropna=False).indices.items():
        req = np.sort(req_all[idx])
        clo = clo_all[idx]
        clo = np.sort(clo[~np.isnat(clo)])
        t = req_all[idx]
        out[idx] = np.searchsorted(req, t, side="left") - np.searchsorted(clo, t, side="left")
    return out


def build_frame(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    if df is None:
        df = load()
    snap = _snapshot(df)
    df = df.sort_values("requested_datetime").reset_index(drop=True)
    t = df["requested_datetime"]

    # Backlog features on ALL rows (admin closures are still workload).
    df["n_subtype_prev_1d"] = _prior_count(df, "service_subtype", "1D")
    df["n_subtype_prev_7d"] = _prior_count(df, "service_subtype", "7D")
    df["n_name_prev_7d"] = _prior_count(df, "service_name", "7D")
    df["_all"] = 0
    df["n_city_prev_1d"] = _prior_count(df, "_all", "1D")
    df["open_subtype_at_t"] = _open_at_t(df, "service_subtype")
    df["open_name_at_t"] = _open_at_t(df, "service_name")
    df["open_agency_at_t"] = _open_at_t(df, "agency_responsible")

    df["hour"] = t.dt.hour
    df["weekday"] = t.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["has_photo"] = df["photo_url"].notna().astype(int)

    # Targets: hours observed so far; open requests have been open (snap - t).
    hours = df["resolution_hours"]
    age_h = (snap - t).dt.total_seconds() / 3600
    open_ = ~df["is_closed"]
    # Closed late, or still open for longer than LATE_H.
    df["late7d"] = np.where(df["is_closed"], hours > LATE_H, age_h > LATE_H).astype(float)
    df.loc[open_ & (age_h <= LATE_H), "late7d"] = np.nan
    capped = np.where(df["is_closed"], np.minimum(hours, CAP_H), np.where(age_h > CAP_H, CAP_H, np.nan))
    df["hours_cap30"] = capped
    df["log_hours_cap30"] = np.log1p(df["hours_cap30"])
    df["log_hours_closed"] = np.log1p(hours.where(df["is_closed"]))
    df = df.drop(columns="_all")
    return df, snap


def make_categories(train: pd.DataFrame) -> dict[str, list[str]]:
    cats = {}
    for c in CAT_COLS:
        vc = train[c].fillna("NA").value_counts()
        if c == "service_details":
            vc = vc.head(TOP_DETAILS)
        cats[c] = sorted(vc[vc >= 5].index.tolist())
    return cats


def to_X(df: pd.DataFrame, cats: dict[str, list[str]], features: list[str] = FEATURES) -> pd.DataFrame:
    """Unseen category values become NaN (LightGBM treats them as missing)."""
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in cats:
            v = df[c].fillna("NA")
            X[c] = pd.Categorical(v.where(v.isin(cats[c])), categories=cats[c])
        else:
            X[c] = df[c].astype(float)
    return X


# ----------------------------------------------------------------------------- metrics


def reg_metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    yh, ph = np.expm1(y_log), np.expm1(p_log)
    return {
        "n": int(len(y_log)),
        "MAE_h": float(mean_absolute_error(yh, ph)),
        "MedAE_h": float(np.median(np.abs(yh - ph))),
        "MAE_log": float(mean_absolute_error(y_log, p_log)),
        "R2_log": float(r2_score(y_log, p_log)),
        "R2_h": float(r2_score(yh, ph)),
    }


def clf_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = y.astype(int)
    pred = (p >= thr).astype(int)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "late_rate": float(y.mean()),
        "n_late": int(y.sum()),
        "ROC_AUC": float(roc_auc_score(y, p)) if both else float("nan"),
        "PR_AUC": float(average_precision_score(y, p)) if both else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "late_caught": int(((pred == 1) & (y == 1)).sum()),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y.astype(int), p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


# ----------------------------------------------------------------------------- baselines


def group_stat(train: pd.DataFrame, target: str, how: str) -> callable:
    """Per-subtype statistic with fallback to service_name, then global."""
    g = train.groupby("service_subtype")[target].agg([how, "size"])
    sub = g[g["size"] >= 20][how]
    name = train.groupby("service_name")[target].agg(how)
    glob = float(getattr(train[target], how)())

    def f(df: pd.DataFrame) -> np.ndarray:
        v = df["service_subtype"].map(sub)
        v = v.fillna(df["service_name"].map(name))
        return v.fillna(glob).to_numpy(dtype=float)

    return f


# ----------------------------------------------------------------------------- train


def _lgb_params(objective: str) -> dict:
    return dict(
        objective=objective,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        cat_smooth=20,
        max_cat_to_onehot=8,
        verbose=-1,
        num_threads=0,
        seed=42,
    )


def fit_lgb(Xtr, ytr, Xva, yva, objective: str):
    import lightgbm as lgb

    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
    return lgb.train(
        _lgb_params(objective),
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )


def per_category(df: pd.DataFrame, y_col: str, preds: dict[str, np.ndarray], thr: dict[str, float], cats: list[str]):
    rows = []
    for c in cats:
        m = (df["service_name"] == c).to_numpy()
        for name, p in preds.items():
            r = clf_metrics(df.loc[m, y_col].to_numpy(), p[m], thr[name])
            rows.append({"service_name": c, "model": name, **r})
    return pd.DataFrame(rows)


def per_category_reg(df: pd.DataFrame, y_col: str, preds: dict[str, np.ndarray], cats: list[str]):
    rows = []
    for c in cats:
        m = (df["service_name"] == c).to_numpy()
        for name, p in preds.items():
            rows.append({"service_name": c, "model": name, **reg_metrics(df.loc[m, y_col].to_numpy(), p[m])})
    return pd.DataFrame(rows)


def train(shap_n: int = 5000) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("loading + features ...")
    df, snap = build_frame()
    # Data errors: closed before requested, or closed after the snapshot.
    bad = (df["closed_date"].notna() & ~df["is_closed"]) | (df["closed_date"] > snap)
    print(f"  dropping {bad.sum():,} rows with negative or post-snapshot durations")
    df = df[~df["is_admin_close"] & ~bad].reset_index(drop=True)
    t = df["requested_datetime"]
    tr = df[(t >= WARMUP_END) & (t < VAL_START)].reset_index(drop=True)
    va = df[(t >= VAL_START) & (t < TEST_START)].reset_index(drop=True)
    te = df[(t >= TEST_START) & (t < TEST_END)].reset_index(drop=True)
    oc = df[(t >= TEST_END) & (t < OCT_END)].reset_index(drop=True)
    for name, d in [("train", tr), ("val", va), ("test", te), ("oct", oc)]:
        print(f"  {name}: {len(d):,} rows, late7d={d['late7d'].mean():.3f}, unsettled late7d={d['late7d'].isna().sum()}")
    assert te["late7d"].notna().all() and te["log_hours_cap30"].notna().all()
    assert oc["late7d"].notna().all()

    cats = make_categories(tr)
    Xtr, Xva, Xte, Xoc = (to_X(d, cats) for d in (tr, va, te, oc))
    results: dict = {
        "snapshot": str(snap),
        "splits": {
            "train": [str(WARMUP_END.date()), str((VAL_START - pd.Timedelta(days=1)).date()), len(tr)],
            "val": [str(VAL_START.date()), str((TEST_START - pd.Timedelta(days=1)).date()), len(va)],
            "test": [str(TEST_START.date()), str((TEST_END - pd.Timedelta(days=1)).date()), len(te)],
            "oct_7d_check": [str(TEST_END.date()), str((OCT_END - pd.Timedelta(days=1)).date()), len(oc)],
        },
    }

    # ---------------- classification: late7d
    print("training late7d classifier ...")
    clf = fit_lgb(Xtr, tr["late7d"], Xva, va["late7d"], "binary")
    p_va = clf.predict(Xva, num_iteration=clf.best_iteration)
    thr = best_f1_threshold(va["late7d"].to_numpy(), p_va)
    base_rate = group_stat(tr, "late7d", "mean")
    b_va = base_rate(va)
    b_thr = best_f1_threshold(va["late7d"].to_numpy(), b_va)

    # Ablation: drop agency_responsible (can be reassigned after filing -> possible leakage).
    feats_noag = [f for f in FEATURES if f not in ("agency_responsible", "open_agency_at_t")]
    clf_noag = fit_lgb(to_X(tr, cats, feats_noag), tr["late7d"], to_X(va, cats, feats_noag), va["late7d"], "binary")
    thr_noag = best_f1_threshold(va["late7d"].to_numpy(), clf_noag.predict(to_X(va, cats, feats_noag), num_iteration=clf_noag.best_iteration))

    def clf_preds(X, d):
        return {
            "LightGBM": clf.predict(X, num_iteration=clf.best_iteration),
            "LightGBM (no agency)": clf_noag.predict(to_X(d, cats, feats_noag), num_iteration=clf_noag.best_iteration),
            "Subtype late-rate": base_rate(d),
        }

    thrs = {"LightGBM": thr, "LightGBM (no agency)": thr_noag, "Subtype late-rate": b_thr}
    results["late7d"] = {"threshold": thr, "baseline_threshold": b_thr}
    for split_name, X, d in [("test", Xte, te), ("oct_7d_check", Xoc, oc)]:
        ps = clf_preds(X, d)
        results["late7d"][split_name] = {k: clf_metrics(d["late7d"].to_numpy(), p, thrs[k]) for k, p in ps.items()}
        results["late7d"][split_name]["Majority (never late)"] = clf_metrics(d["late7d"].to_numpy(), np.zeros(len(d)), 0.5)
        if split_name == "test":
            p_test = ps
    top = te["service_name"].value_counts().head(TOP_CATS).index.tolist()
    pc = per_category(te, "late7d", {k: p_test[k] for k in ("LightGBM", "Subtype late-rate")}, thrs, top)
    results["late7d"]["per_category"] = pc.to_dict(orient="records")
    # Also every category with >= 30 late cases in test: does any get zero recall?
    big = te.groupby("service_name")["late7d"].sum()
    big = big[big >= 30].index.tolist()
    pc_all = per_category(te, "late7d", {"LightGBM": p_test["LightGBM"]}, thrs, big)
    results["late7d"]["min_recall_categories"] = (
        pc_all.sort_values("recall")[["service_name", "n", "n_late", "recall", "precision", "ROC_AUC"]].head(8).to_dict(orient="records")
    )
    results["late7d"]["n_categories_ge30_late"] = len(big)
    results["late7d"]["n_categories_zero_recall"] = int((pc_all["recall"] == 0).sum())

    # ---------------- regression: log1p(min(hours, 720))
    print("training capped regression ...")
    reg = fit_lgb(Xtr, tr["log_hours_cap30"], Xva, va["log_hours_cap30"], "regression_l1")
    y_te = te["log_hours_cap30"].to_numpy()
    med_glob = np.full(len(te), np.log1p(tr["hours_cap30"].median()))
    med_sub = np.log1p(group_stat(tr, "hours_cap30", "median")(te))
    p_reg = reg.predict(Xte, num_iteration=reg.best_iteration)
    reg_preds = {"LightGBM": p_reg, "Subtype median": med_sub, "Global median": med_glob}
    results["reg_cap30"] = {"test": {k: reg_metrics(y_te, p) for k, p in reg_preds.items()}}
    results["reg_cap30"]["per_category"] = per_category_reg(te, "log_hours_cap30", reg_preds, top).to_dict(orient="records")

    # Secondary: closed-only, uncapped (biased toward short durations for recent months).
    print("training closed-only regression ...")
    trc, vac = tr[tr["is_closed"]], va[va["is_closed"]]
    tec = te[te["is_closed"]].reset_index(drop=True)
    reg_c = fit_lgb(to_X(trc, cats), trc["log_hours_closed"], to_X(vac, cats), vac["log_hours_closed"], "regression")
    yc = tec["log_hours_closed"].to_numpy()
    rc_preds = {
        "LightGBM": reg_c.predict(to_X(tec, cats), num_iteration=reg_c.best_iteration),
        "Subtype median": np.log1p(group_stat(trc, "resolution_hours", "median")(tec)),
        "Global median": np.full(len(tec), np.log1p(trc["resolution_hours"].median())),
    }
    results["reg_closed"] = {
        "test": {k: reg_metrics(yc, p) for k, p in rc_preds.items()},
        "test_closed_share": float(te["is_closed"].mean()),
    }

    # ---------------- feature importance + SHAP
    imp = pd.DataFrame(
        {"feature": clf.feature_name(), "gain_clf": clf.feature_importance("gain"), "gain_reg": reg.feature_importance("gain")}
    )
    imp["gain_clf"] /= imp["gain_clf"].sum()
    imp["gain_reg"] /= imp["gain_reg"].sum()
    results["importance"] = imp.sort_values("gain_clf", ascending=False).to_dict(orient="records")

    print("SHAP ...")
    import shap

    samp = Xte.sample(min(shap_n, len(Xte)), random_state=0)
    for model, fname, title in [
        (clf, "resolution_shap_late7d.png", "SHAP: late (>7 days) classifier, Sep 2026 test sample"),
        (reg, "resolution_shap_reg.png", "SHAP: log time-to-close (capped 30 d), Sep 2026 test sample"),
    ]:
        sv = shap.TreeExplainer(model).shap_values(samp)
        if isinstance(sv, list):
            sv = sv[1]
        plt.figure()
        shap.summary_plot(sv, samp, show=False, max_display=18)
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
        plt.close("all")

    _plots(te, p_test, thrs, top, p_reg, med_sub, y_te, plt)

    # ---------------- save
    clf.save_model(str(MODEL_DIR / "late7d_lgbm.txt"), num_iteration=clf.best_iteration)
    reg.save_model(str(MODEL_DIR / "log_hours_cap30_lgbm.txt"), num_iteration=reg.best_iteration)
    meta = {
        "features": FEATURES,
        "categorical": CAT_COLS,
        "categories": cats,
        "late_hours": LATE_H,
        "cap_hours": CAP_H,
        "threshold_late7d": thr,
        "best_iteration": {"late7d": clf.best_iteration, "reg_cap30": reg.best_iteration},
        "train_window": results["splits"]["train"],
        "snapshot": str(snap),
        "note": "Backlog features must be computed with sf311.resolution.build_frame on the full request history.",
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=1, default=float))
    write_report(results)
    print(json.dumps({k: results[k]["test"] for k in ("late7d", "reg_cap30", "reg_closed")}, indent=1, default=float))
    return results


def _plots(te, p_test, thrs, top, p_reg, med_sub, y_te, plt):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve

    y = te["late7d"].to_numpy().astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in p_test.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax[0].plot(rec, prec, label=name)
    ax[0].axhline(y.mean(), ls=":", c="grey", label=f"base rate {y.mean():.2f}")
    ax[0].set(xlabel="Recall (late)", ylabel="Precision (late)", title="Precision-recall, late > 7 days (Sep 2026)")
    ax[0].legend(fontsize=8)
    for name in ("LightGBM", "Subtype late-rate"):
        fr, mp = calibration_curve(y, p_test[name], n_bins=15, strategy="quantile")
        ax[1].plot(mp, fr, marker="o", ms=3, label=name)
    ax[1].plot([0, 1], [0, 1], ls=":", c="grey")
    ax[1].set(xlabel="Predicted P(late)", ylabel="Observed late share", title="Calibration (Sep 2026)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "resolution_pr_calibration.png", dpi=130)
    plt.close(fig)

    # Per-category recall/precision bars.
    rows = []
    for c in top:
        m = (te["service_name"] == c).to_numpy()
        for name in ("LightGBM", "Subtype late-rate"):
            r = clf_metrics(y[m], p_test[name][m], thrs[name])
            rows.append((c, name, r["recall"], r["precision"], r["late_rate"]))
    r = pd.DataFrame(rows, columns=["cat", "model", "recall", "precision", "late_rate"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    yy = np.arange(len(top))
    for i, name in enumerate(("LightGBM", "Subtype late-rate")):
        s = r[r.model == name].set_index("cat").loc[top]
        ax[0].barh(yy + i * 0.4, s["recall"], height=0.4, label=name)
        ax[1].barh(yy + i * 0.4, s["precision"], height=0.4, label=name)
    lr = r[r.model == "LightGBM"].set_index("cat").loc[top, "late_rate"]
    ax[0].set_yticks(yy + 0.2, [f"{c} ({v:.0%} late)" for c, v in zip(top, lr)], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="Recall on late cases", xlim=(0, 1), title="Late recall by category (Sep 2026)")
    ax[1].set(xlabel="Precision on late flags", xlim=(0, 1), title="Late precision by category")
    ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "resolution_per_category.png", dpi=130)
    plt.close(fig)

    # Predicted vs actual (log scale).
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.hexbin(y_te, p_reg, gridsize=50, bins="log", cmap="viridis")
    ticks = [1, 6, 24, 72, 168, 720]
    tl = np.log1p(ticks)
    ax.plot([0, tl[-1]], [0, tl[-1]], c="w", lw=1, ls="--")
    ax.set_xticks(tl, [str(x) for x in ticks])
    ax.set_yticks(tl, [str(x) for x in ticks])
    ax.set(xlabel="Actual hours (capped 30 d)", ylabel="Predicted hours", title="LightGBM time-to-close, Sep 2026")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "resolution_pred_vs_actual.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- report


def _fmt(x, nd=3):
    if isinstance(x, float):
        return "n/a" if np.isnan(x) else f"{x:.{nd}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def _table(rows: list[dict], cols: list[str], nd=3) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def write_report(res: dict) -> None:
    """Writes the auto-generated numeric section; prose lives in reports/resolution.md around the markers."""
    L = res["late7d"]
    sp = res["splits"]
    lines = []
    lines.append(f"Data snapshot: {res['snapshot']}. Splits (request date, admin closures removed):\n")
    lines.append(_table([{"split": k, "from": v[0], "to": v[1], "rows": v[2]} for k, v in sp.items()], ["split", "from", "to", "rows"]))
    lines.append("\n### Late (> 7 days) classification, test = Sep 2026\n")
    cols = ["model", "n", "late_rate", "ROC_AUC", "PR_AUC", "precision", "recall", "F1"]
    lines.append(_table([{"model": k, **v} for k, v in L["test"].items()], cols))
    lines.append(f"\nThresholds chosen on Aug 2026 (max F1): LightGBM {L['threshold']:.3f}, subtype late-rate {L['baseline_threshold']:.3f}.\n")
    lines.append("\n**Extra check, Oct 1-26 2026** (7-day labels settled; further from training):\n")
    lines.append(_table([{"model": k, **v} for k, v in L["oct_7d_check"].items()], cols))
    lines.append("\n### Late classification by category (top 10 by test volume)\n")
    lines.append(_table(L["per_category"], ["service_name", "model", "n", "late_rate", "n_late", "ROC_AUC", "PR_AUC", "precision", "recall", "F1"]))
    lines.append(
        f"\nOf {L['n_categories_ge30_late']} service categories with at least 30 late cases in the test month, "
        f"{L['n_categories_zero_recall']} have zero late recall. Lowest-recall categories:\n"
    )
    lines.append(_table(L["min_recall_categories"], ["service_name", "n", "n_late", "recall", "precision", "ROC_AUC"]))
    R = res["reg_cap30"]
    lines.append("\n### Time-to-close regression, target log1p(min(hours, 720)), test = Sep 2026\n")
    rcols = ["model", "n", "MAE_h", "MedAE_h", "MAE_log", "R2_log", "R2_h"]
    lines.append(_table([{"model": k, **v} for k, v in R["test"].items()], rcols, 2))
    lines.append("\nBy category (top 10):\n")
    lines.append(_table(R["per_category"], ["service_name", "model", "n", "MAE_h", "MedAE_h", "MAE_log", "R2_log"], 2))
    C = res["reg_closed"]
    lines.append(
        f"\n### Secondary: closed-only, uncapped log1p(hours) (biased: {C['test_closed_share']:.1%} of Sep requests closed by the snapshot)\n"
    )
    lines.append(_table([{"model": k, **v} for k, v in C["test"].items()], rcols, 2))
    lines.append("\n### Feature importance (normalised gain)\n")
    lines.append(_table(res["importance"], ["feature", "gain_clf", "gain_reg"]))
    auto = "\n".join(lines)

    path = REPORT_DIR / "resolution.md"
    start, end = "<!-- AUTO:START -->", "<!-- AUTO:END -->"
    if path.exists() and start in (txt := path.read_text(encoding="utf-8")):
        pre, rest = txt.split(start, 1)
        post = rest.split(end, 1)[1] if end in rest else ""
        path.write_text(f"{pre}{start}\n{auto}\n{end}{post}", encoding="utf-8")
    else:
        path.write_text(f"# Resolution-time prediction\n\n{start}\n{auto}\n{end}\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="build features, train, evaluate, write report + models")
    t.add_argument("--shap-n", type=int, default=5000)
    a = ap.parse_args()
    if a.cmd == "train":
        train(shap_n=a.shap_n)


if __name__ == "__main__":
    main()
