"""
v3_metrics.py — generation-free evaluation metrics for the OD-mobility thesis.

Design contract
---------------
Every function here is a PURE function over already-materialised DataFrames.
None of them train or call a generative model. That is deliberate: it lets you
iterate on metrics in seconds without re-running generation.

Two metric families need an artefact that Script A must persist:
  * improved MIA metrics  -> need `mia_results_df` (per-user probabilities)
  * distance-privacy DCR   -> need a real HOLDOUT set (users NOT in main training)

Column assumptions (your real + synth frames):
  start_centroid_lat / start_centroid_lon / end_centroid_lat / end_centroid_lon
  duration_minutes / centroid_dist
  time: either `start_time` (real) OR decoded `hour` + `day_of_week` (synth,
        once you patch inverse_transform). Temporal metrics require one of these.
"""

from collections import Counter

import numpy as np
import pandas as pd
import h3
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors

START_LAT, START_LON = "start_centroid_lat", "start_centroid_lon"
END_LAT, END_LON = "end_centroid_lat", "end_centroid_lon"
OD_COLS = [START_LAT, START_LON, END_LAT, END_LON]


# =====================================================================
# UTILITY  (real_df vs synth_df, no model needed)
# =====================================================================

def _od_pair_counts(df, h3_res):
    df = df.dropna(subset=OD_COLS)
    pairs = (
        (h3.latlng_to_cell(a, b, h3_res), h3.latlng_to_cell(c, d, h3_res))
        for a, b, c, d in df[OD_COLS].itertuples(index=False)
    )
    return Counter(pairs)


def _jsd(p, q):
    """Return (divergence, distance) with base-2 so distance is bounded [0, 1]."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    dist = float(jensenshannon(p, q, base=2))
    return dist ** 2, dist


def od_flow_jsd(real_df, synth_df, h3_res=6):
    """Joint origin->destination flow distribution similarity. The core RQ1 metric:
    bins every trip into an (origin_cell, dest_cell) pair and compares the two
    distributions. Lower = synthetic reproduces real OD structure."""
    rc, sc = _od_pair_counts(real_df, h3_res), _od_pair_counts(synth_df, h3_res)
    keys = sorted(set(rc) | set(sc))
    if not keys:
        return {"od_flow_jsd": np.nan, "od_flow_js_distance": np.nan}
    r = np.array([rc.get(k, 0) for k in keys], float)
    s = np.array([sc.get(k, 0) for k in keys], float)
    div, dist = _jsd(r, s)
    real_set = set(rc)
    coverage = len(real_set & set(sc)) / len(real_set) if real_set else np.nan
    return {
        "od_flow_jsd": div,
        "od_flow_js_distance": dist,
        "od_flow_coverage": float(coverage),   # fraction of real OD pairs the synth reproduces
        "n_real_od_pairs": int(len(real_set)),
        "n_synth_od_pairs": int(len(set(sc))),
    }


def _hour_dow(df):
    if "start_time" in df.columns:
        t = pd.to_datetime(df["start_time"], errors="coerce")
        return t.dt.hour.to_numpy(), t.dt.dayofweek.to_numpy()
    if "hour" in df.columns and "day_of_week" in df.columns:
        return (df["hour"].round().astype(int).to_numpy(),
                df["day_of_week"].round().astype(int).to_numpy())
    raise KeyError("temporal metric needs `start_time` or `hour`+`day_of_week` "
                   "(patch inverse_transform and regenerate)")


def temporal_jsd(real_df, synth_df):
    """Hour-of-day and day-of-week distribution similarity. Shows whether the
    model captures commute peaks / weekday structure."""
    rh, rd = _hour_dow(real_df)
    sh, sd = _hour_dow(synth_df)

    def hist(x, n):
        c = np.bincount(np.clip(x[~np.isnan(x.astype(float))].astype(int), 0, n - 1),
                        minlength=n).astype(float)
        return c if c.sum() == 0 else c / c.sum()

    _, hour_dist = _jsd(hist(rh, 24), hist(sh, 24))
    _, dow_dist = _jsd(hist(rd, 7), hist(sd, 7))
    return {"hour_js_distance": hour_dist, "dow_js_distance": dow_dist}


def spatial_jsd(real_df, synth_df, h3_res=6):
    """Per-cell visit-frequency similarity for origins and destinations separately.
    Catches mode collapse onto a few popular cells. Finer res than od_flow."""
    out = {}
    for tag, lat, lon in [("origin", START_LAT, START_LON), ("dest", END_LAT, END_LON)]:
        rc = Counter(h3.latlng_to_cell(a, b, h3_res)
                     for a, b in real_df.dropna(subset=[lat, lon])[[lat, lon]].itertuples(index=False))
        sc = Counter(h3.latlng_to_cell(a, b, h3_res)
                     for a, b in synth_df.dropna(subset=[lat, lon])[[lat, lon]].itertuples(index=False))
        keys = sorted(set(rc) | set(sc))
        if not keys:
            out[f"{tag}_cell_js_distance"] = np.nan
            out[f"{tag}_cell_coverage"] = np.nan
            continue
        r = np.array([rc.get(k, 0) for k in keys], float)
        s = np.array([sc.get(k, 0) for k in keys], float)
        _, dist = _jsd(r, s)
        out[f"{tag}_cell_js_distance"] = dist
        out[f"{tag}_cell_coverage"] = float(len(set(rc) & set(sc)) / len(set(rc))) if rc else np.nan
    return out


def correlation_diff(real_df, synth_df, feature_cols):
    """How well pairwise feature dependencies survive (e.g. distance<->duration).
    Compares upper-triangle of the two correlation matrices."""
    cols = [c for c in feature_cols if c in real_df.columns and c in synth_df.columns]
    rc = real_df[cols].corr().to_numpy()
    sc = synth_df[cols].corr().to_numpy()
    d = rc - sc
    iu = np.triu_indices_from(d, k=1)
    return {
        "corr_mean_abs_diff": float(np.nanmean(np.abs(d[iu]))),
        "corr_frobenius": float(np.sqrt(np.nansum(d[iu] ** 2))),
    }


def tstr_destination(real_df, synth_df, h3_res=6, feature_cols=None,
                     top_k=15, test_size=0.3, random_state=42):
    """Train-on-Synthetic-Test-on-Real downstream task: predict the destination
    H3 cell from origin + trip attributes. Reports synthetic macro-F1, the
    real->real ceiling (TRTR), and their ratio (1.0 == synthetic is as useful)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    feature_cols = feature_cols or [START_LAT, START_LON, "duration_minutes", "centroid_dist"]

    def prep(df):
        df = df.dropna(subset=OD_COLS + ["duration_minutes", "centroid_dist"]).copy()
        df["dest_cell"] = [h3.latlng_to_cell(a, b, h3_res)
                           for a, b in df[[END_LAT, END_LON]].itertuples(index=False)]
        return df

    R, S = prep(real_df), prep(synth_df)
    top = R["dest_cell"].value_counts().nlargest(top_k).index
    top = top[R["dest_cell"].isin(top).groupby(R["dest_cell"]).size().reindex(top).fillna(0) >= 2]
    R, S = R[R["dest_cell"].isin(top)], S[S["dest_cell"].isin(top)]
    if R.empty or S.empty:
        return {"tstr_macro_f1": np.nan, "trtr_macro_f1": np.nan, "tstr_ratio": np.nan}

    try:
        R_tr, R_te = train_test_split(R, test_size=test_size,
                                      random_state=random_state, stratify=R["dest_cell"])
    except ValueError:
        R_tr, R_te = train_test_split(R, test_size=test_size, random_state=random_state)

    labels = list(top)

    def fit_eval(train):
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state,
                                     class_weight="balanced")
        clf.fit(train[feature_cols], train["dest_cell"])
        pred = clf.predict(R_te[feature_cols])
        return f1_score(R_te["dest_cell"], pred, labels=labels, average="macro", zero_division=0)

    trtr = float(fit_eval(R_tr))
    tstr = float(fit_eval(S))
    return {"tstr_macro_f1": tstr, "trtr_macro_f1": trtr,
            "tstr_ratio": float(tstr / trtr) if trtr > 0 else np.nan}


# =====================================================================
# PRIVACY — distance based  (synth vs real train [+ real holdout])
# =====================================================================

def dcr_metrics(synth_df, real_train_df, real_holdout_df=None, cols=OD_COLS, low_q=5):
    """Distance to Closest Record. Memorisation shows up as synthetic points
    sitting closer to the TRAINING data than to a never-seen HOLDOUT set, i.e.
    ratio < 1. Without a holdout you only get the raw closeness distribution."""
    S = synth_df.dropna(subset=cols)[cols].to_numpy()
    Tr = real_train_df.dropna(subset=cols)[cols].to_numpy()
    d_tr = NearestNeighbors(n_neighbors=1).fit(Tr).kneighbors(S)[0][:, 0]
    out = {"dcr_train_median": float(np.median(d_tr)),
           "dcr_train_p5": float(np.percentile(d_tr, low_q))}
    if real_holdout_df is not None:
        Ho = real_holdout_df.dropna(subset=cols)[cols].to_numpy()
        d_ho = NearestNeighbors(n_neighbors=1).fit(Ho).kneighbors(S)[0][:, 0]
        out["dcr_holdout_median"] = float(np.median(d_ho))
        out["dcr_ratio_train_over_holdout"] = (
            float(np.median(d_tr) / np.median(d_ho)) if np.median(d_ho) > 0 else np.nan
        )  # < 1.0 => leakage signal
    return out


def nndr_metrics(synth_df, real_train_df, cols=OD_COLS):
    """Nearest-Neighbour Distance Ratio: dist(1st NN) / dist(2nd NN) to real
    records. Low values flag a synthetic point glued to one specific real trip."""
    S = synth_df.dropna(subset=cols)[cols].to_numpy()
    Tr = real_train_df.dropna(subset=cols)[cols].to_numpy()
    d = NearestNeighbors(n_neighbors=2).fit(Tr).kneighbors(S)[0]
    ratio = d[:, 0] / (d[:, 1] + 1e-12)
    return {"nndr_median": float(np.median(ratio)),
            "nndr_p5": float(np.percentile(ratio, 5))}


def duplicate_rate(synth_df, real_train_df, cols=OD_COLS, eps=0.012):
    """Fraction of synthetic trips that are near-copies / exact copies of a real
    training trip. eps=0.012 matches your existing `very_close_trip_rate`."""
    S = synth_df.dropna(subset=cols)[cols].to_numpy()
    Tr = real_train_df.dropna(subset=cols)[cols].to_numpy()
    d = NearestNeighbors(n_neighbors=1).fit(Tr).kneighbors(S)[0][:, 0]
    return {"near_copy_rate": float(np.mean(d < eps)),
            "exact_match_rate": float(np.mean(d < 1e-9))}


# =====================================================================
# PRIVACY — improved MIA scoring  (from a saved mia_results_df)
# =====================================================================

def mia_score_metrics(mia_results_df, prob_col="membership_probability",
                      truth_col="ground_truth", pred_col="prediction",
                      fpr_targets=(0.01, 0.05, 0.10)):
    """Replaces accuracy-only reporting. AUC summarises ranking; TPR at a fixed
    low FPR is the worst-case privacy view (the metric the MIA literature uses)."""
    from sklearn.metrics import (roc_auc_score, roc_curve, f1_score, accuracy_score)
    y = mia_results_df[truth_col].to_numpy()
    p = mia_results_df[prob_col].to_numpy()
    pred = (mia_results_df[pred_col].to_numpy()
            if pred_col in mia_results_df else (p >= 0.5).astype(int))
    out = {"mia_accuracy": float(accuracy_score(y, pred)),
           "mia_f1": float(f1_score(y, pred, zero_division=0)),
           "mia_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan}
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        for t in fpr_targets:
            out[f"mia_tpr_at_fpr_{int(t*100)}"] = float(np.interp(t, fpr, tpr))
    return out


def mia_by_activity_group(mia_results_df, group_col="activity_group_x",
                          prob_col="membership_probability",
                          truth_col="ground_truth", pred_col="prediction"):
    """Per-activity-group attack strength (RQ3). `attack_advantage` = acc - 0.5;
    > 0 means leakage above random. Compute model-vs-baseline 'privacy gain' by
    differencing this column across two models in the runner."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    rows = []
    for g, sub in mia_results_df.groupby(group_col):
        y, pred, p = sub[truth_col], sub[pred_col], sub[prob_col]
        rows.append({
            "activity_group": g,
            "n": int(len(sub)),
            "acc": float(accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
            "attack_advantage": float(accuracy_score(y, pred) - 0.5),
        })
    return pd.DataFrame(rows)


# =====================================================================
# GROUP SLICING  (RQ3: is utility / leakage evenly distributed?)
# =====================================================================
# Synthetic trips carry no user_id, so we cannot slice the SYNTH side by
# activity group. Instead we slice the REAL side and ask how well the single
# synthetic distribution represents each activity group — the utility analogue
# of the privacy question, and directly relevant to the "outliers get less
# protection" finding you cite.

def attach_activity_group(df, user_summary, user_col="user_id"):
    """Add an `activity_group` column to a trip frame via the user summary."""
    ag = user_summary[[user_col, "activity_group"]].drop_duplicates()
    return df.merge(ag, on=user_col, how="left")


def utility_by_group(real_df, synth_df, user_summary, h3_res=6, user_col="user_id"):
    """Per-activity-group utility: real subset (that group) vs the full synth set.
    Lower JS / higher coverage = synth represents that group well."""
    real_g = attach_activity_group(real_df, user_summary, user_col)
    rows = []
    for g, sub in real_g.groupby("activity_group", observed=True):
        if sub.empty:
            continue
        rec = {"activity_group": str(g), "n_real_trips": int(len(sub))}
        rec.update({k: v for k, v in od_flow_jsd(sub, synth_df, h3_res).items()
                    if k in ("od_flow_js_distance", "od_flow_coverage")})
        try:
            rec.update(temporal_jsd(sub, synth_df))
        except KeyError:
            pass
        rows.append(rec)
    return pd.DataFrame(rows)


def distance_privacy_by_group(synth_df, main_train_df, holdout_df, user_summary,
                              cols=OD_COLS, user_col="user_id", eps=0.012):
    """Per-activity-group distance leakage. DCR ratio uses that group's own
    training vs holdout records, so ratio < 1 flags the group whose records the
    model reproduces most closely."""
    tr = attach_activity_group(main_train_df, user_summary, user_col)
    ho = attach_activity_group(holdout_df, user_summary, user_col)
    rows = []
    for g in tr["activity_group"].dropna().unique():
        tr_g = tr[tr["activity_group"] == g]
        ho_g = ho[ho["activity_group"] == g]
        if tr_g.dropna(subset=cols).empty:
            continue
        rec = {"activity_group": str(g), "n_train_trips": int(len(tr_g))}
        rec.update(dcr_metrics(synth_df, tr_g,
                               ho_g if not ho_g.dropna(subset=cols).empty else None,
                               cols=cols))
        rec.update(duplicate_rate(synth_df, tr_g, cols=cols, eps=eps))
        rec.update(nndr_metrics(synth_df, tr_g, cols=cols))
        rows.append(rec)
    return pd.DataFrame(rows)
