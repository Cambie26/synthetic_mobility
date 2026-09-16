
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# --- single source of truth (top of src/user_split.py) ---
ACTIVITY_BINS     = [-np.inf, 9, 20, 63, np.inf]
ACTIVITY_LABELS   = ["<10", "10–20", "21–63", "64+"]
MIA_EXCLUDE_LABEL = "<10"
ACTIVITY_DISPLAY  = {"10–20": "Low (10–20)", "21–63": "Medium (21–63)", "64+": "High (64+)"}


def load_trip_data(path):
    """Load OD trip dataset from CSV."""
    return pd.read_csv(path)


def filter_trip_data(
    df,
    duration_col="duration_minutes",
    distance_col="centroid_dist",
    speed_col="centroid_speed_kmh",
    min_duration=3,
    max_duration=360,
    min_distance=200,
    max_distance=100000,
    min_speed=1,
    max_speed=150,
    reset_index=True,
    verbose=True,
):
    """Basic trip plausibility filtering."""

    filtered = df.copy()
    before_n = len(filtered)

    filtered = filtered[
        (filtered[duration_col] >= min_duration)
        & (filtered[duration_col] <= max_duration)
    ]

    filtered = filtered[
        (filtered[distance_col] >= min_distance)
        & (filtered[distance_col] <= max_distance)
    ]

    filtered = filtered[
        (filtered[speed_col] >= min_speed)
        & (filtered[speed_col] <= max_speed)
    ]

    after_n = len(filtered)

    if verbose:
        print("Removed rows:", before_n - after_n)
        print("Percent removed:", round(100 * (before_n - after_n) / before_n, 2))

    if reset_index:
        filtered = filtered.reset_index(drop=True)

    return filtered


def load_and_filter_trip_data(path, **filter_kwargs):
    """Load trip data from CSV, then apply standard filtering."""

    df = load_trip_data(path)
    df = filter_trip_data(df, **filter_kwargs)

    return df




def build_user_summary(
    trips_df,
    user_col="user_id",
    duration_col="duration_minutes",
    distance_col=None,
    test_size=0.2,
    shadow_size=0.3,
    random_state=42,
    bins=None,
    labels=None,
    exclude_label=None,
):
    """
    Build user-level summary and assign activity groups/splits.

    bins / labels / exclude_label fall back to the module constants when None.
    The None sentinel (rather than bins=ACTIVITY_BINS in the signature) means
    they resolve at *call* time, so re-running the constants cell in a notebook
    actually takes effect.
    """
    bins          = ACTIVITY_BINS     if bins is None          else bins
    labels        = ACTIVITY_LABELS   if labels is None        else labels
    exclude_label = MIA_EXCLUDE_LABEL if exclude_label is None else exclude_label

    # fail loudly if the constants drifted out of sync
    if len(labels) != len(bins) - 1:
        raise ValueError(f"{len(labels)} labels for {len(bins) - 1} bins")
    if exclude_label not in labels:
        raise ValueError(f"exclude_label {exclude_label!r} not in labels {labels}")

    agg_dict = {"n_trips": (user_col, "size")}
    if duration_col is not None and duration_col in trips_df.columns:
        agg_dict["avg_duration_minutes"] = (duration_col, "mean")
    if distance_col is not None and distance_col in trips_df.columns:
        agg_dict["avg_distance"] = (distance_col, "mean")

    user_summary = trips_df.groupby(user_col).agg(**agg_dict).reset_index()

    user_summary["activity_group"] = pd.cut(
        user_summary["n_trips"], bins=bins, labels=labels, right=True,
    )

    user_summary["mia_eligible"] = user_summary["activity_group"] != exclude_label
    user_summary["split"] = "low_activity_main_train"

    eligible = user_summary[user_summary["mia_eligible"]].copy()

    group_counts = eligible["activity_group"].value_counts()
    group_counts = group_counts[group_counts.index != exclude_label]
    rare_groups  = group_counts[group_counts < 2]

    if len(rare_groups) > 0:
        print("Warning: some activity groups have fewer than 2 users.")
        print("Stratified split may be unstable for:", rare_groups.to_dict())
        stratify_col = None
    else:
        stratify_col = eligible["activity_group"]

    train_users, eval_out_users = train_test_split(
        eligible, test_size=test_size, random_state=random_state,
        stratify=stratify_col,
    )
    shadow_users, eval_in_users = train_test_split(
        train_users, test_size=1 - shadow_size, random_state=random_state,
        stratify=train_users["activity_group"],
    )

    user_summary.loc[user_summary[user_col].isin(shadow_users[user_col]),   "split"] = "shadow"
    user_summary.loc[user_summary[user_col].isin(eval_in_users[user_col]),  "split"] = "eval_in"
    user_summary.loc[user_summary[user_col].isin(eval_out_users[user_col]), "split"] = "eval_out"

    print(user_summary["activity_group"].value_counts().sort_index())
    return user_summary


def get_users_by_split(user_summary, split_name, user_col="user_id"):
    return user_summary.loc[
        user_summary["split"] == split_name,
        user_col,
    ].tolist()


def get_main_training_users(user_summary, user_col="user_id"):
    """
    Main generator trains on:
    - low activity users
    - shadow users
    - eval_in users

    It excludes eval_out users.
    """

    return user_summary.loc[
        user_summary["split"].isin([
            "low_activity_main_train",
            "shadow",
            "eval_in",
        ]),
        user_col,
    ].tolist()


def get_user_trips(trips_df, user_ids, user_col="user_id"):
    if not isinstance(user_ids, (list, tuple, set, pd.Series, np.ndarray)):
        user_ids = [user_ids]

    return trips_df[trips_df[user_col].isin(user_ids)].copy()
