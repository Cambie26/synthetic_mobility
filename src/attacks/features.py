
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import h3

def extract_mia_features(
    synthetic_df,
    target_user_trips,

    # column headers
    start_lat_col="start_centroid_lat",
    start_lon_col="start_centroid_lon",
    end_lat_col="end_centroid_lat",
    end_lon_col="end_centroid_lon",
    duration_col="duration_minutes",
    distance_col="centroid_dist",

    h3_res=6 # snaps coords to known grid
):
    """
    Extract MIA features for one target user.

    Compares the target user's real trips against the synthetic dataset.
    Lower OD distances / higher overlap may suggest memorisation.
    """


    synth = synthetic_df.copy()
    target = target_user_trips.copy()

    od_cols = [start_lat_col, start_lon_col, end_lat_col, end_lon_col]

    # Keep only usable rows
    synth = synth.dropna(subset=od_cols)
    target = target.dropna(subset=od_cols)

    if len(synth) == 0 or len(target) == 0:
        return {
            "min_od_distance": np.nan,
            "mean_od_distance": np.nan,
            "duration_similarity": np.nan,
            "distance_similarity": np.nan,
            "od_overlap": np.nan,
        }

    # --- Nearest-neighbour OD distance: target trips -> synthetic trips ---
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(synth[od_cols].values)

    od_distances, _ = nn.kneighbors(target[od_cols].values)
    od_distances = od_distances[:, 0]

    # --- Similarity of user-level averages ---
    def safe_similarity(real_mean, synth_mean):
        if pd.isna(real_mean) or pd.isna(synth_mean):
            return np.nan

        denom = abs(real_mean) + 1e-8
        return 1 / (1 + abs(real_mean - synth_mean) / denom)

    if duration_col in synth.columns and duration_col in target.columns:
        duration_similarity = safe_similarity(
            target[duration_col].mean(),
            synth[duration_col].mean()
        )
    else:
        duration_similarity = np.nan

    if distance_col in synth.columns and distance_col in target.columns:
        distance_similarity = safe_similarity(
            target[distance_col].mean(),
            synth[distance_col].mean()
        )
    else:
        distance_similarity = np.nan

    # --- H3 overlap ---

    def make_h3_od_set(data):
        data = data.dropna(subset=od_cols)

        od_pairs = set()
        for _, row in data.iterrows():
            start_cell = h3.latlng_to_cell(
                row[start_lat_col],
                row[start_lon_col],
                h3_res)
            end_cell = h3.latlng_to_cell(
                row[end_lat_col],
                row[end_lon_col],
                h3_res)
            od_pairs.add((start_cell, end_cell))
        return od_pairs

    synth_od = make_h3_od_set(synth)
    target_od = make_h3_od_set(target)

    if len(target_od) == 0:
        od_overlap = np.nan
    else:
        od_overlap = len(target_od.intersection(synth_od)) / len(target_od)

    return { # od distance is the Euclidean distance between coord pairs
        "min_od_distance": od_distances.min(),
        "p01_od_distance": np.percentile(od_distances, 1),
        "p05_od_distance": np.percentile(od_distances, 5),
        "p25_od_distance": np.percentile(od_distances, 25),
        "median_od_distance": np.median(od_distances),
        "mean_od_distance": od_distances.mean(),
        "std_od_distance": od_distances.std(),
        "close_trip_rate": np.mean(od_distances < 0.025),
        "very_close_trip_rate": np.mean(od_distances < 0.012),

        "duration_similarity": duration_similarity,  # comparison of means
        "distance_similarity": distance_similarity, # comparison of means
        "od_overlap": od_overlap, # percentage of same coordinate pairs

        "n_target_trips": len(target), # how much evidence stats are based on
        }
