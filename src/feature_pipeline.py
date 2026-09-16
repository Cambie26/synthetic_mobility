

import h3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# =========================
# Low-level reusable transforms
# =========================

#standardisation
def fit_standardisation(df, cols):
    means = df[cols].mean()
    stds = df[cols].std(ddof=0).replace(0, 1e-8)
    return {"cols": list(cols), "means": means, "stds": stds}

def apply_standardisation(df, params):
    df = df.copy()
    cols = params["cols"]
    df[cols] = (df[cols] - params["means"]) / params["stds"]
    return df

def inverse_standardisation(df, params):
    df = df.copy()
    cols = params["cols"]
    df[cols] = df[cols] * params["stds"] + params["means"]
    return df

# log standardisation
def fit_log_zscore(series):
    x = series.astype(float)
    x_log = np.log1p(x)
    std = max(x_log.std(ddof=0), 1e-8)
    return {
        "mean": float(x_log.mean()),
        "std": float(std),
    }

def apply_log_zscore(series, params):
    x = series.astype(float)
    x_log = np.log1p(x)
    return (x_log - params["mean"]) / params["std"]

def inverse_log_zscore(series_scaled, params):
    x_log = series_scaled * params["std"] + params["mean"]
    return np.expm1(x_log)

# cyclical encoding
def encode_cyclical(values, period, prefix):
    values = np.asarray(values)
    angles = 2 * np.pi * values / period
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(angles),
        f"{prefix}_cos": np.cos(angles),
    })

def decode_cyclical(sin_vals, cos_vals, period):
    angles = np.arctan2(sin_vals, cos_vals)
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)
    return angles * period / (2 * np.pi)


# =========================
# Feature helpers
# =========================

def add_h3_centroid_columns(df, start_hex_col="start_hex", end_hex_col="end_hex"):
    import h3

    df = df.copy()

    df[["start_centroid_lat", "start_centroid_lon"]] = (
        df[start_hex_col].apply(lambda h: pd.Series(h3.cell_to_latlng(h)))
    )

    df[["end_centroid_lat", "end_centroid_lon"]] = (
        df[end_hex_col].apply(lambda h: pd.Series(h3.cell_to_latlng(h)))
    )

    return df


def haversine_distance(lat1, lon1, lat2, lon2, metric="m"):
    """
    Vectorised haversine distance.
    Inputs can be scalars or numpy/pandas arrays.
    metric: 'm' for metres, 'km' for kilometres
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    r = 6371000.0  # metres

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist_m = r * c

    if metric == "m":
        return dist_m
    if metric == "km":
        return dist_m / 1000.0
    raise ValueError("metric must be 'm' or 'km'")


def add_centroid_distance_and_speed(
    df,
    duration_col="duration_minutes",
    start_lat_col="start_centroid_lat",
    start_lon_col="start_centroid_lon",
    end_lat_col="end_centroid_lat",
    end_lon_col="end_centroid_lon",
):
    df = df.copy()

    df["centroid_dist"] = haversine_distance(
        df[start_lat_col].values,
        df[start_lon_col].values,
        df[end_lat_col].values,
        df[end_lon_col].values,
        metric="m"
    )

    duration_hours = df[duration_col].astype(float) / 60.0
    duration_hours = duration_hours.replace(0, 1e-8)

    df["centroid_speed_kmh"] = (df["centroid_dist"] / 1000.0) / duration_hours
    return df


def filter_centroid_speed(df, max_speed_kmh=180, min_speed_kmh=None, speed_col="centroid_speed_kmh"):
    df = df.copy()

    mask = df[speed_col] <= max_speed_kmh
    if min_speed_kmh is not None:
        mask &= df[speed_col] >= min_speed_kmh

    return df[mask].copy()


def check_roundtrip(original_df, recovered_df, cols, atol=1e-8):
    rows = []
    for col in cols:
        abs_err = (original_df[col] - recovered_df[col]).abs()
        rows.append({
            "column": col,
            "mean_abs_error": abs_err.mean(),
            "max_abs_error": abs_err.max(),
            "allclose": np.allclose(original_df[col], recovered_df[col], atol=atol)
        })
    return pd.DataFrame(rows)



# =========================
# High-level feature pipeline
# =========================

@dataclass
class ODFeaturePipeline:
    # raw input columns
    start_lat_col: str = "start_centroid_lat"
    start_lon_col: str = "start_centroid_lon"
    end_lat_col: str = "end_centroid_lat"
    end_lon_col: str = "end_centroid_lon"
    duration_col: str = "duration_minutes"
    distance_col: str = "centroid_dist"
    time_col: str = "start_time"

    # derived column names (also configurable 👇)
    duration_scaled_col: str = "duration_scaled"
    distance_scaled_col: str = "dist_scaled"


    #coord_cols: list = field(default_factory=lambda: [
    #    "start_centroid_lat", "start_centroid_lon",
    #    "end_centroid_lat", "end_centroid_lon",
    #])
    #duration_col: str = "duration_minutes"
    #distance_col: str = "centroid_dist"
    #time_col: str = "start_time"

    coord_params: dict | None = None
    duration_params: dict | None = None
    distance_params: dict | None = None


    @property
    def original_cols(self):
        return [
            self.start_lat_col,
            self.start_lon_col,
            self.end_lat_col,
            self.end_lon_col,
            self.duration_col,
            self.distance_col,
        ]

    @property
    def feature_cols(self):
        return [
            self.start_lat_col,
            self.start_lon_col,
            self.end_lat_col,
            self.end_lon_col,
            self.duration_scaled_col,
            self.distance_scaled_col,
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
        ]

    @property
    def coord_cols(self):
        return [
            self.start_lat_col,
            self.start_lon_col,
            self.end_lat_col,
            self.end_lon_col,
        ]

    def fit(self, df):
        df = df.copy()
        self.coord_params = fit_standardisation(df, self.coord_cols)
        self.duration_params = fit_log_zscore(df[self.duration_col])
        self.distance_params = fit_log_zscore(df[self.distance_col])
        return self

    def transform(self, df):
        if self.coord_params is None:
            raise ValueError("Pipeline has not been fitted yet.")

        df = df.copy()

        # coords
        df = apply_standardisation(df, self.coord_params)

        # duration + distance
        df[self.duration_scaled_col] = apply_log_zscore(
            df[self.duration_col], self.duration_params
        )

        df[self.distance_scaled_col] = apply_log_zscore(
            df[self.distance_col], self.distance_params
        )

        # cyclical time
        df[self.time_col] = pd.to_datetime(df[self.time_col])

        hour_df = encode_cyclical(df[self.time_col].dt.hour, 24, "hour")
        dow_df = encode_cyclical(df[self.time_col].dt.dayofweek, 7, "dow")

        df = pd.concat([df.reset_index(drop=True), hour_df, dow_df], axis=1)

        return df[self.feature_cols].copy()

    def inverse_transform(self, X, original_df=None):
        if self.coord_params is None:
            raise ValueError("Pipeline has not been fitted yet.")

        df = X.copy()

        # coords
        df = inverse_standardisation(df, self.coord_params)

        # duration + distance
        df[self.duration_col] = inverse_log_zscore(
          df[self.duration_scaled_col], self.duration_params)

        df[self.distance_col] = inverse_log_zscore(
            df[self.distance_scaled_col], self.distance_params)

        # cyclical decode
        df["hour"] = decode_cyclical(df["hour_sin"], df["hour_cos"], 24)
        df["day_of_week"] = decode_cyclical(df["dow_sin"], df["dow_cos"], 7)

        # optional IDs
        if original_df is not None:
            for col in ["trip_id", "user_id", "session_id"]:
                if col in original_df.columns:
                    df[col] = original_df[col].values

        # return only original-scale columns
        output_cols = self.original_cols.copy() + ["hour", "day_of_week"] #columns added here 

        if original_df is not None:
            for col in ["trip_id", "user_id", "session_id"]:
                if col in df.columns:
                    output_cols.append(col)

        return df[output_cols].copy()
