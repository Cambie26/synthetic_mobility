
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SPAIN_TZ = ZoneInfo("Europe/Madrid")
EXPERIMENT_LOG_PATH = "outputs/experiment_log_default.json"


def load_experiment_log(path=EXPERIMENT_LOG_PATH):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_experiment_log(log, path=EXPERIMENT_LOG_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


def log_metrics(
    seed,
    model,
    metrics,
    path=EXPERIMENT_LOG_PATH,
    overwrite=False,
    notes=None,
):
    """
    Add or update metrics for one experiment run, identified by seed + model.
    """

    log = load_experiment_log(path)

    # find existing run
    matching_rows = [
        row for row in log
        if row["seed"] == seed and row["model"] == model
    ]

    if len(matching_rows) > 1:
        print(f"⚠️ Warning: found multiple rows for seed={seed}, model={model}")

    if matching_rows:
        row = matching_rows[0]
        print(f"Updating existing log row: seed={seed}, model={model}")
    else:
        row = {
            "seed": seed,
            "model": model,
            "created_at": datetime.now(SPAIN_TZ).isoformat(),
            "metrics": {},
            "notes": [],
        }
        log.append(row)
        print(f"Created new log row: seed={seed}, model={model}")

    row["updated_at"] = datetime.now(SPAIN_TZ).isoformat()

    if notes is not None:
        row.setdefault("notes", [])
        row["notes"].append({
            "timestamp": datetime.now(SPAIN_TZ).isoformat(),
            "text": notes,
        })

    for metric_name, metric_value in metrics.items():

        metric_exists = metric_name in row["metrics"]

        if metric_exists and not overwrite:
            old_value = row["metrics"][metric_name]

            print(
                f"⚠️ Metric already exists for seed={seed}, model={model}: "
                f"{metric_name}={old_value}. "
                f"Not overwriting. Use overwrite=True to replace."
            )

            continue

        if metric_exists and overwrite:
            old_value = row["metrics"][metric_name]

            print(
                f"⚠️ Overwriting metric for seed={seed}, model={model}: "
                f"{metric_name}: {old_value} -> {metric_value}"
            )

        row["metrics"][metric_name] = metric_value

    save_experiment_log(log, path)

    return #row


def experiment_log_to_df(rows):
    flat_rows = []
    for row in rows:
        flat = {
            #"timestamp": row["timestamp"],
            "seed": row["seed"],
            "model": row["model"],
        }
        flat.update(row.get("metrics", {}))
        flat_rows.append(flat)
    return pd.DataFrame(flat_rows)



def summarise_across_runs(results_df, group_col="model"):
    metric_cols = [
        col for col in results_df.columns
        if col not in [group_col, "seed"]
        and pd.api.types.is_numeric_dtype(results_df[col])
    ]
    rows = []
    for metric in metric_cols:
        row = {"metric": metric}

        for model, gdf in results_df.groupby(group_col):
            mean = gdf[metric].mean()
            std = gdf[metric].std()
            row[f"{model}_mean"] = mean
            row[f"{model}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def make_pretty_summary(summary_df, metrics=None):

    """
    Convert summary dataframe into:
        metric | model1 | model2 | ...

    with values formatted as:
        mean ± std
    """

    if metrics is not None:
        summary_df = summary_df[
            summary_df["metric"].isin(metrics)
        ]

    model_names = sorted({
        col.replace("_mean", "")
        for col in summary_df.columns
        if col.endswith("_mean")
    })

    rows = []
    for _, row in summary_df.iterrows():

        out = {"metric": row["metric"]}

        for model in model_names:

            mean_col = f"{model}_mean"
            std_col = f"{model}_std"
            mean = row.get(mean_col, np.nan)
            std = row.get(std_col, np.nan)

            if pd.isna(mean):
                value = "-"
            elif pd.isna(std):
                value = f"{mean:.3f}"
            else:
                value = f"{mean:.3f} ± {std:.3f}"

            out[model] = value

        rows.append(out)

    return pd.DataFrame(rows)


#log_metrics(seed=41, model="VAE", metrics={
#        "attack_auc": 0.26,
#        "attack_accuracy": 0.42
#        })

#experiment_log_df = experiment_log_to_df(load_experiment_log())
#experiment_summary_df = summarise_across_runs(experiment_log_df)
#make_pretty_summary(experiment_summary_df)
