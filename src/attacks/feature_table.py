
import pandas as pd

from src.attacks.features import extract_mia_features


def build_feature_table(users, synthetic_datasets, ground_truth_labels):
    """
    Create an attack feature table for the MIA attack model.

    users : dict
        {user_id: user_trips_df}

    synthetic_datasets : pd.DataFrame or list[pd.DataFrame]

    ground_truth_labels : dict or list[dict]
        If training with shadow datasets:
            list of {user_id: label}
        If evaluating one main synthetic dataset:
            {user_id: label}
    """

    if not isinstance(synthetic_datasets, list):
        synthetic_datasets = [synthetic_datasets]

    if not isinstance(ground_truth_labels, list):
        ground_truth_labels = [ground_truth_labels]

    if len(synthetic_datasets) != len(ground_truth_labels):
        raise ValueError(
            "ground_truth_labels must be same length as synthetic_datasets"
        )

    rows = []

    for synth_id, synth_df in enumerate(synthetic_datasets):

        dataset_labels = ground_truth_labels[synth_id]

        for user_id, user_trips in users.items():

            if user_id not in dataset_labels:
                raise KeyError(
                    f"user_id {user_id} missing from labels for synth_id {synth_id}"
                )

            feats = extract_mia_features(
                synthetic_df=synth_df,
                target_user_trips=user_trips
            )

            row = {
                "user_id": user_id,
                "synth_id": synth_id,
                "ground_truth": dataset_labels[user_id],
                **feats
            }

            rows.append(row)

    return pd.DataFrame(rows)
