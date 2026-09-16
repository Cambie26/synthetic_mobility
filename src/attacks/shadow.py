
import numpy as np
from tqdm.auto import tqdm

from src.models.wrappers import GeneratorWrapper
from src.models.vae import VAEModel
from src.user_split import get_user_trips


def generate_shadow_splits(shadow_users, n_shadow=20, train_frac=0.5, random_state=42):
    """
    Create membership labels for shadow-model training.
    Returns a list of dictionaries:

        shadow_labels[shadow_id][user_id] = 1 if user is in that shadow model's training set,
                                            0 otherwise
    """
    rng = np.random.default_rng(random_state)

    shadow_users = list(shadow_users)

    shadow_labels = []

    for _ in range(n_shadow):

        train_size = int(train_frac * len(shadow_users))

        shadow_in = set(
            rng.choice(shadow_users, size=train_size, replace=False)
        )

        labels = {
            user_id: int(user_id in shadow_in)
            for user_id in shadow_users
        }

        shadow_labels.append(labels)

    return shadow_labels


def verify_shadow_labels(shadow_labels, shadow_users):
    shadow_users = set(shadow_users)

    for synth_id, labels in enumerate(shadow_labels):

        label_users = set(labels.keys())

        assert label_users == shadow_users, f"synth_id {synth_id}: label users mismatch"

        assert set(labels.values()).issubset({0, 1}), f"synth_id {synth_id}: non-binary labels"

        n_in = sum(labels.values())
        n_out = len(labels) - n_in


def train_shadow_generators(
    df,
    shadow_labels,
    # model args (we re-recreate new generator instance for each shadow dataset)
    generator_cls=GeneratorWrapper,
    model_cls=VAEModel,
    model_params=None,
    train_params=None,
    pipeline=None,
    user_col="user_id"
):

    shadow_train_users_log = []
    shadow_synth_datasets = []

    for shadow_id, labels in tqdm(enumerate(shadow_labels),
        total=len(shadow_labels),
        desc="Training shadow models"
    ):

        # get shadow training data
        shadow_in_users = [
            user_id for user_id, label in labels.items()
            if label == 1]

        shadow_train_users_log.append(set(shadow_in_users))

        shadow_train_df = get_user_trips(
            df,
            shadow_in_users,
            user_col=user_col)

        # model training
        generator = generator_cls(
            model_cls,
            model_params,
            train_params,
            pipeline)

        generator.fit(shadow_train_df)

        # synthetic data generation
        shadow_synth_df = generator.generate(len(shadow_train_df))

        shadow_synth_datasets.append(shadow_synth_df)

    return shadow_synth_datasets, shadow_train_users_log


def verify_shadow_training_matches_labels(shadow_labels, shadow_train_users_log):
    for synth_id, labels in enumerate(shadow_labels):

        labelled_in = {
            user_id for user_id, label in labels.items()
            if label == 1
        }

        actual_in = shadow_train_users_log[synth_id]

        assert labelled_in == actual_in, f"Mismatch at synth_id {synth_id}"

    print("✅ Shadow labels match actual shadow training users")
