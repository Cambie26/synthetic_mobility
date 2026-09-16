import numpy as np
import pandas as pd

class BootstrapModel:
    def __init__(self, input_dim, random_state=42, replace=True, noise_std=0.0, device=None):
        self.input_dim = input_dim
        self.random_state = random_state
        self.replace = replace
        self.noise_std = noise_std
        self.device = device

        self.X_train = None
        self.rng = np.random.default_rng(random_state)

        # dummy attribute so save_generator does not break
        self.model = None

    def fit(self, train_loader, val_loader=None, **kwargs):
        batches = []

        for (batch,) in train_loader:
            batches.append(batch.cpu().numpy())

        self.X_train = np.vstack(batches)

        if self.X_train.shape[1] != self.input_dim:
            raise ValueError(
                f"input_dim={self.input_dim}, but training data has "
                f"{self.X_train.shape[1]} columns"
            )

    def generate(self, n_samples):
        if self.X_train is None:
            raise ValueError("BootstrapModel must be fitted before calling generate().")

        idx = self.rng.choice(
            len(self.X_train),
            size=n_samples,
            replace=self.replace
        )

        X_synth = self.X_train[idx].copy()

        if self.noise_std > 0:
            X_synth += self.rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=X_synth.shape
            )

        return pd.DataFrame(X_synth)


bootstrap_params = {
    "random_state": 42,
    "replace": True,
    "noise_std": 0.0
}

bootstrap_train_params = {
    "batch_size": 64
}
