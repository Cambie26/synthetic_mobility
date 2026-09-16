import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta=0.01):
    """Returns (total, recon, kl). recon and kl are the raw, beta-independent terms."""
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


class VAEModel:
    def __init__(self, input_dim, latent_dim=8, lr=1e-3, beta=0.01, device=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.beta = beta  # target (final) KL weight; annealing ramps up to this
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.history = None

    def _beta_at(self, epoch, warmup):
        """Linear KL annealing: 0 -> self.beta over `warmup` epochs, then held flat."""
        if warmup <= 0:
            return self.beta
        return self.beta * min(1.0, epoch / warmup)

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=100,
        batch_size=64,
        val_size=0.1,
        random_state=42,
        kl_warmup_frac=0.4,
    ):
        """
        kl_warmup_frac: fraction of total epochs over which the KL weight ramps
        linearly from 0 to self.beta. Set to 0.0 to disable annealing (constant beta).

        Returns a history dict of per-epoch lists. recon/kl are logged raw
        (no beta), so they read as a clean reconstruction-vs-KL decomposition;
        loss is the actual optimised objective at that epoch's beta_t.
        """
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        warmup = int(kl_warmup_frac * epochs)

        self.history = {
            k: []
            for k in [
                "epoch", "beta",
                "loss", "recon", "kl",
                "val_loss", "val_recon", "val_kl",
            ]
        }

        pbar = tqdm(range(epochs), desc="Training VAE", position=1, leave=False)

        for epoch in pbar:
            beta_t = self._beta_at(epoch, warmup)

            self.model.train()
            train_loss = train_recon = train_kl = 0.0

            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                x_hat, mu, logvar = self.model(batch)
                loss, recon, kl = vae_loss(batch, x_hat, mu, logvar, beta=beta_t)

                loss.backward()
                self.optimizer.step()

                bs = batch.size(0)
                train_loss += loss.item() * bs
                train_recon += recon.item() * bs
                train_kl += kl.item() * bs

            n_train = len(train_loader.dataset)
            train_loss /= n_train
            train_recon /= n_train
            train_kl /= n_train

            self.history["epoch"].append(epoch)
            self.history["beta"].append(beta_t)
            self.history["loss"].append(train_loss)
            self.history["recon"].append(train_recon)
            self.history["kl"].append(train_kl)

            logs = {
                "beta": f"{beta_t:.4f}",
                "loss": f"{train_loss:.4f}",
                "rec": f"{train_recon:.4f}",
                "kl": f"{train_kl:.4f}",
            }

            if val_loader is not None:
                self.model.eval()
                val_loss = val_recon = val_kl = 0.0

                with torch.no_grad():
                    for (batch,) in val_loader:
                        batch = batch.to(self.device)
                        x_hat, mu, logvar = self.model(batch)
                        loss, recon, kl = vae_loss(
                            batch, x_hat, mu, logvar, beta=beta_t
                        )

                        bs = batch.size(0)
                        val_loss += loss.item() * bs
                        val_recon += recon.item() * bs
                        val_kl += kl.item() * bs

                n_val = len(val_loader.dataset)
                val_loss /= n_val
                val_recon /= n_val
                val_kl /= n_val

                self.history["val_loss"].append(val_loss)
                self.history["val_recon"].append(val_recon)
                self.history["val_kl"].append(val_kl)

                logs.update({
                    "v_loss": f"{val_loss:.4f}",
                    "v_rec": f"{val_recon:.4f}",
                    "v_kl": f"{val_kl:.4f}",
                })

            pbar.set_postfix(logs)

        pbar.close()
        return self.history

    def generate(self, n_samples):
        self.model.eval()

        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            X_synth = self.model.decode(z).cpu().numpy()

        return pd.DataFrame(X_synth)