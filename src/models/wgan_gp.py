

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class WGANGenerator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_dim=128):
        super().__init__()

        # BatchNorm on the hidden layers. The generator is NOT under the gradient
        # penalty, so BatchNorm is fine here and helps it express diverse modes.
        # No norm on the output layer.
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class WGANCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        # LayerNorm (NOT BatchNorm) on the hidden layers. The gradient penalty is a
        # per-sample constraint; BatchNorm would couple samples across the batch and
        # invalidate it. LayerNorm normalises within each sample, so GP stays valid.
        # No norm on the output layer.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class WGANModel:
    def __init__(
        self,
        input_dim,
        noise_dim=64,
        hidden_dim=128,
        lr=1e-4,
        betas=(0.5, 0.9),
        lambda_gp=10.0,
        critic_steps=5,
        device=None,
    ):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.betas = betas
        self.lambda_gp = lambda_gp
        self.critic_steps = critic_steps

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.generator = WGANGenerator(
            noise_dim=noise_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = WGANCritic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas,
        )
        self.c_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=betas,
        )

        # for compatibility with save_generator()
        self.model = self.generator
        self.history = None

    def _gradient_penalty(self, real_batch, fake_batch):
        batch_size = real_batch.size(0)

        epsilon = torch.rand(batch_size, 1, device=self.device)
        interpolated = epsilon * real_batch + (1 - epsilon) * fake_batch
        interpolated.requires_grad_(True)

        interpolated_scores = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=100,
        batch_size=64,
        val_size=0.1,
        random_state=42,
    ):
        # seed parity with the VAE; pass the experiment seed in via run_seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.history = {"epoch": [], "c_loss": [], "g_loss": []}

        pbar = tqdm(range(epochs), desc="Training WGAN-GP", position=1, leave=False)

        for epoch in pbar:
            # BatchNorm in the generator needs train mode during training;
            # generate() switches it back to eval.
            self.generator.train()
            self.critic.train()

            critic_losses = []
            generator_losses = []

            for (real_batch,) in train_loader:
                real_batch = real_batch.to(self.device)
                current_batch_size = real_batch.size(0)

                # BatchNorm can't compute batch stats over a single sample;
                # skip a stray size-1 final batch rather than crash.
                if current_batch_size < 2:
                    continue

                # ----- train critic -----
                for _ in range(self.critic_steps):
                    z = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                    fake_batch = self.generator(z).detach()

                    real_score = self.critic(real_batch).mean()
                    fake_score = self.critic(fake_batch).mean()
                    gp = self._gradient_penalty(real_batch, fake_batch)
                    critic_loss = fake_score - real_score + self.lambda_gp * gp

                    self.c_optimizer.zero_grad()
                    critic_loss.backward()
                    self.c_optimizer.step()
                    critic_losses.append(critic_loss.item())

                # ----- train generator -----
                z = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                fake_batch = self.generator(z)
                generator_loss = -self.critic(fake_batch).mean()

                self.g_optimizer.zero_grad()
                generator_loss.backward()
                self.g_optimizer.step()
                generator_losses.append(generator_loss.item())

            avg_c = sum(critic_losses) / max(len(critic_losses), 1)
            avg_g = sum(generator_losses) / max(len(generator_losses), 1)

            self.history["epoch"].append(epoch)
            self.history["c_loss"].append(avg_c)
            self.history["g_loss"].append(avg_g)

            pbar.set_postfix({"c_loss": f"{avg_c:.4f}", "g_loss": f"{avg_g:.4f}"})

        pbar.close()
        return self.history

    def generate(self, n_samples):
        self.generator.eval()  # BatchNorm uses running stats, works for any n_samples

        with torch.no_grad():
            z = torch.randn(n_samples, self.noise_dim, device=self.device)
            X_synth = self.generator(z).cpu().numpy()

        return pd.DataFrame(X_synth)