from typing import Literal

import lightning as L
import torch
from torch import nn, optim

from scientalab import utils
from scientalab.module import mlp


class inVAE(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        invariant_dim: int,
        spurious_dim: int,
        hidden_dims: list[int] = [128],
        invariant_latent_dim: int = 30,
        spurious_latent_dim: int = 5,
        beta: float = 1,
        n_warmup_epochs: int = 50,
        learning_rate: float = 1e-3,
        scheduler: Literal["ReduceOnPlateau", "CosineAnnealing"] = "CosineAnnealing",
    ) -> None:
        super().__init__()

        if latent_dim != spurious_latent_dim + invariant_latent_dim:
            raise ValueError(
                f"latent dimension {latent_dim} does not match invariant latent dimension"
                f"{invariant_latent_dim} and spurious latent dimension {spurious_latent_dim}"
            )
        if input_dim != output_dim + invariant_dim + spurious_dim:
            raise ValueError(
                f"input dimension {input_dim} does not match invariant dimension "
                f"{invariant_dim}, spurious dimension {spurious_dim} and "
                f"output dimension {output_dim}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # encoder
        self.encoder = mlp.MLP(
            in_dim=input_dim,
            out_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
        )
        self.encoder_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_var = nn.Linear(hidden_dims[-1], latent_dim)

        # decoder
        self.decoder = mlp.MLP(
            in_dim=latent_dim,
            out_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
        )
        self.decoder_mean = nn.Linear(hidden_dims[-1], output_dim)
        self.decoder_var = nn.Linear(hidden_dims[-1], output_dim)

        # prior
        self.invariant_dim = invariant_dim
        self.spurious_dim = spurious_dim
        self.prior_invariant = mlp.MLP(
            in_dim=self.invariant_dim,
            out_dim=invariant_latent_dim,
            hidden_dims=hidden_dims,
        )
        self.prior_spurious = mlp.MLP(
            in_dim=self.spurious_dim,
            out_dim=spurious_latent_dim,
            hidden_dims=hidden_dims,
        )
        self.prior_invariant_mean = torch.zeros(invariant_latent_dim)
        self.prior_spurious_mean = torch.zeros(spurious_latent_dim)

        self.invariant_latent_dim = invariant_latent_dim
        self.spurious_latent_dim = spurious_latent_dim
        self.beta = beta

        self.lr = learning_rate
        self.scheduler = scheduler
        self.beta = beta
        self.current_beta = 0
        self.n_warmup_epochs = n_warmup_epochs

        self.save_hyperparameters()

    def forward(
        self,
        batch: dict[str, torch.Tensor | list[str | float]],
    ) -> tuple[torch.Tensor, ...]:
        invariant_var = self.prior_invariant(
            batch["x"][:, -self.invariant_dim - self.spurious_dim : -self.spurious_dim]
        )
        spurious_var = self.prior_spurious(batch["x"][:, -self.spurious_dim :])

        enc = self.encoder(batch["x"])
        enc_mu, enc_var = self.encoder_mean(enc), self.encoder_var(enc)

        latent_sample = utils.reparametrize(mean=enc_mu, variance=enc_var)

        dec = self.decoder(latent_sample)
        dec_mu, dec_var = self.decoder_mean(dec), self.decoder_var(dec)
        return invariant_var, spurious_var, enc_mu, enc_var, dec_mu, dec_var, latent_sample

    def training_step(
        self,
        batch: dict[str, torch.Tensor | list[str | float]],
    ) -> torch.Tensor:
        invariant_var, spurious_var, enc_mu, enc_var, dec_mu, dec_var, latent = self(batch)

        recon = (
            utils.gaussian_likelihood(
                x=batch["x"][:, : self.output_dim], mu=dec_mu, std=torch.exp(0.5 * dec_var) + 1e-10
            )
            .sum(-1)
            .mean()
        )
        invariant_latent, spurious_latent = (
            latent[:, : self.invariant_latent_dim],
            latent[:, self.invariant_latent_dim :],
        )

        kl = (
            utils.gaussian_likelihood(
                x=latent,
                mu=enc_mu,
                std=torch.exp(0.5 * enc_var) + 1e-10,
            ).sum(-1)
            - utils.gaussian_likelihood(
                x=invariant_latent,
                mu=self.prior_invariant_mean.to(invariant_latent.device),
                std=torch.exp(0.5 * invariant_var) + 1e-10,
            ).sum(-1)
            - utils.gaussian_likelihood(
                x=spurious_latent,
                mu=self.prior_spurious_mean.to(spurious_latent.device),
                std=torch.exp(0.5 * spurious_var) + 1e-10,
            ).sum(-1)
        ).mean()

        elbo_loss = -(recon - self.current_beta * kl)

        self.log("train_recon", recon, on_step=True, on_epoch=False)
        self.log("train_kl", kl, on_step=True, on_epoch=False)
        self.log("train_loss", elbo_loss, on_step=True, on_epoch=False)

        return elbo_loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor | list[str | float]],
    ) -> None:
        invariant_var, spurious_var, enc_mu, enc_var, dec_mu, dec_var, latent = self(batch)
        recon = (
            utils.gaussian_likelihood(
                x=batch["x"][:, : self.output_dim], mu=dec_mu, std=torch.exp(0.5 * dec_var) + 1e-10
            )
            .sum(-1)
            .mean()
        )
        invariant_latent, spurious_latent = (
            latent[:, : self.invariant_latent_dim],
            latent[:, self.invariant_latent_dim :],
        )

        kl = (
            utils.gaussian_likelihood(
                x=latent,
                mu=enc_mu,
                std=torch.exp(0.5 * enc_var) + 1e-10,
            ).sum(-1)
            - utils.gaussian_likelihood(
                x=invariant_latent,
                mu=self.prior_invariant_mean.to(invariant_latent.device),
                std=torch.exp(0.5 * invariant_var) + 1e-10,
            ).sum(-1)
            - utils.gaussian_likelihood(
                x=spurious_latent,
                mu=self.prior_spurious_mean.to(spurious_latent.device),
                std=torch.exp(0.5 * spurious_var) + 1e-10,
            ).sum(-1)
        ).mean()

        elbo_loss = -(recon - self.current_beta * kl)

        self.log("val_recon", recon, on_step=False, on_epoch=True)
        self.log("val_kl", kl, on_step=False, on_epoch=True)
        self.log("val_loss", elbo_loss, on_step=False, on_epoch=True)

    def test_step(
        self,
        batch: dict[str, torch.Tensor | list[str | float]],
    ) -> None:
        invariant_var, spurious_var, enc_mu, enc_var, dec_mu, dec_var, latent = self(batch)
        recon = (
            utils.gaussian_likelihood(
                x=batch["x"][:, : self.output_dim], mu=dec_mu, std=torch.exp(0.5 * dec_var) + 1e-10
            )
            .sum(-1)
            .mean()
        )
        invariant_latent, spurious_latent = (
            latent[:, : self.invariant_latent_dim],
            latent[:, self.invariant_latent_dim :],
        )

        kl = (
            utils.gaussian_likelihood(
                x=latent,
                mu=enc_mu,
                std=torch.exp(0.5 * enc_var) + 1e-10,
            ).sum(-1)
            - utils.gaussian_likelihood(
                x=invariant_latent,
                mu=self.prior_invariant_mean.to(invariant_latent.device),
                std=torch.exp(0.5 * invariant_var) + 1e-10,
            ).sum(-1)
            - utils.gaussian_likelihood(
                x=spurious_latent,
                mu=self.prior_spurious_mean.to(spurious_latent.device),
                std=torch.exp(0.5 * spurious_var) + 1e-10,
            ).sum(-1)
        ).mean()

        elbo_loss = -(recon - self.current_beta * kl)

        self.log("test_recon", recon, on_step=False, on_epoch=True)
        self.log("test_kl", kl, on_step=False, on_epoch=True)
        self.log("test_loss", elbo_loss, on_step=False, on_epoch=True)

    def predict_step(
        self,
        batch: dict[str, torch.Tensor | list[str | float]],
    ) -> dict[str, torch.Tensor | str]:
        _, _, _, _, dec_mu, dec_var, latent = self(batch)
        sample_decoder = utils.reparametrize(mean=dec_mu, variance=dec_var)
        prediction = {
            "input": batch["x"][:, : self.output_dim].squeeze().cpu().numpy().tolist(),
            "latent": latent.squeeze().cpu().numpy().tolist(),
            "latent_invariant": latent.squeeze().cpu().numpy().tolist()[: self.invariant_latent_dim],
            "latent_spurious": latent.squeeze().cpu().numpy().tolist()[self.invariant_latent_dim :],
            "dec_mu": dec_mu.squeeze().cpu().numpy().tolist(),
            "dec_var": dec_var.squeeze().cpu().numpy().tolist(),
            "output": sample_decoder.squeeze().cpu().numpy().tolist(),
        }
        prediction.update({label: batch[label][0] for label in batch.keys() if label != "x"})
        return prediction

    def on_train_epoch_start(self) -> None:
        self.current_beta = min(self.current_epoch / self.n_warmup_epochs, self.beta)
        self.log("beta", self.current_beta)

    def configure_optimizers(
        self,
    ) -> dict[str, optim.Optimizer | optim.lr_scheduler.ReduceLROnPlateau | optim.lr_scheduler.CosineAnnealingLR | str]:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler == "ReduceOnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)
        elif self.scheduler == "CosineAnnealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs if self.trainer.max_epochs is not None else 100
            )
        else:
            raise ValueError(f"Scheduler: {self.scheduler} not implemented")

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
