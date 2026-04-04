import math
import torch
import torch.nn as nn

from .losses import compute_fa_loss


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)


def _sigmoid_beta_schedule(timesteps: int) -> torch.Tensor:
    x = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(x)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    betas = betas * (0.02 - 1e-4) + 1e-4
    return betas.clamp(1e-4, 0.999)


def get_beta_schedule(name: str, timesteps: int) -> torch.Tensor:
    name = name.lower()
    if name == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if name == "cosine":
        return _cosine_beta_schedule(timesteps)
    if name == "sigmoid":
        return _sigmoid_beta_schedule(timesteps)
    raise ValueError(f"Unknown beta schedule: {name}")


def compute_per_feature_alpha_bar(
    betas_global: torch.Tensor,
    feature_alpha: torch.Tensor,
    feature_gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-feature cumulative alpha_bar for forward diffusion.
    """
    T = len(betas_global)
    d_c = len(feature_alpha)
    betas_global_exp = betas_global.unsqueeze(1).expand(T, d_c)
    scale = torch.exp(feature_alpha).unsqueeze(0)
    shift = feature_gamma.unsqueeze(0)
    betas_pf = (betas_global_exp * scale + shift).clamp(1e-5, 0.999)
    alphas_pf = 1.0 - betas_pf
    alpha_bar_pf = torch.cumprod(alphas_pf, dim=0)
    return alpha_bar_pf


class GaussianDiffusion:
    """
    Minimal DDPM implementation for tabular data.
    """

    def __init__(
        self,
        model,
        timesteps=500,
        device="cpu",
        schedule: str = "linear",
        beta_scale: float = 1.0,
        continuous_idx: list[int] | None = None,
        categorical_idx: list[int] | None = None,
        categorical_cardinalities: list[int] | None = None,
        feature_weights_cont: torch.Tensor | None = None,
        feature_weights_cat: torch.Tensor | None = None,
        lambda_cat: float = 0.5,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
        feature_alpha: torch.Tensor | None = None,
        feature_gamma: torch.Tensor | None = None,
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.lambda_cat = float(lambda_cat)
        self.continuous_idx = list(continuous_idx or [])
        self.categorical_idx = list(categorical_idx or [])
        self.categorical_cardinalities = list(categorical_cardinalities or [])
        if len(self.categorical_idx) != len(self.categorical_cardinalities):
            self.categorical_idx = []
            self.categorical_cardinalities = []

        self.feature_weights_cont = (
            feature_weights_cont.to(device) if feature_weights_cont is not None else None
        )
        self.feature_weights_cat = (
            feature_weights_cat.to(device) if feature_weights_cat is not None else None
        )
        self.x_mean = x_mean.to(device) if x_mean is not None else None
        self.x_std = x_std.to(device) if x_std is not None else None

        betas = get_beta_schedule(schedule, timesteps).to(device)
        betas = torch.clamp(betas * beta_scale, min=1e-5, max=0.999)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.feature_alpha = (
            feature_alpha.to(device=device, dtype=torch.float32)
            if feature_alpha is not None
            else None
        )
        self.feature_gamma = (
            feature_gamma.to(device=device, dtype=torch.float32)
            if feature_gamma is not None
            else None
        )
        self.alpha_bar_pf = None
        if (
            self.feature_alpha is not None
            and self.feature_gamma is not None
            and len(self.continuous_idx) > 0
            and self.feature_alpha.numel() == len(self.continuous_idx)
            and self.feature_gamma.numel() == len(self.continuous_idx)
        ):
            self.alpha_bar_pf = compute_per_feature_alpha_bar(
                betas,
                self.feature_alpha,
                self.feature_gamma,
            )

    def _extract(self, a, t, shape):
        """
        Extracts coefficients a[t] for each sample in a batch.
        """
        batch_size = t.shape[0]
        out = a.gather(0, t).reshape(batch_size, 1)
        return out.expand(shape)

    def q_sample(self, x_start, t, noise):
        """
        q(x_t | x_0)
        """
        sqrt_ac = torch.sqrt(self.alphas_cumprod)
        sqrt_om = torch.sqrt(1 - self.alphas_cumprod)

        sqrt_ac_t = self._extract(sqrt_ac, t, x_start.shape)
        sqrt_om_t = self._extract(sqrt_om, t, x_start.shape)

        x_noisy = sqrt_ac_t * x_start + sqrt_om_t * noise

        if self.alpha_bar_pf is not None and self.continuous_idx:
            cont_idx = torch.as_tensor(
                self.continuous_idx,
                device=self.device,
                dtype=torch.long,
            )
            x_cont = x_start.index_select(1, cont_idx)
            noise_cont = noise.index_select(1, cont_idx)
            alpha_bar_t_pf = self.alpha_bar_pf.index_select(0, t)
            x_cont_noisy = (
                torch.sqrt(alpha_bar_t_pf) * x_cont
                + torch.sqrt(1.0 - alpha_bar_t_pf) * noise_cont
            )
            x_noisy = x_noisy.clone()
            x_noisy.index_copy_(1, cont_idx, x_cont_noisy)

        return x_noisy

    def training_loss(self, x_start):
        """
        Feature-Aggregated loss with per-feature weighting.
        """
        B = x_start.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        sq_err = (predicted_noise - noise) ** 2

        if not self.continuous_idx and not self.categorical_idx:
            return nn.MSELoss()(predicted_noise, noise)

        noise_pred_cont = None
        noise_cont = None
        if self.continuous_idx:
            cont_idx = torch.as_tensor(self.continuous_idx, device=self.device, dtype=torch.long)
            noise_pred_cont = predicted_noise.index_select(1, cont_idx)
            noise_cont = noise.index_select(1, cont_idx)

        logits_cat: list[torch.Tensor] = []
        targets_cat: list[torch.Tensor] = []
        if self.categorical_idx:
            sqrt_ac = torch.sqrt(self.alphas_cumprod)
            sqrt_om = torch.sqrt(1 - self.alphas_cumprod)
            sqrt_ac_t = self._extract(sqrt_ac, t, x_start.shape)
            sqrt_om_t = self._extract(sqrt_om, t, x_start.shape)
            x0_pred = (x_noisy - sqrt_om_t * predicted_noise) / (sqrt_ac_t + 1e-8)

            if self.x_mean is not None and self.x_std is not None:
                x0_pred_raw = x0_pred * self.x_std.unsqueeze(0) + self.x_mean.unsqueeze(0)
                x0_true_raw = x_start * self.x_std.unsqueeze(0) + self.x_mean.unsqueeze(0)
            else:
                x0_pred_raw = x0_pred
                x0_true_raw = x_start

            for j, (feat_idx, cardinality) in enumerate(
                zip(self.categorical_idx, self.categorical_cardinalities)
            ):
                if cardinality <= 1:
                    continue
                class_vals = torch.arange(
                    cardinality, device=self.device, dtype=x0_pred_raw.dtype
                ).unsqueeze(0)
                pred_col = x0_pred_raw[:, feat_idx].unsqueeze(1)
                logits_j = -((pred_col - class_vals) ** 2)
                target_j = (
                    x0_true_raw[:, feat_idx]
                    .round()
                    .long()
                    .clamp(min=0, max=cardinality - 1)
                )
                logits_cat.append(logits_j)
                targets_cat.append(target_j)

        loss = compute_fa_loss(
            noise_pred_cont=noise_pred_cont,
            noise_cont=noise_cont,
            logits=logits_cat,
            targets_cat=targets_cat,
            feature_weights_cont=self.feature_weights_cont,
            feature_weights_cat=self.feature_weights_cat,
            lambda_cat=self.lambda_cat,
        )
        if torch.isnan(loss):
            return sq_err.mean()
        return loss

    def sample(self, shape):
        """
        Reverse diffusion sampling.
        """
        x = torch.randn(shape).to(self.device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t], device=self.device)
            predicted_noise = self.model(x, t_batch)
            beta = self.betas[t]

            x = (x - beta * predicted_noise) / torch.sqrt(1 - beta)

        return x
