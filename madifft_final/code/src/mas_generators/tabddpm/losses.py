from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_auto_lambda_cat(n_cont: int, n_cat: int) -> float:
    """
    Compute dataset-adaptive lambda_cat based on feature type ratio.

    Formula: 0.3 + 0.5 * (n_cat / (n_cat + n_cont))
    Clamped to [0.25, 0.85].
    """
    total = int(n_cont) + int(n_cat)
    if total == 0:
        return 0.5
    ratio = float(n_cat) / float(total)
    lam = 0.3 + 0.5 * ratio
    return float(max(0.25, min(0.85, lam)))


def compute_adaptive_epochs(
    n_cont: int,
    n_cat: int,
    base_epochs: int = 50,
) -> int:
    """
    Double per-agent training budget for predominantly categorical datasets.
    """
    total = int(n_cont) + int(n_cat)
    if total == 0:
        return int(base_epochs)
    ratio = float(n_cat) / float(total)
    if ratio > 0.5:
        return int(base_epochs) * 2
    return int(base_epochs)


def compute_feature_weights(
    X_train_cont: np.ndarray | None,
    X_train_cat: np.ndarray | None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute per-feature inverse-std weights for continuous features
    and inverse-entropy weights for categorical features from D_train.
    """
    if X_train_cont is None or X_train_cont.size == 0:
        weights_cont = np.array([], dtype=np.float32)
    else:
        Xc = np.asarray(X_train_cont, dtype=float)
        stds = Xc.std(axis=0) + eps
        weights_cont = 1.0 / stds
        weights_cont = weights_cont / max(weights_cont.mean(), eps)
        weights_cont = np.clip(weights_cont, 0.5, 2.0)
        weights_cont = weights_cont / weights_cont.mean()

    if X_train_cat is None or X_train_cat.size == 0:
        weights_cat = np.array([], dtype=np.float32)
    else:
        Xd = np.asarray(X_train_cat)
        weights_cat_list: list[float] = []
        for j in range(Xd.shape[1]):
            col = Xd[:, j]
            _, counts = np.unique(col, return_counts=True)
            probs = counts / max(counts.sum(), 1)
            entropy_j = -(probs * np.log(probs + eps)).sum()
            weights_cat_list.append(1.0 / (entropy_j + eps))
        weights_cat = np.asarray(weights_cat_list, dtype=np.float32)
        if weights_cat.size:
            weights_cat = weights_cat / max(weights_cat.mean(), eps)
            weights_cat = np.clip(weights_cat, 0.5, 2.0)
            weights_cat = weights_cat / weights_cat.mean()

    return (
        torch.tensor(weights_cont, dtype=torch.float32),
        torch.tensor(weights_cat, dtype=torch.float32),
    )


def compute_snr_sampling_probs(
    betas: np.ndarray,
    low_quantile: float = 0.05,
    high_quantile: float = 0.95,
) -> np.ndarray:
    """
    Compute timestep sampling probabilities that up-weight the
    informative mid-SNR region (SNR near 1.0) where the denoising
    task is hardest and gradients are most informative.
    """
    betas = np.clip(np.array(betas, dtype=np.float64), 1e-6, 0.999)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)

    snr_low = np.quantile(snr, low_quantile)
    snr_high = np.quantile(snr, high_quantile)
    in_band = (snr >= snr_low) & (snr <= snr_high)

    weights = np.where(
        in_band,
        1.0 / (np.sqrt(snr) + 1e-4),
        0.01,
    )

    probs = weights / weights.sum()
    return probs.astype(np.float32)


def sample_timesteps_adaptive(
    batch_size: int,
    snr_probs_tensor: torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample timesteps according to SNR importance weights and return
    inverse-probability correction factors for unbiased gradients.
    """
    T = int(snr_probs_tensor.numel())
    t = torch.multinomial(
        snr_probs_tensor,
        num_samples=int(batch_size),
        replacement=True,
    ).to(device)
    p_t = snr_probs_tensor.index_select(0, t).to(device)
    correction = 1.0 / (p_t * T + 1e-8)
    correction = correction / correction.mean()
    return t, correction


def compute_fa_loss(
    noise_pred_cont: torch.Tensor | None,
    noise_cont: torch.Tensor | None,
    logits: list[torch.Tensor] | None,
    targets_cat: list[torch.Tensor] | None,
    feature_weights_cont: torch.Tensor | None,
    feature_weights_cat: torch.Tensor | None,
    lambda_cat: float,
    reduce: bool = True,
) -> torch.Tensor:
    """
    Feature-Aggregated Loss with per-feature weighting.

    Continuous branch: weighted MSE.
    Categorical branch: weighted per-feature cross-entropy.
    """
    device = None
    if noise_pred_cont is not None:
        device = noise_pred_cont.device
    elif logits:
        device = logits[0].device
    else:
        raise ValueError("compute_fa_loss requires at least one non-empty branch.")

    batch_size = 0
    if noise_pred_cont is not None:
        batch_size = int(noise_pred_cont.shape[0])
    elif logits:
        batch_size = int(logits[0].shape[0])

    loss_cont = torch.zeros(batch_size, device=device)
    if noise_pred_cont is not None and noise_cont is not None and noise_pred_cont.numel() > 0:
        sq_err = (noise_pred_cont - noise_cont) ** 2
        if feature_weights_cont is not None and feature_weights_cont.numel() == sq_err.shape[1]:
            weighted_sq_err = sq_err * feature_weights_cont.unsqueeze(0)
        else:
            weighted_sq_err = sq_err
        loss_cont = weighted_sq_err.mean(dim=1)

    loss_cat = torch.zeros(batch_size, device=device)
    if logits and targets_cat:
        losses = []
        for j, (logits_j, targets_j) in enumerate(zip(logits, targets_cat)):
            ce_j = F.cross_entropy(logits_j, targets_j, reduction="none")
            if feature_weights_cat is not None and feature_weights_cat.numel() > j:
                ce_j = feature_weights_cat[j] * ce_j
            losses.append(ce_j)
        if losses:
            loss_cat = torch.stack(losses, dim=1).mean(dim=1)

    per_sample_loss = loss_cont + float(lambda_cat) * loss_cat
    if reduce:
        return per_sample_loss.mean()
    return per_sample_loss
