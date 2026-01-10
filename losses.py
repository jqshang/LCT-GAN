from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.stft import STFTConfig, ComplexSTFT, magnitude


@dataclass
class MRSTFTLossConfig:
    fft_sizes: Tuple[int, ...] = (320, 512, 768)
    hop_factors: Tuple[float, ...] = (0.5, 0.5, 0.5)
    mag_weight: float = 1.0
    complex_weight: float = 1.0
    main_fft_size: int = 512
    main_fft_weight: float = 2.0
    default_weight: float = 1.0


class MultiResolutionSTFTLoss(nn.Module):

    def __init__(self, cfg: Optional[MRSTFTLossConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = MRSTFTLossConfig()
        self.cfg = cfg

        self.stfts = nn.ModuleList()
        self.weights: List[float] = []

        for n_fft, hop_factor in zip(cfg.fft_sizes, cfg.hop_factors):
            hop_length = int(round(n_fft * hop_factor))
            win_length = n_fft

            stft_cfg = STFTConfig(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
            ).finalize()
            self.stfts.append(ComplexSTFT(stft_cfg))

            if n_fft == cfg.main_fft_size:
                self.weights.append(cfg.main_fft_weight)
            else:
                self.weights.append(cfg.default_weight)

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if y_hat.dim() != 2 or y.dim() != 2:
            raise ValueError(
                f"Expected y_hat, y of shape [B, T], got {y_hat.shape}, {y.shape}"
            )

        total_loss = 0.0
        mag_total = 0.0
        complex_total = 0.0
        weight_sum = 0.0

        for stft, w, n_fft in zip(self.stfts, self.weights,
                                  self.cfg.fft_sizes):
            Y_hat = stft(y_hat)
            Y = stft(y)

            mag_hat = magnitude(Y_hat)
            mag = magnitude(Y)

            mag_loss = F.mse_loss(mag_hat, mag)

            diff = Y_hat - Y
            complex_loss = (diff.real.pow(2) + diff.imag.pow(2)).mean()

            l_res = (self.cfg.mag_weight * mag_loss +
                     self.cfg.complex_weight * complex_loss)

            total_loss = total_loss + w * l_res
            mag_total = mag_total + w * mag_loss
            complex_total = complex_total + w * complex_loss
            weight_sum += w

        if weight_sum > 0:
            total_loss = total_loss / weight_sum
            mag_total = mag_total / weight_sum
            complex_total = complex_total / weight_sum

        details = {
            "mrstft_total": total_loss.detach(),
            "mrstft_mag": mag_total.detach(),
            "mrstft_complex": complex_total.detach(),
        }
        return total_loss, details


def _flatten_logits_lists(*logits_lists):
    flat: List[torch.Tensor] = []
    for lst in logits_lists:
        flat.extend(list(lst))
    return flat


def discriminator_loss(
    real_logits: Sequence[torch.Tensor],
    fake_logits: Sequence[torch.Tensor],
    loss_type: str = "ls",
):
    if len(real_logits) != len(fake_logits):
        raise ValueError(
            "real_logits and fake_logits must have the same length.")

    loss = 0.0
    for r, f in zip(real_logits, fake_logits):
        if loss_type == "ls":

            loss_real = F.mse_loss(r, torch.ones_like(r))
            loss_fake = F.mse_loss(f, torch.zeros_like(f))
            loss = loss + (loss_real + loss_fake)
        elif loss_type == "hinge":

            loss_real = torch.mean(F.relu(1.0 - r))
            loss_fake = torch.mean(F.relu(1.0 + f))
            loss = loss + (loss_real + loss_fake)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    loss = loss / max(len(real_logits), 1)
    return loss


def generator_adv_loss(fake_logits, loss_type="ls"):
    loss = 0.0
    for f in fake_logits:
        if loss_type == "ls":

            loss = loss + F.mse_loss(f, torch.ones_like(f))
        elif loss_type == "hinge":

            loss = loss - torch.mean(f)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    loss = loss / max(len(fake_logits), 1)
    return loss


def feature_matching_loss(real_fmaps, fake_fmaps):

    if len(real_fmaps) != len(fake_fmaps):
        raise ValueError(
            "real_fmaps and fake_fmaps must have the same outer length.")

    loss = 0.0
    count = 0

    for real_disc_fmaps, fake_disc_fmaps in zip(real_fmaps, fake_fmaps):
        if len(real_disc_fmaps) != len(fake_disc_fmaps):
            raise ValueError(
                "Mismatched feature map list lengths for a discriminator.")
        for r, f in zip(real_disc_fmaps, fake_disc_fmaps):
            loss = loss + F.l1_loss(f, r)
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=real_fmaps[0][0].device)
    return loss / count


def mask_mse_loss(pred_mask_c, target_mask_c):
    if pred_mask_c.shape != target_mask_c.shape:
        raise ValueError(
            f"Shape mismatch: pred_mask_c {pred_mask_c.shape} vs target_mask_c {target_mask_c.shape}"
        )
    return F.mse_loss(pred_mask_c, target_mask_c)
