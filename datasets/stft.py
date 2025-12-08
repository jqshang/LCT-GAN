# stft.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class STFTConfig:
    """
    Configuration for STFT/iSTFT.

    This is intentionally generic so you can reuse it for:
      - the main TF representation used by the generator, and
      - multi-resolution STFTs used in the loss (with different n_fft, etc.).
    """
    n_fft: int = 512
    hop_length: Optional[int] = None  # default: n_fft // 2 if None
    win_length: Optional[int] = None  # default: n_fft if None
    window: str = "hann"
    center: bool = True
    pad_mode: str = "reflect"
    normalized: bool = False
    onesided: bool = True

    def finalize(self) -> "STFTConfig":
        """Fill in hop_length/win_length defaults if they are None."""
        if self.hop_length is None:
            self.hop_length = self.n_fft // 2
        if self.win_length is None:
            self.win_length = self.n_fft
        return self


class ComplexSTFT(nn.Module):
    """
    Differentiable STFT/iSTFT module using torch.stft / torch.istft.

    - Input:  waveform [B, T]
    - STFT:   complex tensor [B, F, T_frames] (torch.complex64 by default)
    - iSTFT:  waveform [B, T] (length specified or derived by torch.istft)

    The window is stored as a buffer so it moves with .to(device).
    """

    def __init__(self, cfg: STFTConfig):
        super().__init__()
        self.cfg = cfg.finalize()

        if self.cfg.window.lower() != "hann":
            raise ValueError("Only 'hann' window is currently supported.")

        # Window is registered as a buffer; we will cast to the right dtype/device in forward.
        window = torch.hann_window(self.cfg.win_length)
        self.register_buffer("window", window)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute complex STFT.

        Args:
            waveform: [B, T] float tensor

        Returns:
            stft: complex tensor [B, F, T_frames]
        """
        if waveform.dim() != 2:
            raise ValueError(
                f"Expected waveform of shape [B, T], got {waveform.shape}")

        window = self.window.to(device=waveform.device, dtype=waveform.dtype)

        stft = torch.stft(
            waveform,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=self.cfg.center,
            pad_mode=self.cfg.pad_mode,
            normalized=self.cfg.normalized,
            onesided=self.cfg.onesided,
            return_complex=True,
        )
        # stft: [B, F, T_frames], complex
        return stft

    def istft(
        self,
        stft_matrix: torch.Tensor,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inverse STFT.

        Args:
            stft_matrix: complex tensor [B, F, T_frames]
            length: optional target length for the reconstructed waveform.

        Returns:
            waveform: [B, T]
        """
        if not torch.is_complex(stft_matrix):
            raise ValueError("stft_matrix must be a complex tensor.")

        if stft_matrix.dim() != 3:
            raise ValueError(
                f"Expected stft_matrix of shape [B, F, T], got {stft_matrix.shape}"
            )

        # torch.istft expects [F, T, 2] real if return_complex=False; but we pass complex and return_complex=True in stft
        # so here we call torch.istft with complex input.
        window = self.window.to(
            device=stft_matrix.device,
            dtype=stft_matrix.real.dtype,  # real float dtype
        )

        waveform = torch.istft(
            stft_matrix,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            center=self.cfg.center,
            normalized=self.cfg.normalized,
            onesided=self.cfg.onesided,
            length=length,
        )
        # waveform: [B, T]
        return waveform


# ====== Magnitude / compression helpers ======


def magnitude(
    stft_matrix: torch.Tensor,
    power: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute magnitude or power spectrogram from complex STFT.

    Args:
        stft_matrix: complex tensor [..., F, T]
        power: 1.0 for magnitude, 2.0 for power, etc.
        eps: numerical stability

    Returns:
        mag: real tensor [..., F, T]
    """
    if not torch.is_complex(stft_matrix):
        raise ValueError("stft_matrix must be a complex tensor.")

    mag = stft_matrix.abs().clamp_min(eps)
    if power != 1.0:
        mag = mag**power
    return mag


def compress(x: torch.Tensor,
             c: float = 0.3,
             eps: float = 1e-12) -> torch.Tensor:
    """
    Apply magnitude compression: x_c = (max(x, eps)) ** c
    """
    return x.clamp_min(eps)**c


def decompress(x_c: torch.Tensor,
               c: float = 0.3,
               eps: float = 1e-12) -> torch.Tensor:
    """
    Undo magnitude compression: x = (max(x_c, eps)) ** (1 / c)
    """
    return x_c.clamp_min(eps)**(1.0 / c)


# ====== IRM computation & mask utilities ======


def compute_compressed_irm(
    clean_stft: torch.Tensor,
    noisy_stft: torch.Tensor,
    c: float = 0.3,
    gamma: float = 1e-12,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the compressed Ideal Ratio Mask (IRM^c) in the magnitude domain:

        IRM^c(k,l) = |S(k,l)|^c / (|X(k,l)|^c + gamma)

    where S is the clean STFT and X is the noisy STFT.

    Args:
        clean_stft: complex tensor [B, F, T]
        noisy_stft: complex tensor [B, F, T]
        c: compression exponent (e.g., 0.3 in the paper)
        gamma: small constant to avoid division by zero
        eps: small constant for magnitude floor

    Returns:
        irm_c: real tensor [B, F, T] with values in [0, 1] (approximately)
    """
    if not (torch.is_complex(clean_stft) and torch.is_complex(noisy_stft)):
        raise ValueError("clean_stft and noisy_stft must be complex tensors.")

    clean_mag = clean_stft.abs().clamp_min(eps)
    noisy_mag = noisy_stft.abs().clamp_min(eps)

    clean_mag_c = clean_mag**c
    noisy_mag_c = noisy_mag**c

    irm_c = clean_mag_c / (noisy_mag_c + gamma)
    return irm_c


def decompress_mask(
    mask_c: torch.Tensor,
    c: float = 0.3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Decompress a compressed mask (IRM^c or network output in compressed domain)
    to the linear domain.

        mask_lin = (max(mask_c, eps)) ** (1 / c)

    Args:
        mask_c: compressed mask [B, F, T] or [B, 1, F, T]
        c: compression exponent
        eps: numerical stability

    Returns:
        mask_lin: same shape, values ideally in [0, 1]
    """
    return decompress(mask_c, c=c, eps=eps)


def apply_mask(
    noisy_stft: torch.Tensor,
    mask: torch.Tensor,
    compressed: bool = False,
    c: float = 0.3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Apply a (possibly compressed) real-valued TF mask to a noisy STFT.

    If compressed=True, 'mask' is assumed to be in the compressed domain (IRM^c),
    so we first decompress it to the linear domain.

    Args:
        noisy_stft: complex tensor [B, F, T]
        mask: real tensor [B, F, T] or [B, 1, F, T]
        compressed: whether 'mask' is compressed (IRM^c) or already linear
        c: compression exponent (only used if compressed=True)
        eps: numerical stability

    Returns:
        enhanced_stft: complex tensor [B, F, T]
    """
    if not torch.is_complex(noisy_stft):
        raise ValueError("noisy_stft must be a complex tensor.")

    # Ensure mask broadcast shape matches [B, F, T]
    if mask.dim() == 4:
        # [B, 1, F, T] -> [B, F, T]
        if mask.size(1) != 1:
            raise ValueError(
                f"Expected mask shape [B, 1, F, T], got {mask.shape}")
        mask = mask[:, 0, :, :]

    if mask.dim() != 3:
        raise ValueError(
            f"Expected mask shape [B, F, T] (or [B, 1, F, T]), got {mask.shape}"
        )

    if compressed:
        mask = decompress_mask(mask, c=c, eps=eps)

    # Clamp mask to non-negative range; typical training uses sigmoid so this is mostly safety.
    mask = mask.clamp_min(0.0)

    # Broadcast to complex dtype and apply
    enhanced_stft = noisy_stft * mask.to(noisy_stft.dtype)
    return enhanced_stft


def make_lct_stft(
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> ComplexSTFT:
    """
    Convenience function to create the STFT module used by the generator
    (the main 512-point STFT in the paper, with 50% overlap by default).
    """
    cfg = STFTConfig(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    ).finalize()
    return ComplexSTFT(cfg)
