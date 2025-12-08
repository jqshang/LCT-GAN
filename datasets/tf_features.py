from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from datasets.stft import (
    STFTConfig,
    ComplexSTFT,
    make_lct_stft,
    magnitude,
    compress,
    compute_compressed_irm,
)


@dataclass
class TFFeaturesConfig:
    """
    Configuration for TF feature & target computation.

    - n_fft, hop_length, win_length define the main STFT used by the generator.
    - c is the magnitude compression exponent (0.3 in the paper).
    - compress_input controls whether the input magnitude to the generator
      should also be compressed (optional; you can keep it False and use
      linear magnitude as input).
    """
    n_fft: int = 512
    hop_length: Optional[int] = None
    win_length: Optional[int] = None

    c: float = 0.3
    compress_input: bool = False  # if True, produce noisy_mag_c as primary feature

    # whether to keep complex STFTs in the output dict (useful if you want
    # to do STFT-domain losses directly on features; can be disabled to save memory)
    return_stfts: bool = True


class TFFeatures(nn.Module):
    """
    Compute TF-domain features and IRM targets for LCT-GAN training.

    Forward:
        noisy_wave: [B, T]
        clean_wave: [B, T]

    Returns a dict with (at least):
        "noisy_mag"  : [B, F, T_frames]   (input magnitude feature, linear or compressed)
        "irm_c"      : [B, F, T_frames]   (compressed IRM target, IRM^c)

    And, if return_stfts=True:
        "noisy_stft" : complex [B, F, T_frames]
        "clean_stft" : complex [B, F, T_frames]

    Additionally:
        "noisy_mag_c": [B, F, T_frames] if you want compressed mag explicitly.
    """

    def __init__(self, cfg: Optional[TFFeaturesConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = TFFeaturesConfig()
        self.cfg = cfg

        if (cfg.n_fft == 512 and cfg.hop_length is None
                and cfg.win_length is None):
            # Use helper with sensible defaults (50% overlap, Hann window)
            self.stft = make_lct_stft(n_fft=cfg.n_fft)
        else:
            stft_cfg = STFTConfig(
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                window="hann",
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
            ).finalize()
            self.stft = ComplexSTFT(stft_cfg)

        self.c = cfg.c

    def forward(
        self,
        noisy_wave: torch.Tensor,
        clean_wave: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            noisy_wave: [B, T] float
            clean_wave: [B, T] float (aligned with noisy_wave)

        Returns:
            feats: dict with keys described in the class docstring.
        """
        if noisy_wave.dim() != 2 or clean_wave.dim() != 2:
            raise ValueError(
                f"Expected noisy_wave and clean_wave of shape [B, T], "
                f"got {noisy_wave.shape}, {clean_wave.shape}")
        if noisy_wave.shape != clean_wave.shape:
            raise ValueError(
                f"noisy_wave and clean_wave must have same shape, "
                f"got {noisy_wave.shape} vs {clean_wave.shape}")

        # 1) STFTs (complex)
        noisy_stft = self.stft(noisy_wave)  # [B, F, T_frames]
        clean_stft = self.stft(clean_wave)  # [B, F, T_frames]

        # 2) Magnitudes (linear)
        noisy_mag = magnitude(noisy_stft)  # [B, F, T_frames]

        # 3) Compressed IRM target (IRM^c)
        irm_c = compute_compressed_irm(
            clean_stft=clean_stft,
            noisy_stft=noisy_stft,
            c=self.c,
        )  # [B, F, T_frames]

        # 4) Optional compression of input magnitude
        #    - Some setups feed compressed magnitude to the generator.
        noisy_mag_c = compress(noisy_mag, c=self.c)

        # Decide which mag to expose as the primary feature
        if self.cfg.compress_input:
            mag_for_input = noisy_mag_c
        else:
            mag_for_input = noisy_mag

        feats: Dict[str, torch.Tensor] = {
            # Primary input feature for the generator (you will unsqueeze a channel dim in training)
            "noisy_mag": mag_for_input,  # [B, F, T_frames]

            # IRM target (compressed domain)
            "irm_c": irm_c,  # [B, F, T_frames]

            # Always include this for potential auxiliary losses
            "noisy_mag_c": noisy_mag_c,  # [B, F, T_frames]
        }

        if self.cfg.return_stfts:
            feats["noisy_stft"] = noisy_stft  # complex [B, F, T_frames]
            feats["clean_stft"] = clean_stft  # complex [B, F, T_frames]

        return feats
