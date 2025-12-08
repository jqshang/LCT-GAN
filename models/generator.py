# models/generator.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.stft import (
    STFTConfig,
    ComplexSTFT,
    make_lct_stft,
    magnitude,
    apply_mask,
)


class GroupedGRU(nn.Module):
    """
    Grouped GRU along the last (channel) dimension.

    Idea:
      - Split the feature dimension into `num_groups` smaller chunks.
      - Run *one* GRU on each chunk, sharing GRU parameters across groups
        (by flattening groups into the batch dimension).
      - This reduces parameter count vs a full GRU over all channels.

    Expected input shape: [B, L, C]
      - B: batch (can be B*T or B*F)
      - L: sequence length (freq or time)
      - C: channels (must be divisible by num_groups)

    Output shape: [B, L, C] (same as input).
    """

    def __init__(self, channels: int, num_groups: int = 4):
        super().__init__()
        assert channels % num_groups == 0, \
            f"channels ({channels}) must be divisible by num_groups ({num_groups})"
        self.channels = channels
        self.num_groups = num_groups
        self.group_dim = channels // num_groups

        # Single GRU shared across all groups
        self.gru = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]

        Returns:
            out: [B, L, C]
        """
        B, L, C = x.shape
        G = self.num_groups
        D = self.group_dim

        # [B, L, C] -> [B, L, G, D] -> [B, G, L, D] -> [B*G, L, D]
        x = x.view(B, L, G, D).permute(0, 2, 1, 3).contiguous()
        x = x.view(B * G, L, D)

        out, _ = self.gru(x)  # [B*G, L, D]

        # Back to [B, L, C]
        out = out.view(B, G, L, D).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, L, C)
        return out


class FreqTransformerBlock(nn.Module):
    """
    Frequency transformer block (F):

    - Operates along the frequency axis for each time frame independently.
    - Uses:
        Grouped GRU (across freq bins)
        Multi-Head Attention (across freq bins)
        Residual + LayerNorm around each.

    Input / Output shape: [B, C, T, F]
      - B: batch
      - C: channels
      - T: time frames
      - F: frequency bins
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        gru_groups: int = 4,
    ):
        super().__init__()
        self.channels = channels

        self.norm1 = nn.LayerNorm(channels)
        self.gru = GroupedGRU(channels, num_groups=gru_groups)

        self.norm2 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,  # [batch, seq, embed]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F]

        Returns:
            out: [B, C, T, F]
        """
        B, C, T, F = x.shape

        # Treat each time frame independently: sequence over frequency
        # [B, C, T, F] -> [B, T, F, C]
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        # Flatten time into batch
        seq = x_perm.view(B * T, F, C)  # [B*T, F, C]

        # GRU block with residual
        seq_res = seq
        seq_norm = self.norm1(seq)  # LayerNorm over last dim (C)
        seq_gru = self.gru(seq_norm)  # [B*T, F, C]
        seq = seq_res + seq_gru

        # MHA block with residual
        seq_res2 = seq
        seq_norm2 = self.norm2(seq)
        # MHA expects (batch, seq, embed) due to batch_first=True
        attn_out, _ = self.mha(seq_norm2, seq_norm2,
                               seq_norm2)  # self-attention
        seq = seq_res2 + attn_out

        # Back to [B, C, T, F]
        seq = seq.view(B, T, F, C)
        out = seq.permute(0, 3, 1, 2).contiguous()
        return out


class TimeTransformerBlock(nn.Module):
    """
    Time transformer block (T):

    - Operates along the time axis for each frequency bin independently.
    - Uses:
        Grouped GRU (across time frames)
        Causal Multi-Head Attention (with limited past context)
        Residual + LayerNorm around each.

    Input / Output shape: [B, C, T, F]

    Causality:
      - GRU is naturally causal (processes sequence in order).
      - MHA uses a causal + truncated attention mask so each time step
        can only attend to past (and optionally limited history).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        gru_groups: int = 4,
        max_context: Optional[int] = None,
    ):
        """
        Args:
            channels: feature dimension.
            num_heads: MHA heads.
            gru_groups: how many channel groups for GroupedGRU.
            max_context: if not None, each time step attends to at most
                         `max_context` previous frames (including itself).
                         This approximates the 1-second window in the paper
                         and makes complexity independent of utterance length.
        """
        super().__init__()
        self.channels = channels
        self.max_context = max_context

        self.norm1 = nn.LayerNorm(channels)
        self.gru = GroupedGRU(channels, num_groups=gru_groups)

        self.norm2 = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def _build_causal_mask(self, L: int, device, dtype) -> torch.Tensor:
        """
        Build a (L, L) attention mask for causal (and optionally truncated)
        self-attention along time:

            mask[i, j] = 0      if j is allowed to be attended to from i
                          -inf  otherwise

        Allowed positions:
            j <= i (no look-ahead)
            and, if max_context is not None, i - j <= max_context
        """
        # diff[i, j] = i - j
        idx = torch.arange(L, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # [L, L]

        # Allowed: diff >= 0 (j <= i)
        allowed = diff >= 0
        if self.max_context is not None:
            allowed = allowed & (diff <= self.max_context)

        mask = torch.zeros((L, L), device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, F]

        Returns:
            out: [B, C, T, F]
        """
        B, C, T, F = x.shape

        # Treat each frequency bin independently: sequence over time
        # [B, C, T, F] -> [B, F, T, C]
        x_perm = x.permute(0, 3, 2, 1).contiguous()
        # Flatten freq into batch
        seq = x_perm.view(B * F, T, C)  # [B*F, T, C]

        # GRU block
        seq_res = seq
        seq_norm = self.norm1(seq)
        seq_gru = self.gru(seq_norm)  # [B*F, T, C]
        seq = seq_res + seq_gru

        # MHA block with causal mask
        seq_res2 = seq
        seq_norm2 = self.norm2(
            seq_norm if False else seq)  # ensure LN after GRU+res

        L = T
        attn_mask = self._build_causal_mask(
            L=L,
            device=seq.device,
            dtype=seq.dtype,
        )  # [L, L]

        attn_out, _ = self.mha(
            seq_norm2,  # query
            seq_norm2,  # key
            seq_norm2,  # value
            attn_mask=attn_mask,
        )
        seq = seq_res2 + attn_out

        # Back to [B, C, T, F]
        seq = seq.view(B, F, T, C)
        out = seq.permute(0, 3, 2, 1).contiguous()
        return out


class FTFBottleneck(nn.Module):
    """
    Frequency-Time-Frequency (FTF) bottleneck:

        x -> FreqTransformerBlock -> TimeTransformerBlock -> FreqTransformerBlock

    Input / Output shape: [B, C, T, F]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        gru_groups: int = 4,
        max_time_context: Optional[int] = None,
    ):
        super().__init__()
        self.freq1 = FreqTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            gru_groups=gru_groups,
        )
        self.time = TimeTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            gru_groups=gru_groups,
            max_context=max_time_context,
        )
        self.freq2 = FreqTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            gru_groups=gru_groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq1(x)
        x = self.time(x)
        x = self.freq2(x)
        return x


# =========================
#  U-Net encoder / decoder
# =========================


class DownBlock(nn.Module):
    """
    One encoder (downsampling) block:

    - 2D conv over (time, freq) but with kernel_size=(1, 3):
        → mixes across frequency only (time-local, trivially causal).
    - stride = (1, 2): downsample frequency by 2, keep time resolution.
    - LeakyReLU activation.

    Input / Output shape: [B, C_in, T, F] -> [B, C_out, T, F_out]
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int] = (1, 3),
            stride: Tuple[int, int] = (1, 2),
            negative_slope: float = 0.03,
    ):
        super().__init__()
        k_t, k_f = kernel_size
        assert k_t == 1, "We keep k_t=1 so conv is time-local."

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 1),  # pad only along freq, keep time as-is
        )
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x


class UpBlock(nn.Module):
    """
    One decoder (upsampling) block:

    - Nearest-neighbor upsample along frequency by factor 2.
    - Concatenate encoder skip features.
    - 2D conv with kernel_size=(1, 3) and stride=1 to mix channels.

    Input:
        x:    [B, C_in, T, F_in]
        skip: [B, C_skip, T_skip, F_skip]

    Output:
        out: [B, C_out, T_out, F_out]
    """

    def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int] = (1, 3),
            negative_slope: float = 0.03,
    ):
        super().__init__()
        k_t, k_f = kernel_size
        assert k_t == 1, "We keep k_t=1 so conv is time-local."

        self.upsample = nn.Upsample(scale_factor=(1, 2),
                                    mode="nearest")  # upsample freq by 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(0, 1),  # keep time, 'same' along freq
        )
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    @staticmethod
    def _align(x: torch.Tensor,
               skip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align time and freq dimensions of x and skip by simple cropping to the minimum size.
        """
        _, _, T_x, F_x = x.shape
        _, _, T_s, F_s = skip.shape

        T_min = min(T_x, T_s)
        F_min = min(F_x, F_s)

        if T_x != T_min:
            x = x[:, :, :T_min, :]
        if F_x != F_min:
            x = x[:, :, :, :F_min]
        if T_s != T_min:
            skip = skip[:, :, :T_min, :]
        if F_s != F_min:
            skip = skip[:, :, :, :F_min]

        return x, skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x, skip = self._align(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    """
    Simple 3-layer encoder built from DownBlocks.
    """

    def __init__(
            self,
            in_channels: int = 1,
            channels: Tuple[int, int, int] = (16, 32, 64),
    ):
        super().__init__()
        c1, c2, c3 = channels

        self.blocks = nn.ModuleList([
            DownBlock(in_channels, c1),
            DownBlock(c1, c2),
            DownBlock(c2, c3),
        ])

    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return x, skips  # bottleneck, list of skips (low→high freq)


class Decoder(nn.Module):
    """
    Simple 3-layer decoder built from UpBlocks, mirroring Encoder.
    """

    def __init__(
            self,
            out_channels: int = 1,
            channels: Tuple[int, int, int] = (64, 32, 16),
    ):
        super().__init__()
        c3, c2, c1 = channels  # bottleneck, then up

        # UpBlocks: (in_channels, skip_channels, out_channels)
        self.up3 = UpBlock(in_channels=c3, skip_channels=c3, out_channels=c2)
        self.up2 = UpBlock(in_channels=c2, skip_channels=c2, out_channels=c1)
        self.up1 = UpBlock(in_channels=c1,
                           skip_channels=c1,
                           out_channels=out_channels)

    def forward(self, x: torch.Tensor,
                skips: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: bottleneck features from encoder. [B, C3, T, F3]
            skips: list of encoder outputs [s1, s2, s3] with increasing depth.

        Returns:
            out: [B, out_channels, T, F_final]
        """
        # Skips are in order [after_down1, after_down2, after_down3]
        s1, s2, s3 = skips  # shallowest to deepest

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return x


# =========================
#  LCT Generator & wrapper
# =========================


@dataclass
class LCTGeneratorConfig:
    """
    Config for the LCTGenerator.
    """
    in_channels: int = 1
    out_channels: int = 1  # mask channel
    enc_channels: Tuple[int, int, int] = (16, 32, 64)
    dec_channels: Tuple[int, int, int] = (64, 32, 16)
    num_heads: int = 4
    gru_groups: int = 4
    max_time_context: Optional[int] = None
    output_activation: str = "sigmoid"  # 'sigmoid' or 'none'


class LCTGenerator(nn.Module):
    """
    Core LCT-GAN generator (without STFT front-end):

    Input:
        noisy_mag: [B, 1, F, T]  (magnitude or compressed magnitude)

    Output:
        mask_c: [B, 1, F, T]     (compressed mask, values in [0,1] if sigmoid)
    """

    def __init__(self, cfg: LCTGeneratorConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(
            in_channels=cfg.in_channels,
            channels=cfg.enc_channels,
        )
        bottleneck_channels = cfg.enc_channels[-1]

        self.ftf = FTFBottleneck(
            channels=bottleneck_channels,
            num_heads=cfg.num_heads,
            gru_groups=cfg.gru_groups,
            max_time_context=cfg.max_time_context,
        )

        self.decoder = Decoder(
            out_channels=cfg.out_channels,
            channels=cfg.dec_channels,
        )

    def forward(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_mag: [B, 1, F, T]

        Returns:
            mask_c: [B, 1, F, T]
        """
        if noisy_mag.dim() != 4:
            raise ValueError(
                f"Expected noisy_mag [B, 1, F, T], got {noisy_mag.shape}")
        B, C, F, T = noisy_mag.shape
        assert C == 1, "Generator currently expects a single-channel magnitude input."

        # Internal layout: [B, C, T, F] for conv convenience
        x = noisy_mag.permute(0, 1, 3, 2).contiguous()  # [B, 1, T, F]
        T_orig, F_orig = x.shape[2], x.shape[3]

        # Encoder
        bottleneck, skips = self.encoder(x)

        # FTF bottleneck
        bottleneck = self.ftf(bottleneck)

        # Decoder
        out = self.decoder(bottleneck, skips)  # [B, 1, T_dec, F_dec]

        # Align back to original TF size (cropping if necessary)
        B2, C2, T_dec, F_dec = out.shape
        assert B2 == B and C2 == self.cfg.out_channels

        # Crop time and freq to original size if bigger
        if T_dec > T_orig:
            out = out[:, :, :T_orig, :]
        if F_dec > F_orig:
            out = out[:, :, :, :F_orig]

        # If smaller (rare), we can pad with zeros
        if T_dec < T_orig or F_dec < F_orig:
            out_padded = torch.zeros(
                B,
                C2,
                T_orig,
                F_orig,
                device=out.device,
                dtype=out.dtype,
            )
            out_padded[:, :, :T_dec, :F_dec] = out
            out = out_padded

        # [B, 1, T, F] -> [B, 1, F, T]
        out = out.permute(0, 1, 3, 2).contiguous()

        # Optional output non-linearity (e.g., sigmoid to keep mask in [0,1])
        if self.cfg.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.cfg.output_activation == "none":
            pass
        else:
            raise ValueError(
                f"Unknown output_activation: {self.cfg.output_activation}")

        return out  # [B, 1, F, T]


class LCTEnhancer(nn.Module):
    """
    End-to-end enhancer wrapper:

    - Takes noisy waveform(s) as input.
    - STFT -> magnitude -> LCTGenerator -> compressed mask.
    - Apply mask (decompressing if needed) to noisy STFT.
    - iSTFT -> enhanced waveform(s).

    This uses the helpers from datasets.stft.py and keeps the generator itself
    independent of the STFT front-end and IRM target definition.
    """

    def __init__(
        self,
        gen_cfg: Optional[LCTGeneratorConfig] = None,
        stft_cfg: Optional[STFTConfig] = None,
        c: float = 0.3,
    ):
        """
        Args:
            gen_cfg: LCTGeneratorConfig (defaults to a reasonable config).
            stft_cfg: STFTConfig for the main STFT (defaults to 512-pt with 50% overlap).
            c: compression exponent used for IRM and mask (0.3 in the paper).
        """
        super().__init__()

        if gen_cfg is None:
            gen_cfg = LCTGeneratorConfig()
        self.gen = LCTGenerator(gen_cfg)

        if stft_cfg is None:
            self.stft = make_lct_stft(n_fft=512)
        else:
            self.stft = ComplexSTFT(stft_cfg)

        self.c = c

    def forward(
        self,
        noisy_wave: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_wave: [B, T] waveform tensor.

        Returns:
            enhanced_wave: [B, T] enhanced waveform.
            mask_c: [B, 1, F, T_frames] compressed mask predicted by generator.
        """
        if noisy_wave.dim() != 2:
            raise ValueError(
                f"Expected noisy_wave [B, T], got {noisy_wave.shape}")

        # STFT of noisy
        noisy_stft = self.stft(noisy_wave)  # [B, F, T_frames]
        noisy_mag = magnitude(noisy_stft)  # [B, F, T_frames]
        noisy_mag = noisy_mag.unsqueeze(1)  # [B, 1, F, T_frames]

        # Generator predicts compressed mask
        mask_c = self.gen(noisy_mag)  # [B, 1, F, T_frames]

        # Apply mask in compressed domain (stft.apply_mask will decompress)
        enhanced_stft = apply_mask(
            noisy_stft,
            mask_c,
            compressed=True,
            c=self.c,
        )  # [B, F, T_frames]

        # iSTFT back to waveform (match input length)
        enhanced_wave = self.stft.istft(
            enhanced_stft,
            length=noisy_wave.shape[-1],
        )  # [B, T]

        return enhanced_wave, mask_c
