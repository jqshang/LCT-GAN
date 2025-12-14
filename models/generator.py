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


@dataclass
class LCTGeneratorConfig:
    in_channels: int = 1
    out_channels: int = 1
    enc_channels: Tuple[int, int, int] = (16, 32, 64)
    dec_channels: Tuple[int, int, int] = (64, 32, 16)
    num_heads: int = 4
    gru_groups: int = 4
    max_time_context: Optional[int] = None
    output_activation: str = "sigmoid"


class GRUblockf(nn.Module):
    """
    Frequency transformer block (GRUf*) with:
      - 4 bidirectional GRUs (gru1..gru4), each (input=16, hidden=16)
      - Multihead self-attention (embed_dim=64, 4 heads)
      - Two LayerNorms
      - One Linear(128 -> 64)
      - LeakyReLU activation

    Input / output: [B, C=64, T, F]
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        assert channels == 64, "This implementation assumes 64 channels."

        self.channels = channels
        self.num_groups = 4
        self.group_dim = channels // self.num_groups  # 16

        # 4 bidirectional GRUs, each working on 16-dim groups
        self.gru1 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.gru2 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.gru3 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.gru4 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=True,
        )

        # MultiheadAttention over frequency axis
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=4,
            batch_first=True,
        )

        self.activationtrans = nn.LeakyReLU(0.2, inplace=True)
        self.layernorm1 = nn.LayerNorm(channels)
        self.layernorm2 = nn.LayerNorm(channels)
        self.lin = nn.Linear(2 * channels, channels)  # 128 -> 64

    def _run_grus(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Run grouped bidirectional GRUs over frequency axis.

        seq: [B*T, F, C=64]
        return: [B*T, F, C=64]
        """
        BTF, F, C = seq.shape
        assert C == self.channels

        groups = torch.chunk(seq, self.num_groups, dim=-1)  # 4 x [B*T, F, 16]
        outs = []
        for g, gru in zip(groups,
                          [self.gru1, self.gru2, self.gru3, self.gru4]):
            # g: [B*T, F, 16]
            y, _ = gru(g)  # [B*T, F, 2*16]
            y_fwd = y[..., :self.group_dim]
            y_bwd = y[..., self.group_dim:]
            y_sum = y_fwd + y_bwd  # [B*T, F, 16]
            outs.append(y_sum)

        seq_gru = torch.cat(outs, dim=-1)  # [B*T, F, 64]
        return seq_gru

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 64, T, F]
        """
        B, C, T, F = x.shape
        assert C == self.channels

        # [B, C, T, F] -> [B, T, F, C] -> [B*T, F, C]
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        seq = x_perm.view(B * T, F, C)  # [B*T, F, 64]

        # GRU block with residual
        seq_res = seq
        seq_norm = self.layernorm1(seq)  # LN over last dim
        seq_gru = self._run_grus(seq_norm)  # [B*T, F, 64]
        seq = seq_res + seq_gru

        # Attention block
        seq_res2 = seq
        seq_norm2 = self.layernorm2(seq)
        attn_out, _ = self.attn(seq_norm2, seq_norm2,
                                seq_norm2)  # [B*T, F, 64]

        # Concatenate GRU and attention outputs, project back to 64 dims
        combined = torch.cat([seq_gru, attn_out], dim=-1)  # [B*T, F, 128]
        combined = self.lin(combined)  # [B*T, F, 64]
        combined = self.activationtrans(combined)
        seq = seq_res2 + combined

        # Back to [B, C, T, F]
        seq = seq.view(B, T, F, C)
        seq = seq.permute(0, 3, 1, 2).contiguous()
        return seq


class GRUblockt(nn.Module):
    """
    Time transformer block (GRUt*) with:
      - 4 unidirectional GRUs (gru1..gru4), each (input=16, hidden=16)
      - Multihead self-attention (embed_dim=64, 4 heads)
      - Two LayerNorms
      - One Linear(64 -> 64)
      - LeakyReLU activation

    Input / output: [B, C=64, T, F]
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        assert channels == 64, "This implementation assumes 64 channels."

        self.channels = channels
        self.num_groups = 4
        self.group_dim = channels // self.num_groups  # 16

        # 4 unidirectional GRUs for causal time modelling
        self.gru1 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.gru2 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.gru3 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.gru4 = nn.GRU(
            input_size=self.group_dim,
            hidden_size=self.group_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=4,
            batch_first=True,
        )

        self.activationtrans = nn.LeakyReLU(0.2, inplace=True)
        self.layernorm1 = nn.LayerNorm(channels)
        self.layernorm2 = nn.LayerNorm(channels)
        self.lin = nn.Linear(channels, channels)  # 64 -> 64

    def _run_grus(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Run grouped unidirectional GRUs over time axis.

        seq: [B*F, T, C=64]
        return: [B*F, T, C=64]
        """
        BFT, T, C = seq.shape
        assert C == self.channels

        groups = torch.chunk(seq, self.num_groups, dim=-1)  # 4 x [B*F, T, 16]
        outs = []
        for g, gru in zip(groups,
                          [self.gru1, self.gru2, self.gru3, self.gru4]):
            y, _ = gru(g)  # [B*F, T, 16]
            outs.append(y)

        seq_gru = torch.cat(outs, dim=-1)  # [B*F, T, 64]
        return seq_gru

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 64, T, F]
        """
        B, C, T, F = x.shape
        assert C == self.channels

        # [B, C, T, F] -> [B, F, T, C] -> [B*F, T, C]
        x_perm = x.permute(0, 3, 2, 1).contiguous()
        seq = x_perm.view(B * F, T, C)  # [B*F, T, 64]

        # GRU block with residual
        seq_res = seq
        seq_norm = self.layernorm1(seq)
        seq_gru = self._run_grus(seq_norm)  # [B*F, T, 64]
        seq = seq_res + seq_gru

        # Attention block along time
        seq_res2 = seq
        seq_norm2 = self.layernorm2(seq)
        attn_out, _ = self.attn(seq_norm2, seq_norm2,
                                seq_norm2)  # [B*F, T, 64]

        combined = self.lin(attn_out)  # [B*F, T, 64]
        combined = self.activationtrans(combined)
        seq = seq_res2 + combined

        # Back to [B, C, T, F]
        seq = seq.view(B, F, T, C)
        seq = seq.permute(0, 3, 2, 1).contiguous()
        return seq


class DownBlock(nn.Module):
    """
    One encoder (downsampling) block:

    - 2D conv over (time, freq) but with kernel_size=(2, 3):
        → mixes across frequency only (time-local, trivially causal).
    - stride = (1, 2): downsample frequency by 2, keep time resolution.
    - LeakyReLU activation.

    Input / Output shape: [B, C_in, T, F] -> [B, C_out, T, F_out]
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int] = (2, 3),
            stride: Tuple[int, int] = (1, 2),
            negative_slope: float = 0.03,
    ):
        super().__init__()
        k_t, k_f = kernel_size
        # assert k_t == 1, "We keep k_t=1 so conv is time-local."

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(1, 1),  # pad in time and freq
        )
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x


class UpBlock(nn.Module):
    """
    Decoder block that upsamples along frequency using ConvTranspose2d
    and then fuses encoder skip connections.

    Shapes:
        x:    [B, C_in,  T, F_in]
        skip: [B, C_skip, T_skip, F_skip]

    After deconv:
        x: [B, C_out, T', F']

    We then crop both x and skip to the minimum (T, F) so concatenation
    always works, and finish with a 1×3 Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (2, 3),
        stride: Tuple[int, int] = (1, 2),
        padding: Tuple[int, int] = (0, 1),
        output_padding: Tuple[int, int] = (0, 1),
        negative_slope: float = 0.03,
    ):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        self.conv = nn.Conv2d(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )

        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    @staticmethod
    def _align(
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crop x and skip to the same (T, F) so we can concatenate
        along the channel dimension without size mismatches.
        """
        # x, skip: [B, C, T, F]
        _, _, t_x, f_x = x.shape
        _, _, t_s, f_s = skip.shape

        t_min = min(t_x, t_s)
        f_min = min(f_x, f_s)

        x = x[:, :, :t_min, :f_min]
        skip = skip[:, :, :t_min, :f_min]
        return x, skip

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample with ConvTranspose2d
        x = self.deconv(x)

        # Align time/freq dims with skip
        x, skip = self._align(x, skip)

        # Fuse skip and upsampled feature
        x = torch.cat([x, skip], dim=1)

        # Mix channels
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

    def __init__(
            self,
            out_channels: int,
            channels: Tuple[int, int, int] = (64, 32, 16),
    ):
        super().__init__()
        c3, c2, c1 = channels

        # UpBlocks: (in_channels, skip_channels, out_channels)
        self.up3 = UpBlock(in_channels=c3, skip_channels=c3, out_channels=c2)
        self.up2 = UpBlock(in_channels=c2, skip_channels=c2, out_channels=c1)
        self.up1 = UpBlock(in_channels=c1,
                           skip_channels=c1,
                           out_channels=out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
    ) -> torch.Tensor:
        # skips = [s1, s2, s3] from Encoder
        s1, s2, s3 = skips

        x = self.up3(x, s3)  # deepest skip first
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return x


class LCTGenerator(nn.Module):
    """
    FTFNet-style TF generator that matches the parameter structure of the
    ground-truth FTFNet (conv1/2/3, deconv2/3/4, skip2/3/4, GRUf1/GRUt1/GRUf2).

    Input:  noisy_mag [B, 1, F, T]
    Output: mask_c    [B, 1, F, T] in [0,1] if output_activation="sigmoid"
    """

    def __init__(self, cfg: LCTGeneratorConfig):
        super().__init__()
        self.cfg = cfg

        in_ch = cfg.in_channels
        e1, e2, e3 = cfg.enc_channels
        d3, d2, d1 = cfg.dec_channels
        out_ch = cfg.out_channels

        assert in_ch == 1 and out_ch == 1, "FTFNet is defined for 1→1 masks."

        # Encoder convs: 1→16→32→64, kernel=(2,3), stride=(1,2)
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=e1,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=e1,
            out_channels=e2,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
        )
        self.conv3 = nn.Conv2d(
            in_channels=e2,
            out_channels=e3,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
        )

        # Skip 1x1 convs from magnitude: 1→64, 32, 16
        self.skip2 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=e3,  # 64
            kernel_size=1,
        )
        self.skip3 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=e2,  # 32
            kernel_size=1,
        )
        self.skip4 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=e1,  # 16
            kernel_size=1,
        )

        # FTF bottleneck blocks on 64 channels
        self.GRUf1 = GRUblockf(channels=e3)
        self.GRUt1 = GRUblockt(channels=e3)
        self.GRUf2 = GRUblockf(channels=e3)

        # Decoder: ConvTranspose2d with kernel=(2,3), stride=(1,2)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=e3,
            out_channels=e2,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
            output_padding=(0, 1),
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=e2,
            out_channels=e1,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
            output_padding=(0, 1),
        )
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=e1,
            out_channels=out_ch,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 1),
            output_padding=(0, 1),
        )

        # Shared activations
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.activations = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(e3)
        self.pad = nn.ConstantPad2d((0, 0, 0, 0), 0.0)
        self.act_final = nn.ReLU()

    @staticmethod
    def _align(a: torch.Tensor,
               b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Crop two feature maps to the same (T, F) along spatial dims.
        """
        _, _, Ta, Fa = a.shape
        _, _, Tb, Fb = b.shape
        Tm = min(Ta, Tb)
        Fm = min(Fa, Fb)
        return a[:, :, :Tm, :Fm], b[:, :, :Tm, :Fm]

    def forward(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """
        noisy_mag: [B, 1, F, T]
        returns:   [B, 1, F, T]
        """
        if noisy_mag.dim() != 4 or noisy_mag.size(1) != 1:
            raise ValueError(
                f"Expected noisy_mag [B, 1, F, T], got {noisy_mag.shape}")

        B, C, F_in, T_in = noisy_mag.shape

        # [B, 1, F, T] -> [B, 1, T, F]
        x = noisy_mag.permute(0, 1, 3, 2).contiguous()

        # Global skips from raw magnitude
        skip2 = self.skip2(x)  # [B, 64, T, F]
        skip3 = self.skip3(x)  # [B, 32, T, F]
        skip4 = self.skip4(x)  # [B, 16, T, F]

        # Encoder
        x1 = self.activation(self.conv1(x))  # [B, 16, T, F1]
        x2 = self.activation(self.conv2(x1))  # [B, 32, T, F2]
        x3 = self.activation(self.conv3(x2))  # [B, 64, T, F3]

        # LayerNorm over channel dim
        B3, C3, T3, F3 = x3.shape
        x3_perm = x3.permute(0, 2, 3, 1).contiguous()  # [B, T, F, C]
        x3_norm = self.layernorm(x3_perm)  # LN over last dim
        x3 = x3_norm.permute(0, 3, 1, 2).contiguous()  # [B, 64, T, F3]

        # FTF bottleneck: F → T → F
        h = self.GRUf1(x3)
        h = self.GRUt1(h)
        h = self.GRUf2(h)

        # Decoder with skips
        # Level 2
        skip2_aligned, h_aligned = self._align(skip2, h)
        h2 = h_aligned + skip2_aligned
        y2 = self.activation(self.deconv2(h2))  # [B, 32, T2, F2]

        # Level 3
        skip3_aligned, y2_aligned = self._align(skip3, y2)
        h3 = y2_aligned + skip3_aligned
        y3 = self.activation(self.deconv3(h3))  # [B, 16, T3, F3]

        # Level 4 (output)
        skip4_aligned, y3_aligned = self._align(skip4, y3)
        h4 = y3_aligned + skip4_aligned
        y4 = self.act_final(self.deconv4(h4))  # [B, 1, T_out, F_out]

        # Crop or pad back to original [T_in, F_in]
        B4, C4, T_out, F_out = y4.shape
        assert C4 == self.cfg.out_channels == 1

        # Crop if necessary
        if T_out > T_in:
            y4 = y4[:, :, :T_in, :]
        if F_out > F_in:
            y4 = y4[:, :, :, :F_in]

        # Pad if necessary
        if T_out < T_in or F_out < F_in:
            out_padded = torch.zeros(
                B4,
                C4,
                T_in,
                F_in,
                device=y4.device,
                dtype=y4.dtype,
            )
            out_padded[:, :, :T_out, :F_out] = y4
            y4 = out_padded

        # [B, 1, T, F] -> [B, 1, F, T]
        out = y4.permute(0, 1, 3, 2).contiguous()

        if self.cfg.output_activation == "sigmoid":
            out = self.activations(out)
        elif self.cfg.output_activation == "none":
            pass

        return out  # [B, 1, F, T]


class LCTEnhancer(nn.Module):
    """
    Waveform-level enhancer that wraps the FTFNet-style LCTGenerator
    with STFT / iSTFT and mask application.

    Input:  noisy waveform [B, T]
    Output: enhanced waveform [B, T], mask_c [B, 1, F, T_frames]
    """

    def __init__(
        self,
        gen_cfg: LCTGeneratorConfig,
        c: float = 0.3,
        stft_cfg: Optional[STFTConfig] = None,
    ):
        super().__init__()
        self.gen = LCTGenerator(gen_cfg)
        self.c = c

        if stft_cfg is None:
            self.stft = make_lct_stft(n_fft=512)
        else:
            self.stft = ComplexSTFT(stft_cfg)

    def forward(
        self,
        noisy_wave: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            noisy_wave: [B, T]

        Returns:
            enhanced_wave: [B, T]
            mask_c:        [B, 1, F, T_frames]
        """
        if noisy_wave.dim() != 2:
            raise ValueError(
                f"Expected noisy_wave [B, T], got {noisy_wave.shape}")

        # STFT of noisy
        noisy_stft = self.stft(noisy_wave)  # [B, F, T_frames]
        noisy_mag = magnitude(noisy_stft)  # [B, F, T_frames]
        noisy_mag = noisy_mag.unsqueeze(1)  # [B, 1, F, T_frames]

        # FTFNet-style generator predicts compressed mask
        mask_c = self.gen(noisy_mag)  # [B, 1, F, T_frames]

        # Apply mask in compressed domain
        enhanced_stft = apply_mask(
            noisy_stft,
            mask_c,
            compressed=True,
            c=self.c,
        )  # [B, F, T_frames]

        # iSTFT back to waveform
        enhanced_wave = self.stft.istft(
            enhanced_stft,
            length=noisy_wave.shape[-1],
        )  # [B, T]

        return enhanced_wave, mask_c
