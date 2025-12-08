from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class PeriodDiscriminator(nn.Module):
    """
    Period-based waveform discriminator (HiFi-GAN style, simplified).

    For a given period P, we reshape the waveform so that samples that are
    P apart end up in the same "column":

        x: [B, 1, T]
        -> pad to multiple of P
        -> [B, 1, T', 1] reshape to [B, 1, T'//P, P]

    Then we apply a 2D Conv stack with kernels over the time dimension only
    (kernel_size=(k, 1)), which lets the model exploit periodic structure
    aligned with that period.

    Forward returns:
        logits: [B, 1, H, W] (last conv output)
        feature_maps: list of intermediate feature maps (for feature matching)
    """

    def __init__(self, period: int, use_spectral_norm: bool = False):
        super().__init__()
        self.period = period

        norm_f = spectral_norm if use_spectral_norm else weight_norm

        convs = []
        in_ch = 1
        cfgs = [
            # (out_channels, kernel_size_time, stride_time, groups)
            (32, 5, 3, 1),
            (128, 5, 3, 4),
            (512, 5, 3, 16),
            (1024, 5, 3, 64),
            (1024, 5, 1, 64),
        ]
        for out_ch, k, s, g in cfgs:
            convs.append(
                norm_f(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=(k, 1),
                        stride=(s, 1),
                        padding=(k // 2, 0),
                        groups=g,
                    )))
            in_ch = out_ch

        self.convs = nn.ModuleList(convs)
        self.conv_post = norm_f(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=1,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] or [B, T]

        Returns:
            logits: [B, 1, H, W]
            feature_maps: list of 2D feature maps from each conv (including post)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        B, C, T = x.shape
        assert C == 1, "PeriodDiscriminator expects shape [B, 1, T] or [B, T]."

        # Pad to multiple of period
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")  # pad on the right
            T = T + pad_len

        # Reshape for 2D conv: [B, 1, T//P, P]
        x = x.view(B, 1, T // self.period, self.period)

        feature_maps: List[torch.Tensor] = []
        for conv in self.convs:
            x = self.activation(conv(x))
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)

        # logits shape [B, 1, H, W]
        logits = x
        return logits, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD):

    Holds several PeriodDiscriminators with different periods.
    Forward returns lists of logits and feature maps, one per sub-discriminator.
    """

    def __init__(
        self,
        periods: List[int] = (2, 3, 5, 7, 11),
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p, use_spectral_norm=use_spectral_norm)
            for p in periods
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, 1, T] or [B, T] waveform

        Returns:
            logits_list: list of logits from each PeriodDiscriminator
                each logits[i]: [B, 1, H_i, W_i]
            fmaps_list: list of feature maps lists (nested)
                fmaps_list[i]: List[Tensor] from discriminator i
        """
        logits_list: List[torch.Tensor] = []
        fmaps_list: List[List[torch.Tensor]] = []

        for disc in self.discriminators:
            logits, fmaps = disc(x)
            logits_list.append(logits)
            fmaps_list.append(fmaps)

        return logits_list, fmaps_list


class ScaleDiscriminator(nn.Module):
    """
    Single-scale waveform discriminator (HiFi-GAN style, simplified).

    Operates on 1D waveform: [B, 1, T] with a stack of Conv1d layers.
    Forward returns:
        logits: [B, 1, T']   (last conv output)
        feature_maps: list of intermediate 1D feature maps (for feature matching)
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        convs = []
        in_ch = 1
        cfgs = [
            # (out_channels, kernel_size, stride, groups)
            (16, 15, 1, 1),
            (64, 41, 4, 4),
            (256, 41, 4, 16),
            (1024, 41, 4, 64),
            (1024, 41, 4, 256),
            (1024, 5, 1, 1),
        ]
        for out_ch, k, s, g in cfgs:
            convs.append(
                norm_f(
                    nn.Conv1d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=k,
                        stride=s,
                        padding=k // 2,
                        groups=g,
                    )))
            in_ch = out_ch

        self.convs = nn.ModuleList(convs)
        self.conv_post = norm_f(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] or [B, T]

        Returns:
            logits: [B, 1, T']
            feature_maps: list of 1D feature maps from each conv (including post)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        B, C, T = x.shape
        assert C == 1, "ScaleDiscriminator expects shape [B, 1, T] or [B, T]."

        feature_maps: List[torch.Tensor] = []

        for conv in self.convs:
            x = self.activation(conv(x))
            feature_maps.append(x)

        x = self.conv_post(x)
        feature_maps.append(x)

        logits = x  # [B, 1, T']
        return logits, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD):

    Uses several ScaleDiscriminators applied at different time scales:
        - Original waveform
        - Downsampled by avg-pooling (x2, x4, ...)

    Forward returns lists of logits + feature maps from each scale.
    """

    def __init__(self, num_scales: int = 3, use_spectral_norm: bool = False):
        super().__init__()
        assert num_scales >= 1, "num_scales must be >= 1"
        self.num_scales = num_scales

        # Use spectral norm only on the first (raw-scale) discriminator as in some GAN setups.
        discs: List[ScaleDiscriminator] = []
        for i in range(num_scales):
            discs.append(
                ScaleDiscriminator(
                    use_spectral_norm=(use_spectral_norm and i == 0)))
        self.discriminators = nn.ModuleList(discs)

        # Average pooling used to create lower-resolution scales
        self.avg_pool = nn.AvgPool1d(kernel_size=4,
                                     stride=2,
                                     padding=2,
                                     count_include_pad=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, 1, T] or [B, T] waveform

        Returns:
            logits_list: list of logits from each ScaleDiscriminator
                logits_list[i]: [B, 1, T_i]
            fmaps_list: list of feature maps lists (nested)
                fmaps_list[i]: List[Tensor] from discriminator i
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        logits_list: List[torch.Tensor] = []
        fmaps_list: List[List[torch.Tensor]] = []

        x_scale = x
        for disc in self.discriminators:
            logits, fmaps = disc(x_scale)
            logits_list.append(logits)
            fmaps_list.append(fmaps)

            # Downsample for next scale, except after last
            x_scale = self.avg_pool(x_scale)

        return logits_list, fmaps_list
