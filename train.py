import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import LCTScpDataset, collate_fn
from models.generator import LCTEnhancer, LCTGeneratorConfig
from models.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from datasets.tf_features import TFFeatures, TFFeaturesConfig
from losses import (
    MRSTFTLossConfig,
    MultiResolutionSTFTLoss,
    discriminator_loss,
    generator_adv_loss,
    feature_matching_loss,
    _flatten_logits_lists,
    mask_mse_loss,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(
    data_root: str,
    train_scp: str,
    test_scp: str,
    batch_size: int,
    num_workers: int,
    sample_rate: int,
    segment_length: int,
) -> Dict[str, DataLoader]:
    train_ds = LCTScpDataset(
        data_root=data_root,
        scp_path=train_scp,
        subset="train",
        sample_rate=sample_rate,
        segment_length=segment_length,
        random_segment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    test_ds = LCTScpDataset(
        data_root=data_root,
        scp_path=test_scp,
        subset="test",
        sample_rate=sample_rate,
        segment_length=None,  # full utterances for validation
        random_segment=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )

    return {"train": train_loader, "val": test_loader}


def train_one_epoch(
    epoch: int,
    loaders: Dict[str, DataLoader],
    enhancer: LCTEnhancer,
    mpd: MultiPeriodDiscriminator,
    msd: MultiScaleDiscriminator,
    tf_features: TFFeatures,
    mrstft_loss: MultiResolutionSTFTLoss,
    g_opt: torch.optim.Optimizer,
    d_opt: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
):
    enhancer.train()
    mpd.train()
    msd.train()
    tf_features.train()  # just for consistency (no trainable params)

    train_loader = loaders["train"]

    for step, batch in enumerate(train_loader, 1):
        noisy = batch["noisy"].to(device)  # [B, T]
        clean = batch["clean"].to(device)  # [B, T]

        # ---- TF features & IRM targets (1.3) ----
        # This uses the same STFT config as the generator (512, 50% overlap).
        tf_feats = tf_features(noisy, clean)
        irm_c = tf_feats["irm_c"]  # [B, F, T_frames]

        # --------------------
        #   Discriminator step
        # --------------------
        d_opt.zero_grad(set_to_none=True)

        # Get fake waveform for D without building autograd graph for generator
        with torch.no_grad():
            enhanced_for_d, _ = enhancer(noisy)  # [B, T]

        # MPD
        mpd_real_logits, _ = mpd(clean)
        mpd_fake_logits, _ = mpd(enhanced_for_d)

        # MSD
        msd_real_logits, _ = msd(clean)
        msd_fake_logits, _ = msd(enhanced_for_d)

        d_loss = discriminator_loss(
            real_logits=_flatten_logits_lists(mpd_real_logits,
                                              msd_real_logits),
            fake_logits=_flatten_logits_lists(mpd_fake_logits,
                                              msd_fake_logits),
            loss_type=args.gan_loss,
        )

        d_loss.backward()
        d_opt.step()

        # ----------------
        #   Generator step
        # ----------------
        g_opt.zero_grad(set_to_none=True)

        # Forward generator with grad
        enhanced, mask_c = enhancer(
            noisy)  # enhanced: [B, T], mask_c: [B, 1, F, T_frames]

        # MR-STFT loss on waveforms
        mr_loss, mr_details = mrstft_loss(enhanced, clean)

        # Mask loss in compressed domain (IRM^c)
        pred_mask_c = mask_c[:, 0]  # [B, F, T_mask]
        irm_c_aligned, pred_mask_aligned = _align_tf_targets(
            irm_c, pred_mask_c)
        m_loss = mask_mse_loss(pred_mask_aligned, irm_c_aligned)

        # Adversarial + feature matching losses
        mpd_fake_logits_g, mpd_fake_fmaps_g = mpd(enhanced)
        msd_fake_logits_g, msd_fake_fmaps_g = msd(enhanced)

        # Real feature maps for FM (no gradient needed)
        with torch.no_grad():
            mpd_real_logits_g, mpd_real_fmaps_g = mpd(clean)
            msd_real_logits_g, msd_real_fmaps_g = msd(clean)

        adv_loss = generator_adv_loss(
            fake_logits=_flatten_logits_lists(mpd_fake_logits_g,
                                              msd_fake_logits_g),
            loss_type=args.gan_loss,
        )

        real_fmaps_combined = mpd_real_fmaps_g + msd_real_fmaps_g
        fake_fmaps_combined = mpd_fake_fmaps_g + msd_fake_fmaps_g
        fm_loss = feature_matching_loss(real_fmaps_combined,
                                        fake_fmaps_combined)

        adv_total = adv_loss + args.lambda_fm * fm_loss

        g_loss = (mr_loss + args.lambda_mask * m_loss +
                  args.lambda_adv * adv_total)

        g_loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(enhancer.parameters(),
                                           args.grad_clip)
        g_opt.step()

        if step % args.log_interval == 0:
            print(f"[Epoch {epoch:03d} Step {step:05d}] "
                  f"D_loss={d_loss.item():.4f} | "
                  f"G_loss={g_loss.item():.4f} | "
                  f"MR={mr_loss.item():.4f} | "
                  f"Mask={m_loss.item():.4f} | "
                  f"Adv={adv_loss.item():.4f} | "
                  f"FM={fm_loss.item():.4f}")


@torch.no_grad()
def validate(
    epoch: int,
    loaders: Dict[str, DataLoader],
    enhancer: LCTEnhancer,
    mrstft_loss: MultiResolutionSTFTLoss,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    enhancer.eval()
    mrstft_loss.eval()

    val_loader = loaders["val"]

    total_loss = 0.0
    count = 0

    for batch in val_loader:
        noisy = batch["noisy"].to(device)  # [B, T]
        clean = batch["clean"].to(device)  # [B, T]

        enhanced, _ = enhancer(noisy)
        mr_loss, _ = mrstft_loss(enhanced, clean)

        total_loss += mr_loss.item() * noisy.size(0)
        count += noisy.size(0)

    avg_loss = total_loss / max(count, 1)
    print(f"[Epoch {epoch:03d}] Validation MR-STFT Loss: {avg_loss:.4f}")
    return avg_loss


def _align_tf_targets(
    irm_c: torch.Tensor,
    pred_mask_c: torch.Tensor,
):
    """
    Align compressed IRM target and predicted mask along the time axis
    by cropping both to the minimum frame length.
    Shapes:
        irm_c:       [B, F, T_irm]
        pred_mask_c: [B, F, T_mask]
    """
    if irm_c.dim() != 3 or pred_mask_c.dim() != 3:
        raise ValueError(f"Expected irm_c and pred_mask_c to be [B, F, T], "
                         f"got {irm_c.shape}, {pred_mask_c.shape}")

    B1, F1, T_irm = irm_c.shape
    B2, F2, T_mask = pred_mask_c.shape
    if B1 != B2 or F1 != F2:
        raise ValueError(
            f"Batch/Freq mismatch: irm_c {irm_c.shape}, pred_mask_c {pred_mask_c.shape}"
        )

    T_min = min(T_irm, T_mask)
    irm_c_aligned = irm_c[..., :T_min]
    pred_mask_aligned = pred_mask_c[..., :T_min]
    return irm_c_aligned, pred_mask_aligned


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LCT-GAN (LCTEnhancer + MPD/MSD)")

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=
        "Root dir containing the dataset folders and scp files (e.g. .data).")
    parser.add_argument(
        "--train_scp",
        type=str,
        default="train.scp",
        help="Path to train.scp (relative to data_root or absolute).")
    parser.add_argument(
        "--test_scp",
        type=str,
        default="test.scp",
        help=
        "Path to test.scp (for validation; relative to data_root or absolute)."
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_seconds",
                        type=float,
                        default=2.0,
                        help="Training segment length in seconds.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--betas_g", type=float, nargs=2, default=(0.8, 0.99))
    parser.add_argument("--betas_d", type=float, nargs=2, default=(0.8, 0.99))
    parser.add_argument("--grad_clip", type=float, default=5.0)

    # Loss weights
    parser.add_argument("--lambda_mask",
                        type=float,
                        default=1.0,
                        help="Weight for compressed-mask MSE loss.")
    parser.add_argument("--lambda_adv",
                        type=float,
                        default=1e-2,
                        help="Weight for adversarial + FM loss.")
    parser.add_argument(
        "--lambda_fm",
        type=float,
        default=1.0,
        help="Relative weight of FM vs pure adv inside the adv branch.")
    parser.add_argument("--gan_loss",
                        type=str,
                        default="ls",
                        choices=["ls", "hinge"])

    # Model / STFT
    parser.add_argument(
        "--compress_c",
        type=float,
        default=0.3,
        help="Magnitude compression exponent for IRM and mask.")
    parser.add_argument(
        "--max_time_context",
        type=int,
        default=200,
        help="Max attention context in frames for time transformer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    segment_length = int(args.segment_seconds * args.sample_rate)

    # Data
    loaders = prepare_dataloaders(
        data_root=args.data_root,
        train_scp=args.train_scp,
        test_scp=args.test_scp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        segment_length=segment_length,
    )

    # Models
    gen_cfg = LCTGeneratorConfig(
        in_channels=1,
        out_channels=1,
        enc_channels=(16, 32, 64),
        dec_channels=(64, 32, 16),
        num_heads=4,
        gru_groups=4,
        max_time_context=args.max_time_context,
        output_activation="sigmoid",
    )
    enhancer = LCTEnhancer(
        gen_cfg=gen_cfg,
        c=args.compress_c,
    ).to(device)

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # TF features & targets (for IRM^c)
    tf_cfg = TFFeaturesConfig(
        n_fft=512,
        c=args.compress_c,
        compress_input=False,
        return_stfts=False,
    )
    tf_features = TFFeatures(tf_cfg).to(device)

    # MR-STFT loss
    mr_cfg = MRSTFTLossConfig()
    mrstft_loss = MultiResolutionSTFTLoss(mr_cfg).to(device)

    # Optimizers
    g_opt = torch.optim.AdamW(
        enhancer.parameters(),
        lr=args.lr_g,
        betas=tuple(args.betas_g),
    )
    d_opt = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=args.lr_d,
        betas=tuple(args.betas_d),
    )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            epoch=epoch,
            loaders=loaders,
            enhancer=enhancer,
            mpd=mpd,
            msd=msd,
            tf_features=tf_features,
            mrstft_loss=mrstft_loss,
            g_opt=g_opt,
            d_opt=d_opt,
            device=device,
            args=args,
        )

        val_loss = validate(
            epoch=epoch,
            loaders=loaders,
            enhancer=enhancer,
            mrstft_loss=mrstft_loss,
            device=device,
            args=args,
        )

        # Save latest checkpoint
        ckpt = {
            "epoch": epoch,
            "enhancer": enhancer.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        # Save best so far
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(
                f"New best validation loss: {best_val:.4f} (checkpoint saved)")

    print("Training finished.")


if __name__ == "__main__":
    main()
