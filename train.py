import argparse
import csv
import json
import os
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from pesq import pesq as _pesq
from pystoi import stoi as _stoi

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


def _now_timestamp() -> str:
    """Filesystem-safe timestamp for experiment versioning."""
    # Example: 20260105_142530
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of common config objects into JSON-serializable types."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        # argparse.Namespace, many config classes
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback: string representation
    return str(obj)


def _write_json(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)


def _append_csv_row(csv_path: str,
                    row: Dict[str, Any],
                    fieldnames: Optional[list] = None) -> None:
    """Append a row to a CSV file, creating it with a header if missing."""
    _ensure_dir(os.path.dirname(csv_path) or ".")
    file_exists = os.path.exists(csv_path)

    if fieldnames is None:
        # Stable ordering: if file exists, reuse its header; else use row keys.
        if file_exists:
            with open(csv_path, "r", encoding="utf-8", newline="") as rf:
                reader = csv.reader(rf)
                header = next(reader, None)
            fieldnames = header if header else list(row.keys())
        else:
            fieldnames = list(row.keys())

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
def _si_sdr_torch(reference: torch.Tensor,
                  estimate: torch.Tensor,
                  eps: float = 1e-8) -> float:
    """Vectorized SI-SDR for a single pair. reference/estimate are 1D tensors."""
    # Align length
    min_len = min(reference.shape[-1], estimate.shape[-1])
    reference = reference[..., :min_len]
    estimate = estimate[..., :min_len]

    # Zero mean
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    ref_energy = torch.sum(reference**2) + eps
    scale = torch.sum(reference * estimate) / ref_energy
    s_target = scale * reference
    e_noise = estimate - s_target

    val = 10.0 * torch.log10(
        (torch.sum(s_target**2) + eps) / (torch.sum(e_noise**2) + eps))
    return float(val.item())


@torch.no_grad()
def validate_and_compute_metrics(
    *,
    epoch: int,
    loaders: Dict[str, DataLoader],
    enhancer: LCTEnhancer,
    mrstft_loss: MultiResolutionSTFTLoss,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Run validation and compute metrics.

    Always computes MR-STFT loss and SI-SDR.
    Optionally computes PESQ/STOI if the required packages are available.
    """
    enhancer.eval()
    mrstft_loss.eval()

    val_loader = loaders["val"]

    total_mr = 0.0
    total_si_sdr = 0.0
    total_pesq = 0.0
    total_stoi = 0.0
    n_pesq = 0
    n_stoi = 0
    count = 0

    for batch in val_loader:
        noisy = batch["noisy"].to(device)  # [B, T]
        clean = batch["clean"].to(device)  # [B, T]
        lengths = batch.get("lengths", None)

        enhanced, _ = enhancer(noisy)
        mr_loss, _ = mrstft_loss(enhanced, clean)

        B = noisy.size(0)
        total_mr += mr_loss.item() * B

        for b in range(B):
            # If padded, try to use provided lengths
            if lengths is not None:
                L = int(lengths[b])
                ref = clean[b, :L]
                est = enhanced[b, :L]
            else:
                ref = clean[b]
                est = enhanced[b]

            total_si_sdr += _si_sdr_torch(ref, est)

            # PESQ (wideband) + STOI are optional
            if _pesq is not None:
                ref_np = ref.detach().cpu().numpy()
                est_np = est.detach().cpu().numpy()
                min_len = min(ref_np.shape[-1], est_np.shape[-1])
                if min_len > 0:
                    try:
                        total_pesq += float(
                            _pesq(args.sample_rate, ref_np[:min_len],
                                  est_np[:min_len], "wb"))
                        n_pesq += 1
                    except Exception:
                        # Some edge cases (very short signals, SR mismatch, etc.)
                        pass

            if _stoi is not None:
                ref_np = ref.detach().cpu().numpy()
                est_np = est.detach().cpu().numpy()
                min_len = min(ref_np.shape[-1], est_np.shape[-1])
                if min_len > 0:
                    try:
                        total_stoi += float(
                            _stoi(ref_np[:min_len],
                                  est_np[:min_len],
                                  args.sample_rate,
                                  extended=False))
                        n_stoi += 1
                    except Exception:
                        pass

        count += B

    avg_mr = total_mr / max(count, 1)
    avg_si_sdr = total_si_sdr / max(count, 1)
    avg_pesq = (total_pesq / max(n_pesq, 1)) if n_pesq > 0 else float("nan")
    avg_stoi = (total_stoi / max(n_stoi, 1)) if n_stoi > 0 else float("nan")

    msg = f"[Epoch {epoch:03d}] Val MR-STFT={avg_mr:.4f} | SI-SDR={avg_si_sdr:.3f}"
    if n_pesq > 0:
        msg += f" | PESQ={avg_pesq:.3f}"
    if n_stoi > 0:
        msg += f" | STOI={avg_stoi:.4f}"
    print(msg)

    return {
        "val_mrstft": float(avg_mr),
        "val_si_sdr": float(avg_si_sdr),
        "val_pesq": float(avg_pesq),
        "val_stoi": float(avg_stoi),
    }


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

    # Experiment management
    parser.add_argument(
        "--expr_root",
        type=str,
        default="exprs",
        help="Root directory to store experiment runs (default: exprs/).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=
        ("Path to a checkpoint to resume from (e.g., exprs/<ts>/ckpts/last.pt). "
         "If provided, the existing experiment directory is reused."),
    )

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

    # Validation / checkpointing cadence
    parser.add_argument(
        "--val_interval",
        type=int,
        default=50,
        help="Run validation + metrics every N epochs (default: 50).",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=50,
        help="Save periodic checkpoints every N epochs (default: 50).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # -----------------------
    # Experiment directories
    # -----------------------
    if args.resume is not None:
        resume_path = os.path.abspath(args.resume)
        ckpt_dir = os.path.dirname(resume_path)
        run_dir = os.path.dirname(ckpt_dir)
        if os.path.basename(ckpt_dir) != "ckpts":
            # Still try to be helpful if user points to a non-standard location.
            ckpt_dir = os.path.join(run_dir, "ckpts")
        print(f"Resuming from: {resume_path}")
        print(f"Using existing run_dir: {run_dir}")
    else:
        run_dir = os.path.join(args.expr_root, _now_timestamp())
        ckpt_dir = os.path.join(run_dir, "ckpts")

    _ensure_dir(run_dir)
    _ensure_dir(ckpt_dir)

    configs_path = os.path.join(run_dir, "configs.json")
    metrics_csv = os.path.join(run_dir, "metrics.csv")

    # Device
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

    if args.resume is None:
        payload = {
            "run_dir": run_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "gen_cfg": _to_jsonable(gen_cfg),
            "tf_cfg": _to_jsonable(tf_cfg),
            "mr_cfg": _to_jsonable(mr_cfg),
        }
        print("===== Training configuration =====")
        print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))
        _write_json(configs_path, payload)
        print(f"Saved configs to: {configs_path}")
    else:
        # On resume, do not overwrite configs.json.
        if os.path.exists(configs_path):
            print(f"Found existing configs.json: {configs_path}")

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    if args.resume is not None:
        ckpt = torch.load(os.path.abspath(args.resume), map_location=device)
        enhancer.load_state_dict(ckpt["enhancer"], strict=True)
        mpd.load_state_dict(ckpt["mpd"], strict=True)
        msd.load_state_dict(ckpt["msd"], strict=True)
        if "g_opt" in ckpt:
            g_opt.load_state_dict(ckpt["g_opt"])
        if "d_opt" in ckpt:
            d_opt.load_state_dict(ckpt["d_opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(
            ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
        best_epoch = int(ckpt.get("best_epoch", 0))
        print(
            f"Resumed at epoch {start_epoch} (best_val={best_val:.4f} from epoch {best_epoch})."
        )

    # Main loop
    for epoch in range(start_epoch, args.epochs + 1):
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

        # Validation + metrics every N epochs (and always on the final epoch)
        do_val = (epoch % max(args.val_interval, 1) == 0) or (epoch
                                                              == args.epochs)
        val_metrics: Dict[str, float] = {}
        improved = False
        if do_val:
            val_metrics = validate_and_compute_metrics(
                epoch=epoch,
                loaders=loaders,
                enhancer=enhancer,
                mrstft_loss=mrstft_loss,
                device=device,
                args=args,
            )

            # Best checkpoint (based on MR-STFT validation loss)
            if "val_mrstft" in val_metrics:
                val_mr = float(val_metrics["val_mrstft"])
                if val_mr < best_val:
                    best_val = val_mr
                    best_epoch = epoch
                    improved = True

        # Build checkpoint payload with current best info
        ckpt_payload = {
            "epoch": epoch,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "enhancer": enhancer.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
            "gen_cfg": _to_jsonable(gen_cfg),
            "tf_cfg": _to_jsonable(tf_cfg),
            "mr_cfg": _to_jsonable(mr_cfg),
        }

        # Save latest checkpoint (overwritten every epoch to support resuming)
        torch.save(ckpt_payload, os.path.join(ckpt_dir, "last.pt"))

        # Periodic checkpoints
        if (epoch % max(args.ckpt_interval, 1) == 0) or (epoch == args.epochs):
            torch.save(ckpt_payload,
                       os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))

        # Save best checkpoint (if improved)
        if do_val and improved:
            torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pt"))
            print(
                f"New best val MR-STFT: {best_val:.4f} @ epoch {best_epoch} (saved best.pt)"
            )

        # Log to CSV (after best has potentially been updated)
        if do_val:
            _append_csv_row(
                metrics_csv,
                {
                    "epoch": epoch,
                    **val_metrics,
                    "best_val_mrstft": best_val,
                    "best_epoch": best_epoch,
                },
            )

    print("Training finished.")


if __name__ == "__main__":
    main()
