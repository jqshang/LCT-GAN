import argparse
import os

import torch
import torchaudio
from torch.utils.data import DataLoader

from datasets import LCTScpDataset, collate_fn
from models.generator import LCTEnhancer, LCTGeneratorConfig


def parse_args():
    parser = argparse.ArgumentParser(description="LCT-GAN inference script")

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=
        "Root dir containing 'train/' and 'test/' subfolders (e.g. .data).",
    )
    parser.add_argument(
        "--test_scp",
        type=str,
        default=os.path.join("test", "test.scp"),
        help="Path to test.scp (relative to data_root or absolute).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate to load and save audio.",
    )

    # Inference
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g., checkpoints/best.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="enhanced_test",
        help="Directory to save enhanced wavs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="'cuda' or 'cpu'",
    )

    return parser.parse_args()


def build_test_loader(
    data_root: str,
    test_scp: str,
    sample_rate: int,
    batch_size: int,
    num_workers: int,
):
    test_ds = LCTScpDataset(
        data_root=data_root,
        scp_path=test_scp,
        subset="test",
        sample_rate=sample_rate,
        segment_length=None,  # full utterances
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
    return test_loader


def build_enhancer_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> LCTEnhancer:
    ckpt = torch.load(ckpt_path, map_location=device)

    # Try to recover training-time args (not strictly required, but nice for consistency)
    ckpt_args = ckpt.get("args", {})

    compress_c = ckpt_args.get("compress_c", 0.3)
    max_time_context = ckpt_args.get("max_time_context", 200)

    gen_cfg = LCTGeneratorConfig(
        in_channels=1,
        out_channels=1,
        enc_channels=(16, 32, 64),
        dec_channels=(64, 32, 16),
        num_heads=4,
        gru_groups=4,
        max_time_context=max_time_context,
        output_activation="sigmoid",
    )

    enhancer = LCTEnhancer(
        gen_cfg=gen_cfg,
        c=compress_c,
    ).to(device)

    enhancer.load_state_dict(ckpt["enhancer"])
    enhancer.eval()
    return enhancer


@torch.no_grad()
def run_inference(
    enhancer: LCTEnhancer,
    test_loader: DataLoader,
    output_dir: str,
    device: torch.device,
):
    os.makedirs(output_dir, exist_ok=True)

    total_utts = 0

    for batch_idx, batch in enumerate(test_loader, 1):
        noisy = batch["noisy"].to(device)  # [B, T]
        ids = batch["id"]  # list of strings
        sr = batch["sr"]  # int (same for whole batch)

        enhanced, _ = enhancer(noisy)  # [B, T]
        enhanced = enhanced.cpu()

        for i, utt_id in enumerate(ids):
            wav = enhanced[i].unsqueeze(0)  # [1, T]
            out_path = os.path.join(output_dir, f"{utt_id}.wav")

            # torchaudio.save expects [channels, T]
            torchaudio.save(out_path, wav, sample_rate=sr)

            total_utts += 1

        print(f"Processed batch {batch_idx:04d} "
              f"({len(ids)} utterances) → total {total_utts}")

    print(
        f"Inference done. Enhanced {total_utts} utterances into '{output_dir}'."
    )


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    # Data
    test_loader = build_test_loader(
        data_root=args.data_root,
        test_scp=args.test_scp,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    enhancer = build_enhancer_from_checkpoint(args.checkpoint, device)

    # Inference
    run_inference(
        enhancer=enhancer,
        test_loader=test_loader,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
