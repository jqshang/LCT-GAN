import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio


def si_sdr(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        reference: [T] or [1, T] clean waveform
        estimate:  [T] or [1, T] enhanced waveform
        eps: small constant for numerical stability

    Returns:
        si_sdr_db: float (dB)
    """
    if reference.dim() == 2:
        reference = reference.squeeze(0)
    if estimate.dim() == 2:
        estimate = estimate.squeeze(0)

    # Align length (trim to min)
    min_len = min(reference.shape[-1], estimate.shape[-1])
    reference = reference[..., :min_len]
    estimate = estimate[..., :min_len]

    # Zero-mean
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    # Optimal scaling (projection)
    ref_energy = torch.sum(reference**2) + eps
    scale = torch.sum(reference * estimate) / ref_energy
    s_target = scale * reference
    e_noise = estimate - s_target

    si_sdr_val = 10 * torch.log10(
        (torch.sum(s_target**2) + eps) / (torch.sum(e_noise**2) + eps))
    return float(si_sdr_val.item())


def batch_si_sdr(
    clean_batch: torch.Tensor,
    enhanced_batch: torch.Tensor,
) -> List[float]:
    """
    Compute SI-SDR per example in a batch.

    Args:
        clean_batch: [B, T] or [B, 1, T]
        enhanced_batch: same shape

    Returns:
        list of SI-SDR values (len = B)
    """
    if clean_batch.dim() == 3:
        clean_batch = clean_batch.squeeze(1)
    if enhanced_batch.dim() == 3:
        enhanced_batch = enhanced_batch.squeeze(1)

    assert clean_batch.shape[0] == enhanced_batch.shape[0]
    B = clean_batch.shape[0]
    scores: List[float] = []
    for b in range(B):
        scores.append(si_sdr(clean_batch[b], enhanced_batch[b]))
    return scores


def _require_pesq():
    try:
        from pesq import pesq  # type: ignore
    except ImportError:
        raise ImportError(
            "pesq package is not installed. Install with:\n"
            "  pip install pesq\n"
            "and note that it only supports specific sampling rates (8k, 16k)."
        )
    return pesq


def _require_pystoi():
    try:
        from pystoi import stoi  # type: ignore
    except ImportError:
        raise ImportError("pystoi package is not installed. Install with:\n"
                          "  pip install pystoi")
    return stoi


def pesq_score(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    mode: str = "wb",
) -> float:
    """
    Compute PESQ using the 'pesq' package.

    Args:
        reference: clean signal as 1D numpy array
        estimate:  enhanced signal as 1D numpy array
        sr: sampling rate (typically 16000 for 'wb' mode)
        mode: 'wb' (wide-band) or 'nb' (narrow-band)

    Returns:
        PESQ score (float)
    """
    pesq = _require_pesq()
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    return float(pesq(sr, reference, estimate, mode))


def stoi_score(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    extended: bool = False,
) -> float:
    """
    Compute STOI or ESTOI using the 'pystoi' package.

    Args:
        reference: clean signal as 1D numpy array
        estimate:  enhanced signal as 1D numpy array
        sr: sampling rate
        extended: if True, use extended STOI (ESTOI)

    Returns:
        STOI or ESTOI score (float)
    """
    stoi = _require_pystoi()
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]
    return float(stoi(reference, estimate, sr, extended=extended))


def load_mono_wave(path: str, target_sr: Optional[int] = None):
    """
    Load a mono wav as numpy array with optional resampling to target_sr.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav_np = wav.squeeze(0).numpy()
    return wav_np, sr


def read_scp(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def compute_metrics_for_pair(
    clean_path: str,
    enhanced_path: str,
    sr: int,
    do_si_sdr: bool = True,
    do_pesq: bool = True,
    do_stoi: bool = True,
    estoi: bool = False,
):
    """
    Compute requested metrics for a single utterance pair.

    Args:
        clean_path: path to clean wav
        enhanced_path: path to enhanced wav
        sr: sampling rate for loading / PESQ/STOI
        do_si_sdr, do_pesq, do_stoi: toggles
        estoi: if True, use ESTOI instead of classic STOI

    Returns:
        dict with keys among {"si_sdr", "pesq", "stoi"}
    """
    clean_np, sr_c = load_mono_wave(clean_path, target_sr=sr)
    enh_np, sr_e = load_mono_wave(enhanced_path, target_sr=sr)

    metrics: Dict[str, float] = {}

    if do_si_sdr:
        # Compute SI-SDR in torch
        clean_t = torch.from_numpy(clean_np).unsqueeze(0)  # [1, T]
        enh_t = torch.from_numpy(enh_np).unsqueeze(0)
        metrics["si_sdr"] = si_sdr(clean_t, enh_t)

    if do_pesq:
        metrics["pesq"] = pesq_score(clean_np, enh_np, sr)

    if do_stoi:
        metrics["stoi"] = stoi_score(clean_np, enh_np, sr, extended=estoi)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute speech enhancement metrics.")

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root dir containing 'test/' with clean/ and noisy/ (e.g. .data).",
    )
    parser.add_argument(
        "--test_scp",
        type=str,
        default=os.path.join("test", "test.scp"),
        help="Path to test.scp (relative to data_root or absolute).",
    )
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help="Directory containing enhanced wavs named <id>.wav.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sampling rate (should match training/inference).",
    )

    parser.add_argument(
        "--no_si_sdr",
        action="store_true",
        help="Disable SI-SDR computation.",
    )
    parser.add_argument(
        "--no_pesq",
        action="store_true",
        help="Disable PESQ computation.",
    )
    parser.add_argument(
        "--no_stoi",
        action="store_true",
        help="Disable STOI computation.",
    )
    parser.add_argument(
        "--estoi",
        action="store_true",
        help="Use ESTOI (extended STOI) instead of classic STOI.",
    )

    return parser.parse_args()


def main():
    # Assume the file structure is as follows:
    # ├─.data
    # │  ├─enhanced_dir
    # │  ├─test
    # │  │  │  test.scp
    # │  │  │
    # │  │  ├─clean
    # │  │  │      3.wav
    # │  │  │      4.wav
    # │  │  │
    # │  │  └─noisy
    # │  │          3.wav
    # │  │          4.wav
    # │  │
    # │  └─train
    # │      │  train.scp
    # │      │
    # │      ├─clean
    # │      │      1.wav
    # │      │      2.wav
    # │      │
    # │      └─noisy
    # │              1.wav
    # │              2.wav

    args = parse_args()

    if not os.path.isabs(args.test_scp):
        scp_path = os.path.join(args.data_root, args.test_scp)
    else:
        scp_path = args.test_scp

    ids = read_scp(scp_path)

    clean_dir = os.path.join(args.data_root, "test", "clean")
    enhanced_dir = args.enhanced_dir

    do_si_sdr = not args.no_si_sdr
    do_pesq = not args.no_pesq
    do_stoi = not args.no_stoi

    all_metrics: Dict[str, List[float]] = {}

    num_done = 0
    num_missing = 0

    for utt_id in ids:
        clean_path = os.path.join(clean_dir, f"{utt_id}.wav")
        enh_path = os.path.join(enhanced_dir, f"{utt_id}.wav")

        if not (os.path.exists(clean_path) and os.path.exists(enh_path)):
            print(f"[WARN] Missing files for ID {utt_id}: "
                  f"clean={os.path.exists(clean_path)}, "
                  f"enhanced={os.path.exists(enh_path)}")
            num_missing += 1
            continue

        try:
            m = compute_metrics_for_pair(
                clean_path,
                enh_path,
                sr=args.sample_rate,
                do_si_sdr=do_si_sdr,
                do_pesq=do_pesq,
                do_stoi=do_stoi,
                estoi=args.estoi,
            )
        except ImportError as e:
            # If pesq/pystoi are missing, print message and exit
            print(f"ERROR while computing metrics: {e}")
            return
        except Exception as e:
            print(f"[ERROR] Failed on {utt_id}: {e}")
            continue

        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

        num_done += 1
        if num_done % 10 == 0:
            print(f"Processed {num_done} utterances...")

    print("=====================================")
    print(f"Evaluated {num_done} utterances; {num_missing} missing.")
    print("Averages:")
    for k, vals in all_metrics.items():
        if len(vals) == 0:
            continue
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        print(f"  {k}: {mean_v:.4f} ± {std_v:.4f}")


if __name__ == "__main__":
    main()
