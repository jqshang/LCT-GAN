import os
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as AF


class LCTScpDataset(Dataset):
    """
    Dataset for LCT-GAN style speech enhancement with an .scp file and
    {clean,noisy}_{train,test} folder structure.

    Directory layout (example):
        data_root/
          clean_train/
            p226_001.wav
            ...
          noisy_train/
            p226_001.wav
            ...
          clean_test/
            p226_001.wav
          noisy_test/
            p226_001.wav
          train.scp
          test.scp

    Each line of the .scp is an utterance ID (without extension),
    shared between clean and noisy.

    Returns a dict per sample:
        {
            "id": <str>,
            "noisy": Tensor [T],
            "clean": Tensor [T],
            "sr": int,
        }

    If segment_length is provided, both signals are cropped to that
    many samples (shared start) when longer; shorter signals are
    left as-is and will be padded in the collate function.
    """

    def __init__(
        self,
        data_root: str,
        scp_path: str,
        subset: Optional[str] = None,
        sample_rate: Optional[int] = 16000,
        segment_length: Optional[int] = None,
        random_segment: bool = True,
        transform: Optional[Callable[[Dict], Dict]] = None,
        clean_subdir: Optional[str] = None,
        noisy_subdir: Optional[str] = None,
    ) -> None:
        """
        Args:
            data_root: Root directory containing audio folders and .scp files.
            scp_path: Path to the .scp file (absolute or relative to data_root).
            subset: Optional hint ("train" or "test"); used to pick default
                clean/noisy subdirs if clean_subdir/noisy_subdir are not set.
            sample_rate: If not None, audio is resampled to this rate.
            segment_length: If not None, randomly (or centrally) crop each pair
                to this many samples when they are longer.
            random_segment: If True and segment_length is given, use random crop
                (typical for training). If False, use a centered crop (nice for eval).
            transform: Optional callable that takes and returns the sample dict
                (e.g. to compute STFT/masks).
            clean_subdir: Override default clean_* directory name.
            noisy_subdir: Override default noisy_* directory name.
        """
        super().__init__()

        self.data_root = data_root
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.random_segment = random_segment
        self.transform = transform

        # Resolve scp_path relative to data_root if needed
        if not os.path.isabs(scp_path):
            scp_path = os.path.join(data_root, scp_path)
        self.scp_path = scp_path

        # Infer subset from filename if not explicitly given
        if subset is None:
            base = os.path.basename(scp_path).lower()
            if "train" in base:
                subset = "train"
            elif "test" in base:
                subset = "test"
            else:
                subset = "train"  # default
        self.subset = subset

        # Determine clean/noisy subdirs if not explicitly set
        if clean_subdir is None:
            clean_subdir = os.path.join(subset, "clean")  # e.g. "train/clean"
        if noisy_subdir is None:
            noisy_subdir = os.path.join(subset, "noisy")  # e.g. "train/noisy"

        self.clean_dir = os.path.join(data_root, clean_subdir)
        self.noisy_dir = os.path.join(data_root, noisy_subdir)

        # Read utterance IDs from .scp
        self.utt_ids = self._read_scp(self.scp_path)

        if len(self.utt_ids) == 0:
            raise RuntimeError(f"No IDs found in scp file: {self.scp_path}")

    @staticmethod
    def _read_scp(path: str) -> List[str]:
        """Read a simple one-ID-per-line scp file, ignoring empty/comment lines."""
        ids: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                ids.append(line)
        return ids

    def __len__(self) -> int:
        return len(self.utt_ids)

    def _load_wav(self, wav_path: str) -> torch.Tensor:
        """Load a wav file as a mono 1D float tensor, resampling if needed."""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(wav_path)

        waveform, sr = torchaudio.load(wav_path)  # [C, T]

        # Convert to mono (average channels) if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if requested
        if self.sample_rate is not None and sr != self.sample_rate:
            waveform = AF.resample(waveform, sr, self.sample_rate)
            sr = self.sample_rate

        # Return 1D tensor [T]
        return waveform.squeeze(0), sr

    def _crop_pair(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Optionally crop both signals to self.segment_length (same start index).
        If either is too short, they are left as-is (padding happens in collate).
        """
        if self.segment_length is None:
            return noisy, clean

        seg_len = self.segment_length
        n_len = noisy.shape[-1]
        c_len = clean.shape[-1]
        min_len = min(n_len, c_len)

        if min_len <= seg_len:
            # Too short; we don't crop here, let collate pad.
            return noisy, clean

        # Choose the start index for cropping
        max_start = min_len - seg_len
        if self.random_segment:
            start = torch.randint(low=0, high=max_start + 1, size=(1, )).item()
        else:
            start = max_start // 2  # centered crop

        end = start + seg_len

        noisy = noisy[..., start:end]
        clean = clean[..., start:end]
        return noisy, clean

    def __getitem__(self, index: int) -> Dict:
        utt_id = self.utt_ids[index]

        noisy_path = os.path.join(self.noisy_dir, f"{utt_id}.wav")
        clean_path = os.path.join(self.clean_dir, f"{utt_id}.wav")

        noisy, sr_noisy = self._load_wav(noisy_path)
        clean, sr_clean = self._load_wav(clean_path)

        # Basic sanity check: sample rates should match
        if sr_noisy != sr_clean:
            raise RuntimeError(
                f"Sample rate mismatch for {utt_id}: noisy={sr_noisy}, clean={sr_clean}"
            )

        # Optional cropping
        noisy, clean = self._crop_pair(noisy, clean)

        sample: Dict = {
            "id": utt_id,
            "noisy": noisy,  # [T]
            "clean": clean,  # [T]
            "sr": sr_noisy,
        }

        # Optional user-defined transform (e.g., STFT + mask computation)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_fn(batch: List[Dict], ) -> Dict:
    """
    Collate function to:
      - pad variable-length waveforms to the max length in the batch,
      - stack them into tensors,
      - keep track of original lengths.

    Expects each item from LCTScpDataset to contain keys:
      "id", "noisy", "clean", "sr"
    and optionally additional keys (which will be handled carefully).
    """
    if len(batch) == 0:
        return {}

    # All sample rates in a batch should match; take from first
    sr = batch[0]["sr"]

    ids = [b["id"] for b in batch]
    noisy_list = [b["noisy"] for b in batch]
    clean_list = [b["clean"] for b in batch]

    lengths = torch.tensor([x.shape[-1] for x in noisy_list], dtype=torch.long)

    max_len = int(lengths.max().item())
    batch_size = len(batch)

    padded_noisy = torch.zeros(batch_size, max_len, dtype=noisy_list[0].dtype)
    padded_clean = torch.zeros(batch_size, max_len, dtype=clean_list[0].dtype)

    for i in range(batch_size):
        n = noisy_list[i]
        c = clean_list[i]
        n_len = n.shape[-1]
        c_len = c.shape[-1]

        padded_noisy[i, :n_len] = n
        padded_clean[i, :c_len] = c

    out: Dict = {
        "id": ids,
        "noisy": padded_noisy,  # [B, T_max]
        "clean": padded_clean,  # [B, T_max]
        "lengths": lengths,  # [B]
        "sr": sr,
    }

    return out
