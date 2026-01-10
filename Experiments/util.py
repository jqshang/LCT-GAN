from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt

import os
import librosa
import librosa.display

from torchaudio.transforms import Spectrogram, InverseSpectrogram

from df.enhance import enhance

from metrics import compute_metrics_for_pair

# For per-panel colorbars "aside" each subplot
from mpl_toolkits.axes_grid1 import make_axes_locatable


def resample_tensor(x: torch.Tensor, sr_in, sr_out) -> torch.Tensor:
    if sr_in == sr_out:
        return x
    return torchaudio.functional.resample(x, orig_freq=sr_in, new_freq=sr_out)


def plot_specs_stack(
    panels,  # list[tuple[str, np.ndarray]]
    sr=16000,
    save_path=None,
    show: bool = False,
    dpi=150,
):
    """
    Plot a vertical stack of spectrograms for arbitrary panels.
    Each subplot gets its own colorbar "aside", and all share vmin/vmax.
    """

    def mag_db(x) -> np.ndarray:
        S = np.abs(
            librosa.stft(
                x,
                n_fft=512,
                hop_length=256,
                win_length=512,
                window="hann",
            ))
        return librosa.amplitude_to_db(S, ref=np.max)

    if not panels:
        return

    specs = [(title, mag_db(wav)) for (title, wav) in panels]
    vmin = min(S.min() for _, S in specs)
    vmax = max(S.max() for _, S in specs)

    fig, axes = plt.subplots(
        nrows=len(specs),
        ncols=1,
        figsize=(11, 3.0 * len(specs)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(specs) == 1:
        axes = [axes]  # type: ignore

    for ax, (title, S_db) in zip(axes, specs):
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="hz",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.15)
        fig.colorbar(img, cax=cax, format="%+2.0f dB")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_spec(
    x_np,
    sr=16000,
    title="",
    save_path=None,
    show: bool = False,
    dpi=150,
):
    S = np.abs(
        librosa.stft(
            x_np,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window="hann",
        ))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_specs_triptych(
    noisy,
    ftf=None,
    dfn=None,
    clean=None,
    my_ftf=None,
    sr=16000,
    save_path=None,
    show: bool = False,
    dpi=150,
):
    panels = []
    if clean is not None:
        panels.append((f"Clean (16 kHz)", clean))
    panels.append((f"Noisy (16 kHz)", noisy))
    if ftf is not None:
        panels.append((f"LCT-GAN", ftf))
    if my_ftf is not None:
        panels.append((f"FTFNet", my_ftf))
    if dfn is not None:
        panels.append((f"DeepFilterNet (↓48 to 16 kHz)", dfn))

    plot_specs_stack(
        panels=panels,
        sr=sr,
        save_path=save_path,
        show=show,
        dpi=dpi,
    )


class ModelComparator:

    def __init__(
        self,
        lct=None,
        my_lct=None,
        dfn=None,
        dfn_state=None,
        device=torch.device("cpu"),
        metrics_sr=16000,
        metrics_estoi: bool = False,
    ):
        self.lct = lct
        self.my_lct = my_lct
        self.dfn = dfn
        self.dfn_state = dfn_state
        self.device = device

        self.metrics_sr = metrics_sr
        self.metrics_estoi = metrics_estoi

        if self.lct is not None:
            self.lct.to(self.device)
        if self.my_lct is not None:
            self.my_lct.to(self.device)
        if self.dfn is not None:
            self.dfn.to(self.device)

    def _stft_tools(self):
        # STFT & iSTFT settings (match your existing run_lct_gan)
        framelen = 512
        hoplen = 256
        win = lambda x: torch.sqrt(torch.hann_window(x)).to(self.device
                                                            )  # sqrt-hann
        to_spec = Spectrogram(n_fft=framelen,
                              hop_length=hoplen,
                              power=None,
                              window_fn=win)
        from_spec = InverseSpectrogram(n_fft=framelen,
                                       hop_length=hoplen,
                                       window_fn=win)
        expected_F = framelen // 2 + 1
        return framelen, hoplen, to_spec, from_spec, expected_F

    def _align_1d(self, a, b):
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)

        T = min(a.shape[-1], b.shape[-1])
        return a[..., :T], b[..., :T]

    def _diff_clean(self, clean, est):
        c, e = self._align_1d(clean, est)
        return c - e

    def _write_wav(self, path, wav, sr=16000):
        wav = wav
        if wav.dim() == 2:
            wav = wav[0]
        sf.write(str(path), wav.detach().cpu().numpy(), sr)

    def _save_diff_artifacts(
        self,
        tag,
        clean,
        est,
        out_dir,
        spec_dir,
        sr=16000,
        normalize_listen=True,
        eps=1e-9,
    ):
        diff = self._diff_clean(clean, est)
        diff = torch.clamp(diff, -1.0, 1.0)

        diff_wav_path = out_dir / f"{tag}_diff.wav"
        self._write_wav(diff_wav_path, diff, sr=sr)

        # Plot diff spectrogram
        diff_png_path = spec_dir / f"{tag}_diff.png"
        plot_spec(
            diff.squeeze(0).detach().cpu().numpy(),
            title=f"Diff (clean - {tag}) (16 kHz)",
            save_path=str(diff_png_path),
        )

        out = {
            "diff_dir": str(diff_wav_path),
            "diff_plot": str(diff_png_path),
        }

        if normalize_listen:
            peak = diff.abs().max().item()
            if peak > eps:
                diff_norm = torch.clamp(diff / peak * 0.99, -1.0, 1.0)
            else:
                diff_norm = diff

            diff_norm_path = out_dir / f"{tag}_diff_norm.wav"
            self._write_wav(diff_norm_path, diff_norm, sr=sr)
            out["diff_norm_dir"] = str(diff_norm_path)

        return out

    @torch.no_grad()
    def run_lct_gan(self, noisy: torch.Tensor) -> torch.Tensor:
        assert self.lct is not None, "LCT-GAN model is not loaded."

        _, _, to_spec, from_spec, expected_F = self._stft_tools()

        spec = to_spec(noisy)  # likely complex [B, F, T]
        inputs = spec.permute(0, 2, 1).contiguous()  # [B, T, F]

        out = self.lct(inputs)

        if not torch.is_complex(out):
            if out.dim() == 1:
                out = out.unsqueeze(0)
            return out

        if out.dim() == 2:
            out = out.unsqueeze(0)  # (1, A, B)

        B, d1, d2 = out.shape

        if d1 == expected_F:
            stft_bft = out
        elif d2 == expected_F:
            stft_bft = out.permute(0, 2, 1).contiguous()
        else:
            raise RuntimeError(
                f"Cannot infer STFT layout from shape {out.shape}. "
                f"Expected one dimension to equal n_fft//2+1 = {expected_F}.")

        wav = from_spec(stft_bft)
        return wav

    @torch.no_grad()
    def run_my_lct_gan(self,
                       noisy: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
        assert self.my_lct is not None, "FTFNet model is not loaded."

        _, _, to_spec, from_spec, expected_F = self._stft_tools()

        spec = to_spec(noisy)  # complex [B,F,T] expected
        if not torch.is_complex(spec):
            # If torchaudio config changes, be explicit
            spec = torch.complex(spec, torch.zeros_like(spec))

        mag = spec.abs()  # [B,F,T]
        phase = spec / (mag + eps)  # complex unit phase, [B,F,T]

        # Preferred input layout for your my_gen: [B,1,F,T]
        x_b1ft = mag.unsqueeze(1)  # [B,1,F,T]

        # Try calling with [B,1,F,T], then fallback to [B,1,T,F]
        try:
            out = self.my_lct(x_b1ft)
            out_inferred = out
        except Exception:
            x_b1tf = x_b1ft.permute(0, 1, 3, 2).contiguous()  # [B,1,T,F]
            out = self.my_lct(x_b1tf)
            out_inferred = out

        # If waveform returned directly
        if not torch.is_complex(out_inferred) and out_inferred.dim() in (1, 2):
            if out_inferred.dim() == 1:
                out_inferred = out_inferred.unsqueeze(0)
            return out_inferred

        if torch.is_complex(out_inferred):
            # Normalize to [B,F,T] then iSTFT
            if out_inferred.dim() == 4:
                out_inferred = out_inferred.squeeze(1)
            if out_inferred.dim() != 3:
                raise RuntimeError(
                    f"Unexpected complex output shape: {tuple(out_inferred.shape)}"
                )

            # layout inference: one dim must equal expected_F
            B, d1, d2 = out_inferred.shape
            if d1 == expected_F:
                stft_bft = out_inferred
            elif d2 == expected_F:
                stft_bft = out_inferred.permute(0, 2, 1).contiguous()
            else:
                raise RuntimeError(
                    f"Cannot infer STFT layout from complex output shape {out_inferred.shape}."
                )

            return from_spec(stft_bft)

        # Otherwise treat as magnitude-like output (mask or enhanced magnitude)
        out_mag = out_inferred

        # Reduce to 3D [B,F,T] and infer whether time/freq swapped
        if out_mag.dim() == 4 and out_mag.shape[1] == 1:
            out_mag = out_mag.squeeze(1)  # [B,?,?]
        if out_mag.dim() != 3:
            raise RuntimeError(
                f"Unexpected FTFNet output shape: {tuple(out_mag.shape)}")

        B, d1, d2 = out_mag.shape
        if d1 == expected_F:
            mag_like_bft = out_mag  # [B,F,T]
        elif d2 == expected_F:
            mag_like_bft = out_mag.permute(0, 2, 1).contiguous()  # [B,F,T]
        else:
            raise RuntimeError(
                f"Cannot infer F/T from FTFNet output shape {tuple(out_mag.shape)} "
                f"(expected one dim == {expected_F}).")

        enh_stft = mag_like_bft * phase  # complex [B,F,T]
        wav = from_spec(enh_stft)
        return wav

    @torch.no_grad()
    def run_deepfilternet(self, noisy_48k: torch.Tensor) -> torch.Tensor:
        assert self.dfn is not None, "DeepFilterNet model is not loaded."
        assert self.dfn_state is not None, "DeepFilterNet state is not loaded."
        enhanced = enhance(self.dfn, self.dfn_state, noisy_48k)
        return enhanced

    def process_one_file(self, noisy_path, out_dir, clean_path=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        noisy_path = Path(noisy_path)

        x_np, sr_in = sf.read(noisy_path, dtype="float32")
        if x_np.ndim > 1:
            x_np = x_np.mean(axis=1)

        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)

        noisy = resample_tensor(x, sr_in, 16000)
        noisy = torch.clamp(noisy, -1.0, 1.0)

        # clean
        clean = None
        if clean_path is not None:
            clean_np, sr_clean = sf.read(clean_path, dtype="float32")
            if clean_np.ndim > 1:
                clean_np = clean_np.mean(axis=1)
            clean = resample_tensor(
                torch.from_numpy(clean_np).unsqueeze(0).to(self.device),
                sr_clean,
                16000,
            )
            clean = torch.clamp(clean, -1.0, 1.0)

        # Output dirs
        spec_dir = out_dir / "spectrograms"
        spec_dir.mkdir(parents=True, exist_ok=True)

        noisy_wav_path = out_dir / f"noisy.wav"
        sf.write(noisy_wav_path, noisy.squeeze(0).cpu().numpy(), 16000)

        clean_wav_path = None
        if clean is not None:
            clean_wav_path = out_dir / f"clean.wav"
            sf.write(clean_wav_path, clean.squeeze(0).cpu().numpy(), 16000)

        ftf = None
        ftf_path = None

        my_model = None
        my_path = None

        dfn = None
        dfn_path = None

        if self.lct is not None:
            ftf = self.run_lct_gan(noisy)
            ftf = torch.clamp(ftf, -1.0, 1.0)
            ftf_path = out_dir / f"ftfnet.wav"
            sf.write(ftf_path, ftf.squeeze(0).cpu().numpy(), 16000)

        if self.my_lct is not None:
            my_model = self.run_my_lct_gan(noisy)
            my_model = torch.clamp(my_model, -1.0, 1.0)
            my_path = out_dir / f"my_ftfnet.wav"
            sf.write(my_path, my_model.squeeze(0).cpu().numpy(), 16000)

        if self.dfn is not None:
            dfn_sr = self.dfn_state.sr()  # type: ignore
            noisy_48k = resample_tensor(noisy, 16000, dfn_sr)
            noisy_48k = torch.clamp(noisy_48k, -1.0, 1.0)

            dfn_48k = self.run_deepfilternet(noisy_48k)

            dfn = resample_tensor(dfn_48k, dfn_sr, 16000)
            dfn = torch.clamp(dfn, -1.0, 1.0)

            dfn_path = out_dir / f"dfn.wav"
            sf.write(dfn_path, dfn.squeeze(0).cpu().numpy(), 16000)

        if clean is not None:
            plot_spec(
                clean.squeeze(0).cpu().numpy(),
                title=f"Clean (16 kHz)",
                save_path=str(spec_dir / f"clean.png"),
            )

        plot_spec(
            noisy.squeeze(0).cpu().numpy(),
            title=f"Noisy (16 kHz)",
            save_path=str(spec_dir / f"noisy.png"),
        )

        if ftf is not None:
            plot_spec(
                ftf.squeeze(0).cpu().numpy(),
                title=f"LCT-GAN",
                save_path=str(spec_dir / f"ftfnet.png"),
            )

        if my_model is not None:
            plot_spec(
                my_model.squeeze(0).cpu().numpy(),
                title=f"FTFNet",
                save_path=str(spec_dir / f"my_ftfnet.png"),
            )

        if dfn is not None:
            plot_spec(
                dfn.squeeze(0).cpu().numpy(),
                title=f"DeepFilterNet (↓48→16 kHz)",
                save_path=str(spec_dir / f"dfn.png"),
            )

        plot_specs_triptych(
            noisy=noisy.squeeze(0).cpu().numpy(),
            ftf=ftf.squeeze(0).cpu().numpy() if ftf is not None else None,
            dfn=dfn.squeeze(0).cpu().numpy() if dfn is not None else None,
            clean=clean.squeeze(0).cpu().numpy()
            if clean is not None else None,
            my_ftf=my_model.squeeze(0).cpu().numpy()
            if my_model is not None else None,
            save_path=str(spec_dir / f"all.png"),
            show=False,
        )

        # Metrics output format
        result = {}

        # Clean entry
        if clean_wav_path is not None:
            result["clean"] = {"dir": str(clean_wav_path)}
        else:
            result["clean"] = {"dir": None}

        # Noisy entry (+ metrics vs clean if available)
        result["noisy"] = {"dir": str(noisy_wav_path)}
        if clean_wav_path is not None:
            metrics = compute_metrics_for_pair(
                clean_path=str(clean_wav_path),
                enhanced_path=str(noisy_wav_path),
                sr=self.metrics_sr,
                do_si_sdr=True,
                do_pesq=True,
                do_stoi=True,
                estoi=self.metrics_estoi,
            )
            result["noisy"].update(metrics)

        # LCT-GAN / FTFNet entry
        if ftf_path is not None:
            result["ftfnet"] = {"dir": str(ftf_path)}
            if clean_wav_path is not None:
                metrics: dict = compute_metrics_for_pair(
                    clean_path=str(clean_wav_path),
                    enhanced_path=str(ftf_path),
                    sr=self.metrics_sr,
                    do_si_sdr=True,
                    do_pesq=True,
                    do_stoi=True,
                    estoi=self.metrics_estoi,
                )
                result["ftfnet"].update(metrics)

        # My reimpl entry
        if my_path is not None:
            result["my_ftfnet"] = {"dir": str(my_path)}
            if clean_wav_path is not None:
                metrics: dict = compute_metrics_for_pair(
                    clean_path=str(clean_wav_path),
                    enhanced_path=str(my_path),
                    sr=self.metrics_sr,
                    do_si_sdr=True,
                    do_pesq=True,
                    do_stoi=True,
                    estoi=self.metrics_estoi,
                )
                result["my_ftfnet"].update(metrics)

        # dfn
        if dfn_path is not None:
            result["dfn"] = {"dir": str(dfn_path)}
            if clean_wav_path is not None:
                metrics: dict = compute_metrics_for_pair(
                    clean_path=str(clean_wav_path),
                    enhanced_path=str(dfn_path),
                    sr=self.metrics_sr,
                    do_si_sdr=True,
                    do_pesq=True,
                    do_stoi=True,
                    estoi=self.metrics_estoi,
                )
                result["dfn"].update(metrics)

        diff_panels = []

        if clean is not None:
            # Noisy diff
            noisy_diff_meta = self._save_diff_artifacts(
                tag="noisy",
                clean=clean,
                est=noisy,
                out_dir=out_dir,
                spec_dir=spec_dir,
                sr=16000,
                normalize_listen=False,
            )
            result.setdefault("noisy", {}).update(noisy_diff_meta)
            diff_wav_np = sf.read(noisy_diff_meta["diff_dir"],
                                  dtype="float32")[0]
            diff_panels.append((f"Diff: clean - noisy", diff_wav_np))

            # FTFNet diff
            if ftf is not None:
                ftf_diff_meta = self._save_diff_artifacts(
                    tag="ftfnet",
                    clean=clean,
                    est=ftf,
                    out_dir=out_dir,
                    spec_dir=spec_dir,
                    sr=16000,
                    normalize_listen=False,
                )
                result.setdefault("ftfnet", {}).update(ftf_diff_meta)
                diff_wav_np = sf.read(ftf_diff_meta["diff_dir"],
                                      dtype="float32")[0]
                diff_panels.append((f"Diff: clean - ftfnet", diff_wav_np))
            # My LCT diff
            if my_model is not None:
                my_diff_meta = self._save_diff_artifacts(
                    tag="my_ftfnet",
                    clean=clean,
                    est=my_model,
                    out_dir=out_dir,
                    spec_dir=spec_dir,
                    sr=16000,
                    normalize_listen=False,
                )
                result.setdefault("my_ftfnet", {}).update(my_diff_meta)
                diff_wav_np = sf.read(my_diff_meta["diff_dir"],
                                      dtype="float32")[0]
                diff_panels.append((f"Diff: clean - FTFNet", diff_wav_np))
            # DFN diff (16k)
            if dfn is not None:
                dfn_diff_meta = self._save_diff_artifacts(
                    tag="dfn",
                    clean=clean,
                    est=dfn,
                    out_dir=out_dir,
                    spec_dir=spec_dir,
                    sr=16000,
                    normalize_listen=False,
                )
                result.setdefault("dfn", {}).update(dfn_diff_meta)
                diff_wav_np = sf.read(dfn_diff_meta["diff_dir"],
                                      dtype="float32")[0]
                diff_panels.append((f"Diff: clean - dfn", diff_wav_np))

            # Optional: one stacked plot for all diffs
            if diff_panels:
                plot_specs_stack(
                    panels=diff_panels,
                    sr=16000,
                    save_path=str(spec_dir / "all_diff.png"),
                    show=False,
                )
                result["all_diff"] = {"plot": str(spec_dir / "all_diff.png")}

        return result
