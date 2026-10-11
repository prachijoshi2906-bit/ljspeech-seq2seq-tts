import json, warnings
import numpy as np
import torch
from scipy.io.wavfile import write
from env import AttrDict
from models import Generator
from meldataset import MAX_WAV_VALUE


class Vocoder:
    def __init__(self, ckpt="generator_v1", config="config_v1.json", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with open(config) as f:
            self.h = AttrDict(json.load(f))
        self.gen = Generator(self.h).to(self.device)
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.gen.load_state_dict(state["generator"])
        self.gen.eval()
        self.gen.remove_weight_norm()
        self.sr = self.h.sampling_rate

    @torch.no_grad()
    def mel_to_wav(self, mel):
        """mel: (80, T) or (1, 80, T) log-mel. Returns int16 numpy waveform."""
        if torch.is_tensor(mel):
            mel = mel.detach().cpu().numpy()
        mel = torch.as_tensor(np.asarray(mel), dtype=torch.float32, device=self.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        audio = self.gen(mel).squeeze()
        return (audio * MAX_WAV_VALUE).clamp(-32768, 32767).short().cpu().numpy()

    def save(self, wav, path):
        write(path, self.sr, wav)

    def duration(self, wav):
        return len(wav) / self.sr


def prepare_mel(mel, mean=None, std=None, n_mels=80):
    """Denormalize + validate. Returns float32 array of shape (n_mels, T)."""
    if torch.is_tensor(mel):
        mel = mel.detach().cpu().numpy()
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim == 3 and mel.shape[0] == 1:
        mel = mel[0]
    assert mel.ndim == 2, f"expected 2D mel, got shape {mel.shape}"

    if mel.shape[0] != n_mels and mel.shape[1] == n_mels:
        warnings.warn(f"mel looked transposed {mel.shape}; transposing to ({n_mels}, T)")
        mel = mel.T
    assert mel.shape[0] == n_mels, f"expected ({n_mels}, T), got {mel.shape}"

    if mean is not None and std is not None:
        mel = mel * std + mean

    lo, hi = float(mel.min()), float(mel.max())
    if lo < -12.5 or hi > 3.0:
        warnings.warn(f"mel range [{lo:.2f}, {hi:.2f}] is outside the expected log-mel range "
                      f"(about -11.5 to +2). Check denormalization / log type.")
    return mel