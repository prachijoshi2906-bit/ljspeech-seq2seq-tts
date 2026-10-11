import sys, json
import numpy as np
import torch
from scipy.io.wavfile import read
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from vocoder import Vocoder, prepare_mel

path = sys.argv[1]            # python test_mel.py <file.npy or file.wav> [mean std]
mean = float(sys.argv[2]) if len(sys.argv) > 2 else None
std = float(sys.argv[3]) if len(sys.argv) > 3 else None

v = Vocoder()

if path.endswith(".wav"):
    h = AttrDict(json.load(open("config_v1.json")))
    sr, x = read(path)
    x = torch.FloatTensor(x / MAX_WAV_VALUE).unsqueeze(0)
    mel = mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate,
                          h.hop_size, h.win_size, h.fmin, h.fmax).squeeze(0).numpy()
else:
    mel = np.load(path)

print("raw mel:", mel.shape, f"min={mel.min():.2f} max={mel.max():.2f}")
mel = prepare_mel(mel, mean, std)
wav = v.mel_to_wav(mel)
v.save(wav, "mel_test_out.wav")
print(f"saved mel_test_out.wav ({v.duration(wav):.2f}s)")