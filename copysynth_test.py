import json, torch
from scipy.io.wavfile import read, write
from env import AttrDict
from models import Generator
from meldataset import mel_spectrogram, MAX_WAV_VALUE

device = "cuda" if torch.cuda.is_available() else "cpu"

h = AttrDict(json.load(open("config_v1.json")))
gen = Generator(h).to(device)
state = torch.load("generator_v1", map_location=device, weights_only=False)
gen.load_state_dict(state["generator"])
gen.eval()
gen.remove_weight_norm()

sr, x = read(r"C:\Users\Raghavendra\Desktop\ML_Proj\LJSpeech-1.1\wavs\LJ001-0001.wav")
x = torch.FloatTensor(x / MAX_WAV_VALUE).unsqueeze(0)

mel = mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate,
                      h.hop_size, h.win_size, h.fmin, h.fmax)
print("mel shape:", mel.shape)   # expect (1, 80, T)

with torch.no_grad():
    audio = gen(mel.to(device)).squeeze()

audio = (audio * MAX_WAV_VALUE).clamp(-32768, 32767).short().cpu().numpy()
write("copysynth.wav", h.sampling_rate, audio)
print("saved copysynth.wav")