"""text -> predicted 80-bin log-mel using the trained SVR baseline.

Returns a denormalized (80, T) mel, like text_to_mel_mlp / text_to_mel_mean / text_to_mel.
Debug:  python svr_infer.py "The weather is beautiful today."   -> writes svr_test.wav
"""
import json, sys
import numpy as np

import svr_core
from mlp_common import *          # N_MELS, T_MAX, LOG_FLOOR, DATA_DIR, CharTokenizer

_stats = json.load(open(f"{DATA_DIR}/stats.json"))
_tok = CharTokenizer(_stats["chars"])
_model = svr_core.load("svr_model.npz")
_meta = json.load(open("svr_meta.json"))
_n_tok = len(_stats["chars"]) + 1


def _length(text):
    return int(min(T_MAX, max(20, round(_stats["frames_per_char"] * len(text)))))


def text_to_mel_svr(text):
    x = (np.array(_tok.encode(text), dtype=np.float64) / _n_tok)[None]       # same features as training
    y = svr_core.predict(_model, x)[0]
    mel = svr_core.to_frames(y, N_MELS, _meta["stride"], _length(text))
    mel = mel * _stats["std"] + _stats["mean"]                                # denormalize
    return np.clip(mel, LOG_FLOOR, 3.0).astype(np.float32)


if __name__ == "__main__":
    from vocoder import Vocoder, prepare_mel
    text = sys.argv[1] if len(sys.argv) > 1 else "The weather is beautiful today."
    mel = prepare_mel(text_to_mel_svr(text))
    print("mel", mel.shape, f"min={mel.min():.2f} max={mel.max():.2f}")
    v = Vocoder()
    v.save(v.mel_to_wav(mel), "svr_test.wav")
    print("saved svr_test.wav")
