import json
import os
import sys

import numpy as np
import torch

from mlp_common import DATA_DIR, MAX_TEXT, MelMLP, T_MAX, CharTokenizer


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(REPO_DIR, DATA_DIR, "stats.json"), encoding="utf-8") as f:
    _stats = json.load(f)
_tok = CharTokenizer(_stats["chars"])
_model = MelMLP(len(_tok.chars) + 1)
_model.load_state_dict(
    torch.load(os.path.join(REPO_DIR, "mlp_best.pt"), map_location="cpu", weights_only=True)
)
_model.eval()


def _length(text):
    text_length = len(text)
    if text_length > MAX_TEXT:
        raise ValueError(f"text exceeds the {MAX_TEXT}-character model limit")
    length = round(_stats["frames_per_char"] * text_length)
    if length > T_MAX:
        raise ValueError(f"requested mel length {length} exceeds model limit {T_MAX}")
    return length


@torch.no_grad()
def text_to_mel_mlp(text):
    length = _length(text)
    ids = torch.tensor(_tok.encode(text), dtype=torch.long).unsqueeze(0)
    mel = _model(ids)[0].numpy()[:, :length]
    return mel * _stats["std"] + _stats["mean"]


def text_to_mel_mean(text):
    length = _length(text)
    frame = np.array(_stats["mean_frame"], dtype=np.float32)[:, None]
    return np.tile(frame, (1, length))


if __name__ == "__main__":
    from vocoder import Vocoder

    text = sys.argv[1] if len(sys.argv) > 1 else "The weather is beautiful today."
    v = Vocoder(
        ckpt=os.path.join(REPO_DIR, "generator_v1"),
        config=os.path.join(REPO_DIR, "config_v1.json"),
    )
    for name, function in [
        ("mlp", text_to_mel_mlp),
        ("mean", text_to_mel_mean),
    ]:
        mel = function(text)
        v.save(v.mel_to_wav(mel), os.path.join(REPO_DIR, f"{name}_test.wav"))
        print(name, mel.shape, f"min={mel.min():.2f} max={mel.max():.2f}")