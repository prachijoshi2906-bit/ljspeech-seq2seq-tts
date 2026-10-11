import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlp_common import DATA_DIR, N_MELS
from s2s_common import R, Seq2SeqTTS, Tokenizer


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_model = None
_tokenizer = None
_stats = None


def _load_model():
    global _model, _tokenizer, _stats
    if _model is not None:
        return
    checkpoint_path = os.path.join(REPO_DIR, "s2s_best.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} is missing; train the model with train_s2s.py first"
        )
    with open(os.path.join(REPO_DIR, DATA_DIR, "stats.json"), encoding="utf-8") as file:
        _stats = json.load(file)
    _tokenizer = Tokenizer(_stats["chars"])
    _model = Seq2SeqTTS(_tokenizer.vocab_size)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    _model.load_state_dict(checkpoint["model"])
    _model.eval()


@torch.no_grad()
def text_to_mel(text, return_alignment=False, max_steps=1000, stop_threshold=0.5):
    _load_model()
    tokens = torch.tensor([_tokenizer.encode(text)], dtype=torch.long)
    outputs = _model(
        tokens,
        teacher_forcing_probability=0.0,
        max_steps=max_steps,
        stop_threshold=stop_threshold,
    )
    stop_probs = torch.sigmoid(outputs["stop_logits"][0])
    stop_indices = torch.nonzero(stop_probs >= stop_threshold, as_tuple=False)
    generated_steps = (
        int(stop_indices[0, 0].item()) + 1 if stop_indices.numel() else stop_probs.shape[0]
    )
    frame_count = generated_steps * R
    mel = outputs["mel_after"][0, :, :frame_count].cpu().numpy()
    mel = mel * _stats["std"] + _stats["mean"]
    if return_alignment:
        alignment = outputs["alignments"][0, :generated_steps].cpu().numpy()
        return mel.astype(np.float32, copy=False), alignment
    return mel.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text with Seq2Seq TTS")
    parser.add_argument(
        "text", nargs="?", default="The weather is beautiful today."
    )
    parser.add_argument("--output", default="s2s_test.wav")
    parser.add_argument("--alignment", default="s2s_alignment.png")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.max_steps < 1 or not 0 < args.stop_threshold < 1:
        parser.error("max-steps must be positive and stop-threshold must be between 0 and 1")

    mel, alignment = text_to_mel(
        args.text,
        return_alignment=True,
        max_steps=args.max_steps,
        stop_threshold=args.stop_threshold,
    )
    from vocoder import Vocoder

    vocoder = Vocoder(
        ckpt=os.path.join(REPO_DIR, "generator_v1"),
        config=os.path.join(REPO_DIR, "config_v1.json"),
    )
    output_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_DIR, args.output)
    alignment_path = (
        args.alignment if os.path.isabs(args.alignment)
        else os.path.join(REPO_DIR, args.alignment)
    )
    vocoder.save(vocoder.mel_to_wav(mel), output_path)

    plt.figure(figsize=(12, 5))
    plt.imshow(alignment, aspect="auto", origin="lower", interpolation="nearest")
    plt.xlabel("encoder token")
    plt.ylabel("decoder step")
    plt.title("Seq2Seq attention alignment")
    plt.tight_layout()
    plt.savefig(alignment_path, dpi=150)
    plt.close()
    print(f"mel shape={mel.shape} range=[{mel.min():.2f}, {mel.max():.2f}]")
    print(f"saved {output_path} and {alignment_path}")


if __name__ == "__main__":
    main()
