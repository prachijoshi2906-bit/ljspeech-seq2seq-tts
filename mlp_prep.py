import argparse
import json
import os
import random

import numpy as np
import torch
from scipy.io.wavfile import read

from env import AttrDict
from meldataset import MAX_WAV_VALUE, mel_spectrogram
from mlp_common import DATA_DIR, N_MELS, CharTokenizer


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "LJSpeech-1.1")
SPLIT_FILE = None
SEED = 42


def load_texts(metadata_path):
    texts = {}
    with open(metadata_path, encoding="utf-8") as metadata:
        for line_number, line in enumerate(metadata, start=1):
            row = line.rstrip("\r\n").split("|", 2)
            if len(row) != 3:
                raise ValueError(
                    f"Expected three pipe-delimited columns on metadata line {line_number}"
                )
            texts[row[0]] = row[2].lower()
    return texts


def make_split(ids):
    ids = sorted(ids)
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)
    return {
        "train": ids[:train_end],
        "val": ids[train_end:val_end],
        "test": ids[val_end:],
    }


def validate_split(split, all_ids):
    expected = set(all_ids)
    assigned = []
    for name in ("train", "val", "test"):
        if name not in split or not isinstance(split[name], list):
            raise ValueError(f"Split must contain a list for {name!r}")
        assigned.extend(split[name])
    if len(assigned) != len(set(assigned)):
        raise ValueError("Split contains duplicate ids across partitions")
    if set(assigned) != expected:
        missing = len(expected - set(assigned))
        unknown = len(set(assigned) - expected)
        raise ValueError(f"Split does not match metadata (missing={missing}, unknown={unknown})")


def main(split_file=None):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = os.path.join(REPO_DIR, DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    texts = load_texts(os.path.join(DATASET_DIR, "metadata.csv"))

    if split_file:
        with open(split_file, encoding="utf-8") as f:
            split = json.load(f)
    else:
        split = make_split(texts)
    validate_split(split, texts)
    with open(os.path.join(data_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f)

    with open(os.path.join(REPO_DIR, "config_v1.json"), encoding="utf-8") as f:
        h = AttrDict(json.load(f))

    total = 0.0
    total_sq = 0.0
    frame_count = 0
    char_count = 0
    bin_sum = torch.zeros(N_MELS, dtype=torch.float64)
    train_min = float("inf")
    tokenizer = CharTokenizer.build([texts[clip_id] for clip_id in split["train"]])

    for name in ("train", "val", "test"):
        records = []
        for index, clip_id in enumerate(split[name]):
            wav_path = os.path.join(DATASET_DIR, "wavs", clip_id + ".wav")
            sample_rate, audio = read(wav_path)
            if sample_rate != h.sampling_rate:
                raise ValueError(
                    f"{wav_path} has sample rate {sample_rate}; expected {h.sampling_rate}"
                )
            if audio.ndim != 1:
                raise ValueError(f"Expected mono audio in {wav_path}, got shape {audio.shape}")
            audio_tensor = torch.from_numpy(
                np.asarray(audio, dtype=np.float32) / MAX_WAV_VALUE
            ).unsqueeze(0)
            mel = mel_spectrogram(
                audio_tensor,
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.fmax,
            ).squeeze(0)
            if mel.ndim != 2 or mel.shape[0] != N_MELS:
                raise ValueError(f"Unexpected mel shape for {clip_id}: {tuple(mel.shape)}")

            if name == "train":
                mel64 = mel.double()
                total += mel64.sum().item()
                total_sq += mel64.square().sum().item()
                frame_count += mel.shape[1]
                char_count += len(texts[clip_id])
                bin_sum += mel64.sum(dim=1)
                train_min = min(train_min, mel.min().item())
            records.append(mel.to(torch.float16))
            if (index + 1) % 1000 == 0:
                print(f"{name}: {index + 1}/{len(split[name])}")

        torch.save(
            {"ids": split[name], "texts": [texts[c] for c in split[name]], "mels": records},
            os.path.join(data_dir, name + ".pt"),
        )

    value_count = frame_count * N_MELS
    scalar_mean = total / value_count
    scalar_std = max(total_sq / value_count - scalar_mean**2, 0.0) ** 0.5
    stats = {
        "chars": tokenizer.chars,
        "mean": scalar_mean,
        "std": scalar_std,
        "mean_frame": (bin_sum / frame_count).tolist(),
        "frames_per_char": frame_count / char_count,
    }
    with open(os.path.join(data_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f)

    sizes = {name: len(split[name]) for name in ("train", "val", "test")}
    print(
        f"split sizes: train={sizes['train']} val={sizes['val']} test={sizes['test']}"
    )
    print(
        f"scalar mean={scalar_mean:.6f} std={scalar_std:.6f} "
        f"frames_per_char={stats['frames_per_char']:.6f} "
        f"vocab_size={tokenizer.vocab_size}"
    )
    print(f"saved train mel min={train_min:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", default=SPLIT_FILE)
    main(parser.parse_args().split_file)
