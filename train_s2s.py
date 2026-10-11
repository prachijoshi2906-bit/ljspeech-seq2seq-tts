import argparse
import json
import os
import random
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from mlp_common import DATA_DIR, LOG_FLOOR, N_MELS
from s2s_common import R, Seq2SeqTTS, Tokenizer


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
EPOCHS = 100
BATCH_SIZE = int(os.environ.get("BATCH", "32"))
LEARNING_RATE = 1e-3
TEACHER_FORCING = 1.0
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))
GUIDED_W = float(os.environ.get("GUIDED_W", "0.5"))
AMP = os.environ.get("AMP", "1").strip().lower() not in {"0", "false", "no", "off"}
BUCKET_BATCHES = 50
NUM_WORKERS = 2

torch.backends.cudnn.benchmark = True


class MelDataset(Dataset):
    def __init__(self, path, tokenizer, mean, std, max_frames=0):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.texts = data["texts"]
        self.mels = data["mels"]
        self.tokenizer = tokenizer
        self.mean = mean
        self.std = std
        self.indices = [
            index for index, mel in enumerate(self.mels)
            if not max_frames or mel.shape[1] <= max_frames
        ]
        self.lengths = [self.mels[index].shape[1] for index in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        tokens = torch.tensor(self.tokenizer.encode(self.texts[index]), dtype=torch.long)
        mel = (self.mels[index].float() - self.mean) / self.std
        return tokens, mel, mel.shape[1]


class LengthBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, batches_per_bucket=BUCKET_BATCHES, seed=SEED):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = batch_size * batches_per_bucket
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        generator.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket.sort(key=lambda index: self.dataset.lengths[index])
            batches.extend(
                bucket[offset:offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        generator.shuffle(batches)
        return iter(batches)


def collate_batch(items, pad_value):
    tokens, mels, lengths = zip(*items)
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    padded_frames = ((max(lengths) + R - 1) // R) * R
    batch_mels = torch.full(
        (len(items), N_MELS, padded_frames), pad_value, dtype=torch.float32
    )
    for index, mel in enumerate(mels):
        batch_mels[index, :, :mel.shape[1]] = mel
    return tokens, batch_mels, torch.tensor(lengths, dtype=torch.long)


def masked_mel_l1(prediction, target, lengths):
    frame_mask = (
        torch.arange(target.shape[2], device=target.device)[None, :]
        < lengths[:, None]
    ).unsqueeze(1)
    mask = frame_mask.expand(-1, N_MELS, -1)
    prediction = prediction[:, :, :target.shape[2]]
    return (prediction - target).abs().masked_select(mask).mean()


def stop_loss(logits, lengths):
    step_count = logits.shape[1]
    valid_steps = (lengths + R - 1) // R
    step_ids = torch.arange(step_count, device=logits.device)[None, :]
    valid_mask = step_ids < valid_steps[:, None]
    targets = step_ids == (valid_steps[:, None] - 1)
    per_step = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    return (per_step * valid_mask).sum() / valid_mask.sum().clamp(min=1)


def guided_attention_loss(alignments, text_lengths, mel_lengths, sigma=0.2):
    step_count = alignments.shape[1]
    text_width = alignments.shape[2]
    valid_steps = (mel_lengths + R - 1) // R
    step_positions = torch.arange(step_count, device=alignments.device).float()
    text_positions = torch.arange(text_width, device=alignments.device).float()
    normalized_steps = step_positions[None, :, None] / valid_steps[:, None, None].clamp(min=1)
    normalized_text = text_positions[None, None, :] / text_lengths[:, None, None].clamp(min=1)
    weights = 1.0 - torch.exp(
        -((normalized_text - normalized_steps) ** 2) / (2 * sigma**2)
    )
    valid_mask = (
        (step_positions[None, :, None] < valid_steps[:, None, None])
        & (text_positions[None, None, :] < text_lengths[:, None, None])
    )
    weighted_alignments = alignments.float() * weights
    return weighted_alignments.masked_select(valid_mask).mean()


def save_plots(train_values, val_values, alignment, text, epoch):
    figure, axis = plt.subplots()
    axis.plot(train_values, label="train")
    axis.plot(val_values, label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    figure.savefig(os.path.join(REPO_DIR, "s2s_loss.png"), dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.imshow(alignment, aspect="auto", origin="lower", interpolation="nearest")
    axis.set_xlabel("encoder token")
    axis.set_ylabel("decoder step")
    axis.set_title(f"Validation attention alignment (epoch {epoch})")
    axis.set_xticks(range(min(len(text) + 1, alignment.shape[1])))
    axis.set_xticklabels(list(text[:alignment.shape[1] - 1]) + ["<eos>"] if text else ["<eos>"],
                         rotation=90)
    figure.tight_layout()
    figure.savefig(os.path.join(REPO_DIR, "s2s_align.png"), dpi=150)
    plt.close(figure)


def main(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    data_dir = os.path.join(REPO_DIR, DATA_DIR)
    with open(os.path.join(data_dir, "stats.json"), encoding="utf-8") as file:
        stats = json.load(file)
    tokenizer = Tokenizer(stats["chars"])
    mean, std = stats["mean"], stats["std"]
    if std <= 0:
        raise ValueError(f"Training mel standard deviation must be positive, got {std}")
    normalized_floor = (LOG_FLOOR - mean) / std
    train_set = MelDataset(
        os.path.join(data_dir, "train.pt"), tokenizer, mean, std, MAX_FRAMES
    )
    val_set = MelDataset(os.path.join(data_dir, "val.pt"), tokenizer, mean, std)
    if not train_set:
        raise ValueError(f"No training clips remain after MAX_FRAMES={MAX_FRAMES} filtering")

    def collate(batch):
        return collate_batch(batch, normalized_floor)

    train_sampler = LengthBucketBatchSampler(train_set, batch_size)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = Seq2SeqTTS(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    start_epoch = 1
    best_val = float("inf")
    train_history, val_history = [], []
    drive_last_path = os.path.join(REPO_DIR, "s2s_last.pt")
    local_last_path = os.path.join("/content", "s2s_last.pt")
    resume_path = local_last_path if os.path.exists(local_last_path) else drive_last_path
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]
        train_history = checkpoint.get("train_history", [])
        val_history = checkpoint.get("val_history", [])
        print(f"resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_sampler.set_epoch(epoch)
        model.train()
        train_total = 0.0
        interval_started = time.time()
        for batch_index, (tokens, targets, lengths) in enumerate(train_loader, start=1):
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(
                    tokens, targets, lengths,
                    teacher_forcing_probability=TEACHER_FORCING,
                )
                mel_loss = (
                    masked_mel_l1(outputs["mel_before"], targets, lengths)
                    + masked_mel_l1(outputs["mel_after"], targets, lengths)
                )
                loss = mel_loss + stop_loss(outputs["stop_logits"], lengths)
                if GUIDED_W:
                    text_lengths = tokens.ne(0).sum(dim=1)
                    loss = loss + GUIDED_W * guided_attention_loss(
                        outputs["alignments"],
                        text_lengths,
                        lengths,
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.item()
            if batch_index % 100 == 0:
                elapsed = time.time() - interval_started
                print(
                    f"epoch {epoch:03d} | batch {batch_index}/{len(train_loader)}"
                    f" | {elapsed:.1f}s/100 batches",
                    flush=True,
                )
                interval_started = time.time()
        train_loss = train_total / max(len(train_loader), 1)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for tokens, targets, lengths in val_loader:
                tokens = tokens.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    outputs = model(
                        tokens, targets, lengths,
                        teacher_forcing_probability=1.0,
                    )
                    loss = (
                        masked_mel_l1(outputs["mel_before"], targets, lengths)
                        + masked_mel_l1(outputs["mel_after"], targets, lengths)
                        + stop_loss(outputs["stop_logits"], lengths)
                    )
                    if GUIDED_W:
                        loss = loss + GUIDED_W * guided_attention_loss(
                            outputs["alignments"],
                            tokens.ne(0).sum(dim=1),
                            lengths,
                        )
                val_total += loss.item()
        val_loss = val_total / max(len(val_loader), 1)
        scheduler.step(val_loss)
        train_history.append(train_loss)
        val_history.append(val_loss)

        sample_tokens, sample_target, sample_length = collate([val_set[0]])
        with torch.no_grad():
            sample_outputs = model(
                sample_tokens.to(device),
                sample_target.to(device),
                sample_length.to(device),
                teacher_forcing_probability=1.0,
            )
        valid_token_count = int(sample_tokens[0].ne(0).sum().item())
        save_plots(
            train_history,
            val_history,
            sample_outputs["alignments"][0, :].cpu().numpy(),
            val_set.texts[0][:max(valid_token_count - 1, 0)],
            epoch,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "best_val": best_val,
                 "chars": tokenizer.chars, "mean": mean, "std": std},
                os.path.join(REPO_DIR, "s2s_best.pt"),
            )
        checkpoint_to_save = (
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "train_history": train_history,
                "val_history": val_history,
                "chars": tokenizer.chars,
                "mean": mean,
                "std": std,
            }
        )
        os.makedirs(os.path.dirname(local_last_path), exist_ok=True)
        torch.save(checkpoint_to_save, local_last_path)
        if epoch % 3 == 0 or epoch == epochs:
            shutil.copy2(local_last_path, drive_last_path)
        print(
            f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}"
            f" | {time.time() - started:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    options = parser.parse_args()
    if options.epochs < 1 or options.batch_size < 1 or options.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    main(options.epochs, options.batch_size, options.learning_rate)
