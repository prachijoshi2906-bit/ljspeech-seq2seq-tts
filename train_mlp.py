import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mlp_common import DATA_DIR, LOG_FLOOR, MAX_TEXT, MelMLP, N_MELS, T_MAX, CharTokenizer


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCHS = 40
BATCH = 32
LR = 5e-4
SEED = 42


def main(epochs=EPOCHS, batch_size=BATCH, learning_rate=LR):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data_dir = os.path.join(REPO_DIR, DATA_DIR)
    with open(os.path.join(data_dir, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    tokenizer = CharTokenizer(stats["chars"])
    mean, std = stats["mean"], stats["std"]
    if std <= 0:
        raise ValueError(f"Training mel standard deviation must be positive, got {std}")
    pad_value = (LOG_FLOOR - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MelData(Dataset):
        def __init__(self, name):
            with open(os.path.join(data_dir, name + ".pt"), "rb") as f:
                data = torch.load(f, map_location="cpu", weights_only=True)
            self.texts = data["texts"]
            self.mels = data["mels"]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            ids = torch.tensor(tokenizer.encode(self.texts[index]), dtype=torch.long)
            mel = (self.mels[index].float() - mean) / std
            length = min(mel.shape[1], T_MAX)
            target = torch.full((N_MELS, T_MAX), pad_value, dtype=torch.float32)
            target[:, :length] = mel[:, :length]
            return ids, target, length

    train_loader = DataLoader(
        MelData("train"), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(MelData("val"), batch_size=batch_size, shuffle=False)

    model = MelMLP(len(tokenizer.chars) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )

    def masked_l1(prediction, target, lengths):
        frame_mask = (
            torch.arange(T_MAX, device=device)[None, :] < lengths[:, None]
        ).unsqueeze(1)
        mask = frame_mask.expand(-1, N_MELS, -1)
        return (prediction - target).abs().masked_select(mask).mean()

    print(
        f"params: {sum(parameter.numel() for parameter in model.parameters()) / 1e6:.2f} M"
        f" | device: {device}"
    )
    best_validation = float("inf")
    train_history, validation_history = [], []

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        for ids, targets, lengths in train_loader:
            ids = ids.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            loss = masked_l1(model(ids), targets, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for ids, targets, lengths in val_loader:
                ids = ids.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)
                validation_loss += masked_l1(model(ids), targets, lengths).item()
        validation_loss /= len(val_loader)
        scheduler.step(validation_loss)
        train_history.append(train_loss)
        validation_history.append(validation_loss)

        saved = validation_loss < best_validation
        if saved:
            best_validation = validation_loss
            torch.save(model.state_dict(), os.path.join(REPO_DIR, "mlp_best.pt"))
        print(
            f"epoch {epoch:02d} | train {train_loss:.4f} | val {validation_loss:.4f}"
            f" | {time.time() - start:.0f}s" + ("  *saved" if saved else ""),
            flush=True,
        )

    plt.plot(train_history, label="train")
    plt.plot(validation_history, label="val")
    plt.xlabel("epoch")
    plt.ylabel("masked L1 (normalized)")
    plt.legend()
    plt.savefig(os.path.join(REPO_DIR, "mlp_loss.png"), dpi=150)
    print(f"best val: {best_validation:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--learning-rate", type=float, default=LR)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    main(args.epochs, args.batch_size, args.learning_rate)
