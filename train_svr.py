"""Train the SVR baseline (CS229 paper's baseline 1): one polynomial-kernel SVR per mel value.

Needs mlp_data/ from mlp_prep.py. Run:  python train_svr.py
Optional env vars: SVR_N (training sentences, default 200 as in the paper),
                   SVR_STRIDE (predict every k-th frame and interpolate, default 1 = faithful).
"""
import os, json, random, time
import numpy as np
import torch

import svr_core
from mlp_common import *          # N_MELS, T_MAX, LOG_FLOOR, DATA_DIR, CharTokenizer

N_TRAIN = int(os.environ.get("SVR_N", 200))
STRIDE = int(os.environ.get("SVR_STRIDE", 1))
DEGREE, C, EPS = 3, 1.0, 0.1

stats = json.load(open(f"{DATA_DIR}/stats.json"))
tok = CharTokenizer(stats["chars"])
mean, std = stats["mean"], stats["std"]
pad_val = (LOG_FLOOR - mean) / std
n_tok = len(stats["chars"]) + 1


def feats(text):
    return np.array(tok.encode(text), dtype=np.float64) / n_tok          # (MAX_TEXT,) scaled to ~[0,1]


def norm_mel(mel, pad_to=None):
    m = (mel.float().numpy() - mean) / std
    if pad_to is None:
        return m
    out = np.full((N_MELS, pad_to), pad_val, dtype=np.float32)
    T = min(m.shape[1], pad_to)
    out[:, :T] = m[:, :T]
    return out


# ---------------- data: N_TRAIN random training sentences
tr = torch.load(f"{DATA_DIR}/train.pt", weights_only=False)
idx = random.Random(0).sample(range(len(tr["texts"])), N_TRAIN)
X = np.stack([feats(tr["texts"][i]) for i in idx])                        # (N, MAX_TEXT)
Y = np.stack([norm_mel(tr["mels"][i], T_MAX)[:, ::STRIDE] for i in idx])  # (N, 80, T_S)
T_S = Y.shape[2]
Y = Y.reshape(N_TRAIN, -1).astype(np.float64)                             # (N, 80*T_S)
print(f"training sentences: {N_TRAIN} | outputs: {Y.shape[1]} SVRs ({N_MELS} bins x {T_S} steps)")

# ---------------- fit
t0 = time.time()
model = svr_core.fit(X, Y, degree=DEGREE, C=C, eps=EPS)
print(f"fit done in {time.time() - t0:.0f}s")
svr_core.save(model, "svr_model.npz")
json.dump({"stride": STRIDE, "n_train": N_TRAIN, "degree": DEGREE, "C": C, "epsilon": EPS},
          open("svr_meta.json", "w"))

# ---------------- quick validation: masked L1 on normalized mels (same measure as the MLP's val loss)
va = torch.load(f"{DATA_DIR}/val.pt", weights_only=False)
mean_frame = (np.array(stats["mean_frame"], dtype=np.float32)[:, None] - mean) / std
n_val = min(100, len(va["texts"]))
l1_svr = l1_mean = cnt = 0.0
for i in range(n_val):
    tgt = norm_mel(va["mels"][i])[:, :T_MAX]
    T = tgt.shape[1]
    y = svr_core.predict(model, feats(va["texts"][i])[None])[0]
    pred = svr_core.to_frames(y, N_MELS, STRIDE, T)
    l1_svr += np.abs(pred - tgt).sum()
    l1_mean += np.abs(mean_frame - tgt).sum()
    cnt += tgt.size
print(f"validation masked L1 on {n_val} clips | SVR: {l1_svr / cnt:.4f} | mean-mel: {l1_mean / cnt:.4f}")
print("compare with the MLP's best validation loss (~0.706 in your log)")
print("saved svr_model.npz and svr_meta.json")
