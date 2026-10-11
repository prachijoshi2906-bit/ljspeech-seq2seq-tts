import torch.nn as nn

N_MELS = 80
T_MAX = 880            # max mel frames (LJ Speech clips are <= ~870)
MAX_TEXT = 200         # max characters per sentence
LOG_FLOOR = -11.5129   # log(1e-5), the "silence" value in log-mel
DATA_DIR = "mlp_data"


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.pad_id = 0
        self.c2i = {c: i + 1 for i, c in enumerate(self.chars)}
        self.eos_id = len(self.chars) + 1
        self.vocab_size = self.eos_id + 1

    @staticmethod
    def build(texts):
        return CharTokenizer(sorted(set("".join(texts))))

    def encode(self, text, add_eos=False):
        text = text.lower()[:MAX_TEXT - int(add_eos)]
        ids = [self.c2i.get(c, self.pad_id) for c in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids + [0] * (MAX_TEXT - len(ids))


class MelMLP(nn.Module):
    def __init__(self, n_tokens, emb=64, hidden=512, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, emb, padding_idx=0)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MAX_TEXT * emb, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, N_MELS * T_MAX),
        )

    def forward(self, ids):                       # ids: (B, MAX_TEXT)
        return self.net(self.emb(ids)).view(-1, N_MELS, T_MAX)