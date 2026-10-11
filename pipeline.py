import time
import numpy as np
from vocoder import Vocoder, prepare_mel

# Person 1 must provide this: text_to_mel(text: str) -> array of shape (80, T)
from person1_infer import text_to_mel

# Set these only if Person 1 normalizes mels (he gives you the values / stats file)
MEAN, STD = None, None
# e.g. stats = np.load("mel_stats.npy", allow_pickle=True).item(); MEAN, STD = stats["mean"], stats["std"]

_vocoder = None


def synthesize(text, out_path="out.wav"):
    """text -> wav file. Returns (wav, timings dict) so evaluation can reuse it."""
    global _vocoder
    if _vocoder is None:
        _vocoder = Vocoder()

    t0 = time.perf_counter()
    mel = text_to_mel(text)
    t1 = time.perf_counter()

    mel = prepare_mel(mel, MEAN, STD)
    wav = _vocoder.mel_to_wav(mel)
    t2 = time.perf_counter()

    _vocoder.save(wav, out_path)
    dur = _vocoder.duration(wav)
    return wav, {
        "audio_sec": dur,
        "seq2seq_sec": t1 - t0,
        "hifigan_sec": t2 - t1,
        "rtf_seq2seq": (t1 - t0) / dur,
        "rtf_hifigan": (t2 - t1) / dur,
    }


if __name__ == "__main__":
    text = input("Text: ")
    _, info = synthesize(text)
    print(f"saved out.wav | {info['audio_sec']:.2f}s audio | "
          f"RTF seq2seq {info['rtf_seq2seq']:.3f}, hifigan {info['rtf_hifigan']:.3f}")