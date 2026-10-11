# Text-to-Speech Acoustic Model Comparison

This project compares text-to-mel acoustic models for single-speaker English
speech synthesis on the LJ Speech dataset. Every acoustic model predicts the
same 80-bin log-mel representation and uses the same pretrained HiFi-GAN
generator to synthesize waveform audio. This keeps the acoustic model as the
main experimental variable.

The systems are:

1. **SVR baseline** — polynomial-kernel support vector regression, with one
   output regressor per mel-bin/frame value.
2. **MLP baseline** — character embedding followed by a feed-forward network
   that predicts a fixed-length mel.
3. **Seq2Seq + attention** — character encoder with convolutional and
   bidirectional recurrent layers, location-sensitive attention, an
   autoregressive decoder, and a postnet.
4. **Mean-mel baseline** — the train-set average mel frame repeated for a
   text-dependent duration.

The project uses mel-spectrogram prediction followed by a pretrained vocoder.
It does not use the original paper's acoustic-output implementation.

## Requirements

- Python 3.13 and PyTorch 2.x are the tested target environment.
- NumPy, SciPy, librosa, scikit-learn, joblib, and matplotlib are used by the
  project scripts.
- The pretrained HiFi-GAN V1 generator checkpoint must be available as
  `generator_v1` beside the scripts, along with `config_v1.json`.
- LJ Speech must be extracted so that `metadata.csv` and `wavs/` are available.

**Do not install this repository's upstream `requirements.txt` for this
project.** It pins old versions, including `torch==1.4.0`, which is incompatible
with the current project environment.

The scripts are intended to be run with the HiFi-GAN project directory as the
working directory:

```powershell
cd C:\path\to\ML_Proj\hifi-gan
```

`mlp_prep.py` currently expects the dataset at
`..\LJSpeech-1.1\metadata.csv` and `..\LJSpeech-1.1\wavs\`. Keep that layout or
adjust `DATASET_DIR` in `mlp_prep.py` for your machine.

## Audio and mel interface

The acoustic models and vocoder use these shared features:

| Setting | Value |
|---|---:|
| Sample rate | 22,050 Hz |
| FFT / window | 1,024 samples |
| Hop | 256 samples |
| Mel bins | 80 |
| Mel frequency range | 0-8,000 Hz |
| Mel values | Natural-log magnitude |
| Silence/padding value | `log(1e-5)` = `-11.5129` |

Mels have shape `(80, T)`. They are computed using the repository's
`meldataset.mel_spectrogram`; do not replace this with a different mel
implementation if the pretrained HiFi-GAN checkpoint is to remain compatible.
Training can normalize mels using train-set scalar mean and standard deviation.
Inference functions return denormalized log-mels before passing them to the
vocoder.

## Data preparation and shared split

Run once to build the shared split and precompute the mel features:

```powershell
python mlp_prep.py
```

Optional split override:

```powershell
python mlp_prep.py --split-file C:\path\to\split.json
```

The default split is seeded with 42: 90% train, 5% validation, and 5% test. The
split is saved in `mlp_data/split.json`; precomputed float16 mels and their
matching text/IDs are saved in `mlp_data/train.pt`, `val.pt`, and `test.pt`.
`mlp_data/stats.json` contains the train-only character vocabulary, scalar mel
mean/std, per-bin mean frame, and frames-per-character statistic.

The MLP and Seq2Seq training scripts consume these same prepared files and do
not make their own random data split. The SVR trainer also draws its training
examples from the shared training file. Do not regenerate the split between
experiments if you want results to remain comparable.

Note that the default training-data amounts are not equal: the MLP and Seq2Seq
use the available training set (unless `MAX_FRAMES` filters clips from
Seq2Seq), while the SVR samples 200 training sentences by default. This is a
deliberate small-data SVR baseline, not a compute- or data-matched comparison.
For a comparison at matched training-set size, set `SVR_N` to the number of
training examples used by the other models; fitting that many per-output SVRs
can require substantial time and memory. Record this setting with reported
results.

## Train the baselines

### SVR

```powershell
python train_svr.py
```

By default the SVR trains on 200 randomly selected sentences from the shared
training split (`SVR_N=200`). It fits polynomial-kernel SVRs for mel outputs and
prints a masked validation L1 comparison for up to 100 validation clips.
Optional environment variables:

```powershell
$env:SVR_N = "200"
$env:SVR_STRIDE = "1"
python train_svr.py
```

`SVR_STRIDE=1` predicts every frame; larger strides reduce the output size and
interpolate between predicted frames. The model is saved as `svr_model.npz`
with its settings in `svr_meta.json`.

### MLP

```powershell
python train_mlp.py
```

The default run is 40 epochs with batch size 32. The trainer normalizes mel
targets using train-set statistics, masks padded frames in the L1 loss, clips
gradients, and selects the checkpoint with the best validation loss. It writes
`mlp_best.pt` and `mlp_loss.png`.

For a short run or different hyperparameters:

```powershell
python train_mlp.py --epochs 1 --batch-size 32 --learning-rate 0.0005
```

## Train Seq2Seq + attention

```powershell
python train_s2s.py
```

The default is 100 epochs. The trainer uses length-bucketed training batches,
teacher forcing, masked mel losses before and after the postnet, stop-token
loss, and guided attention regularization. It writes `s2s_best.pt`,
`s2s_last.pt`, `s2s_loss.png`, and `s2s_align.png`. If a last checkpoint is
available, training resumes from it.

Environment variables supported by the trainer:

| Variable | Default | Meaning |
|---|---:|---|
| `BATCH` | `32` | Training batch size |
| `MAX_FRAMES` | `0` | Drop training clips longer than this many mel frames; `0` disables filtering |
| `GUIDED_W` | `0.5` | Guided-attention loss weight; `0` disables it |
| `AMP` | `1` | Enable CUDA float16 autocast and gradient scaling; `0`, `false`, `no`, or `off` disables it |

For example, in Colab:

```bash
BATCH=32 MAX_FRAMES=0 GUIDED_W=0.5 AMP=1 python train_s2s.py
```

In the Colab-oriented trainer, `s2s_last.pt` is written to local
`/content/s2s_last.pt` each epoch and copied beside the script every three
epochs and at the final epoch. Improved `s2s_best.pt` checkpoints and training
plots are written beside the script. For Drive persistence, run with the
project directory on mounted Drive and ensure it is writable.

## Synthesize audio

Each inference script includes a command-line example and uses the same
pretrained HiFi-GAN V1 vocoder:

```powershell
python svr_infer.py "The weather is beautiful today."
python mlp_infer.py "The weather is beautiful today."
python person1_infer.py "The weather is beautiful today."
```

These create `svr_test.wav`, `mlp_test.wav`, and `s2s_test.wav`. The Seq2Seq
command also creates `s2s_alignment.png`. Its checkpoint, `s2s_best.pt`, must
exist; run the trainer first.

The MLP inference module exposes:

```python
from mlp_infer import text_to_mel_mlp, text_to_mel_mean
```

The SVR module exposes `text_to_mel_svr(text)`, and the Seq2Seq module exposes
`text_to_mel(text)`. Each returns a denormalized `(80, T)` log-mel array.

`vocoder.py` contains the `Vocoder` wrapper and `prepare_mel` validation helper.
To smoke-test the pretrained vocoder with a recording or mel file:

```powershell
python test_mel.py ..\LJSpeech-1.1\wavs\LJ001-0002.wav
```

This writes `mel_test_out.wav`.

## Comparing the systems

### Acoustic-model comparison

The controlled comparison is:

```text
same input text -> SVR / MLP / Seq2Seq mel prediction -> same HiFi-GAN -> audio
```

For final listening, use the shared **test** prompts; use validation only for
model selection. Keep the text prompts, mel convention, pretrained vocoder
checkpoint, and playback conditions identical. The mean-mel baseline can be generated with
`text_to_mel_mean` from `mlp_infer.py`. Ground-truth recordings and
ground-truth mel passed through HiFi-GAN are useful reference conditions:
they help separate acoustic-model errors from the vocoder's reconstruction
quality.

For objective checks, `train_svr.py` reports masked validation L1, and the
training scripts report their validation losses. These losses are useful
diagnostics, but they are not human ratings and do not by themselves establish
perceived speech quality.

### Subjective quality (MOS)

The subjective protocol uses a **1-to-5 Mean Opinion Score (MOS)** scale.
Present outputs from each system for the same held-out test prompts in
randomized, blind order, and ask listeners to rate perceived audio quality. Include the
ground-truth recording and (when available) ground-truth-mel-to-HiFi-GAN audio
as references. Report the number of listeners and prompts, the rating
instructions, and each system's mean score with uncertainty (for example,
standard deviation or a confidence interval).

The repository provides model inference and the audio needed for this
comparison, but does not currently include a listener-rating interface or
automated MOS aggregation. Collect actual listener scores before reporting
MOS; do not infer them from validation loss.

## Main project files

| File | Purpose |
|---|---|
| `mlp_prep.py` | Creates the shared split, precomputes HiFi-GAN-compatible mels, and writes train statistics |
| `mlp_common.py` | Shared constants, character tokenizer, and MLP architecture |
| `train_svr.py`, `svr_core.py` | SVR baseline training and kernel-regression implementation |
| `svr_infer.py` | SVR mel inference and WAV smoke test |
| `train_mlp.py` | MLP training and best-validation checkpoint |
| `mlp_infer.py` | MLP and mean-mel inference |
| `s2s_common.py` | Seq2Seq tokenizer, encoder, attention, decoder, postnet, and model |
| `train_s2s.py` | Seq2Seq training, checkpointing, and plots |
| `person1_infer.py` | Seq2Seq mel inference, WAV synthesis, and alignment plot |
| `vocoder.py` | HiFi-GAN generator wrapper and mel validation |
| `meldataset.py` | HiFi-GAN's mel-spectrogram implementation used for compatible features |
| `config_v1.json`, `generator_v1` | Pretrained HiFi-GAN V1 configuration and generator checkpoint |

Upstream HiFi-GAN training and inference utilities (`train.py`, `inference.py`,
and `inference_e2e.py`) remain in the folder, but they are not the text-to-mel
acoustic-model comparison pipeline described above.
