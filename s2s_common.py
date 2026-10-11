from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mlp_common import MAX_TEXT, N_MELS


R = 3
LOG_FLOOR = -11.5129


class Tokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.pad_id = 0
        self.char_to_id = {char: index + 1 for index, char in enumerate(self.chars)}
        self.unk_id = len(self.chars) + 1
        self.eos_id = len(self.chars) + 2
        self.vocab_size = self.eos_id + 1

    def encode(self, text):
        text = text.lower()
        if len(text) + 1 > MAX_TEXT:
            raise ValueError(f"text plus EOS exceeds the {MAX_TEXT}-token encoder limit")
        ids = [self.char_to_id.get(char, self.unk_id) for char in text]
        ids.append(self.eos_id)
        return ids


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, conv_channels=256, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim if i == 0 else conv_channels,
                      conv_channels, kernel_size=5, padding=2)
            for i in range(3)
        )
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(conv_channels) for _ in range(3)
        )
        self.lstm = nn.LSTM(
            conv_channels, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_size = hidden_size * 2

    def forward(self, tokens, debug=False):
        lengths = tokens.ne(0).sum(dim=1).clamp(min=1)
        embedded = self.embedding(tokens)
        if debug:
            print(f"embedding: {tuple(embedded.shape)}")
        encoded = embedded.transpose(1, 2)
        for index, (conv, batch_norm) in enumerate(
            zip(self.convolutions, self.batch_norms), start=1
        ):
            encoded = F.relu(batch_norm(conv(encoded)))
            if debug:
                print(f"encoder conv{index}: {tuple(encoded.shape)}")
        encoded = encoded.transpose(1, 2)
        packed = pack_padded_sequence(
            encoded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=tokens.shape[1]
        )
        return outputs, lengths


class LocationAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, attention_size=128, location_channels=32):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_size, attention_size, bias=False)
        self.decoder_projection = nn.Linear(decoder_size, attention_size, bias=False)
        self.location_conv = nn.Conv1d(2, location_channels, kernel_size=31, padding=15)
        self.location_projection = nn.Linear(location_channels, attention_size, bias=False)
        self.energy_projection = nn.Linear(attention_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_state, previous, cumulative, mask):
        location_features = torch.stack((previous, cumulative), dim=1)
        location_features = self.location_conv(location_features).transpose(1, 2)
        energies = self.energy_projection(
            torch.tanh(
                self.encoder_projection(encoder_outputs)
                + self.decoder_projection(decoder_state).unsqueeze(1)
                + self.location_projection(location_features)
            )
        ).squeeze(-1)
        energies = energies.float()
        energies = energies.masked_fill(~mask, -1e9)
        weights = torch.softmax(energies, dim=-1)
        context = torch.bmm(
            weights.to(encoder_outputs.dtype).unsqueeze(1), encoder_outputs
        ).squeeze(1)
        return context, weights, cumulative + weights


class Prenet(nn.Module):
    def __init__(self, input_size, hidden_sizes=(256, 128), dropout=0.5):
        super().__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            current_size = hidden_size
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.output_size = current_size

    def forward(self, value):
        for layer in self.layers:
            value = F.dropout(F.relu(layer(value)), p=self.dropout, training=True)
        return value


class DecoderOutput(NamedTuple):
    mel: torch.Tensor
    stop_logits: torch.Tensor
    alignments: torch.Tensor


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_size,
        mel_bins=N_MELS,
        frames_per_step=R,
        prenet_sizes=(256, 128),
        attention_size=256,
        decoder_size=256,
    ):
        super().__init__()
        self.frames_per_step = frames_per_step
        self.mel_bins = mel_bins
        self.prenet = Prenet(mel_bins * frames_per_step, prenet_sizes)
        self.attention_cell = nn.LSTMCell(
            self.prenet.output_size + encoder_size, attention_size
        )
        self.attention = LocationAttention(
            encoder_size, attention_size, attention_size=attention_size
        )
        self.decoder_cell = nn.LSTMCell(attention_size + encoder_size, decoder_size)
        self.mel_projection = nn.Linear(decoder_size + encoder_size,
                                        frames_per_step * mel_bins)
        self.stop_projection = nn.Linear(decoder_size + encoder_size, 1)

    def forward(
        self,
        encoder_outputs,
        encoder_lengths,
        targets=None,
        target_lengths=None,
        teacher_forcing_probability=1.0,
        max_steps=None,
        stop_threshold=0.5,
        debug=False,
    ):
        batch_size, encoder_steps, encoder_size = encoder_outputs.shape
        device = encoder_outputs.device
        if targets is not None:
            if target_lengths is None:
                raise ValueError("target_lengths are required with teacher-forced targets")
            steps = (targets.shape[2] + self.frames_per_step - 1) // self.frames_per_step
        else:
            if max_steps is None:
                max_steps = 1000
            steps = max_steps

        mask = torch.arange(encoder_steps, device=device)[None, :] < encoder_lengths[:, None]
        attention_hidden = encoder_outputs.new_zeros(batch_size, self.attention_cell.hidden_size)
        attention_cell = torch.zeros_like(attention_hidden)
        decoder_hidden = encoder_outputs.new_zeros(batch_size, self.decoder_cell.hidden_size)
        decoder_cell = torch.zeros_like(decoder_hidden)
        context = encoder_outputs.new_zeros(batch_size, encoder_size)
        previous_weights = encoder_outputs.new_zeros(batch_size, encoder_steps)
        cumulative_weights = torch.zeros_like(previous_weights)
        previous_mel = encoder_outputs.new_zeros(batch_size, self.frames_per_step * self.mel_bins)

        mel_steps, stop_steps, alignment_steps = [], [], []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(steps):
            prenet_output = self.prenet(previous_mel)
            attention_hidden, attention_cell = self.attention_cell(
                torch.cat((prenet_output, context), dim=-1),
                (attention_hidden, attention_cell),
            )
            context, weights, cumulative_weights = self.attention(
                encoder_outputs, attention_hidden, previous_weights,
                cumulative_weights, mask,
            )
            previous_weights = weights
            decoder_hidden, decoder_cell = self.decoder_cell(
                torch.cat((attention_hidden, context), dim=-1),
                (decoder_hidden, decoder_cell),
            )
            projection_input = torch.cat((decoder_hidden, context), dim=-1)
            mel_group = self.mel_projection(projection_input)
            stop_logit = self.stop_projection(projection_input).squeeze(-1)
            if debug and step == 0:
                print(f"prenet: {tuple(prenet_output.shape)}")
                print(f"attention LSTM state: {tuple(attention_hidden.shape)}")
                print(f"attention context: {tuple(context.shape)}")
                print(f"attention weights: {tuple(weights.shape)}")
                print(f"decoder LSTM state: {tuple(decoder_hidden.shape)}")
                print(f"decoder mel group: {tuple(mel_group.shape)}")
                print(f"decoder stop logit: {tuple(stop_logit.shape)}")
            mel_steps.append(mel_group)
            stop_steps.append(stop_logit)
            alignment_steps.append(weights)

            if targets is not None and step + 1 < steps:
                start = step * self.frames_per_step
                next_target = targets[:, :, start:start + self.frames_per_step]
                if next_target.shape[2] < self.frames_per_step:
                    next_target = F.pad(
                        next_target,
                        (0, self.frames_per_step - next_target.shape[2]),
                        value=LOG_FLOOR,
                    )
                teacher_mask = (
                    torch.rand(batch_size, device=device)
                    < teacher_forcing_probability
                ).unsqueeze(1)
                previous_mel = torch.where(
                    teacher_mask, next_target.reshape(batch_size, -1), mel_group
                )
            else:
                previous_mel = mel_group
                finished |= torch.sigmoid(stop_logit) >= stop_threshold
                if targets is None and bool(finished.all()) and step >= 1:
                    break

        mel = torch.stack(mel_steps, dim=1).reshape(
            batch_size, -1, self.mel_bins
        ).transpose(1, 2)
        stop_logits = torch.stack(stop_steps, dim=1)
        alignments = torch.stack(alignment_steps, dim=1)
        return DecoderOutput(mel, stop_logits, alignments)


class Postnet(nn.Module):
    def __init__(self, mel_bins=N_MELS, channels=256, layers=5, kernel_size=5):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for index in range(layers):
            input_channels = mel_bins if index == 0 else channels
            output_channels = mel_bins if index == layers - 1 else channels
            self.convolutions.append(
                nn.Conv1d(input_channels, output_channels, kernel_size,
                          padding=(kernel_size - 1) // 2)
            )
            self.batch_norms.append(nn.BatchNorm1d(output_channels))

    def forward(self, mel):
        value = mel
        for index, (conv, batch_norm) in enumerate(
            zip(self.convolutions, self.batch_norms)
        ):
            value = batch_norm(conv(value))
            if index < len(self.convolutions) - 1:
                value = torch.tanh(value)
            value = F.dropout(value, p=0.5, training=self.training)
        return mel + value


class Seq2SeqTTS(nn.Module):
    def __init__(self, vocab_size, mel_bins=N_MELS):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(self.encoder.output_size, mel_bins)
        self.postnet = Postnet(mel_bins)

    def forward(
        self,
        tokens,
        targets=None,
        target_lengths=None,
        teacher_forcing_probability=1.0,
        max_steps=None,
        stop_threshold=0.5,
        debug=False,
    ):
        encoder_outputs, encoder_lengths = self.encoder(tokens, debug=debug)
        decoded = self.decoder(
            encoder_outputs,
            encoder_lengths,
            targets=targets,
            target_lengths=target_lengths,
            teacher_forcing_probability=teacher_forcing_probability,
            max_steps=max_steps,
            stop_threshold=stop_threshold,
            debug=debug,
        )
        mel_before = decoded.mel
        mel_after = self.postnet(mel_before)
        if debug:
            print(f"tokens: {tuple(tokens.shape)}")
            print(f"encoder: {tuple(encoder_outputs.shape)}")
            print(f"decoder mel before postnet: {tuple(mel_before.shape)}")
            print(f"stop logits: {tuple(decoded.stop_logits.shape)}")
            print(f"alignments: {tuple(decoded.alignments.shape)}")
            print(f"postnet mel: {tuple(mel_after.shape)}")
        return {
            "mel_before": mel_before,
            "mel_after": mel_after,
            "stop_logits": decoded.stop_logits,
            "alignments": decoded.alignments,
            "encoder_lengths": encoder_lengths,
        }
