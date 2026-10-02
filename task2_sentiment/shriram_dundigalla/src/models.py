"""Three sentiment classifiers, all with embeddings learned from scratch.

The three are chosen to differ in *how they use word order*, not just in width, so the
comparative analysis has something to say beyond "more parameters helped":

  baseline_bow   — no order at all. Averages word vectors, so it can only learn which
                   words are positive or negative. Sets the floor that any
                   order-aware model has to beat to justify its cost.
  exp1_bilstm    — full sequential order via a recurrent state, so it can in principle
                   represent "not good" differently from "good ... not".
  exp2_textcnn   — local order only: fixed-width n-gram detectors (2-5 tokens) with
                   global max-pooling, so it finds the strongest phrase anywhere in the
                   review but has no long-range state.

Every model returns a single logit; the loss is BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0


def _masked_mean(x, mask):
    """Mean over real tokens only. Padding must not drag the average toward zero."""
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


class BagOfEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        layers, d = [], embed_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)
        nn.init.normal_(self.embed.weight, std=0.1)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()

    def forward(self, ids, lengths=None):
        mask = (ids != PAD_ID).float()
        pooled = _masked_mean(self.embed(ids), mask)
        return self.out(self.mlp(pooled)).squeeze(-1)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, bidirectional, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=n_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0,
        )
        d = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        # Concatenating max- and mean-pooled states rather than taking the last hidden
        # state: with reviews up to 250 tokens the final state is dominated by the tail
        # of the review, and Yelp verdicts are as often stated up front as at the end.
        self.out = nn.Linear(d * 2, 1)
        nn.init.normal_(self.embed.weight, std=0.1)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()

    def forward(self, ids, lengths=None):
        mask = (ids != PAD_ID)
        x = self.embed(ids)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        else:
            out, _ = self.lstm(x)

        mean = _masked_mean(out, mask.float())
        mx = out.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        mx = torch.nan_to_num(mx, neginf=0.0)
        return self.out(self.dropout(torch.cat([mean, mx], dim=-1))).squeeze(-1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, kernel_sizes, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, n_filters, k, padding=k - 1) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(kernel_sizes), 1)
        nn.init.normal_(self.embed.weight, std=0.1)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()

    def forward(self, ids, lengths=None):
        # (B, T, E) -> (B, E, T) because Conv1d convolves over the last dimension.
        x = self.embed(ids).transpose(1, 2)
        feats = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.out(self.dropout(torch.cat(feats, dim=1))).squeeze(-1)


def build_model(spec: dict, vocab_size: int) -> nn.Module:
    kind = spec["kind"]
    if kind == "bag_of_embeddings":
        return BagOfEmbeddings(vocab_size, spec["embed_dim"], spec["hidden"], spec["dropout"])
    if kind == "bilstm":
        return BiLSTMClassifier(
            vocab_size, spec["embed_dim"], spec["hidden_size"],
            spec["n_layers"], spec["bidirectional"], spec["dropout"],
        )
    if kind == "textcnn":
        return TextCNN(
            vocab_size, spec["embed_dim"], spec["n_filters"],
            spec["kernel_sizes"], spec["dropout"],
        )
    raise ValueError(f"unknown model kind: {kind}")


def architecture_summary(spec: dict, model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kind = spec["kind"]
    if kind == "bag_of_embeddings":
        body = f"mean-pooled embeddings (dim {spec['embed_dim']}) -> MLP {spec['hidden']}"
    elif kind == "bilstm":
        dirs = "bi" if spec["bidirectional"] else "uni"
        body = (f"embeddings (dim {spec['embed_dim']}) -> {dirs}LSTM hidden {spec['hidden_size']} "
                f"x{spec['n_layers']} -> [mean; max] pool")
    else:
        body = (f"embeddings (dim {spec['embed_dim']}) -> Conv1d kernels {spec['kernel_sizes']} "
                f"x{spec['n_filters']} filters -> global max pool")
    return f"{body} -> 1 logit; dropout {spec['dropout']}; {n:,} params"
