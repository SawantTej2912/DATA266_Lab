"""A decoder-only GPT written from scratch for Task 1.

Nothing here comes from `nn.Transformer`, `nn.TransformerEncoderLayer`,
`nn.MultiheadAttention` or `F.scaled_dot_product_attention`. The attention maths is
spelled out so the causal mask and the head split are visible, which is what the task
is checking. The only `torch.nn` pieces used are the generic building blocks:
`Linear`, `Embedding`, `LayerNorm` and `Dropout`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 256
    d_ff_mult: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_head={self.n_head}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    One `Linear` produces Q, K and V for every head at once and the result is split
    afterwards; that is arithmetically identical to 3*n_head separate projections but
    it is a single matmul instead of many small ones.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular mask, kept as a buffer so it moves with .to(device) but is
        # not a parameter. Row t may only attend to columns 0..t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.shape

        # (B, T, 3C) -> three (B, n_head, T, d_head) tensors.
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot-product scores: (B, n_head, T, T).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal masking happens before the softmax, so the masked positions receive
        # exactly zero probability rather than a small one.
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                    # (B, n_head, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))

        return (out, attn) if return_attn else (out, None)


class FeedForward(nn.Module):
    """Position-wise FFN: expand, GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d_ff = cfg.d_ff_mult * cfg.d_model
        self.fc = nn.Linear(cfg.d_model, d_ff)
        self.proj = nn.Linear(d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm Transformer block.

    Pre-LN (norm inside the residual branch) rather than the Post-LN of the original
    paper: with Post-LN the residual stream is renormalised every layer, which makes
    deep stacks need a long warm-up to avoid diverging. Pre-LN leaves a clean identity
    path from input to output, so gradients reach the early layers directly.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, return_attn: bool = False):
        attn_out, attn_w = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_w


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

        # GPT-2's residual scaling: the two projections that write into the residual
        # stream are scaled down by sqrt(2 * n_layer), so the stream's variance does
        # not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x, _ = block(x)

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 repetition_penalty: float = 1.0, penalty_window: int = 64):
        """Autoregressive sampling. temperature <= 0 is treated as greedy decoding.

        The three truncation controls attack different failure modes, which is why all
        three are here rather than one:

        `top_k` caps the candidate set at a fixed size. `top_p` (nucleus sampling,
        Holtzman et al. 2020) caps it by cumulative probability instead, which adapts:
        mid-word, where the next character is nearly determined, the nucleus is one or
        two characters, while at a word boundary it opens up. For a 98-character
        vocabulary that distinction matters more than it does for word models, because
        the entropy varies enormously between positions.

        `repetition_penalty` (Keskar et al. 2019) divides the logits of characters
        already present in the recent context, which targets the degenerate loop this
        model actually falls into at low temperature -- see failure_analysis.md. It is
        applied over a window rather than the whole context: at character level the
        common letters recur constantly by necessity, so penalising every character
        ever emitted would just suppress `e` and `t` and damage the text.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Context is cropped to block_size because positional embeddings only
            # exist for that many positions.
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                recent = idx[:, -penalty_window:]
                for row in range(logits.size(0)):
                    seen = torch.unique(recent[row])
                    vals = logits[row, seen]
                    # Divide positive logits, multiply negative ones, so the penalty
                    # always moves a score *down* regardless of its sign.
                    logits[row, seen] = torch.where(
                        vals > 0, vals / repetition_penalty, vals * repetition_penalty)

            if temperature <= 0:
                next_idx = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    order = logits.argsort(dim=-1, descending=True)
                    sorted_logits = logits.gather(-1, order)
                    cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    # Shift so the token that crosses the threshold is kept; otherwise
                    # top_p below the argmax probability would mask everything.
                    drop = cum - F.softmax(sorted_logits, dim=-1) >= top_p
                    drop[:, 0] = False
                    logits = logits.masked_fill(
                        drop.gather(-1, order.argsort(dim=-1)), float("-inf"))
                next_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

            idx = torch.cat([idx, next_idx], dim=1)
        return idx
