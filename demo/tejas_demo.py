"""Inference-only demo of Tejas's three models (no training). Run from the repository root.

    python demo/tejas_demo.py task1 --prompt "Once upon a time" --n 300 --temperature 0.8
    python demo/tejas_demo.py task2 --review "The food was cold and the waiter was rude."
    python demo/tejas_demo.py task3 --direction photo2monet --image task3_gan/data/photo_jpg/<file>.jpg

Model classes are copied from the notebooks in task*/tejas/src/.
"""
import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")


# ----------------------------------------------------------------------------- task 1
# From task1_llm/tejas/src/task1_llm.ipynb (vocabulary and model cells).

UNK = "\u00bf"


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        mu = xf.mean(-1, keepdim=True)
        var = xf.var(-1, keepdim=True, unbiased=False)
        return (self.gamma * (xf - mu) / torch.sqrt(var + self.eps) + self.beta).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.hd = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop, self.resid_drop = nn.Dropout(dropout), nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1, self.ln2 = LayerNorm(n_embd), LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, tie_weights=True):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0):
    """temperature == 0 -> greedy decoding; otherwise temperature sampling."""
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size:])[:, -1, :].float()
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def run_task1(args):
    ck = torch.load(REPO / "task1_llm/tejas/checkpoints/best.pt", map_location="cpu", weights_only=False)
    cfg, char_to_idx = ck["cfg"], ck["char_to_idx"]
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    model = GPT(len(char_to_idx), cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"],
                cfg["dropout"], cfg.get("tie_weights", True))
    model.load_state_dict(ck["model"])
    model.to(DEVICE).eval()
    print(f"Loaded best.pt (epoch {ck['epoch']}, val CE {ck['val_loss']:.4f}) on {DEVICE}")

    torch.manual_seed(args.seed)
    unk = char_to_idx[UNK]
    ctx = torch.tensor([[char_to_idx.get(c, unk) for c in args.prompt]], device=DEVICE)
    out = generate(model, ctx, args.n, args.temperature)
    mode = "greedy" if args.temperature == 0 else f"T = {args.temperature}"
    print(f"--- {mode}, {args.n} new characters ---")
    print("".join(idx_to_char[int(i)] for i in out[0].tolist()))


# ----------------------------------------------------------------------------- task 2

def run_task2(args):
    # exp2_best.pt holds only the weights and model config. The vocabulary (45,818 tokens)
    # was written to task2_sentiment/tejas/data_processed/, which is gitignored, so a raw
    # review cannot be encoded the way the model was trained without re-running preprocessing.
    vocab_path = REPO / "task2_sentiment/tejas/data_processed"
    print("Task 2: live classification is not available.")
    print(f"  exp2_best.pt has no vocabulary, and {vocab_path.relative_to(REPO)}/ is not in the repository.")
    if args.review:
        print(f"  Review not classified: {args.review!r}")
    print("  Test-set metrics from task2_sentiment/tejas/metrics_report.csv (38,000 reviews):\n")
    cols = [("accuracy", "accuracy"), ("f1_macro", "macro-F1"), ("roc_auc", "ROC-AUC"),
            ("mcc", "MCC"), ("brier", "Brier"), ("ece", "ECE")]
    with open(REPO / "task2_sentiment/tejas/metrics_report.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {'model':<10}" + "".join(f"{label:>10}" for _, label in cols))
    for r in rows:
        print(f"  {r['model']:<10}" + "".join(f"{float(r[c]):>10.4f}" for c, _ in cols))


# ----------------------------------------------------------------------------- task 3
# From task3_gan/tejas/src/task3_cyclegan.ipynb (model cell and eval transform).

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch))

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_res=9, upsample="resize"):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResBlock(ch) for _ in range(n_res)]
        for _ in range(2):
            if upsample == "resize":
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.ReflectionPad2d(1),
                           nn.Conv2d(ch, ch // 2, 3), nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            else:
                layers += [nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_image(path, size):
    """Same as the notebook's eval transform: bicubic resize of the short side, centre crop, scale to [-1, 1]."""
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - 0.5) / 0.5


def save_pair(x, y, path):
    import numpy as np
    from PIL import Image
    pair = torch.cat([x, y], dim=2).add(1).div(2).clamp(0, 1)
    Image.fromarray((pair.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(path)


def run_task3(args):
    # G_AB: Monet -> Photo, G_BA: Photo -> Monet
    name = "G_AB.pt" if args.direction == "monet2photo" else "G_BA.pt"
    ck = torch.load(REPO / "task3_gan/tejas/checkpoints" / name, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    G = ResnetGenerator(cfg["ngf"], cfg["n_res_blocks"], cfg.get("upsample", "deconv"))
    G.load_state_dict(ck["model"])
    G.to(DEVICE).eval()
    print(f"Loaded {name} (EMA, epoch {ck['epoch']}) on {DEVICE}")

    img_path = Path(args.image)
    x = load_image(img_path, cfg["img_size"]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = G(x)
    out = Path(args.out) if args.out else img_path.with_name(f"{img_path.stem}_{args.direction}.png")
    save_pair(x[0], y[0], out)
    print(f"Saved input | output to {out}")


def main():
    ap = argparse.ArgumentParser(description="Inference-only demo of Tejas's models.")
    sub = ap.add_subparsers(dest="task", required=True)
    p1 = sub.add_parser("task1", help="generate text with the character-level GPT")
    p1.add_argument("--prompt", default="Once upon a time")
    p1.add_argument("--n", type=int, default=300, help="number of new characters")
    p1.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p1.add_argument("--seed", type=int, default=1446)
    p2 = sub.add_parser("task2", help="sentiment classifier (prints test metrics)")
    p2.add_argument("--review", default=None)
    p3 = sub.add_parser("task3", help="translate one image with the CycleGAN")
    p3.add_argument("--direction", choices=["photo2monet", "monet2photo"], required=True)
    p3.add_argument("--image", required=True)
    p3.add_argument("--out", default=None, help="output path (default: next to the input image)")
    args = ap.parse_args()
    {"task1": run_task1, "task2": run_task2, "task3": run_task3}[args.task](args)


if __name__ == "__main__":
    main()
