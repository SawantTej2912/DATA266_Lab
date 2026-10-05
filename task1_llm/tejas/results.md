# Task 1 — GPT-Style Character LLM From Scratch (Tejas)

## What I built
A decoder-only Transformer written from scratch in PyTorch. Multi-head self-attention, causal masking, LayerNorm, and the blocks are all hand-implemented; no `nn.MultiheadAttention`, `nn.Transformer*`, or `scaled_dot_product_attention`. A built-in check confirms the causal mask: changing the last token leaves all earlier logits unchanged.

| Component | Choice |
|---|---|
| Tokenization | Character level, own `char_to_idx` / `idx_to_char`; vocabulary built on training text only (77 symbols; characters seen < 50 times map to one `¿` symbol) |
| Data | Own disjoint split of TinyStories: **100,000 train / 10,000 val** stories (seed 1446), stories joined with `\n\n` |
| Sequences | Non-overlapping 256-character windows; target = input shifted by one character |
| Model | 6 blocks, 6 heads, d_model 384, FFN 4× with GELU, dropout 0.1, **pre-LN**, learned token + positional embeddings, LM head tied to the token embedding |
| Parameters | **10,775,424** |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 on weight matrices only, gradient clipping at 1.0 |
| Schedule | **Linear warmup over 3% of steps, then cosine decay** from 6e-4 to 6e-5 |
| Training | 10 epochs × 5,485 steps (54,850 steps, batch 64 × 256 chars), fp16 mixed precision with GradScaler |

## Why these choices
- **Character level** keeps the vocabulary tiny (77), so all model capacity goes to sequence modelling. It's the setting the brief specifies.
- **Pre-LN** (normalisation before attention and FFN) trains more stably than post-LN and is less sensitive to warmup.
- **Context 256:** attention cost grows with T²; 256 characters covers several sentences, enough for local story structure, while keeping 10 epochs at about 45 minutes.
- **6 × 6 × 384 (~10.8M parameters):** large enough to learn TinyStories' grammar well within the 10-epoch budget; the runtime estimate projected the full run before training.
- **Weight tying** shares the input and output embeddings: fewer parameters and usually lower perplexity.
- **Warmup + cosine:** warmup avoids unstable early updates when Adam's statistics aren't settled yet; cosine decay lets the loss settle at the end.

## Results (best checkpoint = epoch 10)

| Metric | Value |
|---|---|
| Training cross-entropy | 0.5370 |
| Validation cross-entropy | **0.5586** |
| Perplexity (val) | **1.748** |
| Bits per character (val) | **0.806** |
| Generalization gap (val − train) | 0.0216 |
| Top-1 next-character accuracy (val) | **82.1%** |
| Distinct-1 / 2 / 3 (sampled, T = 0.8, 45 samples) | 0.273 / 0.693 / 0.890 |
| Repeated 4-gram rate (sampled / greedy) | 0.004 / 0.080 |
| Gradient norm (mean / max) | 0.28 / 16.8 (the max is at step 0) |
| Loss spikes / non-finite gradient steps | 0 / 18 (see stability) |
| Training throughput | 357,646 tokens/s |
| Generation throughput | 256.7 tokens/s (no KV cache) |
| Peak GPU memory | 3,324 MB |
| Total training time | 45.4 min compute (47.4 min wall clock) |
| Hardware | NVIDIA GeForce RTX 4090 (24 GB), AMD Ryzen 9 7950X |

**Per-epoch validation loss:** 0.709 → 0.657 → 0.631 → 0.613 → 0.601 → 0.588 → 0.577 → 0.568 → 0.562 → **0.559**.
Plots: `outputs/loss_curves.png` (training loss, train vs. val per epoch, gradient norm, learning-rate schedule) · table: `outputs/epoch_history.csv`.

## Analysis
- **Convergence:** validation loss fell every epoch with no plateau. It was still improving at epoch 10 (−0.004 in the last epoch), so more epochs would likely help a little.
- **Generalization:** the train/val gap is only 0.022 nats, so there's no meaningful overfitting. With 100K stories and one pass per epoch over non-overlapping windows, the model sees each character once per epoch.
- **Interpreting the numbers:** perplexity 1.75 means the model is, on average, about as uncertain as choosing between ~1.75 characters. 0.81 bits/char compares with log₂(77) ≈ 6.3 bits for a uniform guess. Most remaining errors are at word starts, where many continuations are valid; mid-word characters are nearly deterministic.
- **Stability:** 0 loss spikes and 0 NaN losses. The 18 "non-finite" steps (0.03% of 54,850) are fp16 **gradient overflows** that the GradScaler detects, skips, and recovers from by lowering its scale. That's normal mixed-precision behaviour, not instability. The gradient norm stayed around 0.28 after the first step.
- **Schedule:** the learning rate warmed up linearly over the first ~1,650 steps, then followed the cosine curve down to exactly 10% of peak (6.0e-5) at the end, as designed.
- **Generation:** greedy outputs are fluent but repetitive (repeated-4-gram rate 0.080) and collapse onto a common story template. Temperature sampling is 20× less repetitive and more diverse (Distinct-2 0.69 vs. 0.63) but makes more semantic errors. See `failure_analysis.md`.

## Limitations and next steps
- The 256-character context is shorter than a 300-character generation, so the prompt eventually falls out of view (one cause of story resets).
- Character-level modelling captures syntax well but weak semantics (hallucinated, illogical events).
- Next: train longer (still improving), a KV cache for faster generation, top-k / nucleus sampling and a repetition penalty, a longer context, or subword tokenization.

## Reproducibility
Config in the notebook's `CFG` cell · raw log `reproducibility/raw_logs/task1_llm/tejas/t1_gpt_tejas_20261001-144225.jsonl` · manifest and frozen requirements in `reproducibility/manifests/task1_llm/tejas/` · checkpoint `checkpoints/best.pt` (weights only) · smoke test: `SMOKE=1` (see the repo README).