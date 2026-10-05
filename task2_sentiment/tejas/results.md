# Task 2 — Yelp Polarity Sentiment Classification (Tejas)

## Data and preprocessing
- **Dataset:** Yelp Polarity, 560,000 train / 38,000 test reviews, perfectly balanced. A 5% stratified validation split is carved from train (used for early stopping); the test set is used only for the final metrics.
- **Cleaning:** missing, empty, or duplicate reviews removed. The check found none: 0 rows dropped from train (560,000 in → 560,000 out) and 0 from test (38,000 in → 38,000 out).
- **Text processing:** HTML unescape, literal "\n" removal, lowercasing, **negation contractions expanded** ("don't" → "do not"), removal of punctuation, digits and special characters, NLTK stopwords **with negations kept** (removing "not" would flip sentiment), WordNet lemmatization (cached per unique word).
- **Tokens:** vocabulary from training data only (min frequency 5, max 50,000), max length = 95th percentile of training length (188 tokens), **head + tail truncation** (reviews often state the verdict at the end).
- **Embeddings:** all learned from scratch (`nn.Embedding`, random initialisation). No pretrained vectors or language models.

## Models (all three differ from Shriram's: no shared model families)

| | Baseline | Exp 1 | Exp 2 |
|---|---|---|---|
| Architecture | **fastText-style n-gram bag:** mean of unigram + hashed-bigram embeddings → linear | **Transformer encoder from scratch:** 2 layers, 4 heads, d = 128, FFN 256, learned positions, pre-LN, masked mean pooling | **BiGRU** (hidden 128) + **attention pooling** |
| Embedding dim | 64 (+200K hashed bigram buckets) | 128 | 128 |
| Dropout / LR / batch | 0.2 / 2e-3 / 256 | 0.1 / 5e-4 / 128 | 0.3 / 1e-3 / 128 |
| Parameters | 15,732,417 | 6,154,113 | 6,063,362 |

Common to all three: Adam, BCE-with-logits loss, gradient clipping at 5.0, at most 5 epochs, early stopping on validation macro-F1 (patience 2), seed 1446.

**Why these three:**
- **Baseline (n-gram bag):** a strong, very fast reference (Joulin et al., 2016). Hashed bigrams capture short phrases like "not good" with no sequence model at all, so it shows how far mostly order-free features get.
- **Exp 1 (Transformer):** self-attention connects any two words directly, e.g. a "but" clause far from the opinion it reverses. It tests whether global context helps at this scale.
- **Exp 2 (BiGRU + attention):** reads the review in order in both directions, and the attention pooling learns which words matter most for the verdict.

## Results (test set, n = 38,000, threshold 0.5)

| Metric | Baseline (n-gram bag) | Exp 1 (Transformer) | **Exp 2 (BiGRU + attn)** |
|---|---|---|---|
| Accuracy | 0.9327 | 0.9365 | **0.9536** |
| 95% bootstrap CI | 0.9301–0.9351 | 0.9341–0.9388 | 0.9514–0.9557 |
| Macro-F1 | 0.9327 | 0.9365 | **0.9536** |
| 95% bootstrap CI | 0.9301–0.9351 | 0.9341–0.9388 | 0.9514–0.9557 |
| Micro-F1 / weighted-F1 | 0.9327 / 0.9327 | 0.9365 / 0.9365 | 0.9536 / 0.9536 |
| ROC-AUC / PR-AUC | 0.9794 / 0.9788 | 0.9843 / 0.9847 | **0.9906 / 0.9909** |
| MCC (95% CI) | 0.8653 (0.8603–0.8703) | 0.8730 (0.8683–0.8777) | **0.9073** (0.9028–0.9114) |
| Brier score | 0.0507 | 0.0477 | **0.0356** |
| Expected calibration error | **0.0072** | 0.0121 | 0.0120 |
| McNemar vs. baseline | (reference) | χ² = 12.36, p = 4.4e-4 | χ² = 330.18, p = 8.8e-74 |
| Best val macro-F1 | 0.9312 | 0.9348 | 0.9527 |
| Training time | **0.55 min** | 2.97 min | 11.99 min |
| Examples / sec | **80,405** | 14,950 | 3,698 |
| Peak GPU memory | 355 MB | 947 MB | 440 MB |
| Hardware | RTX 4090 + Ryzen 9 7950X | same | same |

Precision, recall, and F1 per averaging method, confusion matrices, ROC/PR/reliability plots, and per-slice tables: `outputs/table_*.csv|.md`, `outputs/*_eval_plots.png`, `outputs/confusion_matrices_all.png`, `outputs/slice_error_rates.png`, `outputs/model_comparison.png`.

## Analysis
- **Ranking:** BiGRU > Transformer > n-gram bag on every quality metric, and the ranking holds statistically. The BiGRU's confidence interval doesn't overlap the others', and McNemar shows both experimental models make significantly different (fewer) errors than the baseline. The BiGRU gain is very large (p ≈ 1e-73); the Transformer gain is small but significant.
- **Why micro-F1 = accuracy and weighted-F1 = macro-F1:** for single-label binary classification, micro-averaging pools TP/FP/FN across both classes, which equals accuracy. With exactly balanced test classes, weighting by class size equals the plain average.
- **Calibration:** all three are well calibrated (ECE ≤ 0.012). The n-gram bag is the best calibrated (0.007), while the BiGRU has the best Brier score (0.036) because it is both accurate and confident.
- **Cost vs. accuracy:** parameter counts are dominated by embedding tables, not by the architectures. The BiGRU trains about 22× slower than the n-gram bag (sequential recurrence) for +2.1 accuracy points. The Transformer sits in between, but at this small size (2 layers, trained from scratch, 5 epochs) it doesn't beat the recurrent model.
- **Robustness by slice** (`outputs/table_slices.csv`, error rate baseline / Exp 1 / Exp 2):
  - **Truncated reviews (> 188 tokens, n = 1,843, 4.9% of test) are the hardest slice for every model:** 0.079 / 0.070 / 0.062, vs. 0.067 / 0.063 / 0.046 on non-truncated reviews (macro-F1 0.911 / 0.922 / 0.929 vs. 0.933 / 0.937 / 0.954).
  - **Contrast words** ("but", "however", …, n = 22,365) are harder for every model: 0.075 / 0.072 / 0.051 vs. 0.057 / 0.051 / 0.039 without. The BiGRU narrows the gap (+1.2 points vs. +1.8 for the baseline); the Transformer does not (+2.1).
  - **Negations** (n = 28,247) raise the error rate for the baseline and Transformer (+1.6 and +1.8 points) but barely for the BiGRU (0.047 vs. 0.044, +0.3).
  - **Length quartiles** are nearly flat for every model (spread ≤ 0.8 points), so the length problem is specific to truncation, not to long reviews in general.
- **Error review:** the 20 reviewed errors of the BiGRU (`failure_analysis.md`) are dominated by truncation of long reviews, contrast/mixed sentiment, sarcasm, and label noise from the 1–2★ vs. 3–4★ binarisation and edited reviews.

## Limitations and next steps
- Head+tail truncation loses the middle of long reviews (the weakest slice). The proposed, testable fix is in `failure_analysis.md` (max_len 188 → 400).
- Removing punctuation and capitals discards sarcasm cues ("A++++", "!!!", ALL CAPS).
- Embeddings learned from Yelp only: rare words are weakly represented. Subword tokens would reduce out-of-vocabulary words.
- The Transformer could improve with more layers, more epochs, and warmup.

## Reproducibility
Config in the notebook's `CFG` cell · raw logs (one per model) in `reproducibility/raw_logs/task2_sentiment/tejas/` · manifest and frozen requirements in `reproducibility/manifests/task2_sentiment/tejas/` · checkpoints `checkpoints/{baseline,exp1,exp2}_best.pt` · test probabilities `outputs/*_test_probs.npy` (every metric can be recomputed without retraining).