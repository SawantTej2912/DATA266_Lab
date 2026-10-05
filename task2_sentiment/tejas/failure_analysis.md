# Task 2 — Error Review (Tejas)

**Model reviewed:** Exp 2, BiGRU (hidden 128) with attention pooling. This is my best model: test accuracy / macro-F1 0.9536, MCC 0.907, ROC-AUC 0.991 on the 38,000-review Yelp Polarity test set. It was selected automatically as the highest test macro-F1.
**Preprocessing:** lowercasing, removal of punctuation, digits and special characters, NLTK stopwords **with negations kept**, WordNet lemmatization, vocabulary built from training data only, max length = 95th percentile of training length (188 tokens) with head+tail truncation.
**Decision threshold:** 0.5 on P(positive).

**Selection (20 errors, from `outputs/error_candidates.csv`):** 5 confident false positives (highest P(pos) among true negatives), 5 confident false negatives (lowest P(pos) among true positives), 5 near-threshold errors (|p − 0.5| smallest), and 5 errors from the weakest data slice, **truncated reviews** (longer than 188 tokens, ~5% of the test set).

**Dataset note:** Yelp Polarity labels 1–2★ reviews as negative and 3–4★ as positive (Zhang et al., 2015). Reviews near the 2★/3★ boundary are often mild or mixed in tone, so their text only weakly reflects the label. This explains several of the "confident" errors below.

---

## 1. Confident false positives (true = negative, predicted positive)

| # | p(pos) | Snippet | Error type | Observation |
|---|---|---|---|---|
| FP1 | 1.000 | "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!" | Label noise / rating–text mismatch | The text is unambiguously positive; the low star rating isn't reflected in the words at all. No text-only model can recover this label. |
| FP2 | 0.999 | "Very small portions, but good food. Had a perfectly cooked fillet minion, but $60 and they could have served with a toothpick…" | Mixed sentiment (quality vs. value) | Food quality is praised ("good", "perfectly cooked", "excellent salmon"); the negative is about price and portion size, expressed through sarcasm ("served with a toothpick") rather than negative words. |
| FP3 | 0.999 | "Do you come here often? This old line would definitely work here. Great for families… entertainingly, loud, home away from home…" | Sarcasm / irony | Surface vocabulary is positive ("great", "laugh-out-loud", "home away from home") but the framing is ironic. The model has no notion of tone. |
| FP4 | 0.999 | "my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!" | Rating–text mismatch (mild praise, low rating) | Almost every sentence is positive; the only complaint is mild. A low-star review written in a polite tone looks positive to a bag of word features. |
| FP5 | 0.999 | "Though I'm a Copper enthusiast… Maharani was a cheaper but tasty option… Copper is definitely still my place, but Maharani was fine enough." | Comparative / lukewarm | Strong praise is directed at a **competitor** ("enthusiast", "definitely still my place"), and the reviewed restaurant only gets "fine enough". The model doesn't know which entity the sentiment targets. |

## 2. Confident false negatives (true = positive, predicted negative)

| # | p(pos) | Snippet | Error type | Observation |
|---|---|---|---|---|
| FN1 | 0.000 | "EDIT: They really did change the service up since I last posted this. Horrible service… We just had an altercation with a server…" | Edited review (temporal mismatch) | The text was rewritten negatively after the fact, but the star rating still reflects the original positive review. |
| FN2 | 0.000 | "UPDATED. My initial very frustrated and dramatic review read as follows: Billing practices are at best negligent and at worst fraudulent…" | Edited review (quotes earlier negative review) | The reviewer **quotes** their old negative review before (presumably) explaining the resolution. The quoted negative text dominates the input. |
| FN3 | 0.000 | "This is more of appreciation than an actual review. As a former long time Az state employee health care was extremely frustrating…" | Negative background framing | The opening sets up a frustrating backstory before the praise. The model weighs the frequent negative background words ("frustrating", "not an improvement") over the stated appreciation. |
| FN4 | 0.001 | "TERRIBLE SERVICE, RUDE WAITERS WITH A PISS POOR ATTITUDE! WOULD EAT HERE AGAIN! A++++" | Sarcasm / irony | Strongly negative words with a positive verdict. Preprocessing removes cues that might help: capitals are lowercased and "A++++" / "!" are stripped as punctuation. |
| FN5 | 0.001 | "Look, we all know Cox sucks. In fact they are a terrible business… However they are a necessary evil if you want Internet that isn't garbage…" | Contrast (negative words, positive verdict) | The positive conclusion comes after "However" and is itself phrased with negative words ("necessary evil", "isn't garbage"). Almost every content word is negative. |

## 3. Near-threshold errors (|p − 0.5| < 0.01)

| # | p(pos) | true | Snippet | Error type | Observation |
|---|---|---|---|---|---|
| NT1 | 0.499 | pos | "It's hard to find a non corporate coffee outlet… Good coffee, not just a conversation piece… No pastries offered here…" | Mild sentiment with negations | Moderate praise mixed with negated phrases ("non corporate", "not just", "no pastries") that look negative out of context. |
| NT2 | 0.502 | neg | "Fancy looks, fancy prices. Not a whole lot of options… The pork was good, but just not a whole lot of it." | Mixed sentiment | Genuinely balanced text: a positive quality statement and negative value statements. A borderline 2★ review. |
| NT3 | 0.498 | pos | "Danielle's professionalism and quality… deserve 5 starshowever the procedures used to estimate and charge… are in need of improvement." | Contrast + tokenization artifact | A contrastive review ("however"), plus a missing space in the source text ("starshowever") that turns two informative words into one out-of-vocabulary token, so both the "5 stars" cue and the contrast marker are lost. |
| NT4 | 0.498 | pos | "I would recommend OBGYN Specialist to anyone!!… I recently suffered an ectopic pregnancy…" | Negative-topic domain vocabulary | Strong praise ("recommend to anyone", "I matter to them") is offset by medical words that are negative in general sentiment ("suffered", "ectopic") but describe the situation, not the business. |
| NT5 | 0.498 | pos | "I could hear some squealing… I thought I needed the brake pads replaced… these guys were honest enough to tell me…" | Implicit sentiment in a problem narrative | The praise is implicit (honesty, not upselling) and surrounded by problem vocabulary ("squealing", "replaced"). |

## 4. Slice-specific failures: truncated reviews (> 188 tokens)
This was the weakest slice for every model (the BiGRU's error rate there is roughly double its overall error rate).

| # | p(pos) | true | Snippet (opening) | Error type | Observation |
|---|---|---|---|---|---|
| TR1 | 0.378 | pos | "Port Authority (formerly known as PATransit…) operates a fairly extensive network of buses…" | Truncation / long descriptive narrative | Mostly neutral description; the evaluative sentences fall in the middle, which head+tail truncation drops. |
| TR2 | 0.466 | pos | "My husband and I decided to spend an overnight in Downtown Pittsburgh last minute…" | Truncation / long descriptive narrative | A long story setup; the verdict on the hotel is likely in the removed middle section. |
| TR3 | 0.415 | pos | "Thoroughly impressed with this airport. Not that I'm some great world traveler…" | Truncation / long descriptive narrative | The positive verdict is in the first sentence, but it's a tiny fraction of a long, digressive review; the attention pooling spreads weight over the neutral narrative. |
| TR4 | 0.796 | neg | "The Sports Pub is the kind of place I tend to find myself at when I get overruled by my friends… I do know that I like the Ale Asylum A LOT." | Truncation (verdict lost from the middle) | The opening praises a **different** place ("Ale Asylum"); the negative judgement of the Sports Pub is likely in the dropped middle. |
| TR5 | 0.485 | pos | "Summer time is supposed to be 'healthy time', so I am trying to watch what I eat…" | Truncation / long descriptive narrative | A personal backstory dominates the kept text; the opinion about the restaurant is mostly cut. |

---

## Error-type summary

| Error type | Count | Fixable by a text model? |
|---|---|---|
| Truncation / long narrative | 5 | ✅ Yes, with longer input (see the fix below) |
| Contrast / mixed / lukewarm / comparative | 6 | Partly: needs better clause-level modelling |
| Sarcasm / irony | 2 | Hard: punctuation and capitals would help a little |
| Edited reviews (temporal mismatch) | 2 | No: the text no longer matches the rating |
| Label noise / rating–text mismatch | 2 | No: inherent noise near the 2★/3★ boundary |
| Domain vocabulary / implicit sentiment / tokenization | 3 | Partly: more data, subword tokens |

**Key patterns:** (1) the largest *fixable* group is long reviews where truncation removes the evaluative part; (2) many confident errors are not really model failures but **label/text mismatches** caused by edited reviews and by the 1–2★ vs 3–4★ binarisation; (3) the model treats sentiment as an aggregate of word polarity, so contrast ("but", "however"), sarcasm, and sentiment aimed at a competitor are systematically hard.

---

## Proposed testable fix: longer inputs for long reviews

**Hypothesis:** errors on long reviews come from head+tail truncation at 188 tokens dropping the evaluative middle section. Raising `max_len` will reduce errors on the truncated slice without hurting the rest.

**Change:** increase `max_len` from 188 (95th percentile) to **400 tokens** (roughly the 99th percentile), still with head+tail truncation, and retrain Exp 2 (BiGRU + attention) with all other settings and the seed unchanged.

**Test:**
1. Compare the **error rate and macro-F1 on the `truncated=True` slice** (as defined at max_len 188) before vs. after.
2. Run a **paired McNemar test** on the full test set (old vs. new Exp 2) to check that the overall change is significant and not a regression.
3. Check cost: training time and peak memory (a BiGRU's cost grows linearly with sequence length).

**Expected result:** a clear drop in slice error rate (it's currently about twice the overall error rate), a small overall gain, and roughly 2× training time for the longest batches.

**Other candidate fixes (not tested):** keep punctuation and an all-caps flag as features (sarcasm cues such as "A++++", "!!", capitals); split reviews into sentences and pool sentence-level predictions so the clause after "but"/"however" can dominate; remove or down-weight reviews starting with "EDIT:" / "UPDATED" when training.