# Blinded human audit — instructions

There are 35 images in `images/`. Score every one on three
dimensions, 1-5 integers, in your own `ratings_<rater>.csv`.

- **style** — 1 = not Monet-like at all, 5 = convincingly Monet-like
- **content** — 1 = original scene unrecognisable, 5 = scene fully preserved
- **artifacts** — 1 = severe artifacts (blobs, banding, colour bleed), 5 = clean

Rules that make the result usable:

- Rate independently. Do not look at the other rater's sheet, and do not
  discuss scores until both sheets are finished — agreement between two people
  who conferred measures the conversation, not the images.
- Do not open `KEY_do_not_open_until_scored.json`. It maps samples to models,
  including which are yours.
- Score every row. Blanks are dropped, and dropping rows non-randomly biases
  the agreement estimate.
- Use the full scale. If everything gets a 4, kappa is undefined no matter how
  carefully you looked.

Then run `python scripts/human_audit.py score`.
