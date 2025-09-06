## Problems
- `dataset.py:115` is still generating GT heatmaps for loss computation. and NO that's not required.
    - Maybe its okay and actually benefical? (exchange time by space)
- generated datasets contains many wasting fields:
    - SubtitleChunk.events
    - DataChunk.original_audio_file
    - DataChunk.original_subtitle_file
- Is chunks overlapping for training? Perhaps they should not overlap?

### Losses
- FocalLoss: Auto-configure config yaml `focal_alpha` & `focal_gamma`.
    - They comes from dataset inbalance.
