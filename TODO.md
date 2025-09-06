## Problems
- `dataset.py:115` is still generating GT heatmaps for loss computation. and NO that's not required.
    - Maybe its okay and actually benefical? (exchange time by space)
- generated datasets contains many wasting fields:
    - SubtitleChunk.events
    - DataChunk.original_audio_file
    - DataChunk.original_subtitle_file
- Is chunks overlapping for training? Perhaps they should not overlap?

### Logging
- Tensorboard somehow bugs when using duplicated experiment name. Append datetime.now?

### Losses
- FocalLoss: Auto-configure config yaml `focal_alpha` & `focal_gamma`.
    - They comes from dataset inbalance.

### Inference
- `_merge_overlapping_chunks` should not directly merge events by time. 1.0s is also too brutal. Many events will get eliminated.
    - At least, only merge events from different chunks. Implement this first.
    - As a more complicated solution:
        - Merge heatmaps and average features instead.
        - If that is against the current code structure, let the model take the responsity of merging.

## Documentation
- Remove mentionings of network structures in README, such as "Conv1D" "Transformers" as they are subject to change.