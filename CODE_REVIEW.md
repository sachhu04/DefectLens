# DefectLens Code Review

Reviewed baseline: `934348fdd86ca2d6b1c7a97f75a731fed4db9695`.

This is a static review of the feature extractor, token fusion, anomaly
detector, dataset loader, evaluation script, and API. Findings describe
specific edge cases and proposed follow-up work; this document does not
change the implementation. Model inference and dataset evaluation were not
executed for this review.

## Findings

### 1. Missing anomaly masks silently become normal masks

**Priority: High**

Location: `src/data/dataset.py`, `MVTecDataset.__init__` and `__getitem__`;
`scripts/evaluate.py`, pixel-label aggregation.

When an anomalous image has no corresponding ground-truth mask, the loader
stores `None`. The same branch used for normal images then creates an
all-zero mask. Evaluation treats every pixel in that anomalous image as
normal, so incomplete datasets can produce misleading pixel AUROC values
without a warning.

Suggested fix: distinguish normal images from anomalous images with missing
masks. Reject missing anomaly masks during pixel evaluation, or explicitly
exclude those images from pixel metrics while retaining image-level metrics.
Report the number of excluded images.

Suggested verification: remove one anomaly mask from a small fixture and
check that pixel evaluation fails clearly or reports its exclusion.

### 2. MCTF accepts settings that fail during token matching

**Priority: Medium**

Location: `src/models/mctf.py`, `MCTFModule.forward`.

With `use_sim`, `use_info`, and `use_size` all disabled, each weight is the
Python float `1.0`. Their product is also a float, so `W.max(dim=-1)` raises
an error. This matters when exploring the criterion switches described in
the README. The supplied configuration enables all three criteria and does
not trigger this case.

Also, reduction requests are not checked against the number of source
tokens. For 196 patch tokens and `reduction_ratio=0.75`, the code requests
147 reductions from only 98 source tokens, causing `topk` to fail.

Suggested fix: validate the supported reduction range and criterion
combination at initialization. Alternatively, define an explicit all-disabled
behavior using a correctly shaped tensor, and cap reductions to available
source tokens after validating the intended semantics.

Suggested verification: exercise all-disabled criteria and a reduction
request larger than the source partition; require documented behavior or a
clear configuration error.

### 3. The sklearn fallback fails for small memory banks

**Priority: Medium**

Location: `src/models/anomaly_detector.py`, `_build_index` and `predict`.

Random coreset sampling can retain fewer entries than `num_neighbors`.
For example, 100 input features with the default sampling ratio of `0.01`
retain one entry, while the default neighbor count is nine. When FAISS is
unavailable, `kneighbors` requests nine neighbors from that one-entry bank
and fails. The current score calculation uses only the nearest distance.

Suggested fix: validate that the bank is nonempty and use an effective
neighbor count no larger than the bank size, consistently across both
backends. If only one neighbor is needed for scoring, document that choice.

Suggested verification: run prediction against a one-entry bank with the
sklearn fallback and check that a finite nearest-neighbor score is returned.

### 4. Earlier-layer features use the final-layer token mapping

**Priority: Medium**

Location: `src/models/vit.py`, `forward` and `extract_patch_features`.

`extract_patch_features` selects `features[layer_idx]` but returns the mapping
after all blocks have completed. If fusion occurs after the selected layer,
the selected feature tensor and mapping describe different token layouts.
This can silently assign scores to the wrong original patches during
`torch.gather` in the anomaly detector.

The supplied MCTF configuration selects the penultimate layer and fuses only
before blocks 0 through 9, so it avoids this case. Selecting an earlier layer
or adding later fusion can expose it.

Suggested fix: retain a mapping snapshot alongside each block's features,
then return the snapshot for the requested layer. An alternative is to reject
layer selections that precede subsequent fusion.

Suggested verification: extract from an early layer with later fusion enabled
and check spatial assignments against that layer's own mapping, including
index bounds and patch coverage.

## Useful existing safeguards

- The dataset loader checks required dataset and split directories.
- FAISS squared distances are square-rooted to match sklearn Euclidean scores.
- The API preserves explicit missing-memory-bank HTTP errors during model loading.
- The integration script covers the main pipeline when its data, weights,
  dependencies, and memory bank are available.

## Recommended follow-up order

1. Make missing anomaly masks explicit before reporting pixel metrics.
2. Validate token-fusion settings and memory-bank neighbor counts.
3. Keep feature tensors and mappings aligned for every supported layer.
4. Add small, isolated checks for these edge cases alongside the integration script.
