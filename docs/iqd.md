# Measuring Intrinsic Quantization Difficulty

The IQD analyzer measures how a query's retrieval ranking changes when its
embedding and the candidate embeddings are quantized. It runs on NumPy arrays;
it does not load a model, raw media, a training checkpoint, or a pickle file.
NumPy and SciPy are required. Matplotlib is optional for `--plots`.

## Input contract

Provide one `.npz` archive with:

- `query_embeddings`: finite, nonzero floating-point matrix `[Nq, D]`.
- `candidate_embeddings`: finite, nonzero matrix `[Nc, D]`, with `Nc >= 2`.
- `correct_indices`: integer vector `[Nq]`. Entry `i` is the index of query
  `i`'s single positive in the candidate matrix. A missing or out-of-range index
  is an error. No diagonal pairing is inferred.
- Optional `query_ids` and `candidate_ids`: unique string arrays aligned to
  their respective matrices, saved with a Unicode dtype rather than `object`.

For gallery retrieval, construct `correct_indices` by matching the annotation's
gold image ID to the deduplicated candidate IDs. Multiple queries may share the
same positive image. For a paired video pool, verify the pairs before explicitly
writing `np.arange(Nq)`; equal matrix sizes do not prove correct pairing.
The analyzer implements a **single-positive** protocol. Do not silently reduce
a multi-positive benchmark to its first label.

The arrays are converted to float32, matching the research measurements. Input
archives should be generated from the specified model revision, preprocessing,
frame sampling, caption selection, query order and candidate pool. Keep those
encoding choices and split IDs with the archive; the sweep manifest alone does
not certify the embedding export.

## Run a sweep

From the installed repository:

```bash
python -m iqd2rad.analysis.cli \
  --embeddings /path/to/qwen_msrvtt.npz \
  --config configs/iqd/qwen_msrvtt.json \
  --output-dir outputs/iqd/qwen_msrvtt \
  --plots
```

For another verified embedding pool, choose dimensions explicitly:

```bash
python -m iqd2rad.analysis.cli \
  --embeddings /path/to/embeddings.npz \
  --dims 128,256,512 --bits 1,2,4,8 --k 10 \
  --output-dir outputs/iqd/custom
```

CLI analysis flags override JSON values. The default direction is query to
candidate (`symmetric=false`). `--symmetric` averages forward and reverse
ranking, margins, recall-loss indicators and loss-gap signals, aligned through
the explicit ground truth. It requires an equal-size, bijective pairing and
rejects a many-to-one gallery. Query reconstruction error and matched-candidate
density/LID remain descriptors of the original query/positive pair, as in the
research protocol. `--no-symmetric` explicitly overrides a symmetric config.

Dimensions are prefix truncations and must fit the archive. **The native
dimension is never appended automatically.** The four supplied main-heatmap
manifests specify 20 Qwen/video, 15 CLIP/video, 20 Qwen/image and 15 CLIP/image
cells, totaling 70. Every grid uses bits `[1, 2, 4, 8, 16]`. Qwen/video dimensions
are `[128, 256, 512, 2048]`; Qwen/image dimensions are
`[128, 256, 512, 1024]`; both CLIP grids use `[128, 256, 512]`.

The appendix's GT-correct mean-correlation tables use a different explicit
image sweep: Qwen dimensions `[128,256,512,1024,2048]` and CLIP dimensions
`[128,256,512]`, with bits `[1,2,4,8]`. Override `--dims`/`--bits` accordingly;
do not average the 70-cell heatmap manifests and call it the appendix table.

**CLIP video export provenance is still unverified.** The final paper describes
frame-level scores without temporal mean pooling. The research repository also
contains later mean-pooled CLIP encoders. They are not interchangeable. The
CLIP/video manifest preserves the analysis grid and emits a warning; running
it on an arbitrary mean-pooled export does not reproduce the paper's CLIP cells.
No CLIP training implementation is required by this analyzer.

## Quantization and statistics

The offline IQD quantizer first L2-normalizes each prefix vector. For bits > 1,
it rounds onto `2**bits` uniformly spaced levels on fixed `[-1, 1]`; 1-bit uses
`numpy.sign`, including zero for exactly zero coordinates. Similarities are
cosines after normalizing the quantized vectors again. The 16-bit grid is a
near-float control, distinct from the unquantized reference.

This is **not a fitted per-vector min/max quantizer**. It also is not the
evaluation-time candidate-calibrated affine SQ codebook. Training uses a fixed
symmetric grid with STE gradients; this offline tool uses NumPy with no
gradients. Changing the IQD quantizer to the deployment evaluator's codebook
would change the measured scientific experiment.

The nine predictors retain the research definitions:

- `quant_error`: query L2 error between the normalized float vector and the
  raw grid output, before normalizing that output again for cosine similarity.
- `density`: mean cosine to the gold candidate's `k` nearest other candidates.
- `iqd_combined`: query quantization error times that matched-candidate density.
- `margin_float`: gold score minus the largest non-gold float score.
- `inv_margin_float`: reciprocal of the float margin clipped below at `1e-6`.
- `lid`: matched-candidate local intrinsic dimensionality from up to 20 nearest
  neighbors, using the research MLE and its `[0,1000]` clipping.
- `margin_erosion`: float margin minus quantized margin.
- `focal_weight`: `(1 - p_quant(gold))**2` at InfoNCE temperature `0.07`.
- `online_diff`: quantized minus float per-query InfoNCE loss at temperature
  `0.07`, evaluated on this whole candidate pool.

Density/LID are candidate properties gathered at each query's explicit positive
index; they are not aligned by row position in a gallery. The neighbor count
is capped at `Nc-1` for small pools and recorded in each result.

`rank_drop` is the quantized minus float **zero-based** rank, where rank counts
scores strictly greater than the gold score. Positive means degradation;
negative means improvement. `recall_drop` is the indicator that a float top-k
hit is lost after quantization, not a signed difference of corpus recalls.
Top-k membership preserves the research `numpy.argpartition` rule. Boundary
ties can depend on candidate order; the archive must preserve that order.

Spearman correlations use finite predictor/target pairs and require at least
10 valid samples. Constant predictors and unchanged rankings produce a JSON
`null` correlation with an explicit status, not an invented zero; such cells
remain in the sweep. Values are stratified by the 1/3 and 2/3 quantiles, with
`easy <= q1`, `q1 < medium <= q2`, and `hard > q2`. Tied values can make unequal
or empty groups. Empty group means are `null`. This differs from rank-splitting
samples into artificially equal thirds.

The analyzer computes `online_diff` over the full candidate pool. The optional
training extension computes it within each local microbatch and applies the
weight normalization described in the [training guide](training.md).

## Outputs

`summary.json` records the input archive's SHA-256, exact grid, quantization and
tie policies, per-predictor signed correlations/p-values/valid counts, and
stratified statistics. Each `samples_d<D>_b<B>.npz` contains aligned per-query
predictors, float/quantized ranks and hits, rank/recall losses, explicit GT, and
IDs when supplied. `--plots` adds a labeled nine-predictor correlation heatmap.
Undefined correlations are gray. Existing results require an explicit
`--overwrite` or a fresh output directory.

The implementation is extracted from the Apache-2.0 HiQ research scripts
`validate_iqd.py` and `validate_iqd_gt.py`, retaining their measurement definitions
while replacing implicit diagonal pairing with validated explicit ground truth.
