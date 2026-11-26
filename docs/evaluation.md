# Retrieval evaluation

The evaluator requires only NumPy. Model encoding is a separate command and can
be run once before sweeping dimensions and bit widths. It implements the final
paper's candidate-only affine scalar-quantization protocol. There is no query
calibration or alternative calibration switch.

## Shared embedding archive

An `.npz` archive must contain these arrays:

- `query_embeddings`: finite numeric array, shape `(N_query, D)`.
- `candidate_embeddings`: finite numeric array, shape `(N_candidate, D)`.
- `correct_indices`: integer array, shape `(N_query,)` for single positives, or
  `(N_query, max_positives)` padded with `-1` for multiple positives. Every query
  must have a valid positive. Ground truth is never assumed to be diagonal.
- Optional `query_ids` and `candidate_ids`: unique nonempty Unicode string arrays
  aligned with the embedding rows. Object/pickled arrays are not accepted.

For faithful scalar-quantized results, export **raw, unnormalized** embeddings.
The shared `iqd2rad-encode` command does this. Normalizing each full vector before
the per-channel grid is fitted changes the protocol, even though float cosine
scores would be unchanged.

The preparation and encoder commands resolve `positive_ids` against the gallery
before producing `correct_indices`. A reordered candidate archive requires
remapping these indices as well; metadata IDs alone do not repair stale indices.

Encoding defaults to the reference **FlashAttention 2** backend and bf16 on CUDA.
Install FlashAttention 2 in the training environment before using this path.
For CPU encoding, select `--device cpu --dtype float32 --attn-implementation sdpa`
explicitly. SDPA is also available on CUDA; record the selected backend when
comparing results, since different kernels can introduce numerical differences.

When an adapter includes `preprocessor_config.json`, encoding loads its exported
processor. A weights-only adapter uses the base processor; `--processor-source`
can override this selection. The embedding metadata records the processor source,
requested revision, selection rule, attention backend and dtype. Base and adapter
revisions should be pinned to the checkpoints being evaluated.

## Run

```bash
iqd2rad-evaluate --embeddings outputs/embeddings/msrvtt_test.npz \
  --dims 128 256 512 1024 2048 \
  --backends float sign sq2 sq4 sq8 \
  --output outputs/evaluation/msrvtt_test.json

iqd2rad-evaluate --config configs/eval/mscoco_1k.json
```

Paths in recipes are relative to the command's working directory. Explicit flags
override the JSON recipe. Omitting dimensions evaluates the full available
dimension; an oversized or nonpositive dimension raises an error.

Each dimension takes the first `d` coordinates of the raw embedding:

1. **Float:** L2-normalize each prefix, then compute cosine similarity.
2. **Sign:** map negative coordinates to `-1` and nonnegative coordinates to
   `+1`; rank by their dot product (equivalent to inverse Hamming distance).
3. **SQ2/SQ4/SQ8:** fit per-channel minima and maxima on candidate prefixes
   **only**; freeze that grid; clip and quantize both candidate and query
   prefixes on the same grid; reconstruct and L2-normalize before cosine scoring.

The affine grid has `2**bits` levels. Constant and numerically tiny-range channels
follow the original implementation's scale fallback (`scale < 1e-12 -> 1.0`).
This evaluation quantizer is distinct from the fixed symmetric grid used in
training and in the IQD experiments; see the IQD documentation for that protocol.

Results contain R@1, R@5, R@10 and NDCG@10 as **fractions in [0,1]**. Multiply by
100 for percentage tables. Recall means the fraction of queries with at least
one relevant item in top-k. NDCG uses binary relevance, including multiple
positives when present. Candidate pools smaller than k are supported. Ties use
NumPy quicksort, preserving the research evaluator's convention; sign ties are
common, so candidate order and the NumPy version should be recorded for exact
reproduction. The output labels this convention as `numpy_quicksort`.

`--batch-size` controls the query-by-gallery score block, not the retrieval
candidate set. It does not partition or resample the gallery. This is exhaustive
evaluation rather than an ANN or latency benchmark.

## Query bootstrap

```bash
iqd2rad-evaluate --config configs/eval/mscoco_1k.json \
  --bootstrap 10000 --seed 42 --output outputs/evaluation/mscoco_ci.json
```

Bootstrap resamples the already evaluated per-query metrics and reports percentile
95% intervals. The candidate-fitted quantizer remains fixed throughout; queries
never refit it. These intervals quantify evaluation-query sampling uncertainty,
not training-seed variability. They are single-model intervals, not a paired
significance test between checkpoints.

## Reproduction boundaries

The data recipes cover the paper's Qwen image/video datasets. Exact table
reproduction still requires the matching checkpoint, raw embedding export,
query text, query/gallery IDs, input preprocessing, model revision and bit/dim
configuration. Standard dataset splits with different caption choices or gallery
membership are not interchangeable with the released experiment metadata.

Media must be complete and decodable. The portable pipeline reports missing or
corrupt inputs instead of substituting blank images, removing failed frames or
encoding placeholder text. Repair such inputs before comparing a run with an
existing embedding export.
