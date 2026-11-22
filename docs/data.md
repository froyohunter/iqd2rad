# Local data preparation

This release distributes preparation code and the public MSR-VTT split IDs.
Acquire images, videos and annotations from their dataset providers under their
terms. No raw media is downloaded or redistributed by these commands.

All recipes read local annotations and write portable JSONL. Media paths in
JSONL are relative to your `raw_root`; absolute machine paths are not embedded.
Preparation fails on missing media, missing split IDs, conflicting identities or
ground truth outside the gallery. It never silently drops unavailable examples.

## Formats and dataset scope

The supported datasets are MSR-VTT, MSCOCO, MSVD, DiDeMo, Flickr30k, ActivityNet
Captions and VisualNews.

- **MSR-VTT:** official `MSRVTT_train.7k.csv` / `MSRVTT_JSFUSION_test.csv`
  (`video_id`, `sentence`), or the original JSON with `videos` and `sentences`
  (`video_id`, `caption`, optional `sen_id`). The bundled 7,010 train and 1,000
  JSFusion test video IDs are disjoint. Their source URLs, counts and hashes are
  in `configs/data/splits/msrvtt_provenance.json`. The first caption per video
  produces the paired-video protocol; supplying a different caption collection
  is not guaranteed to reproduce the paper's query text.
- **MSCOCO training:** Karpathy `dataset_coco.json`, with `images` records
  containing `cocoid`, `filename`, `filepath`, `split` and `sentences` with `raw`
  text. The train recipe includes `train` and `restval` and takes the first
  caption per image, matching the one-pair-per-image training loader. Standard
  COCO captions JSON (`images` + `annotations`, joined through `image_id`) also
  works; an explicit split ID list is then needed for the intended split.
- **MSCOCO 1K-query evaluation:** use the exact public MMEB-style metadata
  export, not a newly sampled 1K Karpathy image subset. Required fields are
  `qry_text` and `tgt_img_path` (a list, positive first); optional `id` and
  `qry_inst` are preserved. The gallery is the union of candidate paths across
  rows and may have more than 1,000 images. `tgt_text` is never fed to the visual
  encoder. The release does not bundle or reconstruct this metadata export.
- **MSCOCO Karpathy 5K:** the test split of Karpathy annotations, first caption
  per image (5K queries, 5K gallery). `--caption-mode all` instead produces the
  different 25K-query protocol. Use the supplied recipe for first-caption mode.
- **Flickr30k evaluation:** the paper uses the `AnyModal/flickr30k` test export,
  with `filename` and `alt_text`. In that export `alt_text` contains one caption
  per image, giving **1,000 queries and 1,000 candidates**. The test recipe
  explicitly selects this metadata format, uses its first supplied caption, and
  validates both counts. Supply a JSON/JSONL export of these exact fields;
  `metadata/anymodal_flickr30k_test.jsonl` is the example local destination.
  A generic Karpathy annotation's first caption is not established as the same
  query and is not substituted by this recipe. For other experiments, the native
  adapter also accepts Karpathy JSON or the token file
  `image.jpg#caption_index<TAB>caption`; use `--caption-mode all` explicitly for
  a five-caption/5K-query protocol. Token files need explicit split IDs because
  they carry no split labels. The training recipe uses one caption per image.
- **MSVD:** a JSON mapping `video_id -> [captions]`, rows with `video_id` and
  `caption`/`captions`, or the official multilingual CSV with `YouTubeID`,
  `Start`, `End`, `Language`, `Description`. CSV input keeps English captions
  and identifies a clip as `YouTubeID_Start_End`. Use a test-only annotation
  file or `--split-ids`. Evaluation uses the first caption per video. Map IDs
  to your extracted frame folders with `--media-map` if their names differ.
- **DiDeMo:** official list entries with `video`, `description`, and optional
  temporal annotation fields. Descriptions are grouped by video in annotation
  order and joined into one paragraph query. Segment timestamps are not used
  in full-video retrieval. Test-only files or explicit split IDs are required.
- **ActivityNet Captions:** official `train.json` / `val_2.json` mappings from
  video ID to `sentences`, `timestamps` and `duration`, or rows with `video_id`
  and a prepared caption. Sentences are joined into one paragraph. The recipes
  use 32 sampled frames and require all selected videos; they do not reproduce
  an older availability-filtered subset unless its exact IDs are supplied.
- **VisualNews:** the paper-facing recipe consumes the exact MMEB-style query
  and gallery metadata as described for MSCOCO. Native VisualNews `data`/list
  records with `id`, `caption`, `image_path` and optional `split` are also
  supported via `--annotation-format native`; this defines a different gallery
  unless the original evaluation IDs are explicitly selected.

For JSON files without split labels, `--split` is only a recorded label: choose
the correct split-specific annotation file or provide `--split-ids`. When split
labels are present, all records must have labels and the selected label is
enforced. No random or hash split is generated.

## Prepare and validate

The files in `configs/data/` are editable JSON recipes. Their `metadata/...`
filenames and `data/raw` root are examples of where **you** place the files.
Run from the repository root; CLI flags override recipe values.

```bash
iqd2rad-data prepare --config configs/data/msrvtt_train.json \
  --annotations /path/to/MSRVTT_train.7k.csv --raw-root /path/to/media

iqd2rad-data prepare --config configs/data/mscoco_1k.json \
  --annotations /path/to/mscoco_mmeb_test.jsonl --raw-root /path/to/images

iqd2rad-data validate \
  --queries data/prepared/mscoco_1k/queries.jsonl \
  --candidates data/prepared/mscoco_1k/candidates.jsonl \
  --raw-root /path/to/images

iqd2rad-data check-splits \
  --split train=configs/data/splits/msrvtt_train_ids.txt \
  --split test=configs/data/splits/msrvtt_test_ids.txt
```

Preparation writes `metadata.json` with the annotation SHA-256, counts, caption
policy and whether explicit identity lists were supplied. It refuses to
overwrite an existing prepared split. Keep the original metadata export and its
hash when matching a published table.

`--split-ids` and `--exclude-ids` accept a JSON list or a text file with one media
ID per line; overlap is an error. With MMEB input, media IDs are relative image
paths and exclusion checks include the entire gallery. `--query-ids` and
`--candidate-ids` select and order the query/gallery records explicitly. Every
query's positive must still be present. These lists are useful for publishing
an exact benchmark subset without shipping media.

Split-ID lists filter membership while retaining annotation order. MMEB query
instructions are taken from `qry_inst`; native formats use dataset-specific
retrieval instructions. `--query-instruction` explicitly overrides either, and
is recorded in the preparation metadata.

## Manifest schemas

Training produces `train.jsonl`:

```json
{"id":"42:caption:0:0","text":"A cyclist crosses a bridge.","image":"images/42.jpg"}
{"id":"video1:caption:0:0","text":"A person opens a door.","frames":["frames/video1/frame_000000.jpg","frames/video1/frame_000001.jpg"]}
```

Retrieval produces `queries.jsonl` and `candidates.jsonl` separately:

```json
{"id":"q1","text":"A cyclist crosses a bridge.","instruction":"Find an image that matches the description.","positive_ids":["42"]}
```

```json
{"id":"42","image":"images/42.jpg"}
```

Candidate records never contain the query caption. Positive IDs are resolved
against candidate IDs when encoding, so shuffling the gallery does not change
ground truth. The evaluator supports multiple positives; the initial IQD
analyzer requires exactly one positive per query.

## Frame preparation

By default videos use `raw_root/frames/<video_id>/`. Existing frame filenames
are naturally sorted and uniformly subsampled up to the requested maximum;
fewer frames are retained, not duplicated. Main MSR-VTT/MSVD/DiDeMo recipes use
up to 64 frames; ActivityNet uses 32.

With `ffmpeg` and `ffprobe` installed, a local video can be sampled once:

```bash
iqd2rad-data extract-frames --video /path/to/video1.mp4 \
  --output-dir /path/to/media/frames/video1 --num-frames 64
```

This counts decoded frames and selects uniformly spaced frame indices including
both endpoints. It does not download videos or silently reuse a partial output.
Keep the same extraction and sampling policy for reference and reproduced runs.

`--media-map /path/to/map.json` resolves nonstandard layouts without embedding
absolute paths in the public manifests. For example:

```json
{"v_example":{"frame_dir":"frames/example"},"42":{"image":"train2014/COCO_train2014_000000000042.jpg"}}
```

A video map can instead give an explicit `frames` array. Paths must remain under
`raw_root`; parent traversal and escaping symlinks are rejected.
