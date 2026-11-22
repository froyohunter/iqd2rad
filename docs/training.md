# Qwen training

The Qwen reference recipes provide the a–f experiment ladder, an optional
image-only 8-bit control, and portable JSONL data input.

## Experiment arms

- **a — Base + PTQ:** evaluate the pretrained base model; no training.
- **b — FT + PTQ:** train float-space symmetric InfoNCE.
- **c — STE-QAT:** b plus quantized symmetric InfoNCE, averaged over 2/4/8 bits.
- **d — MSE:** c plus embedding MSE distillation.
- **e — RAD:** c plus ranking KL distillation.
- **f — ODARD:** e plus detached online difficulty weights.

The presets in `configs/qwen/video_[a-f].json` and
`configs/qwen/image_[a-f].json` provide the MSR-VTT video and MSCOCO image
recipes. They specify 3,001 optimizer steps, learning rate
1e-5, 300 LR warmup steps, 200 distillation warmup steps, AdamW weight decay
0.01, gradient clipping 1.0, batch size 8 per GPU, and effective batch size
128. LoRA uses rank 32, alpha 64, dropout 0.05, and q/k/v/o plus
gate/up/down projection targets. Gradient checkpointing is off. Default
precision is bf16 and attention is FlashAttention 2, matching the source
saved training configurations.

The effective batch is obtained by gradient accumulation:
`128 / (8 × world_size)`. It must be an integer. For example, eight GPUs use
two microbatches per optimizer update. **The contrastive negative pool remains
the local microbatch of eight pairs:** neither gradient accumulation nor DDP
adds cross-device negatives. A different per-device batch size changes the
training objective even when effective batch size stays 128.

The historical main runs and the 8-bit control used **16 GPUs (two nodes of
eight), with one microbatch per optimizer update**. The eight-GPU examples
below use accumulation of two and are portable launch examples, not a claim
of an identical stochastic training trajectory. Use matching world size,
seeds, manifests and model revision when comparing to the recorded runs.

## Optional 8-bit-only image control

`configs/qwen/image_d_8bit.json` and `image_e_8bit.json` train the matched MSE
and RAD arms with only the 8-bit branch. The loss, temperatures, LoRA setup,
batch sizes, learning rate and warmup schedule are unchanged from the
corresponding image d/e recipe. Only `bit_levels: [8]` changes the objective;
the runtime option `gradient_checkpointing: true` matches the recorded
control launch. The main 12 configs retain their original setting.

These controls have separate output directories. They train only the 8-bit
branch, although the resulting embeddings may still be evaluated at other
bit widths by the same retrieval evaluator.

```bash
torchrun --standalone --nproc_per_node=8 -m iqd2rad.training.cli \
  --config configs/qwen/image_e_8bit.json \
  --data-root /data/raw
```

For an exact hardware comparison, launch d/e and their 8-bit counterparts
under the same 16-GPU distributed setup. Merely changing accumulation to
match the global batch does not reproduce the historical sample partition
or dropout streams.

## Training inputs

Training supplies the paired caption to both the image/video branch and the
text-only branch, configured by `visual_input_includes_text: true`. Retrieval
evaluation encodes corpus images/videos without the paired caption.

Setting this flag to `false` creates a different training protocol; it should
be reported and evaluated as a new experiment. The flag is recorded in the
exported metadata. A small CPU test checks the actual inputs passed by the
training forward so this behavior cannot change unnoticed.

## Data manifests

Each line is one visual/text training pair. The `id` and nonempty `text` fields
are required; include exactly one of `image` or `frames`:

```json
{"id":"coco_42_caption_0","text":"A person rides a bicycle.","image":"coco/train2014/COCO_train2014_000000000042.jpg"}
{"id":"video42_caption_0","text":"A person opens a door.","frames":["msrvtt/video42/000001.jpg","msrvtt/video42/000002.jpg"]}
```

Relative media paths resolve against `--data-root`, not against the manifest
directory. Absolute local paths are accepted; remote media URLs are not.
Frame lists must be in temporal order. Sequences longer than 64 frames are
uniformly subsampled while retaining the endpoints. Missing media raises an
error instead of silently changing the sample population. Multiple captions
may refer to the same visual sample; preserve the source split and caption
sampling policy in your manifest.

The same format supports MSCOCO/MSR-VTT and user-prepared Flickr30k/ActivityNet
training data. Record the dataset, split and frame policy with each run and use
a distinct output directory.

The supplied training configs read `data/prepared/msrvtt_train/train.jsonl`
or `data/prepared/mscoco_train/train.jsonl`, directly matching the default
outputs of the data-preparation recipes. Keep `--data-root` equal to the
`--raw-root` used during preparation. Override `--manifest` only when the
prepared manifest is stored elsewhere.

## Installation and launch

From the repository root:

```bash
python -m pip install -e '.[train]'
python -m iqd2rad.training.cli --config configs/qwen/video_e.json --validate-config
```

The `train` extra does not automatically install `flash-attn`, because that
package requires a compatible Linux/CUDA build environment. For the paper
attention backend, install a FlashAttention 2 build compatible with your
PyTorch and CUDA installation. Otherwise select `--attn-implementation sdpa`
explicitly; this changes the attention implementation and needs its own
numerical validation. The launcher reports a clear error if the preset asks
for FlashAttention 2 and it is unavailable.

```bash
torchrun --standalone --nproc_per_node=8 -m iqd2rad.training.cli \
  --config configs/qwen/video_e.json \
  --manifest data/prepared/msrvtt_train/train.jsonl \
  --data-root /data/raw \
  --output-dir outputs/qwen-video-e

torchrun --standalone --nproc_per_node=8 -m iqd2rad.training.cli \
  --config configs/qwen/image_f.json \
  --manifest data/prepared/mscoco_train/train.jsonl \
  --data-root /data/raw \
  --output-dir outputs/qwen-image-f
```

Use ordinary torchrun rendezvous options for multiple nodes. Set a known
base commit with `--base-revision COMMIT_SHA`. No historical base commit is
invented by the presets: their `base_revision` is null until you pin it.
Offline operation requires previously downloaded model/processor assets and
`local_files_only: true` in the config. The launcher never changes system
packages, terminates other jobs, or relies on an internal scheduler.

Optional W&B logging uses the public client. Install `.[wandb]`, configure
authentication through the standard client, and set `wandb_project` in your
JSON config. Logging is disabled by default.

## Loss details that affect reproduction

Training quantization normalizes each embedding and uses a fixed symmetric
range [-1, 1] with straight-through rounding. This differs from the calibrated
per-channel affine SQ used in retrieval evaluation. The loss averages the
quantized contrastive terms and distillation terms across bits; it does not
sum them without rescaling.

For RAD, the implementation minimizes `KL(p_float || p_quant)` in both
retrieval directions, using a distillation temperature of 0.1. The float
teacher and opposite-modality float corpus are detached. The InfoNCE
temperature is fixed at 0.07; there is no learned logit scale. The MSE arm
preserves the source implementation's raw float teacher embeddings.

For f, weights come from detached per-pair quantized-minus-float InfoNCE,
sample-standard-deviation z-scoring, softmax temperature 1.0, multiplication
by the local batch size, and clamping to 5. The original implementation does
not renormalize weights after clamping. The distillation coefficient ramps
from `0.5 / 200` to `0.5` over the first 200 optimizer steps. No auxiliary
SorterHead, EMA teacher, or bit-depth consistency loss is used.

## Saved artifacts

Every 500 steps and at `output_dir/final`, the launcher exports the LoRA
adapter and processor/tokenizer using safetensors. It writes the experimental
configuration to `training_config.json`, leaving model configuration filenames
to the model/processor libraries. `metadata.json` records the requested and
resolved base revision, manifest SHA-256, arm, modality, optimizer step,
world size, accumulation count, package versions, and visual-caption flag.

These are inference exports, not optimizer-resume checkpoints. Optimizer and
scheduler state are not exported. Existing nonempty output directories are
rejected to prevent a run from mixing with old artifacts. No Qwen GPU run is
claimed solely from the CPU loss and orchestration tests.
