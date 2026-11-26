"""LoRA training for the Qwen main experiments with ordinary torchrun/DDP."""

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import TrainConfig
from .data import ManifestDataset, collate_pairs, model_inputs
from .loss import RankingDistillLoss


class PairEncoder(nn.Module):
    """One DDP forward contains both modality forwards and their shared parameters."""

    def __init__(self, encoder, visual_input_includes_text=True):
        super().__init__()
        # QwenEncoder is a preprocessing wrapper; register its model exactly once.
        object.__setattr__(self, "encoder", encoder)
        self.model = encoder.model
        self.visual_input_includes_text = visual_input_includes_text

    def forward(self, records):
        visuals, texts = model_inputs(records, self.visual_input_includes_text)
        visual_embeddings = self.encoder.encode_batch(visuals, normalize=False)
        text_embeddings = self.encoder.encode_batch(texts, normalize=False)
        return visual_embeddings, text_embeddings


def _lr_multiplier(step, warmup, total):
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = float(step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _seed_training(seed, rank):
    """Seed once before model/LoRA initialization; DDP broadcasts rank-zero weights."""
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def _training_loader(dataset, config, world_size, rank):
    # The original DistributedSampler uses its default seed=0 independently of
    # the model seed, pads to equal rank lengths, and retains the last batch.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=0, drop_last=False) if world_size > 1 else None
    local_samples = len(sampler) if sampler is not None else len(dataset)
    if local_samples < 2 or local_samples % config.per_device_batch_size == 1:
        raise ValueError("The retained final microbatch would contain fewer than two pairs; "
                         "choose a dataset/batch/world-size combination without a singleton tail")
    loader = DataLoader(dataset, batch_size=config.per_device_batch_size,
                        shuffle=sampler is None, sampler=sampler, drop_last=False,
                        num_workers=config.num_workers, collate_fn=collate_pairs,
                        pin_memory=True, prefetch_factor=4 if config.num_workers > 0 else None,
                        persistent_workers=config.num_workers > 0)
    return loader, sampler


def _periodic_checkpoint(step, save_every):
    # Source loop checks zero-based loop index after incrementing global_step.
    return bool(save_every and step > 1 and (step - 1) % save_every == 0)


def _wandb_metrics(metrics, config, step, lr):
    result = {f"train/{key}": value for key, value in metrics.items()}
    result.update({"train/lr": lr, "train/global_step": step})
    if config.arm in "cdef":
        weight = config.distill_weight if config.arm in "def" else 0.0
        # Preserve the source logging point: global_step has already advanced,
        # so this field is the next-step coefficient during warmup.
        scale = min(1.0, (step + 1) / config.distill_warmup_steps) if config.distill_warmup_steps else 1.0
        result["train/distill_weight"] = weight * scale
    return result


def _versions():
    result = {}
    for package in ("torch", "transformers", "peft", "qwen-vl-utils", "safetensors"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def export_adapter(encoder, config, destination, *, step, world_size, manifest_sha256,
                   num_training_samples=None, num_training_batches=None, current_lr=None,
                   is_final=False):
    """Export inference weights and provenance; never overwrite model config.json."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    encoder.model.save_pretrained(destination, safe_serialization=True)
    encoder.processor.save_pretrained(destination)
    with (destination / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
        handle.write("\n")
    training_info = {
        "global_step": step,
        "num_training_samples": num_training_samples,
        "num_training_batches": num_training_batches,
        "num_val_samples": 0,
        "world_size": world_size,
        "effective_batch_size": config.global_batch_size,
        "per_device_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.accumulation_steps(world_size),
        "current_lr": current_lr,
        "is_final": is_final,
    }
    with (destination / "training_info.json").open("w", encoding="utf-8") as handle:
        json.dump(training_info, handle, indent=2)
        handle.write("\n")
    metadata = {
        **training_info,
        "format": "iqd2rad-lora-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": config.arm,
        "modality": config.modality,
        "base_model": config.base_model,
        "requested_base_revision": config.base_revision,
        "resolved_base_revision": getattr(encoder.model.config, "_commit_hash", None),
        "training_manifest_sha256": manifest_sha256,
        "training_pairs_include_caption_in_visual_input": config.visual_input_includes_text,
        "optimizer_state_included": False,
        "versions": _versions(),
    }
    with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def train(config: TrainConfig):
    """Run b–f; arm a requires only evaluation of the pretrained base model."""
    from peft import LoraConfig, get_peft_model
    from iqd2rad.models.qwen import QwenEncoder

    config.validate()
    if config.arm == "a":
        raise ValueError("Arm a does not train; evaluate the base model directly")
    if config.attn_implementation == "flash_attention_2" and find_spec("flash_attn") is None:
        raise RuntimeError("The paper recipe uses FlashAttention 2. Install flash-attn on supported Linux/CUDA, "
                           "or explicitly select --attn-implementation sdpa for a different attention backend.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    accumulation = config.accumulation_steps(world_size)
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Qwen training requires a suitable GPU installation.")
        device = torch.device(f"cuda:{local_rank}" if world_size > 1 or config.device == "cuda" else config.device)
        torch.cuda.set_device(device)
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bf16; choose fp16 or fp32 explicitly")
    elif config.device == "cpu":
        device = torch.device("cpu")
        if config.precision == "fp16":
            raise ValueError("CPU training does not support the fp16 recipe")
    else:
        raise ValueError("Supported training devices are cuda[:index] and cpu")
    distributed = world_size > 1
    initialized_here = False
    run = None
    try:
        if distributed and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
            initialized_here = True
        _seed_training(config.seed, rank)
        destination = Path(config.output_dir).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {destination}. Choose a new output_dir.")
        if rank == 0:
            destination.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()
        dataset = ManifestDataset(config.manifest, config.data_root, config.modality, config.max_frames)
        loader, sampler = _training_loader(dataset, config, world_size, rank)
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[config.precision]
        encoder = QwenEncoder(
            config.base_model, revision=config.base_revision, device=str(device), torch_dtype=dtype,
            attn_implementation=config.attn_implementation, local_files_only=config.local_files_only,
            max_frames=config.max_frames,
        )
        peft_config = LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules, bias="none", task_type="FEATURE_EXTRACTION",
            revision=config.base_revision,
        )
        encoder.model = get_peft_model(encoder.model, peft_config)
        encoder.model.peft_config["default"].base_model_name_or_path = config.base_model
        if config.gradient_checkpointing:
            encoder.model.enable_input_require_grads()
            encoder.model.gradient_checkpointing_enable()
        encoder.model.config.use_cache = False
        pair_model = PairEncoder(encoder, config.visual_input_includes_text).to(device)
        pair_model.train()
        if distributed:
            pair_model = DistributedDataParallel(
                pair_model, device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
            if config.gradient_checkpointing:
                pair_model._set_static_graph()
        criterion = RankingDistillLoss(
            arm=config.arm, temperature=config.temperature,
            distill_temperature=config.distill_temperature, distill_weight=config.distill_weight,
            online_diff_temperature=config.online_diff_temperature,
            online_diff_zscore=config.online_diff_zscore, online_diff_clamp=config.online_diff_clamp,
            bit_levels=config.bit_levels,
        ).to(device)
        parameters = [parameter for parameter in pair_model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable LoRA parameters were attached")
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                                      fused=device.type == "cuda")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _lr_multiplier(step, config.warmup_steps, config.max_steps),
        )
        if rank == 0 and config.wandb_project:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("Install 'iqd2rad[wandb]' to enable W&B") from exc
            run = wandb.init(project=config.wandb_project, name=config.wandb_run_name,
                             config=config.to_dict(), dir=str(destination))
        manifest_sha256 = hashlib.sha256(Path(config.manifest).expanduser().read_bytes()).hexdigest()
        epoch, step = 0, 0
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print(json.dumps({"event": "training_start", "arm": config.arm, "pairs": len(dataset),
                              "world_size": world_size, "gradient_accumulation_steps": accumulation,
                              "negative_pool": config.per_device_batch_size,
                              "visual_input_includes_text": config.visual_input_includes_text}), flush=True)
        while step < config.max_steps:
            totals = {}
            for micro in range(accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                sync = micro == accumulation - 1
                sync_context = (pair_model.no_sync() if distributed and not sync and not config.gradient_checkpointing
                                else nullcontext())
                with sync_context:
                    # The reference runs use the loaded backbone dtype directly;
                    # autocast would promote loss operations and change BF16 results.
                    visual, text = pair_model(batch)
                    scale = min(1.0, (step + 1) / config.distill_warmup_steps) if config.distill_warmup_steps else 1.0
                    metrics = criterion(visual, text, distill_scale=scale)
                    loss = metrics["loss"] / accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
                    loss.backward()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {step + 1}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0:
                if run is not None:
                    run.log(_wandb_metrics(totals, config, step, scheduler.get_last_lr()[0]), step=step)
                if (step - 1) % config.log_every == 0:
                    logged = dict(totals, step=step, epoch=epoch, lr=scheduler.get_last_lr()[0],
                                  grad_norm=float(grad_norm))
                    print(json.dumps(logged), flush=True)
            if _periodic_checkpoint(step, config.save_every):
                if rank == 0:
                    export_adapter(encoder, config, destination / f"checkpoint-{step}", step=step,
                                   world_size=world_size, manifest_sha256=manifest_sha256,
                                   num_training_samples=len(dataset), num_training_batches=len(loader),
                                   current_lr=scheduler.get_last_lr()[0])
                if distributed:
                    dist.barrier()
        if rank == 0:
            export_adapter(encoder, config, destination / "final", step=step,
                           world_size=world_size, manifest_sha256=manifest_sha256,
                           num_training_samples=len(dataset), num_training_batches=len(loader),
                           current_lr=scheduler.get_last_lr()[0], is_final=True)
            print(json.dumps({"event": "training_complete", "output": str(destination / "final")}), flush=True)
        if distributed:
            dist.barrier()
    finally:
        if run is not None:
            run.finish()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
