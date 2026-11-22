"""Portable, explicit configuration for the original Qwen a–f arms."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class TrainConfig:
    arm: str = "e"
    modality: str = "video"
    base_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    base_revision: str | None = None
    manifest: str = "data/prepared/msrvtt_train/train.jsonl"
    data_root: str = "."
    output_dir: str = "outputs/qwen-video-e"
    seed: int = 42
    max_steps: int = 3001
    per_device_batch_size: int = 8
    global_batch_size: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 300
    distill_warmup_steps: int = 200
    gradient_clip: float = 1.0
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    bit_levels: list[int] = field(default_factory=lambda: [2, 4, 8])
    temperature: float = 0.07
    distill_temperature: float = 0.1
    distill_weight: float = 0.5
    online_diff_temperature: float = 1.0
    online_diff_zscore: bool = True
    online_diff_clamp: float = 5.0
    use_global_negatives: bool = False
    # The source experiments feed the caption to BOTH the visual and text branch.
    # Keep this visible rather than silently changing the training protocol.
    visual_input_includes_text: bool = True
    max_frames: int = 64
    precision: str = "bf16"
    device: str = "cuda"
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = False
    local_files_only: bool = False
    num_workers: int = 0
    save_every: int = 500
    log_every: int = 10
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Training configuration must be a JSON object.")
        try:
            config = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"Unknown or invalid training configuration field: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.arm not in tuple("abcdef"):
            raise ValueError("arm must be one of a, b, c, d, e, f")
        if self.modality not in {"image", "video"}:
            raise ValueError("modality must be image or video")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        for name in ("max_steps", "per_device_batch_size", "global_batch_size", "lora_r",
                     "lora_alpha", "max_frames", "log_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "distill_warmup_steps", "num_workers", "save_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        for name in ("learning_rate", "temperature", "distill_temperature", "online_diff_temperature",
                     "online_diff_clamp", "gradient_clip"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0 or self.distill_weight < 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid decay, distillation weight, or LoRA dropout")
        if self.bit_levels not in ([2, 4, 8], [8]):
            raise ValueError("bit_levels must be [2, 4, 8] for the main protocol or [8] for the optional control")
        if self.use_global_negatives:
            raise ValueError("The released P03/P03_img recipes use local in-batch negatives")
        if self.per_device_batch_size < 2:
            raise ValueError("Contrastive training requires at least two pairs per device")
        if not self.lora_target_modules:
            raise ValueError("At least one LoRA target module is required")

    def accumulation_steps(self, world_size: int) -> int:
        batch = self.per_device_batch_size * world_size
        if world_size < 1 or self.global_batch_size % batch:
            raise ValueError("global_batch_size must be divisible by per_device_batch_size × world_size")
        steps = self.global_batch_size // batch
        if steps < 1:
            raise ValueError("global_batch_size is smaller than one distributed microbatch")
        return steps

    def to_dict(self) -> dict:
        return asdict(self)
