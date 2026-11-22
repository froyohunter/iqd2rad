"""Qwen3-VL embeddings, adapted from QwenLM/Qwen3-VL-Embedding (Apache-2.0).

See NOTICE. The differentiable batch path preserves the upstream chat template,
vision processing and last-token pooling. Preprocessing errors are not hidden.
"""

from pathlib import Path
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from qwen_vl_utils.vision_process import process_vision_info
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Backbone with the same checkpoint key layout as the upstream embedder."""

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_decoder(self):
        return self.model.get_decoder()

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # PEFT may forward an unused language-model argument to the embedding backbone.
        kwargs.pop("logits_to_keep", None)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def resolve_model_source(value: str) -> str:
    """A Hub ID such as Qwen/Qwen3-VL-Embedding-2B is not a local path."""
    path = Path(value).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if value.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {value}")
    return value


def last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or not torch.all(mask.any(dim=1)):
        raise ValueError("Every encoded sample needs at least one non-padding token")
    col = mask.shape[1] - 1 - mask.flip(dims=[1]).long().argmax(dim=1)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), col]


class QwenEncoder:
    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-VL-Embedding-2B",
        *,
        revision: str | None = None,
        adapter: str | None = None,
        adapter_revision: str | None = None,
        processor_source: str | None = None,
        device: str | torch.device | None = None,
        torch_dtype: str | torch.dtype | None = None,
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
        max_length: int = 8192,
        min_pixels: int = 4 * 32 * 32,
        max_pixels: int = 1800 * 32 * 32,
        total_pixels: int = 10 * 768 * 32 * 32,
        fps: float = 1,
        max_frames: int = 64,
        default_instruction: str = "Represent the user's input.",
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype.removeprefix("torch."))
        dtype = torch_dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        source = resolve_model_source(base_model)
        self.model = Qwen3VLForEmbedding.from_pretrained(
            source, revision=revision, torch_dtype=dtype,
            attn_implementation=attn_implementation, use_safetensors=True,
            local_files_only=local_files_only,
        ).to(self.device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, resolve_model_source(adapter), revision=adapter_revision,
                local_files_only=local_files_only,
            )
            self.model = self.model.merge_and_unload()
        # The upstream processor is the default. An exported processor can be passed
        # explicitly, avoiding the training_config/model_config filename ambiguity.
        proc_source = resolve_model_source(processor_source) if processor_source else source
        self.processor = Qwen3VLProcessor.from_pretrained(
            proc_source, revision=None if processor_source else revision,
            padding_side="right", local_files_only=local_files_only,
        )
        self.base_model = base_model
        self.revision = revision
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        self.model.eval()

    @property
    def hidden_size(self):
        config = self.model.config
        return getattr(getattr(config, "text_config", config), "hidden_size")

    @staticmethod
    def _media_path(value):
        if not isinstance(value, (str, Path)):
            return value
        value = str(value)
        if value.startswith(("http://", "https://", "file://")):
            return value
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Media file does not exist: {path}")
        # qwen-vl-utils strips file:// but does not URL-decode percent escapes.
        # Keep local Unicode and spaces intact for PIL and frame readers.
        return str(path)

    def format_input(self, row: dict) -> list[dict]:
        instruction = row.get("instruction") or self.default_instruction
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        content = []
        video = row.get("video")
        if video is not None:
            if isinstance(video, list):
                if not video:
                    raise ValueError("Video frame list is empty")
                limit = int(row.get("max_frames", self.max_frames))
                if limit < 1:
                    raise ValueError("max_frames must be positive")
                if len(video) > limit:
                    video = [video[i] for i in np.linspace(0, len(video) - 1, limit, dtype=int)]
                content.append({"type": "video", "video": [self._media_path(x) for x in video],
                                "total_pixels": self.total_pixels})
            else:
                content.append({"type": "video", "video": self._media_path(video),
                                "fps": row.get("fps", self.fps),
                                "max_frames": row.get("max_frames", self.max_frames)})
        image = row.get("image")
        if image is not None:
            for item in image if isinstance(image, list) else [image]:
                content.append({"type": "image", "image": self._media_path(item),
                                "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})
        text = row.get("text")
        if text is not None:
            for item in text if isinstance(text, list) else [text]:
                content.append({"type": "text", "text": item})
        if not content:
            raise ValueError("Each input must contain text, image, or video")
        return [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content}]

    def prepare_inputs(self, inputs: list[dict]):
        if not inputs:
            raise ValueError("Cannot encode an empty batch")
        conversations = [self.format_input(row) for row in inputs]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False,
        )
        images, video_inputs, video_kwargs = process_vision_info(
            conversations, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True,
        )
        videos, metadata = None, None
        if video_inputs:
            videos, metadata = zip(*video_inputs)
            videos, metadata = list(videos), list(metadata)
        batch = self.processor(
            text=text, images=images, videos=videos, video_metadata=metadata,
            truncation=True, max_length=self.max_length, padding=True,
            do_resize=False, return_tensors="pt", **video_kwargs,
        )
        return {key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in batch.items()}

    def encode_batch(self, inputs: list[dict], normalize: bool = True) -> torch.Tensor:
        """Differentiable: leaves model.train()/eval() and autograd to the caller."""
        batch = self.prepare_inputs(inputs)
        output = self.model(**batch)
        embeddings = last_token_pool(output.last_hidden_state, batch["attention_mask"])
        return F.normalize(embeddings, p=2, dim=-1) if normalize else embeddings

    def encode(self, inputs: list[dict], batch_size: int = 8, normalize: bool = True):
        if batch_size < 1 or not inputs:
            raise ValueError("A nonempty input list and positive batch_size are required")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                parts = [self.encode_batch(inputs[i:i + batch_size], normalize).float().cpu().numpy()
                         for i in range(0, len(inputs), batch_size)]
            return np.concatenate(parts)
        finally:
            self.model.train(was_training)
