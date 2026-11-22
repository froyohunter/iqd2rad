from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from iqd2rad.training.trainer import PairEncoder, export_adapter
from iqd2rad.training.config import TrainConfig


def test_pair_forward_keeps_both_inputs_differentiable():
    class Encoder:
        def __init__(self):
            self.model = torch.nn.Linear(2, 3, bias=False)
            self.calls = []

        def encode_batch(self, inputs, normalize=True):
            self.calls.append((inputs, normalize, self.model.training))
            return self.model(torch.ones(len(inputs), 2))

    encoder = Encoder()
    pair = PairEncoder(encoder)
    pair.train()
    visual, text = pair([{"id": "one", "image": "/data/a.jpg", "text": "caption"}])
    (visual.sum() + text.sum()).backward()
    assert encoder.model.weight.grad is not None
    assert encoder.calls == [
        ([{"image": "/data/a.jpg", "text": "caption"}], False, True),
        ([{"text": "caption"}], False, True),
    ]
    assert list(pair.parameters()) == list(encoder.model.parameters())


def test_export_preserves_model_config_and_records_protocol(tmp_path):
    import json

    class Model:
        config = SimpleNamespace(_commit_hash="test-revision")

        def save_pretrained(self, destination, safe_serialization):
            assert safe_serialization
            (destination / "adapter_config.json").write_text('{"base_model_name_or_path":"Qwen/Qwen3-VL-Embedding-2B"}')

    class Processor:
        def save_pretrained(self, destination):
            (destination / "preprocessor_config.json").write_text("{}")

    encoder = SimpleNamespace(model=Model(), processor=Processor())
    destination = tmp_path / "final"
    export_adapter(encoder, TrainConfig(), destination, step=1, world_size=1, manifest_sha256="abc")
    assert not (destination / "config.json").exists()
    assert json.loads((destination / "training_config.json").read_text())["visual_input_includes_text"]
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["resolved_base_revision"] == "test-revision"
    assert metadata["optimizer_state_included"] is False


def test_cpu_training_accumulates_and_exports_without_model_download(tmp_path, monkeypatch):
    """Exercise the real optimizer loop with an in-memory tiny encoder, not Qwen."""
    import json
    import sys
    from types import ModuleType
    from iqd2rad.training.trainer import train

    instances = []

    class TinyModel(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 4, bias=False)
            self.config = SimpleNamespace(use_cache=True, _commit_hash="unit-test")

        def save_pretrained(self, destination, safe_serialization=True):
            (destination / "adapter_config.json").write_text("{}")

    class Processor:
        def save_pretrained(self, destination):
            (destination / "preprocessor_config.json").write_text("{}")

    class TinyEncoder:
        def __init__(self, *args, **kwargs):
            self.model = TinyModel()
            self.processor = Processor()
            self.calls = 0
            instances.append(self)

        def encode_batch(self, inputs, normalize=False):
            self.calls += 1
            features = torch.tensor([[float(record["text"].split()[-1]) + 1, 1.0] for record in inputs])
            return self.model(features)

    def attach_lora(model, config):
        model.peft_config = {"default": config}
        return model

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_peft.get_peft_model = attach_lora
    fake_qwen = ModuleType("iqd2rad.models.qwen")
    fake_qwen.QwenEncoder = TinyEncoder
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "iqd2rad.models.qwen", fake_qwen)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    rows = []
    for index in range(4):
        image = tmp_path / f"{index}.jpg"
        image.touch()
        rows.append({"id": str(index), "text": f"caption {index}", "image": image.name})
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    config = TrainConfig(arm="e", modality="image", manifest=str(manifest), data_root=str(tmp_path),
                         output_dir=str(tmp_path / "run"), max_steps=2, warmup_steps=0,
                         per_device_batch_size=2, global_batch_size=4, device="cpu",
                         precision="fp32", attn_implementation="sdpa", save_every=1)
    train(config)
    assert instances[0].calls == 8  # 2 updates × 2 microbatches × 2 modalities
    assert (tmp_path / "run/checkpoint-1/training_config.json").is_file()
    metadata = json.loads((tmp_path / "run/final/metadata.json").read_text())
    assert metadata["global_step"] == 2
    assert metadata["gradient_accumulation_steps"] == 2
    assert metadata["training_pairs_include_caption_in_visual_input"] is True
