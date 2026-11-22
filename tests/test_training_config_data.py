import json

import pytest

from iqd2rad.training.config import TrainConfig
from iqd2rad.training.data import ManifestDataset, model_inputs


def test_effective_batch_and_reject_fractional_accumulation():
    config = TrainConfig()
    config.validate()
    assert config.accumulation_steps(8) == 2
    assert config.accumulation_steps(1) == 16
    with pytest.raises(ValueError, match="divisible"):
        config.accumulation_steps(3)


def test_paper_protocol_explicit_caption_in_both_branches():
    record = {"id": "sample", "image": "/images/sample.jpg", "text": "a caption"}
    visual, text = model_inputs([record], TrainConfig().visual_input_includes_text)
    assert visual == [{"image": "/images/sample.jpg", "text": "a caption"}]
    assert text == [{"text": "a caption"}]
    visual_only, _ = model_inputs([record], False)
    assert visual_only == [{"image": "/images/sample.jpg"}]


def test_manifest_root_frame_order_and_missing_file(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    for index in range(5):
        (media / f"{index}.jpg").touch()
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(json.dumps({"id": "clip", "text": "a video", "frames": [f"{i}.jpg" for i in range(5)]}) + "\n")
    dataset = ManifestDataset(manifest, media, modality="video", max_frames=3)
    assert dataset[0]["video"] == [str(media / f"{i}.jpg") for i in (0, 2, 4)]
    (media / "2.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="Missing manifest media"):
        dataset[0]


def test_manifest_rejects_ambiguous_modality(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps({"id": "1", "text": "caption", "image": "a.jpg", "frames": ["a.jpg"]}))
    with pytest.raises(ValueError, match="exactly one"):
        ManifestDataset(manifest)


def test_configs_are_the_main_protocol():
    from pathlib import Path
    configs = sorted(path for path in (Path(__file__).resolve().parents[1] / "configs" / "qwen").glob("*.json")
                     if not path.stem.endswith("_8bit"))
    assert len(configs) == 12
    for path in configs:
        config = TrainConfig.load(path)
        assert config.max_steps == 3001
        assert config.lora_r == 32 and config.lora_alpha == 64
        assert config.bit_levels == [2, 4, 8]
        assert config.visual_input_includes_text
        assert not config.use_global_negatives


@pytest.mark.parametrize("arm", ["d", "e"])
def test_eight_bit_control_retains_main_loss_schedule(arm):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "configs" / "qwen"
    main = TrainConfig.load(root / f"image_{arm}.json")
    control = TrainConfig.load(root / f"image_{arm}_8bit.json")
    assert control.bit_levels == [8]
    assert control.output_dir != main.output_dir
    assert control.gradient_checkpointing is True  # Recorded W6 launch setting.
    for key in ("arm", "modality", "temperature", "distill_temperature", "distill_weight",
                "distill_warmup_steps", "warmup_steps", "max_steps", "learning_rate",
                "per_device_batch_size", "global_batch_size", "lora_r", "lora_alpha"):
        assert getattr(control, key) == getattr(main, key)
