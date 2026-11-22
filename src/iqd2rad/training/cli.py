"""Run with python -m iqd2rad.training.cli --config configs/qwen/video_e.json."""

import argparse
import json

from .config import TrainConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen main-protocol LoRA training (arms a–f)")
    parser.add_argument("--config", required=True, help="JSON training configuration")
    parser.add_argument("--manifest", help="Override the JSONL training manifest")
    parser.add_argument("--data-root", help="Root for relative media paths in the manifest")
    parser.add_argument("--output-dir", help="Override the run output directory (must be empty)")
    parser.add_argument("--attn-implementation", choices=["flash_attention_2", "sdpa", "eager"], help="Override attention backend")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="Override numeric precision")
    parser.add_argument("--device", help="Override training device, e.g. cuda:0 or cpu")
    parser.add_argument("--base-revision", help="Pin the base model to a Hugging Face commit")
    parser.add_argument("--validate-config", action="store_true", help="Validate and print configuration without loading models")
    args = parser.parse_args(argv)
    try:
        config = TrainConfig.load(args.config)
        for name in ("manifest", "data_root", "output_dir", "attn_implementation", "precision", "device", "base_revision"):
            if getattr(args, name) is not None:
                setattr(config, name, getattr(args, name))
        config.validate()
        if args.validate_config:
            print(json.dumps(config.to_dict(), indent=2))
            return
        if config.arm == "a":
            print(f"Arm a uses the untrained base model {config.base_model}; run retrieval evaluation with PTQ. No training is performed.")
            return
        try:
            from .trainer import train
            train(config)
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"Missing training dependency {exc.name!r}. Install 'iqd2rad[train]'.") from exc
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Training error: {exc}\n")


if __name__ == "__main__":
    main()
