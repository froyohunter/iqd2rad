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


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cpu_training_accumulates_and_exports_without_model_download(tmp_path, monkeypatch, precision):
    """Exercise the real optimizer loop with an in-memory tiny encoder, not Qwen."""
    import json
    import sys
    from types import ModuleType
    from iqd2rad.training.trainer import train

    instances = []
    logged = []
    seeds = []
    import iqd2rad.training.trainer as trainer_module
    original_seed = trainer_module._seed_training

    def record_seed(seed, rank):
        seeds.append((seed, rank))
        original_seed(seed, rank)

    monkeypatch.setattr(trainer_module, "_seed_training", record_seed)

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
            self.model = TinyModel().to(dtype=kwargs["torch_dtype"])
            self.processor = Processor()
            self.calls = 0
            instances.append(self)

        def encode_batch(self, inputs, normalize=False):
            self.calls += 1
            assert not torch.is_autocast_enabled("cpu")
            features = torch.tensor([[float(record["text"].split()[-1]) + 1, 1.0] for record in inputs],
                                    dtype=self.model.weight.dtype)
            return self.model(features)

    def attach_lora(model, config):
        model.peft_config = {"default": config}
        return model

    fake_peft = ModuleType("peft")
    fake_peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_peft.get_peft_model = attach_lora
    fake_wandb = ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: SimpleNamespace(
        log=lambda metrics, step: logged.append((step, dict(metrics))), finish=lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
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
                         output_dir=str(tmp_path / "run"), max_steps=3, warmup_steps=0,
                         per_device_batch_size=2, global_batch_size=4, device="cpu",
                         precision=precision, attn_implementation="sdpa", save_every=1,
                         wandb_project="test", log_every=10)
    train(config)
    assert instances[0].calls == 12  # 3 updates × 2 microbatches × 2 modalities
    assert seeds == [(42, 0)]  # No reseed after LoRA/model initialization.
    assert not (tmp_path / "run/checkpoint-1").exists()
    for step in (2, 3):
        assert (tmp_path / f"run/checkpoint-{step}/training_config.json").is_file()
    assert [step for step, _ in logged] == [1, 2, 3]  # Independent of console log_every.
    for step, metrics in logged:
        assert all(key.startswith("train/") for key in metrics)
        assert metrics["train/global_step"] == step
        assert metrics["train/distill_weight"] == pytest.approx(0.5 * (step + 1) / 200)
        assert metrics["train/loss_term_contrastive_term"] == pytest.approx(
            metrics["train/contrastive_float"] + metrics["train/contrastive_quant"], abs=0.05)
        assert metrics["train/loss_term_matryoshka_term"] == 0
    metadata = json.loads((tmp_path / "run/final/metadata.json").read_text())
    assert metadata["global_step"] == 3
    assert metadata["gradient_accumulation_steps"] == 2
    assert metadata["training_pairs_include_caption_in_visual_input"] is True
    info = json.loads((tmp_path / "run/final/training_info.json").read_text())
    assert info == {"global_step": 3, "num_training_samples": 4, "num_training_batches": 2,
                    "num_val_samples": 0, "world_size": 1, "effective_batch_size": 4,
                    "per_device_batch_size": 2, "gradient_accumulation_steps": 2,
                    "current_lr": 0.0, "is_final": True}
    periodic = json.loads((tmp_path / "run/checkpoint-3/training_info.json").read_text())
    assert periodic == dict(info, is_final=False)


@pytest.mark.parametrize("size, batches, tail", [(7010, 55, 7), (82783, 647, 6)])
def test_reference_distributed_sampler_keeps_order_and_tail(size, batches, tail):
    from torch.utils.data import BatchSampler, DistributedSampler
    from iqd2rad.training.trainer import _training_loader

    config = TrainConfig(per_device_batch_size=8, seed=42)
    dataset = range(size)
    loader, sampler = _training_loader(dataset, config, world_size=16, rank=0)
    reference = DistributedSampler(dataset, num_replicas=16, rank=0, shuffle=True)
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        reference.set_epoch(epoch)
        actual = list(loader.batch_sampler)
        expected = list(BatchSampler(reference, batch_size=8, drop_last=False))
        assert actual == expected
        assert len(actual) == batches and len(actual[-1]) == tail
    assert loader.generator is None


def test_singleton_tail_is_rejected_without_dropping_samples():
    from iqd2rad.training.trainer import _training_loader

    with pytest.raises(ValueError, match="singleton tail"):
        _training_loader(range(9), TrainConfig(per_device_batch_size=8), world_size=1, rank=0)


def test_reference_periodic_checkpoint_times():
    from iqd2rad.training.trainer import _periodic_checkpoint

    assert [step for step in range(1, 3002) if _periodic_checkpoint(step, 500)] == [
        501, 1001, 1501, 2001, 2501, 3001,
    ]
    assert not any(_periodic_checkpoint(step, 0) for step in range(10))


class _GlooEncoder:
    def __init__(self, checkpointing=False):
        self.model = torch.nn.Linear(2, 4, bias=False)
        self.checkpointing = checkpointing

    def encode_batch(self, inputs, normalize=False):
        from torch.utils.checkpoint import checkpoint
        values = [[float(row["text"]), 1.0 if "image" in row else -1.0] for row in inputs]
        features = torch.tensor(values)
        if self.checkpointing:
            return checkpoint(self.model, features.requires_grad_(), use_reentrant=True)
        return self.model(features)


def _gloo_gradient_worker(rank, init_file, output_dir, checkpointing):
    from contextlib import nullcontext
    from pathlib import Path
    from torch.nn.parallel import DistributedDataParallel
    import torch.distributed as dist
    from iqd2rad.training.loss import RankingDistillLoss

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42 + rank)
        pair = PairEncoder(_GlooEncoder(checkpointing))
        ddp = DistributedDataParallel(pair, find_unused_parameters=False)
        if checkpointing:
            ddp._set_static_graph()
        reference = PairEncoder(_GlooEncoder(checkpointing))
        reference.load_state_dict(pair.state_dict())
        criterion = RankingDistillLoss(arm="e")
        optimizer = torch.optim.AdamW(pair.parameters(), lr=1e-5, weight_decay=0.01)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-5, weight_decay=0.01)

        def batch(worker, micro, update):
            return [{"image": "unused.jpg", "text": str(update * 8 + worker * 4 + micro * 2 + i + 1)}
                    for i in range(2)]

        for update in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                context = ddp.no_sync() if micro == 0 and not checkpointing else nullcontext()
                with context:
                    visual, text = ddp(batch(rank, micro, update))
                    (criterion(visual, text)["loss"] / 2).backward()
            # Average the same local losses across ranks and accumulated
            # microbatches, retaining each local negative pool.
            for worker in range(2):
                for micro in range(2):
                    visual, text = reference(batch(worker, micro, update))
                    (criterion(visual, text)["loss"] / 4).backward()
            torch.testing.assert_close(pair.model.weight.grad, reference.model.weight.grad,
                                       rtol=2e-5, atol=2e-6)
            optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(pair.model.weight, reference.model.weight,
                                       rtol=2e-5, atol=2e-6)
        torch.save(pair.model.weight.grad, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gloo_pair_ddp_matches_local_loss_gradient_average(tmp_path, checkpointing):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable in this Torch build")
    torch.multiprocessing.spawn(
        _gloo_gradient_worker,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), checkpointing),
        nprocs=2, join=True,
    )
    first = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    second = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
