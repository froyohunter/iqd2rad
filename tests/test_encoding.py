import numpy as np
import pytest

from iqd2rad.models.encode import correct_indices, model_input


def test_gallery_ground_truth_is_id_based():
    rows = [{"id": "q0", "positive_ids": ["b"]}, {"id": "q1", "positive_ids": ["a", "b"]}]
    np.testing.assert_array_equal(correct_indices(rows, ["a", "b"]), [[1, -1], [0, 1]])
    with pytest.raises(ValueError, match="missing"):
        correct_indices(rows, ["a"])


def test_visual_candidates_do_not_encode_captions(tmp_path):
    with pytest.raises(ValueError, match="caption"):
        model_input({"id": "image", "image": "x.jpg", "text": "the answer"}, tmp_path, candidate=True)
    assert model_input({"id": "v", "frames": ["a.jpg"]}, tmp_path, candidate=True) == {
        "video": [str(tmp_path / "a.jpg")]}


def test_last_token_pool_handles_left_and_right_padding():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from iqd2rad.models.qwen import last_token_pool
    hidden = torch.arange(12.0).view(2, 3, 2).requires_grad_()
    actual = last_token_pool(hidden, torch.tensor([[1, 1, 0], [0, 1, 1]]))
    assert actual.tolist() == [[2., 3.], [10., 11.]]
    actual.sum().backward()
    assert hidden.grad.sum().item() == 4


def test_local_media_with_spaces_and_unicode(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("qwen_vl_utils")
    from PIL import Image
    from qwen_vl_utils.vision_process import fetch_image
    from iqd2rad.models.qwen import QwenEncoder

    path = tmp_path / "a frame 中文.png"
    Image.new("RGB", (64, 64), color="blue").save(path)
    resolved = QwenEncoder._media_path(path)
    assert resolved == str(path)
    image = fetch_image({"image": resolved}, image_patch_size=16)
    assert image.size == (64, 64)


@pytest.mark.parametrize("backend", [None, "sdpa"])
def test_encoding_cli_passes_backend_and_records_processor(tmp_path, monkeypatch, backend):
    import json
    import sys
    from types import ModuleType, SimpleNamespace
    from iqd2rad.models.encode import main

    calls = []

    class Encoder:
        def __init__(self, base, **kwargs):
            calls.append(kwargs)
            self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="base-commit"))
            self.processor_source = base
            self.processor_revision = "base-commit"
            self.processor_source_kind = "base_without_adapter_processor"
            self.attn_implementation = kwargs["attn_implementation"]
            self.torch_dtype = "torch.bfloat16"

        def encode(self, rows, **kwargs):
            assert kwargs["normalize"] is False
            return np.ones((len(rows), 4), dtype=np.float32)

    fake = ModuleType("iqd2rad.models.qwen")
    fake.QwenEncoder = Encoder
    monkeypatch.setitem(sys.modules, "iqd2rad.models.qwen", fake)
    queries, candidates = tmp_path / "queries.jsonl", tmp_path / "candidates.jsonl"
    queries.write_text(json.dumps({"id": "q", "text": "a query", "positive_ids": ["c"]}))
    candidates.write_text(json.dumps({"id": "c", "image": "image.jpg"}))
    output = tmp_path / "embeddings.npz"
    args = ["iqd2rad-encode", "--queries", str(queries), "--candidates", str(candidates),
            "--data-root", str(tmp_path), "--output", str(output)]
    if backend:
        args += ["--attn-implementation", backend, "--device", "cpu", "--dtype", "float32"]
    monkeypatch.setattr(sys, "argv", args)
    main()
    expected = backend or "flash_attention_2"
    assert calls[0]["attn_implementation"] == expected
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["attn_implementation"] == expected
    assert metadata["processor_source_kind"] == "base_without_adapter_processor"
    assert metadata["processor_revision"] == "base-commit"


def test_processor_selection_prefers_export_and_supports_weights_only_adapter(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from iqd2rad.models.qwen import resolve_processor_source
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError

    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    kwargs = {"revision": "base-commit", "adapter_source": str(adapter),
              "adapter_revision": "adapter-commit"}
    assert resolve_processor_source(str(base), **kwargs) == (
        str(base), "base-commit", "base_without_adapter_processor")
    (adapter / "preprocessor_config.json").write_text("{}")
    assert resolve_processor_source(str(base), **kwargs) == (str(adapter), "adapter-commit", "adapter")
    assert resolve_processor_source(str(base), processor_source=str(base), **kwargs) == (
        str(base), "base-commit", "explicit")

    def download(repo, filename, **kwargs):
        assert repo == "example/adapter" and filename == "preprocessor_config.json"
        assert kwargs == {"revision": "adapter-commit", "local_files_only": True}
        raise EntryNotFoundError("Processor not included")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    kwargs.update(adapter_source="example/adapter", local_files_only=True)
    assert resolve_processor_source(str(base), **kwargs) == (
        str(base), "base-commit", "base_without_adapter_processor")

    def uncached(*args, **kwargs):
        raise LocalEntryNotFoundError("Processor is not cached for offline use")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", uncached)
    assert resolve_processor_source(str(base), **kwargs) == (
        str(base), "base-commit", "base_adapter_processor_unavailable_offline")

    def denied(*args, **kwargs):
        raise HfHubHTTPError("Authorization failed")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", denied)
    with pytest.raises(HfHubHTTPError, match="Authorization"):
        resolve_processor_source(str(base), **kwargs)
