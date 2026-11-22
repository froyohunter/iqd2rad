"""Exercise the real Transformers/PEFT integration without downloading weights."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from transformers import Qwen3VLConfig
from iqd2rad.models.qwen import Qwen3VLForEmbedding


def test_tiny_qwen_adapter_backward_and_reload(tmp_path):
    config = Qwen3VLConfig(
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 8,
                     "max_position_embeddings": 128,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32, "deepstack_visual_indexes": []},
    )
    torch.manual_seed(42)
    model = Qwen3VLForEmbedding(config)
    model.save_pretrained(tmp_path / "base")
    model = Qwen3VLForEmbedding.from_pretrained(tmp_path / "base")
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        task_type="FEATURE_EXTRACTION", lora_dropout=0,
    ))
    inputs = {"input_ids": torch.tensor([[1, 2, 3], [2, 3, 4]]),
              "attention_mask": torch.ones(2, 3, dtype=torch.long)}
    output = model(**inputs).last_hidden_state
    assert output.shape == (2, 3, 32)
    output.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in model.named_parameters() if "lora_" in n)
    model.eval()
    model.save_pretrained(tmp_path / "adapter")
    restored = peft.PeftModel.from_pretrained(
        Qwen3VLForEmbedding.from_pretrained(tmp_path / "base"), tmp_path / "adapter",
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs).last_hidden_state,
                                   restored(**inputs).last_hidden_state)
