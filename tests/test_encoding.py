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
