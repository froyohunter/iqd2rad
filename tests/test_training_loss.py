import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from iqd2rad.training.loss import RankingDistillLoss, uniform_ste_quantize


def test_quantization_forward_levels_and_straight_through_gradient():
    source = torch.tensor([[0.2, 0.3, -0.7], [-0.1, 0.8, 0.3]], requires_grad=True)
    result = uniform_ste_quantize(source, 2)
    levels = (result.detach() + 1) * 1.5
    torch.testing.assert_close(levels, levels.round())
    upstream = torch.tensor([[0.1, -0.3, 0.7], [0.2, 0.4, 0.8]])
    actual = torch.autograd.grad((result * upstream).sum(), source)[0]
    expected = torch.autograd.grad((F.normalize(source, dim=-1) * upstream).sum(), source)[0]
    torch.testing.assert_close(actual, expected)
    assert actual.norm() > 0


def test_ranking_kl_detaches_teacher_and_corpus():
    generator = torch.Generator().manual_seed(7)
    teacher = torch.randn(4, 6, generator=generator, requires_grad=True)
    student = torch.randn(4, 6, generator=generator, requires_grad=True)
    corpus = torch.randn(4, 6, generator=generator, requires_grad=True)
    loss = RankingDistillLoss(arm="e")._ranking_kl(teacher, student, corpus).mean()
    loss.backward()
    assert teacher.grad is None and corpus.grad is None
    assert student.grad is not None and student.grad.norm() > 0


def test_online_weights_match_sample_zscore_and_clamp_without_renormalization():
    criterion = RankingDistillLoss(arm="f", online_diff_clamp=1.5)
    float_loss = torch.zeros(4, requires_grad=True)
    quant_loss = torch.tensor([0.0, 0.0, 0.0, 8.0], requires_grad=True)
    diff = quant_loss.detach()
    expected = (torch.softmax((diff - diff.mean()) / (diff.std() + 1e-8), dim=0) * 4).clamp(0, 1.5)
    weights = criterion._online_weights(float_loss, quant_loss)
    torch.testing.assert_close(weights, expected)
    assert not weights.requires_grad
    assert weights.sum() < 4


@pytest.mark.parametrize("arm", list("bcdef"))
def test_main_arms_have_finite_loss_and_gradients(arm):
    generator = torch.Generator().manual_seed(3)
    visual = torch.randn(4, 9, generator=generator, requires_grad=True)
    text = torch.randn(4, 9, generator=generator, requires_grad=True)
    metrics = RankingDistillLoss(arm=arm)(visual, text, distill_scale=0.5)
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(visual.grad).all() and visual.grad.norm() > 0
    assert torch.isfinite(text.grad).all() and text.grad.norm() > 0
    if arm == "b":
        assert metrics["contrastive_quant"] == 0
    if arm in "bc":
        assert metrics["distill"] == 0


def test_zero_distillation_warmup_recovers_qat_loss():
    generator = torch.Generator().manual_seed(23)
    visual = torch.randn(4, 11, generator=generator)
    text = torch.randn(4, 11, generator=generator)
    qat = RankingDistillLoss(arm="c")(visual, text)["loss"]
    rad_at_zero = RankingDistillLoss(arm="e")(visual, text, distill_scale=0)["loss"]
    torch.testing.assert_close(qat, rad_at_zero)


@pytest.mark.parametrize("arm", ["d", "e"])
def test_eight_bit_control_uses_no_lower_bit_branch(arm):
    generator = torch.Generator().manual_seed(9)
    visual = torch.randn(4, 13, generator=generator, requires_grad=True)
    text = torch.randn(4, 13, generator=generator, requires_grad=True)
    criterion = RankingDistillLoss(arm=arm, bit_levels=[8])
    metrics = criterion(visual, text, distill_scale=0.25)
    assert "quant_8bit" in metrics and "distill_8bit" in metrics
    assert not any(key.endswith(("2bit", "4bit")) for key in metrics)
    expected = metrics["contrastive_float"] + metrics["quant_8bit"] + 0.5 * 0.25 * metrics["distill_8bit"]
    torch.testing.assert_close(metrics["loss"].detach(), expected)
    metrics["loss"].backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all() and visual.grad.norm() > 0
    assert text.grad is not None and torch.isfinite(text.grad).all() and text.grad.norm() > 0
