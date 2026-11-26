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
        assert set(metrics) == {"loss", "iqd_mean", "iqd_std", "unweighted_loss"}
        assert metrics["iqd_mean"] == metrics["iqd_std"] == 0
        torch.testing.assert_close(metrics["loss"], metrics["unweighted_loss"])
    if arm == "c":
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


@pytest.mark.parametrize("arm", ["b", "c", "e", "f"])
def test_scalar_infonce_keeps_reference_bf16_reduction_order(arm):
    generator = torch.Generator().manual_seed(8)
    visual = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    text = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    criterion = RankingDistillLoss(arm=arm)
    similarity = F.normalize(visual, dim=-1) @ F.normalize(text, dim=-1).T
    logits = similarity / 0.07 if arm == "b" else criterion.logit_scale * similarity
    labels = torch.arange(8)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
    actual = criterion._infonce(visual, text)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, (visual, text), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (visual, text))
    for value, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_bf16_bit_losses_use_reference_sequential_sum():
    generator = torch.Generator().manual_seed(7)
    visual = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator)
    text = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator)
    metrics = RankingDistillLoss(arm="e")(visual, text, distill_scale=0.37)
    quant = sum(metrics[f"quant_{bits}bit"] for bits in (2, 4, 8)) / 3
    distill = sum(metrics[f"distill_{bits}bit"] for bits in (2, 4, 8)) / 3
    torch.testing.assert_close(metrics["contrastive_quant"], quant, rtol=0, atol=0)
    torch.testing.assert_close(metrics["distill"], distill, rtol=0, atol=0)
    torch.testing.assert_close(metrics["loss"], metrics["contrastive_float"] + quant + (0.5 * 0.37) * distill,
                               rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["b", "c", "e", "f"])
@pytest.mark.parametrize("bits", [[2, 4, 8], [8]])
def test_bf16_total_dtype_matches_disabled_reference_terms(arm, bits):
    generator = torch.Generator().manual_seed(7)
    visual = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    text = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator, requires_grad=True)
    metrics = RankingDistillLoss(arm=arm, bit_levels=bits)(visual, text)
    expected = torch.float32 if arm == "c" or (arm != "b" and bits == [8]) else torch.bfloat16
    assert metrics["loss"].dtype == expected
    metrics["loss"].backward()
    assert visual.grad.dtype == text.grad.dtype == torch.bfloat16
    assert torch.isfinite(visual.grad).all() and torch.isfinite(text.grad).all()
