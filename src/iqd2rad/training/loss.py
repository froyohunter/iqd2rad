"""Main Qwen loss, extracted from the original HiQ RankingDistillLoss.

The quantized contrastive and distillation losses are averaged across bits.
Ranking distillation is KL(teacher || student); both the float teacher and
the opposite-modality corpus are detached. No auxiliary head is trained.
"""

import torch
from torch import nn
from torch.nn import functional as F


def uniform_ste_quantize(embeddings: torch.Tensor, bits: int) -> torch.Tensor:
    """L2 normalize, then uniform STE on the fixed symmetric interval [-1, 1]."""
    if bits not in (2, 4, 8):
        raise ValueError("Main-protocol training supports 2, 4, and 8 bits")
    x = F.normalize(embeddings, dim=-1)
    levels = 2**bits - 1
    continuous = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0) * levels
    rounded = continuous + (torch.round(continuous) - continuous).detach()
    return (rounded / levels) * 2.0 - 1.0


class RankingDistillLoss(nn.Module):
    def __init__(self, arm="e", temperature=0.07, distill_temperature=0.1,
                 distill_weight=0.5, online_diff_temperature=1.0,
                 online_diff_zscore=True, online_diff_clamp=5.0, bit_levels=None):
        super().__init__()
        if arm not in tuple("bcdef"):
            raise ValueError("Arm a is the untrained base model; training arms are b–f")
        self.arm = arm
        self.temperature = temperature
        self.bit_levels = tuple([2, 4, 8] if bit_levels is None else bit_levels)
        if self.bit_levels not in ((2, 4, 8), (8,)):
            raise ValueError("bit_levels must be [2, 4, 8] or [8]")
        self.distill_temperature = distill_temperature
        self.distill_weight = distill_weight
        self.online_diff_temperature = online_diff_temperature
        self.online_diff_zscore = online_diff_zscore
        self.online_diff_clamp = online_diff_clamp
        self.register_buffer("logit_scale", torch.tensor(1.0 / temperature))

    def _infonce(self, visual, text):
        similarities = F.normalize(visual, dim=-1) @ F.normalize(text, dim=-1).T
        logits = similarities / max(1e-6, self.temperature) if self.arm == "b" else self.logit_scale * similarities
        labels = torch.arange(visual.shape[0], device=visual.device)
        # Keep the source scalar reduction order, including BF16 rounding.
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

    def _per_sample_infonce(self, visual, text):
        logits = self.logit_scale * (F.normalize(visual, dim=-1) @ F.normalize(text, dim=-1).T)
        labels = torch.arange(visual.shape[0], device=visual.device)
        return (F.cross_entropy(logits, labels, reduction="none")
                + F.cross_entropy(logits.T, labels, reduction="none")) / 2.0

    def _ranking_kl(self, teacher, student, corpus):
        teacher = F.normalize(teacher.detach(), dim=-1)
        student = F.normalize(student, dim=-1)
        corpus = F.normalize(corpus.detach(), dim=-1)
        teacher_logits = teacher @ corpus.T / self.distill_temperature
        student_logits = student @ corpus.T / self.distill_temperature
        p = F.softmax(teacher_logits, dim=-1)
        logq = F.log_softmax(student_logits, dim=-1)
        return (p * (p.log() - logq)).sum(dim=-1)

    def _online_weights(self, float_loss, quant_loss):
        diff = (quant_loss - float_loss).detach()
        if self.online_diff_zscore:
            # sample std (correction=1) matches the original; PBS >= 2 is required.
            diff = (diff - diff.mean()) / (diff.std() + 1e-8)
        else:
            diff = diff - diff.mean()
        weights = F.softmax(diff / self.online_diff_temperature, dim=0) * diff.shape[0]
        # Deliberately do not renormalize after clamping: original P03 behavior.
        return weights.clamp(min=0.0, max=self.online_diff_clamp)

    def forward(self, visual, text, distill_scale=1.0):
        if visual.shape != text.shape or visual.ndim != 2 or visual.shape[0] < 2:
            raise ValueError("Expected matching visual/text [batch >= 2, dimension] tensors")
        narrow_visual, narrow_text = {}, {}
        if self.arm != "b":
            # Preserve the source graph construction and double normalization.
            visual_cont = F.normalize(visual, dim=-1)
            narrow_visual = {bits: uniform_ste_quantize(visual_cont, bits) for bits in self.bit_levels}
            text_cont = F.normalize(text, dim=-1)
            narrow_text = {bits: uniform_ste_quantize(text_cont, bits) for bits in self.bit_levels}
        float_loss = self._infonce(visual, text)
        if self.arm == "b":
            return {"loss": float_loss, "iqd_mean": float_loss.detach().new_zeros(()),
                    "iqd_std": float_loss.detach().new_zeros(()), "unweighted_loss": float_loss.detach()}
        quant_losses = {bits: self._infonce(narrow_visual[bits], narrow_text[bits])
                        for bits in self.bit_levels if bits in narrow_visual}
        quant_loss = sum(quant_losses.values()) / len(quant_losses) if quant_losses else float_loss.new_zeros(())
        contrastive_loss = float_loss + quant_loss
        distill_losses, online_diff = {}, {}
        if self.arm == "d":
            for bits in self.bit_levels:
                per = (F.mse_loss(visual.detach(), narrow_visual[bits], reduction="none").mean(-1)
                       + F.mse_loss(text.detach(), narrow_text[bits], reduction="none").mean(-1)) / 2.0
                distill_losses[bits] = per.mean()
        elif self.arm in ("e", "f"):
            float_per = self._per_sample_infonce(visual, text).detach()
            for bits in self.bit_levels:
                per = (self._ranking_kl(visual, narrow_visual[bits], text)
                       + self._ranking_kl(text, narrow_text[bits], visual)) / 2.0
                if self.arm == "f":
                    quant_per = self._per_sample_infonce(narrow_visual[bits], narrow_text[bits]).detach()
                    per = self._online_weights(float_per, quant_per) * per
                    online_diff[bits] = (quant_per - float_per).mean().detach()
                distill_losses[bits] = per.mean()
        distill_loss = (sum(distill_losses.values()) / len(distill_losses) if distill_losses
                        else torch.tensor(0.0, device=visual.device))
        distill_term = (self.distill_weight * distill_scale) * distill_loss
        # Preserve the disabled auxiliary term's scalar dtype without computing
        # an additional objective: the source returns an FP32 zero for one bit.
        auxiliary_zero = (torch.tensor(0.0, device=visual.device) if len(self.bit_levels) == 1
                          else float_loss.new_zeros(()))
        loss = contrastive_loss + distill_term + auxiliary_zero
        metrics = {"contrastive_float": float_loss.detach(), "contrastive_quant": quant_loss.detach(),
                   "distill": distill_loss.detach(), "loss": loss,
                   "loss_term_contrastive_term": contrastive_loss.detach(),
                   "loss_term_distill_term": distill_term.detach(),
                   "loss_term_matryoshka_term": auxiliary_zero}
        metrics.update({f"quant_{bits}bit": value.detach() for bits, value in quant_losses.items()})
        metrics.update({f"distill_{bits}bit": value.detach() for bits, value in distill_losses.items()})
        metrics.update({f"online_diff_{bits}bit": value for bits, value in online_diff.items()})
        return metrics
