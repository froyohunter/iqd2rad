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
        self.bit_levels = tuple([2, 4, 8] if bit_levels is None else bit_levels)
        if self.bit_levels not in ((2, 4, 8), (8,)):
            raise ValueError("bit_levels must be [2, 4, 8] or [8]")
        self.distill_temperature = distill_temperature
        self.distill_weight = distill_weight
        self.online_diff_temperature = online_diff_temperature
        self.online_diff_zscore = online_diff_zscore
        self.online_diff_clamp = online_diff_clamp
        self.register_buffer("logit_scale", torch.tensor(1.0 / temperature))

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
        float_per = self._per_sample_infonce(visual, text)
        float_loss = float_per.mean()
        quant_losses, distill_losses = [], []
        metrics = {"contrastive_float": float_loss.detach()}
        if self.arm != "b":
            # Preserve the two normalization stages in original _build_narrow_dict.
            visual_cont = F.normalize(visual, dim=-1)
            text_cont = F.normalize(text, dim=-1)
            for bits in self.bit_levels:
                vq = uniform_ste_quantize(visual_cont, bits)
                tq = uniform_ste_quantize(text_cont, bits)
                quant_per = self._per_sample_infonce(vq, tq)
                quant_losses.append(quant_per.mean())
                metrics[f"quant_{bits}bit"] = quant_per.mean().detach()
                if self.arm == "d":
                    # MSE uses RAW float embeddings as teacher, matching source.
                    per = (F.mse_loss(visual.detach(), vq, reduction="none").mean(-1)
                           + F.mse_loss(text.detach(), tq, reduction="none").mean(-1)) / 2
                elif self.arm in ("e", "f"):
                    per = (self._ranking_kl(visual, vq, text)
                           + self._ranking_kl(text, tq, visual)) / 2
                    if self.arm == "f":
                        per = self._online_weights(float_per, quant_per) * per
                        metrics[f"online_diff_{bits}bit"] = (quant_per - float_per).mean().detach()
                else:
                    continue
                distill_losses.append(per.mean())
                metrics[f"distill_{bits}bit"] = per.mean().detach()
        quant_loss = torch.stack(quant_losses).mean() if quant_losses else float_loss.new_zeros(())
        distill_loss = torch.stack(distill_losses).mean() if distill_losses else float_loss.new_zeros(())
        loss = float_loss + quant_loss + self.distill_weight * distill_scale * distill_loss
        metrics.update(contrastive_quant=quant_loss.detach(), distill=distill_loss.detach(), loss=loss)
        return metrics
