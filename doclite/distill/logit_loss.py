"""
Logits distillation loss.

THEORY:
The student is trained to match the teacher's output distribution
using KL divergence with temperature scaling.
"""

import torch
import torch.nn.functional as F


def logits_loss(teacher_logits, student_logits, temperature: float = 2.0, attention_mask=None):
    """
    Compute KL-divergence distillation loss on logits.

    Args:
        teacher_logits: [batch, seq_len, num_labels]
        student_logits: [batch, seq_len, num_labels]
        temperature: softening factor
        attention_mask: [batch, seq_len] optional mask to exclude padding

    Returns:
        scalar tensor
    """
    T = temperature

    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)

    if attention_mask is not None:
        # Mask out padding tokens so they don't contribute to loss
        mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        kl_masked = (kl_per_token * mask).sum() / mask.sum()
        return kl_masked * (T ** 2)

    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    return loss * (T ** 2)