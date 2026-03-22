"""
Combined SlimDoc-style distillation loss.

THEORY:
This combines internal and external supervision:
- hidden-state loss
- attention loss
- logits loss
"""

from doclite.configs.core import WEIGHTS
from doclite.distill.hidden_loss import hidden_state_loss
from doclite.distill.attn_loss import attention_loss
from doclite.distill.logit_loss import logits_loss


def compute_distill_loss(teacher_out, student_out, attention_mask=None):
    """
    Compute total weighted distillation loss.

    Args:
        teacher_out: dict with keys
            - hidden_states
            - attentions
            - logits
        student_out: dict with same keys
        attention_mask: [batch, seq_len] optional mask to exclude padding

    Returns:
        dict with individual loss terms and total loss
    """
    loss_hidden = hidden_state_loss(
        teacher_out["hidden_states"],
        student_out["hidden_states"]
    )

    loss_attn = attention_loss(
        teacher_out["attentions"],
        student_out["attentions"]
    )

    loss_logits = logits_loss(
        teacher_out["logits"],
        student_out["logits"],
        attention_mask=attention_mask,
    )

    total_loss = (
        WEIGHTS.ALPHA_HIDDEN * loss_hidden
        + WEIGHTS.BETA_ATTN * loss_attn
        + WEIGHTS.GAMMA_LOGITS * loss_logits
    )

    return {
        "loss_hidden": loss_hidden,
        "loss_attn": loss_attn,
        "loss_logits": loss_logits,
        "loss_total": total_loss,
    }