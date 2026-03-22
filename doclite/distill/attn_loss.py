"""
Attention distillation loss.

THEORY:
The student is trained to match the teacher's attention maps,
so it learns similar token-to-token interaction patterns.
"""

import torch.nn.functional as F
from doclite.distill.layer_mapper import build_layer_map


def attention_loss(teacher_attentions, student_attentions):
    """
    Compute attention distillation loss with uniform layer mapping.

    Args:
        teacher_attentions: tuple/list of tensors
        student_attentions: tuple/list of tensors

    Returns:
        scalar tensor
    """
    t_num = len(teacher_attentions)
    s_num = len(student_attentions)

    mapping = build_layer_map(t_num, s_num)

    total_loss = 0.0
    for s_idx, t_idx in enumerate(mapping):
        total_loss += F.mse_loss(student_attentions[s_idx], teacher_attentions[t_idx])

    return total_loss / s_num