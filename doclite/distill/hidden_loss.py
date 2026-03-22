"""
Hidden-state distillation loss.

THEORY:
The student is trained to match the teacher's internal representations.
We compare transformer-layer hidden states using MSE.
"""

import torch.nn.functional as F
from doclite.distill.layer_mapper import build_layer_map


def hidden_state_loss(teacher_hidden_states, student_hidden_states):
    """
    Compute hidden-state distillation loss with uniform layer mapping.

    Args:
        teacher_hidden_states: tuple/list of tensors
            Includes embedding output at index 0
        student_hidden_states: tuple/list of tensors
            Includes embedding output at index 0

    Returns:
        scalar tensor
    """
    # Exclude embedding outputs; compare only transformer layers
    teacher_layers = teacher_hidden_states[1:]
    student_layers = student_hidden_states[1:]

    t_num = len(teacher_layers)
    s_num = len(student_layers)

    mapping = build_layer_map(t_num, s_num)

    total_loss = 0.0
    for s_idx, t_idx in enumerate(mapping):
        total_loss += F.mse_loss(student_layers[s_idx], teacher_layers[t_idx])

    return total_loss / s_num