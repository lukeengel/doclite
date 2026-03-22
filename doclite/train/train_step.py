import torch
from doclite.distill.distill_loss import compute_distill_loss


def run_train_step(teacher, student, batch, optimizer, device="cpu"):
    """
    Run one distillation training step.

    Args:
        teacher: frozen teacher model
        student: trainable student model
        batch: dict with input_ids, attention_mask, bbox, ...
        optimizer: torch optimizer
        device: cpu or cuda

    Returns:
        dict of losses
    """
    teacher.eval()  # put teacher in evaluation mode
    student.train()

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer.zero_grad()

    # Teacher forward (already no_grad inside teacher wrapper)
    teacher_out = teacher(**batch)

    # Student forward
    student_out = student(**batch)

    # Distillation loss
    losses = compute_distill_loss(teacher_out, student_out)
    total_loss = losses["loss_total"]

    # Backprop through student only
    total_loss.backward()
    optimizer.step()

    return losses