import torch


def save_checkpoint(student, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )