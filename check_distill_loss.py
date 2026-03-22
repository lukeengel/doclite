import torch

from doclite.models.teacher import TeacherModel
from doclite.models.student import StudentModel
from doclite.distill.distill_loss import compute_distill_loss

teacher = TeacherModel("microsoft/layoutlmv3-base", num_labels=3)
student = StudentModel("SCUT-DLVCLab/lilt-roberta-en-base", num_labels=3)

batch_size = 1
seq_len = 8

# Valid bbox generation
x0 = torch.randint(0, 800, (batch_size, seq_len), dtype=torch.long)
y0 = torch.randint(0, 800, (batch_size, seq_len), dtype=torch.long)
w = torch.randint(1, 200, (batch_size, seq_len), dtype=torch.long)
h = torch.randint(1, 200, (batch_size, seq_len), dtype=torch.long)

x1 = torch.clamp(x0 + w, max=1000)
y1 = torch.clamp(y0 + h, max=1000)

bbox = torch.stack([x0, y0, x1, y1], dim=-1)

inputs = {
    "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
    "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    "bbox": bbox,
}

teacher_out = teacher(**inputs)
student_out = student(**inputs)

losses = compute_distill_loss(teacher_out, student_out)

print("Hidden loss:", losses["loss_hidden"].item())
print("Attention loss:", losses["loss_attn"].item())
print("Logits loss:", losses["loss_logits"].item())
print("Total loss:", losses["loss_total"].item())