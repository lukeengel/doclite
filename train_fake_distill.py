import torch

from doclite.models.teacher import TeacherModel
from doclite.models.student import StudentModel
from doclite.train.train_step import run_train_step

device = "cuda" if torch.cuda.is_available() else "cpu"

teacher = TeacherModel("microsoft/layoutlmv3-base", num_labels=3).to(device)
student = StudentModel("SCUT-DLVCLab/lilt-roberta-en-base", num_labels=3).to(device)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

batch_size = 1
seq_len = 8


def make_fake_batch():
    x0 = torch.randint(0, 800, (batch_size, seq_len), dtype=torch.long)
    y0 = torch.randint(0, 800, (batch_size, seq_len), dtype=torch.long)
    w = torch.randint(1, 200, (batch_size, seq_len), dtype=torch.long)
    h = torch.randint(1, 200, (batch_size, seq_len), dtype=torch.long)

    x1 = torch.clamp(x0 + w, max=1000)
    y1 = torch.clamp(y0 + h, max=1000)

    bbox = torch.stack([x0, y0, x1, y1], dim=-1)

    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "bbox": bbox,
    }


for step in range(5):
    batch = make_fake_batch()
    losses = run_train_step(teacher, student, batch, optimizer, device=device)

    print(
        f"Step {step} | "
        f"hidden={losses['loss_hidden'].item():.4f} | "
        f"attn={losses['loss_attn'].item():.4f} | "
        f"logits={losses['loss_logits'].item():.4f} | "
        f"total={losses['loss_total'].item():.4f}"
    )