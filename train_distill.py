import torch
from torch.utils.data import DataLoader

from doclite.models.teacher import TeacherModel
from doclite.models.student import StudentModel
from doclite.train.train_step import run_train_step
from doclite.data.document_dataset import DocumentDataset
from doclite.data.collate import collate_document_batch

from doclite.utils.checkpoint import save_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

# Replace this with real preprocessed examples later
dummy_examples = [
    {
        "input_ids": [10, 11, 12, 13, 0, 0, 0, 0],
        "attention_mask": [1, 1, 1, 1, 0, 0, 0, 0],
        "bbox": [[0, 0, 10, 10]] * 8,
        "labels": [0, 1, 0, 2, -100, -100, -100, -100],
    }
    for _ in range(10)
]

dataset = DocumentDataset(dummy_examples)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_document_batch)

teacher = TeacherModel("microsoft/layoutlmv3-base", num_labels=3).to(device)
student = StudentModel("SCUT-DLVCLab/lilt-roberta-en-base", num_labels=3).to(device)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

num_epochs = 2

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch in loader:
        # labels are not used yet in distill loss, so remove for forward compatibility
        batch_for_models = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "bbox": batch["bbox"],
        }

        losses = run_train_step(teacher, student, batch_for_models, optimizer, device=device)
        total_loss += losses["loss_total"].item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1} | avg total loss = {avg_loss:.4f}")


    save_checkpoint(
        student,
        optimizer,
        epoch,
        f"checkpoints/student_epoch_{epoch + 1}.pt"
    )