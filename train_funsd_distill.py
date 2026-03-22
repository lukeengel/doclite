import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F

from doclite.configs.core import ENV, WEIGHTS
from doclite.data.document_dataset import DocumentDataset
from doclite.data.collate import collate_document_batch
from doclite.eval.evaluate import evaluate_student
from doclite.utils.checkpoint import save_checkpoint
from build_funsd_examples import load_funsd_split
from doclite.models.teacher import TeacherModel
from doclite.models.student import StudentModel
from doclite.distill.distill_loss import compute_distill_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
FUNSD_ROOT = ENV.DATA / "funsd"
TRAIN_ANN_DIR = FUNSD_ROOT / "training_data" / "annotations"
TEST_ANN_DIR = FUNSD_ROOT / "testing_data" / "annotations"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

# Load processed examples
train_examples = load_funsd_split(TRAIN_ANN_DIR, tokenizer)
test_examples = load_funsd_split(TEST_ANN_DIR, tokenizer)

# Dataset / DataLoader
train_dataset = DocumentDataset(train_examples)
test_dataset = DocumentDataset(test_examples)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_document_batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_document_batch
)

print("Number of train examples:", len(train_dataset))
print("Number of test examples:", len(test_dataset))
print("Number of train batches:", len(train_loader))
print("Number of test batches:", len(test_loader))
print("Using device:", device, flush=True)

# Models
teacher = TeacherModel("microsoft/layoutlmv3-base", num_labels=4).to(device)
student = StudentModel("SCUT-DLVCLab/lilt-roberta-en-base", num_labels=4).to(device)

# Freeze teacher explicitly
for param in teacher.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, weight_decay=0.01)

num_epochs = 3
best_f1 = -1.0

for epoch in range(num_epochs):
    teacher.eval()
    student.train()

    total_loss = 0.0
    total_distill = 0.0
    total_task = 0.0

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # Inputs without labels for teacher forward
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "bbox": batch["bbox"],
        }

        # Teacher outputs (frozen)
        teacher_out = teacher(**model_inputs)

        # Student HuggingFace forward for supervised token loss + internals
        student_hf_out = student.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            labels=batch["labels"],
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        student_out = {
            "logits": student_hf_out.logits,
            "hidden_states": student_hf_out.hidden_states,
            "attentions": student_hf_out.attentions,
        }

        # Distillation losses
        distill_losses = compute_distill_loss(teacher_out, student_out)
        distill_loss = distill_losses["loss_total"]

        # Supervised task loss on student
        task_loss = student_hf_out.loss

        # Final loss: distillation + supervised token classification
        loss = distill_loss + WEIGHTS.DELTA_TASK * task_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_distill += distill_loss.item()
        total_task += task_loss.item()

        if step % 10 == 0:
            print(
                f"Epoch {epoch + 1} | Step {step} | "
                f"total={loss.item():.4f} | "
                f"distill={distill_loss.item():.4f} | "
                f"task={task_loss.item():.4f}",
                flush=True
            )

    avg_total = total_loss / len(train_loader)
    avg_distill = total_distill / len(train_loader)
    avg_task = total_task / len(train_loader)

    metrics = evaluate_student(student, test_loader, device=device)

    print(
        f"Epoch {epoch + 1} | "
        f"train_total={avg_total:.4f} | "
        f"train_distill={avg_distill:.4f} | "
        f"train_task={avg_task:.4f} | "
        f"val_token_acc={metrics['token_acc']:.4f} | "
        f"val_token_f1={metrics['token_f1']:.4f}",
        flush=True
    )

    # Save every epoch
    save_checkpoint(
        student,
        optimizer,
        epoch,
        f"checkpoints/funsd_distill_epoch_{epoch + 1}.pt"
    )

    # Save best student by F1
    if metrics["token_f1"] > best_f1:
        best_f1 = metrics["token_f1"]
        save_checkpoint(
            student,
            optimizer,
            epoch,
            "checkpoints/funsd_distill_best.pt"
        )
        print(f"New best F1: {best_f1:.4f}", flush=True)