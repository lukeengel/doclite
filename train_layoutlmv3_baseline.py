import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LayoutLMv3Config, LayoutLMv3ForTokenClassification

from doclite.configs.core import ENV
from doclite.data.document_dataset import DocumentDataset
from doclite.data.collate import collate_document_batch
from doclite.eval.evaluate import evaluate_student
from doclite.utils.checkpoint import save_checkpoint
from build_funsd_examples import load_funsd_split

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

# LayoutLMv3 supervised model
config = LayoutLMv3Config.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=4,
    output_hidden_states=True,
    output_attentions=True,
)

teacher_model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    config=config
).to(device)

optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-5)

num_epochs = 3

# Small wrapper so we can reuse evaluate_student
class LayoutLMv3EvalWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, bbox):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
        )
        return {"logits": out.logits}

eval_wrapper = LayoutLMv3EvalWrapper(teacher_model).to(device)

for epoch in range(num_epochs):
    teacher_model.train()
    total_loss = 0.0

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        outputs = teacher_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            labels=batch["labels"],
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1} | Step {step} | batch_loss={loss.item():.4f}", flush=True)

    avg_train_loss = total_loss / len(train_loader)
    metrics = evaluate_student(eval_wrapper, test_loader, device=device)

    print(
        f"Epoch {epoch + 1} | "
        f"train_loss={avg_train_loss:.4f} | "
        f"val_token_acc={metrics['token_acc']:.4f} | "
        f"val_token_f1={metrics['token_f1']:.4f}",
        flush=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": teacher_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f"checkpoints/layoutlmv3_baseline_epoch_{epoch + 1}.pt"
    )