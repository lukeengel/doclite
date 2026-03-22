import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate_student(student, dataloader, device="cpu"):
    student.eval()

    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        fwd_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "bbox": batch["bbox"],
        }
        if "pixel_values" in batch:
            fwd_kwargs["pixel_values"] = batch["pixel_values"]

        outputs = student(**fwd_kwargs)

        preds = outputs["logits"].argmax(dim=-1)
        labels = batch["labels"]

        mask = labels != -100

        all_preds.extend(preds[mask].cpu().tolist())
        all_labels.extend(labels[mask].cpu().tolist())

    if len(all_labels) == 0:
        return {"token_acc": 0.0, "token_f1": 0.0}

    correct = sum(int(p == y) for p, y in zip(all_preds, all_labels))
    token_acc = correct / len(all_labels)
    token_f1 = f1_score(all_labels, all_preds, average="micro")

    return {
        "token_acc": token_acc,
        "token_f1": token_f1,
    }