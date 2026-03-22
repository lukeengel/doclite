import torch
from doclite.models.student import StudentModel

student = StudentModel("SCUT-DLVCLab/lilt-roberta-en-base", num_labels=3)

batch_size = 1
seq_len = 8

# Create valid bounding boxes
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

out = student(**inputs)

print("logits:", out["logits"].shape)
print("hidden layers:", len(out["hidden_states"]))
print("hidden[0]:", out["hidden_states"][0].shape)
print("hidden[-1]:", out["hidden_states"][-1].shape)
print("attn layers:", len(out["attentions"]))
print("attn[0]:", out["attentions"][0].shape)