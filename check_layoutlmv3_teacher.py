"""
Check whether the LayoutLMv3 teacher can run on a fake document batch.

THEORY:
A document transformer needs:
- input_ids
- attention_mask
- bbox

We use fake tensors here just to verify the model wrapper and output structure.
"""

import torch
from doclite.models.teacher import TeacherModel

# LayoutLMv3 teacher
teacher = TeacherModel("microsoft/layoutlmv3-base", num_labels=3)

# Fake document batch
batch_size = 1
seq_len = 8

inputs = {
    "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
    "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),

    # bbox shape = [batch, seq_len, 4]
    # values usually expected in [0, 1000]
    "bbox": torch.randint(0, 1000, (batch_size, seq_len, 4), dtype=torch.long),
}

out = teacher(**inputs)

print("logits:", out["logits"].shape)
print("hidden layers:", len(out["hidden_states"]))
print("hidden[0]:", out["hidden_states"][0].shape)
print("hidden[-1]:", out["hidden_states"][-1].shape)
print("attn layers:", len(out["attentions"]))
print("attn[0]:", out["attentions"][0].shape)