"""
Student wrapper for DocLite using LiLT.

THEORY:
In SlimDoc, the student is a smaller / more efficient document model.
Here, we use LiLT as the student.

The student must expose the same kinds of outputs as the teacher:
- logits
- hidden_states
- attentions

Unlike the teacher, the student is trainable, so we do NOT use torch.no_grad().
"""

import torch
from transformers import LiltConfig, LiltForTokenClassification


class StudentModel(torch.nn.Module):
    def __init__(self, model_name: str = "SCUT-DLVCLab/lilt-roberta-en-base", num_labels: int = 2):
        super().__init__()

        config = LiltConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
        )

        self.model = LiltForTokenClassification.from_pretrained(
            model_name,
            config=config
        )

    def forward(self, **inputs):
        """
        THEORY:
        Student is trainable, so gradients should flow through it.
        LiLT does not accept pixel_values, so we drop it if present.
        """
        inputs.pop("pixel_values", None)
        out = self.model(**inputs)

        return {
            "logits": out.logits,
            "hidden_states": out.hidden_states,
            "attentions": out.attentions,
        }