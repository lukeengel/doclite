"""
Teacher wrapper for DocLite using LayoutLMv3.

THEORY:
SlimDoc distills from a large, layout-aware document transformer teacher.
Here, the teacher is LayoutLMv3.

The teacher provides:
- INTERNAL signals: hidden_states, attentions
- EXTERNAL signal: logits

Unlike plain text transformers, document transformers can take layout-aware
inputs such as bounding boxes (bbox), and optionally image features.
"""

import torch
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification


class TeacherModel(torch.nn.Module):
    def __init__(self, model_name: str = "microsoft/layoutlmv3-base", num_labels: int = 2):
        super().__init__()

        # THEORY:
        # We explicitly enable hidden states and attentions because SlimDoc
        # uses them for internal distillation losses.
        config = LayoutLMv3Config.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True,
        )

        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            config=config
        )

    @torch.no_grad()
    def forward(self, **inputs):
        """
        THEORY:
        Teacher is frozen during distillation.
        So we do not compute gradients for it.
        """
        out = self.model(**inputs)

        return {
            "logits": out.logits,
            "hidden_states": out.hidden_states,
            "attentions": out.attentions,
        }