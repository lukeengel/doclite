import torch
from torch.utils.data import Dataset


class DocumentDataset(Dataset):
    def __init__(self, examples):
        """
        examples: list of dicts, each containing preprocessed fields:
            - input_ids
            - attention_mask
            - bbox
            - labels
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        item = {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "bbox": torch.tensor(ex["bbox"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }

        if "pixel_values" in ex:
            item["pixel_values"] = torch.tensor(ex["pixel_values"], dtype=torch.float)

        return item