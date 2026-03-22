import torch


def collate_document_batch(batch):
    collated = {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "bbox": torch.stack([x["bbox"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }

    if "pixel_values" in batch[0]:
        collated["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])

    return collated