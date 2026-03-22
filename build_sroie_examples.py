"""
SROIE dataset builder for DocLite.

SROIE (Scanned Receipts OCR and Information Extraction) format:
- OCR files: each line is x1,y1,x2,y2,x3,y3,x4,y4,transcription (8-point polygon)
- Entity files: JSON with keys {company, date, address, total}
- Labels: 5 classes — company, date, address, total, other

Label assignment strategy:
  For each OCR word, check if it appears as a substring in any entity value.
  Multi-word entities (e.g. addresses) are matched word-by-word.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

from doclite.configs.core import ENV

LABEL2ID = {
    "company": 0,
    "date": 1,
    "address": 2,
    "total": 3,
    "other": 4,
}

NUM_LABELS = 5
IGNORE_LABEL = -100


def parse_sroie_doc(ocr_path: Path, entity_path: Path):
    """Parse a single SROIE receipt (OCR file + entity JSON)."""
    with open(entity_path, "r", encoding="utf-8") as f:
        entities = json.load(f)

    # Build word-level lookup for each entity type
    entity_words = {}
    for key in ["company", "date", "address", "total"]:
        val = entities.get(key, "")
        entity_words[key] = [w.lower() for w in val.split()]

    words = []
    bboxes = []
    labels = []

    with open(ocr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",", 8)
            if len(parts) < 9:
                continue

            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                continue

            text = parts[8].strip()
            if not text:
                continue

            # 8-point polygon → axis-aligned bbox [x0, y0, x1, y1]
            xs = coords[0::2]
            ys = coords[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            # Assign label by fuzzy matching against entity values
            label = "other"
            text_lower = text.lower()
            for key, val_words in entity_words.items():
                if text_lower in val_words:
                    label = key
                    break

            words.append(text)
            bboxes.append(bbox)
            labels.append(LABEL2ID[label])

    return words, bboxes, labels


def tokenize_and_align(words, bboxes, labels, tokenizer, max_length=512):
    """Tokenize words with LayoutLMv3 tokenizer and align labels to subwords."""
    encoding = tokenizer(
        words,
        boxes=bboxes,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    word_ids = encoding.word_ids()

    aligned_labels = []
    aligned_bboxes = []

    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(IGNORE_LABEL)
            aligned_bboxes.append([0, 0, 0, 0])
        else:
            if word_idx != prev_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(IGNORE_LABEL)
            aligned_bboxes.append(bboxes[word_idx])

        prev_word_idx = word_idx

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": aligned_bboxes,
        "labels": aligned_labels,
    }


def load_sroie_split(ocr_dir: Path, entity_dir: Path, tokenizer):
    """Load all SROIE receipts from a split directory."""
    examples = []

    for ocr_file in sorted(ocr_dir.glob("*.txt")):
        entity_file = entity_dir / ocr_file.name
        if not entity_file.exists():
            continue

        words, bboxes, labels = parse_sroie_doc(ocr_file, entity_file)
        if words:
            example = tokenize_and_align(words, bboxes, labels, tokenizer)
            examples.append(example)

    return examples


if __name__ == "__main__":
    SROIE_ROOT = ENV.DATA / "sroie"
    TRAIN_OCR_DIR = SROIE_ROOT / "train" / "box"
    TRAIN_ENT_DIR = SROIE_ROOT / "train" / "entities"
    TEST_OCR_DIR = SROIE_ROOT / "test" / "box"
    TEST_ENT_DIR = SROIE_ROOT / "test" / "entities"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

    train_examples = load_sroie_split(TRAIN_OCR_DIR, TRAIN_ENT_DIR, tokenizer)
    test_examples = load_sroie_split(TEST_OCR_DIR, TEST_ENT_DIR, tokenizer)

    print("Number of training examples:", len(train_examples))
    print("Number of testing examples:", len(test_examples))

    if train_examples:
        print("One training example keys:", train_examples[0].keys())
        print("Length of input_ids:", len(train_examples[0]["input_ids"]))
