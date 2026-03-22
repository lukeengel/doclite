import json
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, LayoutLMv3ImageProcessor

from doclite.configs.core import ENV

LABEL2ID = {
    "question": 0,
    "answer": 1,
    "header": 2,
    "other": 3,
}

IGNORE_LABEL = -100


def parse_funsd_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    bboxes = []
    labels = []

    for entity in data["form"]:
        entity_label = entity["label"]

        for word_info in entity["words"]:
            word_text = word_info["text"].strip()
            word_box = word_info["box"]

            if word_text == "":
                continue

            words.append(word_text)
            bboxes.append(word_box)
            labels.append(LABEL2ID[entity_label])

    return words, bboxes, labels


def tokenize_and_align(words, bboxes, labels, tokenizer, max_length=512):
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


def load_funsd_split(annotation_dir: Path, tokenizer, image_dir: Path = None):
    """
    Load FUNSD split. If image_dir is provided, also loads document images
    and produces pixel_values for LayoutLMv3's vision encoder.
    """
    image_processor = None
    if image_dir is not None:
        image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")

    examples = []

    for json_file in sorted(annotation_dir.glob("*.json")):
        words, bboxes, labels = parse_funsd_json(json_file)
        example = tokenize_and_align(words, bboxes, labels, tokenizer)

        if image_dir is not None and image_processor is not None:
            # FUNSD images are .png with same stem as the .json
            image_path = image_dir / (json_file.stem + ".png")
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
                pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
                example["pixel_values"] = pixel_values.squeeze(0).tolist()

        examples.append(example)

    return examples


if __name__ == "__main__":
    FUNSD_ROOT = ENV.DATA / "funsd"
    TRAIN_ANN_DIR = FUNSD_ROOT / "training_data" / "annotations"
    TEST_ANN_DIR = FUNSD_ROOT / "testing_data" / "annotations"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

    train_examples = load_funsd_split(TRAIN_ANN_DIR, tokenizer)
    test_examples = load_funsd_split(TEST_ANN_DIR, tokenizer)

    print("Number of training examples:", len(train_examples))
    print("Number of testing examples:", len(test_examples))
    print("One training example keys:", train_examples[0].keys())
    print("Length of input_ids:", len(train_examples[0]["input_ids"]))
    print("Length of bbox:", len(train_examples[0]["bbox"]))
    print("Length of labels:", len(train_examples[0]["labels"]))