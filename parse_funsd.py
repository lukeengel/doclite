import json
from pathlib import Path


LABEL2ID = {
    "question": 0,
    "answer": 1,
    "header": 2,
    "other": 3,
}


def parse_funsd_json(json_path: str):
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

            # Skip empty OCR tokens
            if word_text == "":
                continue

            words.append(word_text)
            bboxes.append(word_box)
            labels.append(entity_label)

    return {
        "words": words,
        "bboxes": bboxes,
        "labels": labels,
        "label_ids": [LABEL2ID[label] for label in labels],
    }


if __name__ == "__main__":
    json_path = r"00040534.json"   # replace later with your local FUNSD path
    parsed = parse_funsd_json(json_path)

    print("Number of words:", len(parsed["words"]))
    print("First 10 words:", parsed["words"][:10])
    print("First 10 bboxes:", parsed["bboxes"][:10])
    print("First 10 labels:", parsed["labels"][:10])
    print("First 10 label ids:", parsed["label_ids"][:10])