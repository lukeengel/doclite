from torch.utils.data import DataLoader

from doclite.configs.core import ENV
from doclite.data.document_dataset import DocumentDataset
from doclite.data.collate import collate_document_batch
from build_funsd_examples import load_funsd_split
from transformers import AutoTokenizer

# Paths
FUNSD_ROOT = ENV.DATA / "funsd"
TRAIN_ANN_DIR = FUNSD_ROOT / "training_data" / "annotations"
TEST_ANN_DIR = FUNSD_ROOT / "testing_data" / "annotations"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

# Build processed examples
train_examples = load_funsd_split(TRAIN_ANN_DIR, tokenizer)
test_examples = load_funsd_split(TEST_ANN_DIR, tokenizer)

# Wrap in Dataset
train_dataset = DocumentDataset(train_examples)
test_dataset = DocumentDataset(test_examples)

# Build DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_document_batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_document_batch
)

# Inspect one batch
batch = next(iter(train_loader))

print("Batch keys:", batch.keys())
print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
print("bbox shape:", batch["bbox"].shape)
print("labels shape:", batch["labels"].shape)

print("First 20 labels of sample 0:", batch["labels"][0][:20].tolist())
print("First 5 bboxes of sample 0:", batch["bbox"][0][:5].tolist())