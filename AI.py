%pip install --upgrade torch torchvision torchaudio
%pip install transformers tokenizers datasets evaluate accelerate
%pip install --upgrade git+https://github.com/dask/s3fs

import datasets
import transformers as tr
from itertools import chain
from pathlib import Path


MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
DATASET_FILE = Path("dataset.txt")
OUTPUT_DIR = "my-dutch-gpt"
BLOCK_SIZE = 128


if not DATASET_FILE.exists():
    raise FileNotFoundError(f"Dataset file not found: {DATASET_FILE}")

if not DATASET_FILE.read_text(encoding="utf-8").strip():
    raise ValueError("dataset.txt is empty. Add one phrase per line before training.")


tokenizer = tr.AutoTokenizer.from_pretrained(MODEL_NAME)
gpt = tr.GPT2LMHeadModel.from_pretrained(MODEL_NAME)

if tokenizer.eos_token is None:
    raise ValueError("The tokenizer must have an EOS token for phrase boundaries.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gpt.config.pad_token_id = tokenizer.pad_token_id
gpt.config.eos_token_id = tokenizer.eos_token_id
gpt.config.bos_token_id = tokenizer.bos_token_id


dataset = datasets.load_dataset("text", data_files=str(DATASET_FILE))
dataset = dataset.filter(lambda x: x["text"] and x["text"].strip() != "")


def add_eos(text):
    text = text.strip()

    if text.endswith(tokenizer.eos_token):
        return text

    return f"{text} {tokenizer.eos_token}"


def tokenize(batch):
    return tokenizer([add_eos(text) for text in batch["text"]])


ds = dataset.map(tokenize, batched=True, remove_columns=["text"])


def group_texts(examples):
    concatenated = {key: list(chain(*examples[key])) for key in examples.keys()}
    total_length = len(concatenated["input_ids"])

    return {
        key: [tokens[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for key, tokens in concatenated.items()
    }


dsb = ds.map(group_texts, batched=True)

args = tr.TrainingArguments(
    output_dir="gpt2-finetune",
    num_train_epochs=4,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

trainer = tr.Trainer(
    model=gpt,
    args=args,
    train_dataset=dsb["train"],
    tokenizer=tokenizer,
    data_collator=tr.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
