import asyncio
import os
import random
import re
import threading
from pathlib import Path

import discord
import torch
import transformers as tr
from discord.ext import tasks


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("DUTCH_AI_MODEL_DIR", "my-dutch-gpt"))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = BASE_DIR / MODEL_DIR

TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

CHECK_INTERVAL_MINUTES = float(os.getenv("CHECK_INTERVAL_MINUTES", "60"))
SEND_CHANCE = float(os.getenv("SEND_CHANCE", "0.35"))

LEARN_FROM_CHAT = os.getenv("LEARN_FROM_CHAT", "1") != "0"
LEARN_USER_IDS = {
    int(user_id)
    for user_id in os.getenv("LEARN_USER_IDS", "").split(",")
    if user_id.strip()
}
LEARN_CHANNEL_IDS = {
    int(channel_id)
    for channel_id in os.getenv("LEARN_CHANNEL_IDS", "").split(",")
    if channel_id.strip()
}
IGNORE_PREFIXES = tuple(
    prefix
    for prefix in os.getenv("IGNORE_PREFIXES", "!,/,$").split(",")
    if prefix
)
MIN_TRAIN_MESSAGES = int(os.getenv("MIN_TRAIN_MESSAGES", "20"))
MAX_TRAIN_MESSAGES_PER_RUN = int(os.getenv("MAX_TRAIN_MESSAGES_PER_RUN", "64"))
TRAIN_INTERVAL_MINUTES = float(os.getenv("TRAIN_INTERVAL_MINUTES", "120"))
ONLINE_BATCH_SIZE = int(os.getenv("ONLINE_BATCH_SIZE", "8"))
ONLINE_EPOCHS = int(os.getenv("ONLINE_EPOCHS", "1"))
ONLINE_LEARNING_RATE = float(os.getenv("ONLINE_LEARNING_RATE", "1e-6"))
TRAIN_MAX_LENGTH = int(os.getenv("TRAIN_MAX_LENGTH", "128"))

TOP_K = int(os.getenv("TOP_K", "50"))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "3"))
MIN_NEW_TOKENS = int(os.getenv("MIN_NEW_TOKENS", "8"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "60"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
GENERATION_ATTEMPTS = int(os.getenv("GENERATION_ATTEMPTS", "5"))
PROMPTS = [
    prompt.strip() + " "
    for prompt in os.getenv(
        "GENERATION_PROMPTS",
        "ну,|короче,|если честно,|вообще,|ладно,|я думаю,",
    ).split("|")
    if prompt.strip()
]

LEARN_LOG_FILE = Path(os.getenv("LEARN_LOG_FILE", "online_dataset.txt"))
if not LEARN_LOG_FILE.is_absolute():
    LEARN_LOG_FILE = BASE_DIR / LEARN_LOG_FILE


if not TOKEN:
    raise RuntimeError("Set DISCORD_TOKEN before starting the bot.")

if not CHANNEL_ID:
    raise RuntimeError("Set DISCORD_CHANNEL_ID before starting the bot.")

if not MODEL_DIR.exists():
    raise RuntimeError(f"Model directory does not exist: {MODEL_DIR}")

if LEARN_FROM_CHAT and not LEARN_USER_IDS:
    raise RuntimeError("Set LEARN_USER_IDS to your Discord user ID before enabling chat learning.")

if not 0 <= SEND_CHANCE <= 1:
    raise RuntimeError("SEND_CHANCE must be between 0 and 1.")

if MIN_NEW_TOKENS > MAX_NEW_TOKENS:
    raise RuntimeError("MIN_NEW_TOKENS must be less than or equal to MAX_NEW_TOKENS.")

if not PROMPTS:
    raise RuntimeError("GENERATION_PROMPTS must contain at least one prompt.")


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = tr.AutoTokenizer.from_pretrained(MODEL_DIR)
model = tr.AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)

if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token_id is None:
    raise RuntimeError("Tokenizer has no PAD token and no EOS token to use as PAD.")

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=ONLINE_LEARNING_RATE)
model_lock = threading.Lock()
train_buffer = []
buffer_lock = asyncio.Lock()

intents = discord.Intents.default()
intents.message_content = LEARN_FROM_CHAT
client = discord.Client(intents=intents)


def is_good_generated_phrase(text: str) -> bool:
    if not 5 <= len(text) <= 300:
        return False

    if re.search(r"(https?://|www\.|/blockquote|<[^>]+>|[{}\[\]\\])", text, re.IGNORECASE):
        return False

    if re.search(r"\b(file|modifier|blockquote|code|number|required|order)\b", text, re.IGNORECASE):
        return False

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    cyrillic = re.findall(r"[А-Яа-яЁё]", text)

    if not letters:
        return False

    return len(cyrillic) / len(letters) >= 0.6


def generate_phrase() -> str:
    for _ in range(GENERATION_ATTEMPTS):
        prompt = random.choice(PROMPTS)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with model_lock:
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    min_new_tokens=MIN_NEW_TOKENS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.9,
                    top_k=TOP_K,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        text = text.splitlines()[0].strip()

        if is_good_generated_phrase(text):
            return text

    return ""


def clean_message(text: str) -> str | None:
    text = " ".join(text.split())

    if not 3 <= len(text) <= 300:
        return None

    if text.startswith(IGNORE_PREFIXES):
        return None

    eos_token = tokenizer.eos_token or ""
    return f"{text} {eos_token}".strip()


def append_to_learn_log(text: str) -> None:
    LEARN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with LEARN_LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(text + "\n")


def train_on_texts(texts: list[str]) -> float:
    if not texts:
        return 0.0

    losses = []
    shuffled_texts = texts[:]
    random.shuffle(shuffled_texts)

    with model_lock:
        model.train()

        try:
            for _ in range(ONLINE_EPOCHS):
                for start in range(0, len(shuffled_texts), ONLINE_BATCH_SIZE):
                    batch = shuffled_texts[start : start + ONLINE_BATCH_SIZE]
                    encoded = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=TRAIN_MAX_LENGTH,
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    labels = encoded["input_ids"].clone()

                    if "attention_mask" in encoded:
                        labels[encoded["attention_mask"] == 0] = -100

                    outputs = model(**encoded, labels=labels)
                    loss = outputs.loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    losses.append(float(loss.detach().cpu()))

            if losses:
                model.save_pretrained(MODEL_DIR)
                tokenizer.save_pretrained(MODEL_DIR)
        finally:
            model.eval()

    return sum(losses) / len(losses) if losses else 0.0


@tasks.loop(minutes=CHECK_INTERVAL_MINUTES)
async def send_random_phrase():
    if random.random() > SEND_CHANCE:
        return

    channel = client.get_channel(int(CHANNEL_ID))

    if channel is None:
        channel = await client.fetch_channel(int(CHANNEL_ID))

    phrase = await asyncio.to_thread(generate_phrase)

    if phrase:
        await channel.send(phrase[:1900])


@tasks.loop(minutes=TRAIN_INTERVAL_MINUTES)
async def train_from_chat_buffer():
    async with buffer_lock:
        if len(train_buffer) < MIN_TRAIN_MESSAGES:
            return

        batch = train_buffer[:MAX_TRAIN_MESSAGES_PER_RUN]
        del train_buffer[: len(batch)]

    try:
        loss = await asyncio.to_thread(train_on_texts, batch)
    except Exception as error:
        async with buffer_lock:
            train_buffer[:0] = batch
        print(f"Online training failed: {error}")
        return

    print(f"Online training finished on {len(batch)} messages, loss={loss:.4f}")


@send_random_phrase.before_loop
async def before_send_random_phrase():
    await client.wait_until_ready()


@train_from_chat_buffer.before_loop
async def before_train_from_chat_buffer():
    await client.wait_until_ready()


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

    if not send_random_phrase.is_running():
        send_random_phrase.start()

    if LEARN_FROM_CHAT and not train_from_chat_buffer.is_running():
        train_from_chat_buffer.start()


@client.event
async def on_message(message):
    if not LEARN_FROM_CHAT:
        return

    if message.author.bot:
        return

    if message.author.id not in LEARN_USER_IDS:
        return

    if LEARN_CHANNEL_IDS and message.channel.id not in LEARN_CHANNEL_IDS:
        return

    text = clean_message(message.clean_content)

    if text is None:
        return

    await asyncio.to_thread(append_to_learn_log, text)

    async with buffer_lock:
        train_buffer.append(text)


client.run(TOKEN)
