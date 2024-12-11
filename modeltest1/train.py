import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import deepspeed
from GPTCustom import GPTCustom  # Предполагается, что модель сохранена в your_model_file.py

# Параметры модели
vocab_size = 50257
embed_size = 4096
num_layers = 48
num_heads = 64
dropout = 0.1
forward_expansion = 4
max_length = 2048

# Инициализация модели
model = GPTCustom(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout=dropout,
    forward_expansion=forward_expansion,
    max_length=max_length
)

# Загрузка конфигурации DeepSpeed из JSON файла
# Создайте файл deepsea_config.json с необходимыми параметрами
ds_config = "deepsea_config.json"

# Инициализация DeepSpeed
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# Загрузка и подготовка данных
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_loader = DataLoader(tokenized_datasets, batch_size=2, shuffle=True)

# Тренировочный цикл
model_engine.train()
num_epochs = 3

for epoch in range(num_epochs):
    for step, batch in enumerate(data_loader):
        inputs = torch.tensor(batch["input_ids"]).to(model_engine.device)
        labels = inputs.clone().detach()

        outputs = model_engine(inputs, mask=None)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, vocab_size), labels.view(-1))

        model_engine.backward(loss)
        model_engine.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Сохранение модели после каждой эпохи
    model_engine.save_model(f"gpt_custom_epoch_{epoch}.pt")
