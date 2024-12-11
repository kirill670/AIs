# train.py

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from model import Text2ImageTransformer
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from io import BytesIO
import requests
from tqdm import tqdm

# Параметры
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
IMG_SIZE = 64
LATENT_DIM = 256
MAX_LENGTH = 32

# Класс датасета для Hugging Face COCO Captions
class COCODataset(Dataset):
    def __init__(self, split='train', transform=None, tokenizer=None, max_length=32, num_samples=None):
        self.dataset = load_dataset("coco_captions", split=split)
        if num_samples:
            self.dataset = self.dataset.select(range(num_samples))
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # Получение изображения через URL
        image_url = data['image']['coco_url']
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Выбор одной случайной подписи (caption)
        captions = data['captions']
        if len(captions) == 0:
            caption = ""
        else:
            caption = captions[0]['caption']  # Можно выбрать случайную подпись для разнообразия

        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask

# Трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Создание датасета и загрузчика
# Можно ограничить количество образцов для ускорения тренировки
NUM_SAMPLES = 50000  # Например, 50,000 образцов
dataset = COCODataset(split='train', transform=transform, tokenizer=tokenizer, max_length=MAX_LENGTH, num_samples=NUM_SAMPLES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Инициализация модели, потерь и оптимизатора
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Text2ImageTransformer(img_size=IMG_SIZE, latent_dim=LATENT_DIM).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Тренировочный цикл
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for idx, (images, input_ids, attention_mask) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}')

    # Сохранение модели после каждой эпохи
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

print('Тренировка завершена.')
