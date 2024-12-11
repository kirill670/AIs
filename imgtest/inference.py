# inference.py

import torch
from model import Text2ImageTransformer
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
import argparse

def generate_image(model, tokenizer, text, device, img_size=64, max_length=32):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    # Преобразование выходного тензора в изображение
    img = output.squeeze(0).cpu()
    img = (img + 1) / 2  # Деинормализация из [-1,1] в [0,1]
    transform = transforms.ToPILImage()
    img = transform(img)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text to Image Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Путь к сохраненной модели')
    parser.add_argument('--text', type=str, required=True, help='Текстовое описание для генерации изображения')
    parser.add_argument('--output_path', type=str, default='generated_image.png', help='Путь для сохранения сгенерированного изображения')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Инициализация модели и загрузка весов
    model = Text2ImageTransformer(img_size=IMG_SIZE, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Генерация и сохранение изображения
    image = generate_image(model, tokenizer, args.text, device, img_size=IMG_SIZE)
    image.save(args.output_path)
    print(f'Изображение сохранено по пути: {args.output_path}')
