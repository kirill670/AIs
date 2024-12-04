import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import io

# Гиперпараметры
latent_size = 128
output_size = 784  # Размер для MNIST
population_size = 50
num_generations = 100

# Преобразование изображений в байты
class ByteDataset(Dataset):
    def __init__(self, transform=None):
        self.data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        
        # Преобразуем изображение в байты
        byte_arr = io.BytesIO()
        torchvision.utils.save_image(image, byte_arr)
        byte_arr.seek(0)
        byte_data = byte_arr.read()
        
        return byte_data, label

# Модель Liquid VQ-VAE
class LiquidVQVAE(nn.Module):
    def __init__(self):
        super(LiquidVQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Генератор GAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Tanh()
        )
        
    def forward(self, z):
        return self.model(z)

# Дискриминатор GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# Основная модель
class SwarmGANModel:
    def __init__(self):
        self.vqvae = LiquidVQVAE()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        
    def train(self, data_loader):
        for epoch in range(num_generations):
            for byte_data, labels in data_loader:
                # Преобразование байтов в тензоры
                # Здесь предполагается, что byte_data содержит изображения в байтах
                real_data = torch.tensor(np.frombuffer(byte_data[0], dtype=np.uint8)).float() / 255.0
                real_data = real_data.view(-1, output_size)

                # Обучение дискриминатора
                self.optimizer_d.zero_grad()
                
                # Генерация фейковых данных
                noise = torch.randn(real_data.size(0), latent_size)
                fake_data = self.generator(noise)
                
                # Расчет потерь
                real_loss = F.binary_cross_entropy(self.discriminator(real_data), torch.ones(real_data.size(0), 1))
                fake_loss = F.binary_cross_entropy(self.discriminator(fake_data.detach()), torch.zeros(real_data.size(0), 1))
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_d.step()
                
                # Обучение генератора
                self.optimizer_g.zero_grad()
                g_loss = F.binary_cross_entropy(self.discriminator(fake_data), torch.ones(real_data.size(0), 1))
                g_loss.backward()
                self.optimizer_g.step()
                
            print(f'Epoch [{epoch + 1}/{num_generations}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    def infer(self, noise):
        with torch.no_grad():
            return self.generator(noise)

    def visualize_generations(self, noise):
        generated_data = self.infer(noise)
        generated_images = generated_data.view(-1, 1, 28, 28)  # Преобразование для визуализации
        grid_img = torchvision.utils.make_grid(generated_images, nrow=8)
        plt.imshow(grid_img.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()

# Создание DataLoader
transform = transforms.Compose([transforms.ToTensor()])
byte_dataset = ByteDataset(transform=transform)
train_loader = DataLoader(byte_dataset, batch_size=32, shuffle=True)

# Пример использования
if __name__ == "__main__":
    model = SwarmGANModel()
    model.train(train_loader)  # Обучение модели
    
    # Инференс
    noise = torch.randn(64, latent_size)  # Генерация 64 "выходов"
    model.visualize_generations(noise)
