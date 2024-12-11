import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Text2ImageTransformer(nn.Module):
    def __init__(self, img_size=64, latent_dim=256, vocab_size=30522, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Text2ImageTransformer, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Токенизатор и BERT для обработки текста
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        # Линейный слой для преобразования текстового представления в латентное пространство
        self.text_linear = nn.Linear(self.text_encoder.config.hidden_size, latent_dim)

        # Декодер трансформера для генерации изображения
        self.transformer_decoder = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        # Финальный слой для преобразования латентного представления в изображение
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, img_size * img_size * 3),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask):
        # Получение текстовых эмбеддингов с помощью BERT
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = self.text_linear(text_outputs.last_hidden_state)  # [batch_size, seq_len, latent_dim]

        # Генерация случайного шума для латентного пространства изображения
        batch_size, seq_len, _ = text_embeds.size()
        noise = torch.randn(batch_size, self.img_size, self.img_size, self.latent_dim).to(text_embeds.device)

        # Flatten изображения для трансформера
        noise = noise.view(batch_size, self.img_size * self.img_size, self.latent_dim)  # [batch_size, img_size*img_size, latent_dim]

        # Трансформер ожидает [sequence_length, batch_size, d_model]
        memory = text_embeds.permute(1, 0, 2)
        tgt = noise.permute(1, 0, 2)

        # Декодирование
        transformer_output = self.transformer_decoder(tgt, memory)  # [img_size*img_size, batch_size, d_model]

        # Преобразование обратно в изображение
        transformer_output = transformer_output.permute(1, 0, 2)  # [batch_size, img_size*img_size, d_model]
        img = self.output_layer(transformer_output)  # [batch_size, img_size*img_size * 3]

        img = img.view(-1, 3, self.img_size, self.img_size)  # [batch_size, 3, img_size, img_size]
        return img
