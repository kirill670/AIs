import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts=4, num_layers=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.num_layers = num_layers

        # Модули
        self.moe_layers = nn.ModuleList([self._make_moe_layer(input_size if i == 0 else hidden_size) for i in range(num_layers)])
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4) # упрощенный Transformer
        self.feedforward = nn.Linear(hidden_size, hidden_size)
        self.crossfeed = nn.Linear(hidden_size, hidden_size) # упрощенный CrossFeed
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

        # Упрощенная имитация разложения Колмогорова-Арнольда
        self.kolmogorov_approx = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )


    def _make_moe_layer(self, input_size):
        experts = nn.ModuleList([nn.Linear(input_size, self.hidden_size) for _ in range(self.num_experts)])
        gating = nn.Linear(input_size, self.num_experts)
        return experts, gating

    def forward(self, x, targets=None):
        # Прямое прохождение через MoE слои
        for experts, gating in self.moe_layers:
            gating_scores = F.softmax(gating(x), dim=-1)
            top_k_scores, top_k_indices = torch.topk(gating_scores, 2, dim=-1)
            expert_outputs = [experts[idx](x) for idx in top_k_indices[0]]
            x = torch.stack(expert_outputs).mean(dim=0) # усредняем выходы экспертов

        # Transformer, FeedForward и CrossFeed
        x = self.transformer(x)
        x = self.feedforward(x)
        x = self.crossfeed(x)
        x = self.kolmogorov_approx(x) #  применение упрощенной аппроксимации Колмогорова-Арнольда

        # Выходной слой
        x = self.output_layer(x)


        if targets is not None:
            loss = F.mse_loss(x, targets)
            return x, loss
        else:
            return x


# Пример использования
model = LiquidModel(input_size=64, hidden_size=128, output_size=10)
input_tensor = torch.randn(1, 64)
output, loss = model(input_tensor, torch.randn(1,10))
print(output.shape, loss)

