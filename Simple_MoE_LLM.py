import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts=4, top_k=2):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_experts)])
        self.gating_network = nn.Linear(input_size, num_experts)

    def forward(self, x):
        gating_scores = torch.softmax(self.gating_network(x), dim=-1)
        top_k_scores, top_k_indices = torch.topk(gating_scores, self.top_k, dim=-1)
        expert_outputs = torch.stack([self.experts[idx](x) for idx in top_k_indices[0]], dim=1)
        top_k_scores = top_k_scores.unsqueeze(-1).expand_as(expert_outputs)
        weighted_outputs = (expert_outputs * top_k_scores).sum(dim=1)
        return weighted_outputs

class ReasoningLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ReasoningLayer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_experts=4, num_layers=3):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.moelayers = nn.ModuleList([MoELayer(embed_size if i == 0 else hidden_size, hidden_size, num_experts) for i in range(num_layers)])
        self.reasoning_layer = ReasoningLayer(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.moelayers:
            x = layer(x)
        x = self.reasoning_layer(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=-1)

# Пример использования
vocab_size = 10000
embed_size = 128
hidden_size = 256
output_size = vocab_size  # Размер словаря для языковой модели

# Создаем объект SimpleLLM
model = SimpleLLM(vocab_size, embed_size, hidden_size, output_size)

# Пример данных
input_data = torch.randint(0, vocab_size, (10, 20))  # 10 последовательностей длиной 20

# Предсказание
output = model(input_data)
print(output.shape)  # Ожидается форма (10, 20, vocab_size)
