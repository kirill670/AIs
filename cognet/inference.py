# inference.py

import torch
import torch.nn as nn

# Определение тех же модулей CogNetX, что и в trainer.py

class PerceptionModule(nn.Module):
    def __init__(self, input_size, feature_size):
        super(PerceptionModule, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return features

class AttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(AttentionModule, self).__init__()
        self.attention_layer = nn.Linear(feature_size, feature_size)
    
    def forward(self, features):
        queries = self.attention_layer(features)
        keys = self.attention_layer(features)
        values = features
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(features.size(-1), dtype=torch.float32))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, values)
        return attended

class MemoryModule(nn.Module):
    def __init__(self, feature_size, memory_size):
        super(MemoryModule, self).__init__()
        self.lstm = nn.LSTM(feature_size, memory_size, batch_first=True)
        self.memory_size = memory_size
    
    def forward(self, attended, hidden):
        output, hidden = self.lstm(attended, hidden)
        return output, hidden

class DecisionModule(nn.Module):
    def __init__(self, memory_size, decision_size):
        super(DecisionModule, self).__init__()
        self.decision_layer = nn.Sequential(
            nn.Linear(memory_size, 128),
            nn.ReLU(),
            nn.Linear(128, decision_size)
        )
    
    def forward(self, memory):
        decision = self.decision_layer(memory)
        return decision

class EthicsModule(nn.Module):
    def __init__(self, decision_size, ethic_size):
        super(EthicsModule, self).__init__()
        self.ethics_layer = nn.Sequential(
            nn.Linear(decision_size, 64),
            nn.ReLU(),
            nn.Linear(64, ethic_size),
            nn.Sigmoid()
        )
    
    def forward(self, decision):
        ethic_score = self.ethics_layer(decision)
        return ethic_score

class ExplainabilityModule(nn.Module):
    def __init__(self, ethic_size, explanation_size):
        super(ExplainabilityModule, self).__init__()
        self.explain_layer = nn.Sequential(
            nn.Linear(ethic_size, 64),
            nn.ReLU(),
            nn.Linear(64, explanation_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, ethic_score):
        explanation = self.explain_layer(ethic_score)
        return explanation

class CogNetX(nn.Module):
    def __init__(self, input_size, feature_size, memory_size, decision_size, ethic_size, explanation_size, output_size):
        super(CogNetX, self).__init__()
        self.perception = PerceptionModule(input_size, feature_size)
        self.attention = AttentionModule(feature_size)
        self.memory = MemoryModule(feature_size, memory_size)
        self.decision = DecisionModule(memory_size, decision_size)
        self.ethics = EthicsModule(decision_size, ethic_size)
        self.explainability = ExplainabilityModule(ethic_size, explanation_size)
        self.output_layer = nn.Linear(decision_size + ethic_size + explanation_size, output_size)
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.zeros(1, 1, self.memory.memory_size),
                torch.zeros(1, 1, self.memory.memory_size))
    
    def forward(self, x):
        features = self.perception(x)
        attended = self.attention(features)
        attended = attended.unsqueeze(0)
        memory_output, self.hidden = self.memory(attended, self.hidden)
        decision = self.decision(memory_output.squeeze(0))
        ethic_score = self.ethics(decision)
        explanation = self.explainability(ethic_score)
        combined = torch.cat((decision, ethic_score, explanation), dim=-1)
        final_output = self.output_layer(combined)
        return final_output, ethic_score, explanation

def load_model(model_path, device='cpu'):
    # Параметры модели (должны совпадать с trainer.py)
    input_size = 100
    feature_size = 64
    memory_size = 128
    decision_size = 64
    ethic_size = 32
    explanation_size = 16
    output_size = 10
    
    model = CogNetX(input_size, feature_size, memory_size, decision_size, ethic_size, explanation_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def inference(model, input_data, device='cpu'):
    """
    Выполняет инференс на входных данных.
    
    :param model: Обученная модель CogNetX.
    :param input_Входные данные в виде тензора.
    :param device: Устройство ('cpu' или 'cuda').
    :return: final_output, ethic_score, explanation
    """
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        final_output, ethic_score, explanation = model(input_tensor)
    return final_output.cpu().numpy(), ethic_score.cpu().numpy(), explanation.cpu().numpy()

def main():
    # Путь к сохраненной модели
    model_path = 'cognetx_model.pth'
    
    # Загрузка модели
    model = load_model(model_path)
    
    # Пример входных данных (замените на реальные данные)
    input_data = torch.randn(1, 100).numpy()  # Батч размером 1, размер входа 100
    
    # Выполнение инференса
    final_output, ethic_score, explanation = inference(model, input_data)
    
    print("Окончательный вывод модели:", final_output)
    print("Оценка этичности:", ethic_score)
    print("Объяснение:", explanation)

if __name__ == "__main__":
    main()
