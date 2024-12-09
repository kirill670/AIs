# trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import evotorch as et  # Убедитесь, что EvoTorch установлен: pip install evotorch

# Определение модулей CogNetX

class PerceptionModule(nn.Module):
    """
    Модуль восприятия: извлекает признаки из входных данных.
    """
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
    """
    Модуль внимания: выделяет релевантные признаки с использованием механизма внимания.
    """
    def __init__(self, feature_size):
        super(AttentionModule, self).__init__()
        self.attention_layer = nn.Linear(feature_size, feature_size)
    
    def forward(self, features):
        queries = self.attention_layer(features)
        keys = self.attention_layer(features)
        values = features
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(features.size(-1), dtype=torch.float32))
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, values)
        return attended

class MemoryModule(nn.Module):
    """
    Модуль памяти: хранит и обновляет состояние памяти с использованием LSTM.
    """
    def __init__(self, feature_size, memory_size):
        super(MemoryModule, self).__init__()
        self.lstm = nn.LSTM(feature_size, memory_size, batch_first=True)
        self.memory_size = memory_size
    
    def forward(self, attended, hidden):
        output, hidden = self.lstm(attended, hidden)
        return output, hidden

class DecisionModule(nn.Module):
    """
    Модуль принятия решений: генерирует выходные данные на основе памяти.
    """
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
    """
    Этический модуль: оценивает этичность принимаемых решений.
    """
    def __init__(self, decision_size, ethic_size):
        super(EthicsModule, self).__init__()
        self.ethics_layer = nn.Sequential(
            nn.Linear(decision_size, 64),
            nn.ReLU(),
            nn.Linear(64, ethic_size),
            nn.Sigmoid()  # Выход в диапазоне [0,1] для оценки этичности
        )
    
    def forward(self, decision):
        ethic_score = self.ethics_layer(decision)
        return ethic_score

class ExplainabilityModule(nn.Module):
    """
    Модуль объяснимости: предоставляет обоснования принятых решений.
    """
    def __init__(self, ethic_size, explanation_size):
        super(ExplainabilityModule, self).__init__()
        self.explain_layer = nn.Sequential(
            nn.Linear(ethic_size, 64),
            nn.ReLU(),
            nn.Linear(64, explanation_size),
            nn.Softmax(dim=-1)  # Вероятностное распределение по типам объяснений
        )
    
    def forward(self, ethic_score):
        explanation = self.explain_layer(ethic_score)
        return explanation

class CogNetX(nn.Module):
    """
    Полная модель CogNetX, объединяющая все модули, включая этический и объяснимости.
    """
    def __init__(self, input_size, feature_size, memory_size, decision_size, ethic_size, explanation_size, output_size):
        super(CogNetX, self).__init__()
        self.perception = PerceptionModule(input_size, feature_size)
        self.attention = AttentionModule(feature_size)
        self.memory = MemoryModule(feature_size, memory_size)
        self.decision = DecisionModule(memory_size, decision_size)
        self.ethics = EthicsModule(decision_size, ethic_size)
        self.explainability = ExplainabilityModule(ethic_size, explanation_size)
        self.output_layer = nn.Linear(decision_size + ethic_size + explanation_size, output_size)  # Комбинирование выходов
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.zeros(1, 1, self.memory.memory_size),
                torch.zeros(1, 1, self.memory.memory_size))
    
    def forward(self, x):
        # Этап восприятия
        features = self.perception(x)
        
        # Этап внимания
        attended = self.attention(features)
        
        # Этап памяти
        attended = attended.unsqueeze(0)  # Добавляем размерность последовательности
        memory_output, self.hidden = self.memory(attended, self.hidden)
        
        # Этап принятия решений
        decision = self.decision(memory_output.squeeze(0))
        
        # Этап оценки этичности
        ethic_score = self.ethics(decision)
        
        # Этап объяснимости
        explanation = self.explainability(ethic_score)
        
        # Комбинирование всех выходов для генерации окончательного решения
        combined = torch.cat((decision, ethic_score, explanation), dim=-1)
        final_output = self.output_layer(combined)
        
        return final_output, ethic_score, explanation

# Функция для эволюционной оптимизации с использованием EvoTorch

def optimize_hyperparameters(model, dataloader, generations=10, population_size=20):
    """
    Оптимизирует гиперпараметры модели с использованием эволюционных алгоритмов через EvoTorch.
    
    :param model: Модель нейронной сети.
    :param dataloader: DataLoader с обучающими данными.
    :param generations: Количество поколений.
    :param population_size: Размер популяции.
    """
    # Определение пространства гиперпараметров
    hyperparams = {
        'learning_rate': {'min': 1e-5, 'max': 1e-2},
        'batch_size': {'min': 16, 'max': 128},
        'feature_size': {'min': 32, 'max': 256},
        'memory_size': {'min': 64, 'max': 512},
        'decision_size': {'min': 32, 'max': 256},
        'ethic_size': {'min': 16, 'max': 128},
        'explanation_size': {'min': 8, 'max': 64}
    }
    
    # Создание популяции гиперпараметров
    population = et.population_create(generation_size=population_size, length=len(hyperparams))
    
    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            # Назначение гиперпараметров модели
            hp = {}
            for i, key in enumerate(hyperparams.keys()):
                hp[key] = hyperparams[key]['min'] + individual[i] * (hyperparams[key]['max'] - hyperparams[key]['min'])
            
            # Применение гиперпараметров
            optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
            batch_size = int(hp['batch_size'])
            feature_size = int(hp['feature_size'])
            memory_size = int(hp['memory_size'])
            decision_size = int(hp['decision_size'])
            ethic_size = int(hp['ethic_size'])
            explanation_size = int(hp['explanation_size'])
            
            # Тренировка модели на одном батче для оценки фитнеса
            try:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)
                outputs, _, _ = model(inputs)
                loss = F.mse_loss(outputs, targets)
                fitness = 1.0 / (loss.item() + 1e-6)  # Чем меньше ошибка, тем выше фитнес
            except StopIteration:
                fitness = 0.0
            fitness_scores.append(fitness)
        
        # Эволюция популяции
        population = et.population_evolve(population, fitness_scores)
        print(f"Поколение {generation + 1}/{generations}, Лучший фитнес: {max(fitness_scores)}")
    
    # Выбор лучшего индивидуума
    best_individual = et.population_get_best(population, fitness_scores)
    best_hp = {}
    for i, key in enumerate(hyperparams.keys()):
        best_hp[key] = hyperparams[key]['min'] + best_individual[i] * (hyperparams[key]['max'] - hyperparams[key]['min'])
    
    print("Лучшие гиперпараметры:", best_hp)
    return best_hp

def main():
    # Параметры модели
    input_size = 100        # Размер входных данных
    feature_size = 64       # Размер признаков
    memory_size = 128       # Размер памяти
    decision_size = 64      # Размер принятия решений
    ethic_size = 32         # Размер оценки этичности
    explanation_size = 16   # Размер объяснимости
    output_size = 10        # Размер выходных данных
    
    # Создание модели
    model = CogNetX(input_size, feature_size, memory_size, decision_size, ethic_size, explanation_size, output_size)
    
    # Подготовка данных (Замените на реальные данные)
    # Пример случайных данных для демонстрации
    inputs = torch.randn(1000, input_size)
    targets = torch.randn(1000, output_size)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Эволюционная оптимизация гиперпараметров
    best_hyperparams = optimize_hyperparameters(model, dataloader, generations=10, population_size=20)
    
    # Обучение модели с оптимальными гиперпараметрами
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams['learning_rate'])
    criterion = nn.MSELoss()
    num_epochs = 20
    batch_size = int(best_hyperparams['batch_size'])
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs, _, _ = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Эпоха {epoch + 1}/{num_epochs}, Потери: {avg_loss}")
    
    # Сохранение модели
    torch.save(model.state_dict(), 'cognetx_model.pth')
    print("Обучение завершено. Модель сохранена как 'cognetx_model.pth'.")

if __name__ == "__main__":
    main()
