import torch
import evotorch
from evotorch import Problem
from evotorch.algorithms import CMAES, PSO
import numpy as np
from typing import List
import logging

class NeuroEvolutionModel:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Создаем архитектуру сети
        self.model = self._build_network(input_size, hidden_sizes, output_size).to(device)
        self.model_params_count = sum(p.numel() for p in self.model.parameters())
        
        # Настройка логирования
        self.logger = self._setup_logging()
        self.history = {'fitness': [], 'best_fitness': []}
        
    def _build_network(self, input_size: int, hidden_sizes: List[int], output_size: int) -> torch.nn.Module:
        layers = []
        prev_size = input_size

        # LSF Layers (Концептуальная реализация)
        lsf_size = hidden_sizes[0] // 2  # Пример: половина размера первого скрытого слоя
        layers.extend([
            torch.nn.Conv2d(1, lsf_size, kernel_size=3, padding=1),  # Предполагаем одноканальный ввод для пространственных данных
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)  # Снижение пространственных размеров до одной точки
        ])
        prev_size = lsf_size

        # Создание скрытых слоев
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(torch.nn.Linear(prev_size, output_size))
        return torch.nn.Sequential(*layers)

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def fitness_function(self, solution: torch.Tensor, batch_data: torch.Tensor, target: torch.Tensor) -> float:
        # Обновляем параметры модели
        with torch.no_grad():
            idx = 0
            for param in self.model.parameters():
                param_size = param.numel()
                param.data = solution[idx:idx + param_size].reshape(param.shape)
                idx += param_size

        # Оценка производительности модели
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_data)
            loss = torch.nn.functional.mse_loss(outputs, target)
        return -loss.item()  # Чем меньше ошибка, тем лучше

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_generations: int,
        checkpoint_freq: int = 10
    ):
        # Определяем задачу
        problem = Problem(
            self.model_params_count,
            fitness_function=self.fitness_function,
            data=train_loader.dataset.data,
            target=train_loader.dataset.targets
        )

        # Комбинируем CMA-ES и PSO
        searcher = CMAES(problem, population_size=self.population_size)
        pso_searcher = PSO(problem, population_size=self.population_size // 2)  # Половина популяции для PSO

        for generation in range(num_generations):
            # Запускаем CMA-ES
            cma_results = searcher.step()

            # Запускаем PSO
            pso_results = pso_searcher.step()

            # Объединяем и оцениваем результаты
            combined_population = torch.cat([cma_results.population, pso_results.population])
            combined_fitnesses = torch.cat([cma_results.fitnesses, pso_results.fitnesses])

            # Выбираем лучших индивидуумов для следующего поколения
            best_indices = torch.argsort(combined_fitnesses, descending=True)[:self.population_size]
            searcher.population = combined_population[best_indices]
            pso_searcher.population = combined_population[best_indices[:self.population_size // 2]]

            # Сохраняем историю фитнеса
            best_fitness = combined_fitnesses[best_indices[0]].item()
            self.history['fitness'].append(best_fitness)
            self.history['best_fitness'].append(best_fitness)

            # Логирование
            self.logger.info(f'Generation {generation+1}/{num_generations}: Best Fitness = {best_fitness}')

            # Чекпоинт
            if (generation + 1) % checkpoint_freq == 0:
                torch.save(self.model.state_dict(), f'checkpoint_gen_{generation+1}.pt')

    def infer(self, input_data: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(input_data)

# Пример использования
if __name__ == "__main__":
    # Параметры
    input_size = 784  # Пример для MNIST
    hidden_sizes = [128, 64]
    output_size = 10   # Классы от 0 до 9
    population_size = 50
    num_generations = 100

    # Инициализация модели
    model = NeuroEvolutionModel(input_size, hidden_sizes, output_size, population_size)

    # Загрузка данных (предполагается, что train_loader уже создан)
    # Пример: train_loader = DataLoader(dataset=..., batch_size=32, shuffle=True)

    # Обучение модели
    # model.train(train_loader, num_generations)

    # Инференс (пример использования)
    # input_data = torch.randn(1, 1, 28, 28)  # Пример входных данных для изображения
    # output = model.infer(input_data)
    # print(output)
