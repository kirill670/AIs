import asyncio
import random
import uuid
import logging
import os
import shutil
import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime
import traceback

# Импорт необходимых библиотек для GUI
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox, simpledialog

# Импорт GPT-4 Free библиотеки
from g4f import ChatCompletion, Provider

# Настройка логирования
logging.basicConfig(level=logging.INFO, filename='system.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация системы
CONFIG = {
    "initial_population": 10,
    "max_population": 50,
    "mutation_rate": 0.1,
    "task": "Analyze the provided document.",
    "evaluation_cycles": 100
}

# Путь к базе данных для долгосрочной памяти
DB_PATH = 'agents_memory.db'


def initialize_database():
    """Инициализация базы данных для хранения агентов и проектов."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    strategy TEXT,
                    fitness REAL,
                    generation INTEGER,
                    role TEXT,
                    last_updated TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    description TEXT,
                    status TEXT,
                    last_updated TEXT
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Ошибка инициализации базы данных: {e}")
        raise


initialize_database()


class Agent:
    """Класс, представляющий агента."""

    def __init__(self, agent_id: str, strategy: Dict[str, Any], generation: int = 0, role: str = "employee"):
        self.agent_id = agent_id
        self.strategy = strategy
        self.fitness = 0
        self.generation = generation
        self.role = role  # Роль агента: 'employee', 'debugger', 'fixer'
        self.swarm = None  # Связь с роем при инициализации

    async def perform_task(self, task: str):
        """
        Метод для агента, выполняющего задачу.
        Интеграция с GPT-4 Free для обработки задачи.
        """
        logger.info(f"Agent {self.agent_id} ({self.role}) выполняет задачу: {task}")
        try:
            # Формирование сообщений для GPT-4
            messages = [
                {"role": "system", "content": "You are an intelligent agent."},
                {"role": "user", "content": task}
            ]

            # Вызов GPT-4 Free API
            response = ChatCompletion.create(
                model="o1-mini",
                messages=messages,
                provider=Provider.Airforce  # Используйте правильный провайдер из g4f
            )

            # Обработка ответа
            reply = response.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            logger.info(f"Agent {self.agent_id} получил ответ: {reply}")

            # Оценка фитнеса на основе ответа
            self.fitness = self.evaluate_response(reply)
            logger.info(f"Фитнес агента {self.agent_id} установлен на: {self.fitness:.4f}")

            # Сохранение ответа в файловую систему проекта
            self.swarm.project_fs.save_file(self.swarm.get_project_id_by_agent(self.agent_id), 'response.txt', reply)

        except Exception as e:
            logger.error(f"Агент {self.agent_id} не смог выполнить задачу: {e}")
            logger.error(traceback.format_exc())
            self.fitness = 0  # При ошибке фитнес устанавливается в 0 или другое значение

        await asyncio.sleep(random.uniform(0.1, 0.5))  # Симуляция времени выполнения

    def evaluate_response(self, response: str) -> float:
        """
        Оценка ответа GPT-4 для определения фитнеса агента.
        В реальной системе эта функция должна быть более сложной.
        """
        # Пример простой оценки: длина ответа
        return float(len(response))

    def mutate(self, mutation_rate: float):
        """
        Мутация стратегии агента с заданной вероятностью.
        """
        if random.random() < mutation_rate:
            # Простейшая мутация: случайное изменение одного из параметров стратегии
            key_to_mutate = random.choice(list(self.strategy.keys()))
            old_value = self.strategy[key_to_mutate]
            mutation = random.uniform(-0.1, 0.1)
            self.strategy[key_to_mutate] = round(self.strategy[key_to_mutate] + mutation, 4)
            logger.info(f"Agent {self.agent_id} мутировал {key_to_mutate} с {old_value:.4f} на {self.strategy[key_to_mutate]:.4f}")

    def save_to_db(self, generation: int):
        """
        Сохранение состояния агента в базу данных.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO agents (agent_id, strategy, fitness, generation, role, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    self.agent_id,
                    json.dumps(self.strategy),
                    self.fitness,
                    generation,
                    self.role,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Ошибка базы данных при сохранении агента {self.agent_id}: {e}")
            logger.error(traceback.format_exc())

    def set_swarm(self, swarm):
        """
        Связывание агента с роем для доступа к файловой системе.
        """
        self.swarm = swarm


class Project:
    """Класс, представляющий проект."""

    def __init__(self, project_id: str, description: str):
        self.project_id = project_id
        self.description = description
        self.status = "ongoing"  # Статус проекта: 'ongoing', 'completed', 'failed'
        self.creation_time = datetime.now()
        self.last_updated = datetime.now()

    def update_status(self, status: str):
        self.status = status
        self.last_updated = datetime.now()

    def save_to_db(self):
        """
        Сохранение проекта в базу данных.
        """
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO projects (project_id, description, status, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (
                    self.project_id,
                    self.description,
                    self.status,
                    self.last_updated.isoformat()
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Ошибка базы данных при сохранении проекта {self.project_id}: {e}")
            logger.error(traceback.format_exc())


class ProjectFS:
    """
    Класс для управления файловой системой проектов.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def create_project(self, project_id: str):
        """
        Создание файловой системы для нового проекта.
        """
        project_path = os.path.join(self.base_path, project_id)
        os.makedirs(project_path, exist_ok=True)
        with open(os.path.join(project_path, 'README.txt'), 'w') as f:
            f.write(f"Проект {project_id} инициализирован.\n")

    def save_file(self, project_id: str, filename: str, content: str):
        """
        Сохранение файла в проекте.
        """
        project_path = os.path.join(self.base_path, project_id)
        if not os.path.exists(project_path):
            self.create_project(project_id)
        file_path = os.path.join(project_path, filename)
        try:
            with open(file_path, 'a') as f:
                f.write(content + '\n')
        except IOError as e:
            logger.error(f"Ошибка при записи в файл {file_path}: {e}")
            logger.error(traceback.format_exc())

    def load_file(self, project_id: str, filename: str) -> str:
        """
        Загрузка содержимого файла из проекта.
        """
        file_path = os.path.join(self.base_path, project_id, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except IOError as e:
                logger.error(f"Ошибка при чтении файла {file_path}: {e}")
                logger.error(traceback.format_exc())
        return ""


class Swarm:
    """Класс, представляющий рой агентов."""

    def __init__(self, config: Dict[str, Any], gui):
        self.agents: List[Agent] = []
        self.config = config
        self.generation = 0
        self.gui = gui  # Ссылка на GUI для обновления интерфейса
        self.project_fs = ProjectFS("projects")
        self.swarm_lock = asyncio.Lock()  # Для безопасного доступа к агентам
        self.projects: Dict[str, Project] = {}  # Хранилище проектов
        self.running = True

    def initialize_population(self):
        logger.info("Инициализация популяции...")
        for _ in range(self.config["initial_population"]):
            agent_id = str(uuid.uuid4())
            strategy = self.initialize_strategy()
            role = "employee"
            agent = Agent(agent_id, strategy, self.generation, role)
            agent.set_swarm(self)
            self.agents.append(agent)
            self.project_fs.create_project(agent_id)
            agent.save_to_db(self.generation)
        logger.info(f"Популяция инициализирована с {len(self.agents)} агентами.")

    def initialize_strategy(self) -> Dict[str, Any]:
        """
        Инициализация стратегии агента случайными параметрами.
        """
        return {
            "param1": round(random.uniform(0, 1), 4),
            "param2": round(random.uniform(0, 1), 4),
            "param3": round(random.uniform(0, 1), 4)
        }

    async def evaluate_agents(self):
        """
        Оценка всех агентов путем выполнения задачи.
        """
        logger.info("Оценка всех агентов...")
        tasks = [agent.perform_task(self.config["task"]) for agent in self.agents]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Все агенты выполнили свои задачи.")

    def select_top_agents(self) -> List[Agent]:
        """
        Выбор лучших агентов на основе их фитнеса.
        """
        sorted_agents = sorted(self.agents, key=lambda agent: agent.fitness, reverse=True)
        top_agents = sorted_agents[:max(2, len(self.agents) // 2)]
        logger.info(f"Выбрано {len(top_agents)} лучших агентов для воспроизводства.")
        return top_agents

    def reproduce(self, top_agents: List[Agent]):
        """
        Воспроизводство новых агентов на основе топовых агентов.
        """
        offspring = []
        while len(self.agents) + len(offspring) < self.config["max_population"]:
            parent = random.choice(top_agents)
            child_strategy = parent.strategy.copy()
            child_agent = Agent(str(uuid.uuid4()), child_strategy, self.generation, parent.role)
            child_agent.set_swarm(self)
            child_agent.mutate(self.config["mutation_rate"])
            offspring.append(child_agent)
            self.project_fs.create_project(child_agent.agent_id)
            child_agent.save_to_db(self.generation)
        self.agents.extend(offspring)
        logger.info(f"Добавлено {len(offspring)} новых агентов в популяцию.")

    def cull_population(self):
        """
        Ограничение размера популяции до максимального значения.
        """
        if len(self.agents) > self.config["max_population"]:
            self.agents = self.agents[:self.config["max_population"]]
            logger.info(f"Популяция уменьшена до {self.config['max_population']} агентов.")

    def get_project_id_by_agent(self, agent_id: str) -> str:
        """
        Получение project_id по agent_id.
        Предполагается, что каждый агент соответствует одному проекту.
        """
        return agent_id

    def add_project(self, project_id: str, description: str):
        """
        Добавление нового проекта.
        """
        if project_id in self.projects:
            logger.warning(f"Проект {project_id} уже существует.")
            self.gui.log(f"Проект {project_id} уже существует.\n")
            return
        project = Project(project_id, description)
        self.projects[project_id] = project
        project.save_to_db()
        logger.info(f"Добавлен новый проект: {project_id}")
        self.gui.log(f"Добавлен новый проект: {project_id}\n")

    def assign_task_to_project(self, project_id: str, task: str):
        """
        Назначение задачи проекту.
        """
        if project_id not in self.projects:
            logger.error(f"Проект {project_id} не существует.")
            self.gui.log(f"Проект {project_id} не существует.\n")
            return
        project = self.projects[project_id]
        if project.status != "ongoing":
            logger.warning(f"Нельзя назначить задачу проекту {project_id} со статусом {project.status}.")
            self.gui.log(f"Нельзя назначить задачу проекту {project_id} со статусом {project.status}.\n")
            return
        # Назначение задачи топовым агентам
        top_agents = self.select_top_agents()
        for agent in top_agents:
            asyncio.create_task(agent.perform_task(task))
        logger.info(f"Назначена задача проекту {project_id}: {task}")
        self.gui.log(f"Назначена задача проекту {project_id}: {task}\n")

    async def run_generation(self):
        """
        Запуск одной генерации эволюционного процесса.
        """
        async with self.swarm_lock:
            self.generation += 1
            await self.evaluate_agents()
            top_agents = self.select_top_agents()
            self.reproduce(top_agents)
            self.cull_population()
            self.save_agents()
            self.gui.update_gui(self)
            # Обновление GUI через метод log
            best_agents = sorted(self.agents, key=lambda agent: agent.fitness, reverse=True)[:5]
            top_info = "\n".join([f"ID: {agent.agent_id}, Fitness: {agent.fitness:.4f}" for agent in best_agents])
            message = f"--- Generation {self.generation} ---\nTop Agents:\n{top_info}"
            logger.info(message)
            self.gui.log(message + "\n")

    def save_agents(self):
        """
        Сохранение состояния всех агентов в базу данных.
        """
        for agent in self.agents:
            agent.save_to_db(self.generation)

    async def notify_telegram(self, message: str):
        """
        Отправка уведомлений.
        Для локальной версии можно заменить отправку сообщений на логирование или
        обновление GUI
        """
        # Пример реализации: логирование сообщения
        logger.info(f"Telegram уведомление: {message}")
        self.gui.log(f"Telegram уведомление: {message}\n")

    async def run_evolution(self):
        """
        Запуск эволюционного процесса.
        """
        self.initialize_population()
        for cycle in range(self.config["evaluation_cycles"]):
            if not self.running:
                break
            logger.info(f"Запуск генерации {cycle + 1}/{self.config['evaluation_cycles']}")
            await self.run_generation()
            await asyncio.sleep(0.1)  # Позволяет другим задачам выполняться
        logger.info("Эволюционный процесс завершен.")
        best_agents = sorted(self.agents, key=lambda agent: agent.fitness, reverse=True)[:5]
        for agent in best_agents:
            logger.info(f"Agent ID: {agent.agent_id}, Fitness: {agent.fitness}, Strategy: {agent.strategy}")
        await self.notify_telegram("Эволюционный процесс завершен.")
        messagebox.showinfo("EvoSwarms", "Эволюционный процесс завершен.")

    def stop_evolution_process(self):
        """Остановка эволюционного процесса."""
        self.running = False
        logger.info("Эволюционный процесс остановлен.")


class GUI:
    """Класс, представляющий графический интерфейс пользователя."""

    def __init__(self, root, swarm: Swarm):
        self.root = root
        self.swarm = swarm
        self.root.title("AI IDE - EvoSwarms Agent System")
        self.root.geometry("1000x700")
        self.create_widgets()
        self.current_project_id = None
        self.editor = None

    def create_widgets(self):
        """Создание элементов интерфейса."""

        # Главный фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Левый фрейм для панели проектов
        left_frame = ttk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Панель проектов
        projects_label = ttk.Label(left_frame, text="Проекты")
        projects_label.pack(pady=5)

        self.projects_listbox = tk.Listbox(left_frame)
        self.projects_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.projects_listbox.bind('<<ListboxSelect>>', self.on_project_select)

        # Кнопки управления проектами
        create_project_button = ttk.Button(left_frame, text="Создать проект", command=self.create_project)
        create_project_button.pack(pady=2)

        view_project_button = ttk.Button(left_frame, text="Просмотреть проект", command=self.view_project)
        view_project_button.pack(pady=2)

        delete_project_button = ttk.Button(left_frame, text="Удалить проект", command=self.delete_project)
        delete_project_button.pack(pady=2)

        # Центральный фрейм для редактора и логов
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Редактор кода
        editor_label = ttk.Label(center_frame, text="Редактор кода")
        editor_label.pack(pady=5)

        self.editor_text = scrolledtext.ScrolledText(center_frame, wrap=tk.NONE, height=20)
        self.editor_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Кнопки редактора
        editor_buttons_frame = ttk.Frame(center_frame)
        editor_buttons_frame.pack(pady=2)

        open_file_button = ttk.Button(editor_buttons_frame, text="Открыть файл", command=self.open_file)
        open_file_button.pack(side=tk.LEFT, padx=5)

        save_file_button = ttk.Button(editor_buttons_frame, text="Сохранить файл", command=self.save_file)
        save_file_button.pack(side=tk.LEFT, padx=5)

        # Нижний фрейм для логов и управления эволюцией
        bottom_frame = ttk.Frame(main_frame, height=200)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Логовая область
        log_label = ttk.Label(bottom_frame, text="Логи")
        log_label.pack(pady=5)

        self.log_area = scrolledtext.ScrolledText(bottom_frame, width=80, height=10, state='disabled')
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)

        # Кнопки управления эволюцией и задачами
        controls_frame = ttk.Frame(bottom_frame)
        controls_frame.pack(pady=5)

        self.start_button = ttk.Button(controls_frame, text="Начать эволюцию", command=self.start_evolution)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(controls_frame, text="Остановить эволюцию", command=self.stop_evolution, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)

        schedule_task_button = ttk.Button(controls_frame, text="Назначить задачу", command=self.schedule_task)
        schedule_task_button.pack(side=tk.LEFT, padx=5)

        run_debugger_button = ttk.Button(controls_frame, text="Запустить отладчик", command=self.run_debugger)
        run_debugger_button.pack(side=tk.LEFT, padx=5)

    def start_evolution(self):
        """Запуск процесса эволюции."""
        if not self.swarm:
            messagebox.showerror("Ошибка", "Swarm не инициализирован.")
            return
        if hasattr(self, 'swarm_task') and not self.swarm_task.done():
            messagebox.showwarning("Предупреждение", "Эволюция уже запущена.")
            return
        self.swarm.running = True
        self.swarm_task = asyncio.create_task(self.swarm.run_evolution())
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.log("Эволюция запущена...\n")

    def stop_evolution(self):
        """Остановка процесса эволюции."""
        if hasattr(self, 'swarm_task') and not self.swarm_task.done():
            self.swarm.stop_evolution_process()
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.log("Эволюция остановлена.\n")

    def log(self, message: str):
        """Запись сообщения в логовую область."""
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message)
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def update_gui(self, swarm: Swarm):
        """Обновление интерфейса GUI на основе состояния роев."""
        if not swarm:
            return
        # Обновление списка проектов
        self.projects_listbox.delete(0, tk.END)
        for project_id, project in swarm.projects.items():
            display_text = f"{project_id} - {project.status}"
            self.projects_listbox.insert(tk.END, display_text)

    def create_project(self):
        """Создание нового проекта."""
        project_description = simpledialog.askstring("Создать проект", "Введите описание проекта:")
        if project_description:
            project_id = str(uuid.uuid4())
            self.swarm.add_project(project_id, project_description)
            self.log(f"Создан проект {project_id} с описанием: {project_description}\n")
            self.projects_listbox.insert(tk.END, f"{project_id} - ongoing")

    def view_project(self):
        """Просмотр выбранного проекта."""
        selected = self.projects_listbox.curselection()
        if not selected:
            self.log("Проект не выбран.\n")
            return
        project_display = self.projects_listbox.get(selected[0])
        project_id = project_display.split(" - ")[0]
        readme_content = self.swarm.project_fs.load_file(project_id, 'README.txt')
        response_content = self.swarm.project_fs.load_file(project_id, 'response.txt')
        content_display = f"Проект {project_id} README:\n{readme_content}\n\nОтветы:\n{response_content}\n"
        self.editor_text.delete(1.0, tk.END)
        self.editor_text.insert(tk.END, content_display)
        self.log(f"Просмотр проекта {project_id}.\n")

    def delete_project(self):
        """Удаление выбранного проекта."""
        selected = self.projects_listbox.curselection()
        if not selected:
            self.log("Проект не выбран для удаления.\n")
            return
        project_display = self.projects_listbox.get(selected[0])
        project_id = project_display.split(" - ")[0]
        confirm = messagebox.askyesno("Подтверждение", f"Вы уверены, что хотите удалить проект {project_id}?")
        if confirm:
            # Удаление из базы данных
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM projects WHERE project_id = ?', (project_id,))
                    conn.commit()
                # Удаление из памяти
                del self.swarm.projects[project_id]
                # Удаление файловой системы
                project_path = os.path.join(self.swarm.project_fs.base_path, project_id)
                shutil.rmtree(project_path, ignore_errors=True)
                # Обновление интерфейса
                self.projects_listbox.delete(selected[0])
                self.log(f"Проект {project_id} удален.\n")
            except Exception as e:
                logger.error(f"Ошибка при удалении проекта {project_id}: {e}")
                logger.error(traceback.format_exc())
                self.log(f"Ошибка при удалении проекта {project_id}.\n")

    def open_file(self):
        """Открытие файла из выбранного проекта."""
        if not self.current_project_id:
            self.log("Проект не выбран.\n")
            return
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(self.swarm.project_fs.base_path, self.current_project_id),
            title="Выберите файл"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                self.editor_text.delete(1.0, tk.END)
                self.editor_text.insert(tk.END, content)
                self.log(f"Файл {os.path.basename(file_path)} открыт.\n")
            except Exception as e:
                logger.error(f"Ошибка при открытии файла {file_path}: {e}")
                logger.error(traceback.format_exc())
                self.log(f"Ошибка при открытии файла {file_path}.\n")

    def save_file(self):
        """Сохранение содержимого редактора в файл."""
        if not self.current_project_id:
            self.log("Проект не выбран.\n")
            return
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.join(self.swarm.project_fs.base_path, self.current_project_id),
            title="Сохранить файл",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if file_path:
            try:
                content = self.editor_text.get(1.0, tk.END)
                with open(file_path, 'w') as f:
                    f.write(content)
                self.log(f"Файл сохранен: {os.path.basename(file_path)}.\n")
            except Exception as e:
                logger.error(f"Ошибка при сохранении файла {file_path}: {e}")
                logger.error(traceback.format_exc())
                self.log(f"Ошибка при сохранении файла {file_path}.\n")

    def on_project_select(self, event):
        """Обработчик выбора проекта из списка."""
        selected = self.projects_listbox.curselection()
        if selected:
            project_display = self.projects_listbox.get(selected[0])
            self.current_project_id = project_display.split(" - ")[0]
            self.log(f"Выбран проект {self.current_project_id}.\n")
        else:
            self.current_project_id = None

    def schedule_task(self):
        """Назначение задачи проекту."""
        project_id = simpledialog.askstring("Назначить задачу", "Введите ID проекта:")
        if not project_id:
            self.log("ID проекта не предоставлен.\n")
            return
        task = simpledialog.askstring("Назначить задачу", "Введите описание задачи:")
        if not task:
            self.log("Описание задачи не предоставлено.\n")
            return
        self.swarm.assign_task_to_project(project_id, task)

    def run_debugger(self):
        """Запуск отладчика для всех активных проектов."""
        # Пример: запуск агента-отладчика для всех активных проектов
        for project_id, project in self.swarm.projects.items():
            if project.status == "ongoing":
                task = f"Проверьте ответы для проекта {project_id} на наличие ошибок."
                top_agents = self.swarm.select_top_agents()
                for agent in top_agents:
                    asyncio.create_task(agent.perform_task(task))
        self.log("Задачи отладки были назначены.\n")

    def set_swarm(self, swarm: Swarm):
        """
        Связывание GUI с роем и обновление GUI.
        """
        self.swarm = swarm
        self.update_gui(self.swarm)
        # Обновление списка проектов
        self.update_projects_list()

    def update_projects_list(self):
        """Обновление списка проектов в интерфейсе."""
        self.projects_listbox.delete(0, tk.END)
        for project_id, project in self.swarm.projects.items():
            display_text = f"{project_id} - {project.status}"
            self.projects_listbox.insert(tk.END, display_text)


async def run_gui(root):
    """
    Асинхронно обновляет GUI без блокировки главного потока.
    """
    try:
        while True:
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                # GUI был закрыт
                logger.info("GUI был закрыт.")
                break
            await asyncio.sleep(0.01)  # Небольшая задержка для предотвращения высокого использования ЦП
    except Exception as e:
        logger.error(f"Ошибка в run_gui: {e}")
        logger.error(traceback.format_exc())
        raise


async def main_async():
    # Инициализация GUI
    root = tk.Tk()
    gui = GUI(root, None)  # Изначально передаем None

    # Инициализация Swarm с привязкой к GUI
    swarm = Swarm(CONFIG, gui)
    gui.set_swarm(swarm)  # Связывание Swarm и GUI

    # Запуск как GUI, так и эволюционного процесса роя одновременно
    await asyncio.gather(
        run_gui(root),
        swarm.run_evolution()
    )


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Программа завершена пользователем.")
    except Exception as e:
        logger.error(f"Произошла неожиданная ошибка: {e}")
        # Поскольку Tkinter может не быть инициализирован, используем временный скрытый root для отображения сообщения об ошибке.
        try:
            temp_root = tk.Tk()
            temp_root.withdraw()
            messagebox.showerror("Ошибка", f"Произошла неожиданная ошибка:\n{e}")
        except Exception as gui_e:
            logger.error(f"Не удалось отобразить messagebox: {gui_e}")
            logger.error(traceback.format_exc())
        finally:
            temp_root.destroy()


if __name__ == "__main__":
    main()
