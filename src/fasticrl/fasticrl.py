
from typing import Callable
from fasticrl.models.model_provider import ModelProvider


class Runtime:
    def __init__(self, model: ModelProvider, reward_func: Callable[[str], int], episodes: int, buffer, task_description: str):
        self.model = model
        self.reward_func = reward_func
        self.episodes = episodes
        self.buffer = buffer
        self.task_description = task_description
    
    
    def run(k=1):
        pass