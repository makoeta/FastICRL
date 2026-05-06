
from typing import Callable
from fasticrl.models.model_provider import ModelProvider


class Runtime:
    def __init__(self, model_provider: ModelProvider, reward_func: Callable[[str], int], total_episodes: int, buffer: list[str], task_description: str, icrl_instruction: str, tasks: list[str]):
        self.model = model_provider
        self.reward_func = reward_func
        self.episodes = total_episodes
        self.buffer = buffer
        self.task_description = task_description
        self.tasks = tasks
        self.icrl_instruction = icrl_instruction
        
        self.episode = 0
    
    
    def run(self):
        
        for k in range(self.episodes):
            example = self._generate_example(k)
            
    
    
    def _generate_example(self, k) -> list[(str, str)]:
        pass
    
    
    @classmethod
    def __initial_prompt(self) -> str:
        
        attempts = ""
        for attempt in self.buffer:
            attempts += "<attempt>\n" + attempt + "\n</attempt>\n"
        
        attempts = attempts.strip()
        
            
        
        return f"""
            {attempts} 
            **Task**: {self.task_description}
            **Answer**:
            **Prompt**: {self.tasks[self.episode % len(self.tasks)]}
            """