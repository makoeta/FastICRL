from fasticrl.models.attempt import Attempt
from fasticrl.models.learning_output import LearningOutput
from enum import Enum
import enum
from fasticrl import prompts
from typing import Callable
from fasticrl.model_providers.model_provider import ModelProvider
import fasticrl.prompts
import json
import fasticrl.models.sentinel as sentinel
import yaml
import tqdm

class ICRLMode(Enum):
    LEARNING = 1
    EVALUATE = 2


class ICRLLearner:
    def __init__(
        self,
        model_provider: ModelProvider,
        reward_func: Callable[[str, str], int],
        total_episodes: int,
        buffer: list[Attempt],
        task_description: str,
        tasks: list[str],
    ):
        self.model = model_provider
        self.reward_func = reward_func
        self.episodes = total_episodes
        self.buffer = buffer
        self.task_description = task_description
        self.tasks = tasks

        self.mode = ICRLMode.LEARNING
        self.episode = 0

    def _learn_step(self):
        attempt: Attempt = self.generate_attempt_by_task(self.__present_learning_task)

        if attempt.reward != -1:
            self.buffer.append(attempt)
        self.episode += 1

    def auto_learn(self, rerun_tasks=False, CLI_MODE=False):
        
        if CLI_MODE:
                
            for k in tqdm.tqdm(range(self.episodes)):
                self._learn_step()
            return
        for k in range(self.episodes):
            self._learn_step()



    def generate_attempt_by_task(
        self, task: str = sentinel._sentinel
    ) -> Attempt:

        if task is sentinel._sentinel:
            raise ValueError("Task must be provided")

        model_out = self.model.invoke(self.__present_prompt)

        try:
            learning_out: LearningOutput = LearningOutput.model_validate(json.loads(model_out.output_text))
            reward: int = self.reward_func(
                learning_out.output,
                f"{self.task_description}\n{self.__present_learning_task}",
            )
            
            return Attempt(task=self.__present_learning_task, output=learning_out.output, reward=reward)
        except Exception as e:
            print(e)

    @property
    def __present_learning_task(self) -> str:
        return str(self.tasks[self.episode % len(self.tasks)])

    @property
    def __present_prompt(self) -> str:

        attempts = ""
        for attempt in self.buffer:
            attempts += (
                "<attempt>\n"
                + f"\tTask: {attempt.task}\n\tOutput: {attempt.output}\n\tReward: {str(attempt.reward)}"
                + "\n</attempt>\n"
            )

        attempts = attempts.strip()

        if self.mode == ICRLMode.LEARNING:
            return prompts.ICRL_LEARNING_PROMPT.format(
                attempts=attempts,
                task=self.__present_learning_task,
                task_description=self.task_description,
            )

        if self.mode == ICRLMode.EVALUATE:
            return prompts.ICRL_EXPLOITATION_PROMPT.format(
                attempts=attempts,
                task=self.__present_learning_task,
                task_description=self.task_description,
            )

        raise NotImplementedError("Mode not implemented")

    def save_to_file(self, path: str):
        save_state = dict()
        save_state["taskDescription"] = self.task_description
        save_state["buffer"] = [dict(a.model_dump(mode="json")) for  a in self.buffer]

        if not path.endswith(".yaml"):
            path += ".yaml"

        with open(path, "w") as outfile:
            yaml.dump(save_state, outfile, default_flow_style=False)
