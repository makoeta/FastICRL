import json
from json import JSONDecodeError
from typing import Callable

import tqdm
import yaml
from agno.agent import Agent
from agno.run.agent import RunOutput

from fasticrl.models.icrl_mode import ICRLMode
import fasticrl.models.sentinel as sentinel
from fasticrl import prompts
from fasticrl.models.agent_save_state import AgentSaveState
from fasticrl.models.attempt import Attempt
from fasticrl.models.learning_output import LearningOutput

class ICRLLearner:
    def __init__(
        self,
        agent: Agent = sentinel._sentinel,
        reward_func: Callable[[str, str], int] = sentinel._sentinel,
        task_description: str = sentinel._sentinel,
        tasks: list[str] = [],
        buffer: list[Attempt] = [],
        strategy: str = "",
    ):
        self.agent = agent
        self.reward_func = reward_func
        self.buffer = buffer
        self.task_description = task_description
        self.tasks = tasks
        self.strategy = strategy

        self.mode = ICRLMode.LEARN
        self.episode = 0

    @property
    def __is_learnable(self) -> bool:
        return (
            self.agent is not sentinel._sentinel
            and self.reward_func is not sentinel._sentinel
            and self.task_description is not sentinel._sentinel
        )

    def _learn_step(self, retries=3):

        attempt = None
        tries = 0
        while attempt is None:
            tries += 1
            attempt: Attempt = self.generate_attempt_by_task(
                self.__present_learning_task
            )
            if tries > retries:
                break

        if attempt is not None:
            if attempt.reward != -1:
                self.buffer.append(attempt)

        self.episode += 1

    def auto_learn(self, episodes=None, cli_mode=False):

        if episodes is None:
            episodes = len(self.tasks)

        if not self.__is_learnable:
            raise ValueError(
                "Model, reward function, and task description must be provided for learning."
            )

        iterator = tqdm.tqdm(range(episodes)) if cli_mode else range(episodes)

        for _ in iterator:
            self._learn_step()

    def run(self, prompt: str, output_format="") -> RunOutput:
        return self.agent.run(self.build_icrl_prompt(prompt, output_format))

    def generate_attempt_by_task(self, task: str = sentinel._sentinel) -> Attempt:
        if task is sentinel._sentinel:
            raise ValueError("Task must be provided")
        model_out: RunOutput = self.run(prompt=self.__present_learning_task)

        try:
            learning_out: LearningOutput = LearningOutput.model_validate(
                json.loads(model_out.content)
            )
            reward: int = self.reward_func(
                learning_out.output,
                f"{self.task_description}\n{self.__present_learning_task}",
            )

            return Attempt(
                task=self.__present_learning_task,
                output=learning_out.output,
                reward=reward,
            )
        except Exception as e:
            print(e)

    def update_strategy(self, retries=3) -> bool:
        prior_mode = self.mode
        self.mode = ICRLMode.STRATEGIZE

        tries = 0

        while tries < retries:
            tries += 1
            try:
                out: RunOutput = self.run("")

                out_json = json.loads(out.content)
                self.strategy = out_json["strategy"]
                self.mode = prior_mode
                return True
            except JSONDecodeError as _:
                continue

        self.mode = prior_mode
        return False

    @property
    def __present_learning_task(self) -> str:
        return str(self.tasks[self.episode % len(self.tasks)])

    def build_icrl_prompt(self, prompt="", output_format="") -> str:

        attempts = ""
        for attempt in self.buffer:
            attempts += (
                "<attempt>\n"
                + f"\tTask: {attempt.task}\n\tOutput: {attempt.output}\n\tReward: {str(attempt.reward)}"
                + "\n</attempt>\n"
            )

        attempts = attempts.strip()

        if self.mode == ICRLMode.LEARN:
            return prompts.ICRL_LEARNING_PROMPT.format(
                attempts=attempts,
                task=prompt,
                task_description=self.task_description,
                strategy=self.strategy
            ).strip()

        if self.mode == ICRLMode.EVALUATE:
            return prompts.ICRL_EXPLOITATION_PROMPT.format(
                attempts=attempts,
                task=prompt,
                task_description=self.task_description,
                output_format=output_format,
            ).strip()

        if self.mode == ICRLMode.STRATEGIZE:
            return prompts.ICRL_STRATEGY_PROMPT.format(
                attempts=attempts,
                task_description=self.task_description,
                strategy=self.strategy,
            )

        raise NotImplementedError(f"Mode ({self.mode}) not implemented")

    @property
    def agent_save_state(self) -> AgentSaveState:
        save_state: AgentSaveState = AgentSaveState(task_description=self.task_description,
                                                    strategy=self.strategy,
                                                    buffer=[dict(b.model_dump(mode="json")) for b in self.buffer])
        return save_state

    def to_yaml(self, path: str):
        if not path.endswith(".yaml"):
            path += ".yaml"

        with open(path, "w") as outfile:
            yaml.dump(self.agent_save_state.model_dump(mode="json"), outfile, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as savestate:
            save_state = yaml.load(savestate, Loader=yaml.SafeLoader)
            
            agent_state: AgentSaveState = AgentSaveState.model_validate(save_state)

            return ICRLLearner(
                task_description=agent_state.task_description, buffer=agent_state.buffer, strategy=agent_state.strategy
            )

    def evaluation_mode(self):
        self.mode = ICRLMode.EVALUATE

    def learning_mode(self):
        self.mode = ICRLMode.LEARN
