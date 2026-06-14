from fasticrl.strategist.models.strategy_ouput import StrategyOutput
from typing import Callable

import tqdm
import yaml
from agno.models.base import Model

import fasticrl.models.sentinel as sentinel
from fasticrl.learner.core import LearnerAgent
from fasticrl.learner.models.learn_output import LearnerOutput
from fasticrl.models.agent_save_state import AgentSaveState
from fasticrl.models.attempt import Attempt
from fasticrl.reward.core import RewardAgent
from fasticrl.reward.models.reward_output import RewardOutput
from fasticrl.strategist.core import StrategistAgent


class ICRLLearner:
    def __init__(
        self,
        learner_model: Model,
        reward_model: Model,
        strategy_model: Model,
        task_description: str = sentinel._sentinel,
        tasks: list[str] = [],
        buffer: list[Attempt] = [],
        strategy: str = "",
    ):
        self.buffer = buffer
        self.task_description = task_description
        self.tasks = tasks
        self.strategy = strategy

        self.learner_agent = LearnerAgent(model=learner_model)
        self.reward_agent = RewardAgent(model=reward_model)
        self.strategist_agent = StrategistAgent(model=strategy_model)

        self.episode = 0

    def _learn_step(self, retries=3):
        attempt: Attempt = self.generate_attempt_by_present_task()

        self.buffer.append(attempt)

        self.episode += 1

    def auto_learn(self, episodes=None, cli_mode=False):

        if episodes is None:
            episodes = len(self.tasks)

        iterator = tqdm.tqdm(range(episodes)) if cli_mode else range(episodes)

        for _ in iterator:
            self._learn_step()

    def __attempts_as_xml(self) -> str:
        attempts = ""
        for attempt in self.buffer:
            attempts += (
                "<attempt>\n"
                + f"\tTask: {attempt.task}\n\tOutput: {attempt.output}\n\tReward: {str(attempt.reward)}"
                + "\n</attempt>\n"
            )

        return attempts.strip()

    def generate_action(self, task: str):
        attempts = self.__attempts_as_xml()

        learner_output: LearnerOutput = self.learner_agent.generate_learning_output(
            task=task,
            task_description=self.task_description,
            strategy=self.strategy,
            buffer_as_xml=attempts,
        )

        return learner_output

    def generate_reward(self, task: str, action: LearnerOutput) -> RewardOutput:
        return self.reward_agent.generate_reward_output(
            task=task, output=action.learning_output
        )

    def generate_attempt_by_present_task(self) -> Attempt:

        task: str = self.__present_learning_task

        learner_output: LearnerOutput = self.generate_action(task)

        reward_output: RewardOutput = self.generate_reward(task, learner_output)

        return Attempt(
            task=self.__present_learning_task,
            reward=reward_output.reward,
            output=learner_output.learning_output,
        )

    def update_strategy(self):
        attempts: str = self.__attempts_as_xml()

        strategist_out: StrategyOutput = self.strategist_agent.generate_new_strategy(
            task_description=self.task_description,
            existing_strategy=self.strategy,
            buffer_as_xml=attempts,
        )

        self.strategy = strategist_out.strategy

    @property
    def __present_learning_task(self) -> str:
        return str(self.tasks[self.episode % len(self.tasks)])

    @property
    def agent_save_state(self) -> AgentSaveState:
        save_state: AgentSaveState = AgentSaveState(
            task_description=self.task_description,
            strategy=self.strategy,
            buffer=[dict(b.model_dump(mode="json")) for b in self.buffer],
        )
        return save_state

    def to_yaml(self, path: str):
        if not path.endswith(".yaml"):
            path += ".yaml"

        with open(path, "w") as outfile:
            yaml.dump(
                self.agent_save_state.model_dump(mode="json"),
                outfile,
                default_flow_style=False,
            )

    @classmethod
    def from_yaml(
        cls, path: str, learner_model: Model, reward_model: Model, strategy_model: Model
    ):
        with open(path, "r") as savestate:
            save_state = yaml.load(savestate, Loader=yaml.SafeLoader)

            agent_state: AgentSaveState = AgentSaveState.model_validate(save_state)

            return ICRLLearner(
                task_description=agent_state.task_description,
                buffer=agent_state.buffer,
                strategy=agent_state.strategy,
                learner_model=learner_model,
                reward_model=reward_model,
                strategy_model=strategy_model,
            )
