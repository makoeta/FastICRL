from fasticrl.strategist.models.strategy_output import StrategyOutput

import tqdm
import yaml
from agno.models.base import Model
from concurrent.futures import ThreadPoolExecutor

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
        tasks: list[str] = None,
        buffer: list[Attempt] = None,
        strategy: str = "",
    ):
        if task_description is sentinel._sentinel:
            raise ValueError("task_description must be provided")
        if tasks is not None and len(tasks) == 0:
            raise ValueError("tasks list must not be empty")

        self.buffer = buffer if buffer is not None else []
        self.task_description = task_description
        self.tasks = tasks if tasks is not None else []
        self.strategy = strategy

        self.learner_agent = LearnerAgent(model=learner_model)
        self.reward_agent = RewardAgent(model=reward_model)
        self.strategist_agent = StrategistAgent(model=strategy_model)

        self.episode = len(self.buffer)

    def _learn_step(self):
        attempt: Attempt = self.generate_attempt_by_present_task()
        self.buffer.append(attempt)
        self.episode += 1

    def auto_learn(
        self,
        episodes: int = None,
        batch_size: int = 1,
        cli_mode: bool = False,
        strategy_update_interval: int = None,
    ):
        if not self.tasks:
            raise ValueError("tasks list is empty — nothing to learn from")

        if episodes is None:
            episodes = len(self.tasks)

        iterator = tqdm.tqdm(range(episodes)) if cli_mode else range(episodes)

        for i in iterator:
            batch_tasks = [
                str(self.tasks[(self.episode + j) % len(self.tasks)])
                for j in range(batch_size)
            ]

            if batch_size == 1:
                attempts = [self._generate_attempt_for_task(batch_tasks[0])]
            else:
                with ThreadPoolExecutor(max_workers=batch_size) as pool:
                    futures = {
                        pool.submit(self._generate_attempt_for_task, task): idx
                        for idx, task in enumerate(batch_tasks)
                    }
                    attempts = [None] * batch_size
                    for future in futures:
                        attempts[futures[future]] = future.result()

            self.buffer.extend(attempts)
            self.episode += batch_size

            if strategy_update_interval and (i + 1) % strategy_update_interval == 0:
                self.update_strategy()

    def __attempts_as_xml(self) -> str:
        attempts = ""
        for attempt in self.buffer:
            attempts += (
                "<attempt>\n"
                + f"\tTask: {attempt.task}\n\tOutput: {attempt.output}\n\tReward: {str(attempt.reward)}"
                + "\n</attempt>\n"
            )
        return attempts.strip()

    def generate_action(self, task: str) -> LearnerOutput:
        return self.learner_agent.generate_learning_output(
            task=task,
            task_description=self.task_description,
            strategy=self.strategy,
            buffer_as_xml=self.__attempts_as_xml(),
        )

    def generate_reward(self, task: str, action: LearnerOutput) -> RewardOutput:
        return self.reward_agent.generate_reward_output(
            task=task, output=action.learning_output
        )

    def _generate_attempt_for_task(self, task: str) -> Attempt:
        learner_output: LearnerOutput = self.generate_action(task)
        reward_output: RewardOutput = self.generate_reward(task, learner_output)
        return Attempt(
            task=task,
            reward=reward_output.reward,
            output=learner_output.learning_output,
        )

    def generate_attempt_by_present_task(self) -> Attempt:
        return self._generate_attempt_for_task(self._present_learning_task)

    def update_strategy(self):
        strategist_out: StrategyOutput = self.strategist_agent.generate_new_strategy(
            task_description=self.task_description,
            existing_strategy=self.strategy,
            buffer_as_xml=self.__attempts_as_xml(),
        )
        self.strategy = strategist_out.strategy

    @property
    def _present_learning_task(self) -> str:
        return str(self.tasks[self.episode % len(self.tasks)])

    @property
    def agent_save_state(self) -> AgentSaveState:
        return AgentSaveState(
            task_description=self.task_description,
            strategy=self.strategy,
            buffer=[dict(b.model_dump(mode="json")) for b in self.buffer],
        )

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
        with open(path, "r") as f:
            save_state = yaml.load(f, Loader=yaml.SafeLoader)
            agent_state: AgentSaveState = AgentSaveState.model_validate(save_state)

            return ICRLLearner(
                task_description=agent_state.task_description,
                buffer=agent_state.buffer,
                strategy=agent_state.strategy,
                learner_model=learner_model,
                reward_model=reward_model,
                strategy_model=strategy_model,
            )
