from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput

from fasticrl.reward.models.reward_output import RewardOutput
from fasticrl.reward.prompts import ICRL_REWARD_INPUT_PROMPT, ICRL_REWARD_SYSTEM_PROMPT


class RewardAgent(Agent):

    def __init__(self, model: Model, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.output_schema = RewardOutput
        self.system_message = ICRL_REWARD_SYSTEM_PROMPT

    def generate_reward_output(self, task: str, output: str) -> RewardOutput:

        run_out: RunOutput = super().run(
            ICRL_REWARD_INPUT_PROMPT.format(task=task, output=output)
        )

        return run_out.content
