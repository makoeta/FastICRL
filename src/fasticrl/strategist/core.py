from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput

from fasticrl.strategist.models.strategy_ouput import StrategyOutput
from fasticrl.strategist.prompts import (
    ICRL_STRATEGY_INPUT_PROMPT,
    ICRL_STRATEGY_SYSTEM_PROMPT,
)


class StrategistAgent(Agent):

    def __init__(self, model: Model, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.system_message = ICRL_STRATEGY_SYSTEM_PROMPT
        self.output_schema = StrategyOutput

    def generate_new_strategy(
        self, task_description: str, existing_strategy: str, buffer_as_xml: str
    ) -> StrategyOutput:
        run_out: RunOutput = super().run(
            ICRL_STRATEGY_INPUT_PROMPT.format(
                task_description=task_description,
                strategy=existing_strategy,
                attempts=buffer_as_xml,
            )
        )

        return run_out.content
