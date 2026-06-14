from fasticrl.learner.prompts import ICRL_LEARNING_SYSTEM_PROMPT
from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput

from fasticrl.learner.models.learn_output import LearnerOutput
from fasticrl.learner.prompts import ICRL_LEARNING_INPUT_PROMPT


class LearnerAgent(Agent):

    def __init__(self, model: Model, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.output_schema = LearnerOutput
        self.system_message = ICRL_LEARNING_SYSTEM_PROMPT

    def generate_learning_output(
        self, buffer_as_xml: str, task: str, task_description: str, strategy: str
    ) -> LearnerOutput:
        run_out: RunOutput = super().run(
            ICRL_LEARNING_INPUT_PROMPT.format(
                attempts=buffer_as_xml,
                task=task,
                task_description=task_description,
                strategy=strategy,
            )
        )

        return run_out.content
