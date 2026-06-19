from typing import Annotated, Literal
from pydantic import BaseModel, Field


class LearnerOutput(BaseModel):
    decision: Annotated[
        Literal["EXPLOIT", "EXPLORE"],
        Field(description="The explore/exploit decision for this step."),
    ] # Not used in the code again - just to let the model think about the strategies
    reflection: Annotated[
        str,
        Field(description="Brief justification for the decision."),
    ]
    learning_output: Annotated[
        str,
        Field(description="The actual response to the task. No meta-commentary — only the answer."),
    ]
