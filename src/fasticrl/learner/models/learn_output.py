from pydantic import Field
from typing import Annotated
from pydantic import BaseModel


class LearnerOutput(BaseModel):
    learning_output: Annotated[str, Field(description="What was learned.")]
