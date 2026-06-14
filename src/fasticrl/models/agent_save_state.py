from typing import Annotated

from pydantic import BaseModel, Field

from fasticrl.models.attempt import Attempt


class AgentSaveState(BaseModel):
    task_description: Annotated[str, Field()]
    strategy: Annotated[str, Field(default="")]
    buffer: Annotated[list[Attempt], Field(default_factory=list)]
