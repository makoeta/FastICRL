from pydantic import Field
from typing import Annotated
from pydantic import BaseModel


class RewardOutput(BaseModel):
    reward: Annotated[
        int,
        Field(
            description="Reward to the given output. Must be in range from 1 - 10.",
            ge=1,
            le=10,
        ),
    ]
