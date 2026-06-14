from pydantic import Field
from typing import Annotated
from pydantic import BaseModel


class StrategyOutput(BaseModel):

    strategy: Annotated[
        str, Field(description="The strategy extracted from prior action and rewards.")
    ]
