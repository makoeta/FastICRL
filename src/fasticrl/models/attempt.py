from pydantic import Field
from pydantic import BaseModel


class Attempt(BaseModel):
    task: str = Field()
    output: str = Field()
    reward: int = Field()
