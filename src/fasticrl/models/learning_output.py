
from pydantic import Field
from pydantic import BaseModel

class LearningOutput(BaseModel):
    thoughts: str = Field()
    output: str = Field()