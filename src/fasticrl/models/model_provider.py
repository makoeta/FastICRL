
from OpenAI.types.responses import Response


class ModelProvider:

    def __init__(self, model: str):
        self.model = model

    def invoke(self, input: str, instructions: str, **kwargs) -> Response:
        raise NotImplementedError()
