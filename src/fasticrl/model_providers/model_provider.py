
from openai.types.responses import Response


class ModelProvider:

    def __init__(self, model: str, url = ""):
        self.model = model
        self.url = url

    def invoke(self, input: str, instructions: str, **kwargs) -> Response:
        raise NotImplementedError()
