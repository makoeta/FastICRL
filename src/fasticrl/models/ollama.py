
from openai.types.responses.response import Response
from fasticrl.models.model_provider import ModelProvider
from ollama import chat

class OllamaModel(ModelProvider):

    def __init__(self, model: str):
        super().__init__(model, url="")

    def invoke(self, input: str, **kwargs) -> Response:
        response: Response = chat(self.model, **kwargs)
        
        return response
