
from fasticrl.model_providers.model_provider import ModelProvider
from openai.types.responses.response import Response
from openai import OpenAI
import os


class OpenAIModel(ModelProvider):

    def __init__(self, model: str):
        super().__init__(model)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def invoke(self, input: str, instructions = "", **kwargs) -> Response:
        response: Response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input,
            **kwargs
        )
        
        return response
