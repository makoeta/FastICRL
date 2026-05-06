from fasticrl.models.ollama import OllamaModel
from fasticrl.fasticrl import Runtime


model_provider = OllamaModel("gemma4")

print(model_provider.invoke(input="", messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?'
  }]))