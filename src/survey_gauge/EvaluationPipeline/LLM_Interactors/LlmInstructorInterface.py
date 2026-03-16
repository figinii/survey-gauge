from typing import Any, Coroutine
from pydantic import BaseModel
from instructor import Instructor

class LlmInstructorInterface:
  """Interface for interacting with the LLM, sending prompts and receiving responses."""
  def __init__(self, client: Instructor, model_name: str):
    self.client = client
    self.model_name = model_name

  def subscribe_prompt(self, prompt: str, temperature: float, output_model: BaseModel, role:str = 'user') -> Coroutine[Any, Any, Any]:
    """Subscribe a prompt to the engine and return the result along with the request_id."""

    return self.client.chat.completions.create(
      model = self.model_name,
      messages=[{"role": role, "content": prompt}],
      response_model=output_model,
      temperature=temperature
    )
