from abc import ABC, abstractmethod
from typing import Any, Coroutine
from pydantic import BaseModel

class LlmInteract(ABC):
  """Class to interact with the LLM, sending prompts and receiving responses."""
  def __init__(self):
    pass

  @abstractmethod
  def subscribe_prompt(self, prompt: str, temperature: float, output_model: BaseModel, 
                      top_p: float, seed:int, role:str = 'user') -> Coroutine[Any, Any, Any]:
    """Subscribe a prompt to the engine and return the result along with the request_id."""
    raise NotImplementedError("This method should be implemented by subclasses of LLM_Interact.")