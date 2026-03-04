from typing import Optional, List
from pathlib import Path
import yaml
from uuid import UUID, uuid4
from pydantic import BaseModel, ValidationError, Field

class Question(BaseModel):
  id: UUID = Field(default_factory=uuid4)
  question: str

class Questionnaire(BaseModel):
  preamble: str
  default_choices: List[str]
  questions: List[Question]

  @classmethod
  def from_yml(cls, path: str):
    path = Path(path)
    if not path.exists() or not path.is_file():
      raise(FileNotFoundError(f"Questionnaire file not found at {path}"))
    elif path.suffix not in ['.yml', '.yaml']:
      raise(ValueError(f"Questionnaire file must be a .yml or .yaml file, got {path.suffix}"))
    
    with open(path, 'r') as f:
      try:
        data = yaml.safe_load(f)
        return cls(**data)
      except yaml.YAMLError as e:
        raise(ValueError(f"Error parsing YAML file: {e}"))
      except ValidationError as e:
        raise(ValueError(f"Error validating questionnaire data: {e}"))