from typing import List
from pathlib import Path
import yaml
from uuid import UUID, uuid4
from pydantic import BaseModel, ValidationError, Field, ConfigDict
from functools import cached_property

class Question(BaseModel):
  model_config = ConfigDict(frozen=True)
  id: UUID = Field(default_factory=uuid4)
  question: str
  inverted: bool = Field(default=False, description="Whether the question is inverted (i.e. higher score means worse outcome)")

class Choice(BaseModel):
  model_config = ConfigDict(frozen=True)
  choice: str  
  score: float
  inverted_score: float = Field(description="The score to use if the question is inverted.")

class Questionnaire(BaseModel):
  model_config = ConfigDict(frozen=True)
  preamble: str
  default_choices: List[Choice]
  questions: List[Question]

  @cached_property
  def inverted_map(self) -> dict[UUID, bool]:
    """Returns a mapping of question IDs to their inverted boolean values for O(1) lookup."""
    return {q.id: q.inverted for q in self.questions}

  @cached_property
  def choices_list(self) -> List[str]:
    """Returns a list of choice strings for O(1) access."""
    return [c.choice for c in self.default_choices]

  @cached_property
  def choice_score_map(self) -> dict[str, float]:
    """Returns a mapping of choice strings to their corresponding scores for O(1) lookup."""
    return {c.choice: c.score for c in self.default_choices}

  @cached_property
  def inverted_choice_score_map(self) -> dict[str, float]:
    """Returns a mapping of choice strings to their corresponding inverted scores for O(1) lookup."""
    return {c.choice: c.inverted_score for c in self.default_choices}

  @classmethod
  def from_yml(cls, location: str):
    path = Path(location)
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