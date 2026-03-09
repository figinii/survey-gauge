from typing import List
from pathlib import Path
import yaml
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

class Scenario(BaseModel):
  id: UUID = Field(default_factory=uuid4)
  description: str

  @classmethod
  def from_yml(cls, location: str):
    path = Path(location)
    if not path.exists() or not path.is_file():
      raise FileNotFoundError(f"Scenario file not found at {path}")
    elif path.suffix not in ['.yml', '.yaml']:
      raise ValueError(f"Scenario file must be a .yml or .yaml file, got {path.suffix}")
    
    with open(path, 'r') as f:
      try:
        data = yaml.safe_load(f)
        scenarios_data = data if isinstance(data, List) else [data]
        return [cls(**scenario) for scenario in scenarios_data]
      except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
      except ValidationError as e:
        raise ValueError(f"Error validating scenario data: {e}")