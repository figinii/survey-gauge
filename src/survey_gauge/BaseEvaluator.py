from abc import ABC, abstractmethod
from typing import List, Optional, Any
from logging import Logger

from .Data import Questionnaire, Scenario

class BaseEvaluator(ABC):
  def __init__(self, questionnaire: Questionnaire, logger: Logger):
    self.questionnaire = questionnaire
    self.logger = logger

  def _prepare_prompts(self, scenario: Scenario, question_indexes: Optional[List[int]]) -> tuple[List[Any], List[str]]:
    """Shared logic to filter questions and build prompt strings."""
    if question_indexes is None:
      question_indexes = list(range(len(self.questionnaire.questions)))
    
    selected_questions = [self.questionnaire.questions[i] for i in question_indexes]
    prompts = [
      f"{self.questionnaire.preamble}\n\n{scenario.description}\n\n{q.question}" 
      for q in selected_questions
    ]
    return selected_questions, prompts

  def _add_answers(self, prompts: List[str]) -> List[str]:
    """Append the questionnaire preamble and choices to each prompt."""
    choices_block = "\n".join(f"- {choice}" for choice in self.questionnaire.choices_list)
    suffix = f"\nChoices:\n{choices_block}"
    return [f"{prompt}\n\n{suffix}" for prompt in prompts]

  def _map_results_to_scores(self, questions: List[Any], results: List[str]) -> List[float | str]:
    """Shared logic to handle score inversion and mapping."""
    scores: List[float | str] = []
    for q, output in zip(questions, results):
      mapping = (
        self.questionnaire.inverted_choice_score_map 
        if self.questionnaire.inverted_map[q.id] 
        else self.questionnaire.choice_score_map
      )
      scores.append(mapping.get(output, self.questionnaire.failure_indicator))
      
    return scores
  
  @abstractmethod
  async def evaluate_scenario(self, scenario: Scenario, **kwargs) -> List[float | str]:
    """Subclasses must implement the specific execution logic."""
    pass