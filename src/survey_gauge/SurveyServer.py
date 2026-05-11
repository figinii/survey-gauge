from typing import List, Callable, Optional
from logging import Logger

from .BaseEvaluator import BaseEvaluator
from .Data import Questionnaire
from .Data import Scenario

class SurveyServer(BaseEvaluator):
  def __init__(self, prompt_client: Callable[[List[str]], List[str]], questionnaire: Questionnaire, logger: Logger):
    """Initialize the server with the given client function, questionnaire, and logger.
       Args:
          prompt_client: A function that takes a list of prompts, and returns a list of scores corresponding to those prompts. 
                         This function is responsible for executing the prompts and returning the scores. 
          questionnaire: The questionnaire to use for evaluating scenarios.
          logger: The logger to use for logging information.
    """
    super().__init__(questionnaire, logger)
    self.prompt_client = prompt_client


  async def evaluate_scenario(self, scenario: Scenario, question_indexes: Optional[List[int]] = None, **kwargs) -> List[float | str]:
    """Given a scenario and a questionnaire, evaluate the scenario by generating prompts
      for each question and aggregating the scores based on the answers.
      
      Args:
        scenario: The scenario to evaluate
        question_indexes: List of question indexes to evaluate. If None, evaluates all questions.
    """
    if question_indexes is None:
      question_indexes = list(range(len(self.questionnaire.questions)))
    
    selected_questions, prompts = self._prepare_prompts(scenario, question_indexes)
    prompts = self._add_answers(prompts)

    self.logger.info(f"Serving {len(prompts)} prompts for scenario {scenario.id}")

    results = self.prompt_client(prompts)
    self.logger.info(f"Received results for scenario {scenario.id}")
  
    scores = self._map_results_to_scores(selected_questions, results)

    return scores