from typing import List, Any, Literal, Callable, Coroutine, Optional
from logging import Logger
from pydantic import create_model, BaseModel

from asyncio import TaskGroup, sleep
from tqdm import tqdm

from .BaseEvaluator import BaseEvaluator
from .Data import Questionnaire
from .Data import Scenario

class Eval(BaseEvaluator):
  def __init__(self, prompt_execution: Callable[[str, type[BaseModel]], Coroutine[Any, None, BaseModel]], questionnaire: Questionnaire, logger: Logger):
    """Initialize the server with the given client, questionnaire, and logger.
       Args:
          p_execution: A function that takes a prompt and an output model, and returns a coroutine 
                        that will execute the prompt and return the output model.
          questionnaire: The questionnaire to use for evaluating scenarios.
          logger: The logger to use for logging information.
    """
    self.prompt_execution = prompt_execution
    self.questionnaire = questionnaire
    self.logger = logger
    self.QC_output_model_class = "Questionnaire_Choices"

  async def evaluate_scenario(self, scenario: Scenario, delay:int=0, question_indexes: Optional[List[int]] = None, **kwargs) -> List[float | str]:
    """Given a scenario and a questionnaire, evaluate the scenario by generating prompts
      for each question and aggregating the scores based on the answers.
      
      Args:
        scenario: The scenario to evaluate
        delay: Delay between prompts in seconds
        question_indexes: List of question indexes to evaluate. If None, evaluates all questions.
    """
    # Default to all questions if not specified
    if question_indexes is None:
      question_indexes = list(range(len(self.questionnaire.questions)))
    
    # Filter questions and create prompts for selected indexes
    selected_questions, prompts = self._prepare_prompts(scenario, question_indexes)

    choice_model = create_model(self.QC_output_model_class, choices=(Literal[tuple(self.questionnaire.choices_list)], ...))

    self.logger.info(f"Subscribing {len(prompts)} prompts for scenario {scenario.id}")
    tasks: List[Any] = []
    async with TaskGroup() as tg:
      for i, prompt in tqdm(enumerate(prompts)):
        async def wrapped_call(prompt_to_call:str=prompt, id:int=i) -> BaseModel:
          await sleep(delay * id)
          return await self.prompt_execution(prompt_to_call, choice_model)
    
        tasks.append(tg.create_task(wrapped_call()))

    results = [tasks[i].result().choices for i in range(len(tasks))]
    self.logger.info(f"Received results for scenario {scenario.id}")
  
    scores = self._map_results_to_scores(selected_questions, results)

    return scores