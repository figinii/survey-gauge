from typing import List, Any, Literal, Coroutine
from logging import Logger
from pydantic import create_model, BaseModel

from instructor import Instructor
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
from vllm.sampling_params import StructuredOutputsParams
from asyncio import TaskGroup, sleep

from .. import Questionnaire
from .LLM_Interactors import LlmInteract
from .. import Scenario

class Eval():
  def __init__(self, client: LlmInteract, questionnaire: Questionnaire, logger: Logger):
    """Initialize the server with the given client, questionnaire, and logger."""
    self.client = client
    self.questionnaire = questionnaire
    self.logger = logger
    self.QC_output_model_class = "Questionnaire_Choices"


    
  # def __init__(self, config: AsyncEngineArgs, questionnaire: Questionnaire, logger: Logger):
  #   """Initialize the server with the given configurations, questionnaire, and logger."""
  #   self.engine = AsyncLLMEngine.from_engine_args(config)
  #   self.questionnaire = questionnaire
  #   self.logger = logger

  # async def subscribe_prompt(self, prompt: str, sampling_params: SamplingParams, request_id: str) -> Tuple[str, str]:
  #   """Subscribe a prompt to the engine and return the result along with the request_id."""
    
  #   generator = self.engine.generate(prompt, sampling_params, request_id)
  #   result: None | RequestOutput = None
  #   async for out in generator: 
  #     result = out

  #   if result is None:
  #     self.logger.error(f"No result received for prompt with request_id {request_id}")
  #     return (request_id, "")
  #   else:
  #     return (request_id, result.outputs[0].text)

  async def evaluate_scenario(self, scenario: Scenario, temperature:float=0, delay:int=0) -> float:
    """Given a scenario and a questionnaire, evaluate the scenario by generating prompts
      for each question and aggregating the scores based on the answers.
    """
    prompts = [f"{self.questionnaire.preamble}\n\n{scenario.description}\n\n{q.question}" 
              for q in self.questionnaire.questions]

    choice_model = create_model(self.QC_output_model_class, choices=(Literal[tuple(self.questionnaire.choices_list)], ...))


    self.logger.info(f"Subscribing {len(prompts)} prompts for scenario {scenario.id}")
    tasks: List[Any] = []
    async with TaskGroup() as tg:
      for i, prompt in enumerate(prompts):
        async def wrapped_call(prompt_to_call=prompt, id=i):
          await sleep(delay * id)
          return await self.client.subscribe_prompt(prompt_to_call, temperature, choice_model, 'user')
    
        tasks.append(tg.create_task(wrapped_call()))    

    results = [tasks[i].result().choices for i in range(len(tasks))]
    self.logger.info(f"Received results for scenario {scenario.id}")
  
    score = 0.0
    for q_id, output in zip([q.id for q in self.questionnaire.questions], results):
      if self.questionnaire.inverted_map[q_id]:
        score += self.questionnaire.inverted_choice_score_map.get(output, 0.0)
      else:
        score += self.questionnaire.choice_score_map.get(output, 0.0)

    return score