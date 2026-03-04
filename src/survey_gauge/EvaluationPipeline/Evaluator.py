from typing import Coroutine, Tuple, List, Any
from logging import Logger
from uuid import UUID

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, RequestOutput
from vllm.sampling_params import StructuredOutputsParams
import asyncio

from .. import Questionnaire
from .. import Scenario

class Eval():
  def __init__(self, config: AsyncEngineArgs, questionnaire: Questionnaire, logger: Logger):
    """Initialize the server with the given configurations, questionnaire, and logger."""
    self.engine = AsyncLLMEngine.from_engine_args(config)
    self.questionnaire = questionnaire
    self.logger = logger

  async def subscribe_prompt(self, prompt: str, sampling_params: SamplingParams, request_id: str) -> Tuple[str, str]:
    """Subscribe a prompt to the engine and return the result along with the request_id."""
    
    generator = self.engine.generate(prompt, sampling_params, request_id)
    result: None | RequestOutput = None
    async for out in generator: 
      result = out

    if result is None:
      self.logger.error(f"No result received for prompt with request_id {request_id}")
      return (request_id, "")
    else:
      return (request_id, result.outputs[0].text)


  async def evaluate_scenario(self, scenario: Scenario, temperature:float=0) -> float:
    """Given a scenario and a questionnaire, evaluate the scenario by generating prompts
      for each question and aggregating the scores based on the answers.
    """
    id_prompt_pairs = [
      (q.id, f"{self.questionnaire.preamble}\n\n{scenario.description}\n\n{q.question}")
      for q in self.questionnaire.questions
    ]

    sampling_params = SamplingParams(
        temperature=temperature, 
        structured_outputs=StructuredOutputsParams(choice=self.questionnaire.choices_list)
    )

    self.logger.info(f"Subscribing {len(id_prompt_pairs)} prompts for scenario {scenario.id}")
    tasks: List[Coroutine[Any, Any, Tuple[str, str]]] = []
    for q_id, prompt in id_prompt_pairs:
      tasks.append(self.subscribe_prompt(prompt, sampling_params, f"{q_id}#{scenario.id}"))
    
    results = await asyncio.gather(*tasks)
    self.logger.info(f"Received results for scenario {scenario.id}")

    question_answer_pair = [(UUID(request_id.split("#")[0]), output) for request_id, output in results]
    score = 0.0
    for q_id, output in question_answer_pair:
      if self.questionnaire.inverted_map[q_id]:
        score += self.questionnaire.inverted_choice_score_map.get(output, 0.0)
      else:
        score += self.questionnaire.choice_score_map.get(output, 0.0)

    return score