from typing import Tuple, List, Callable
from logging import Logger, log
from uuid import UUID

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import asyncio

from .. import Questionnaire
from .. import Scenario

class Server():
  def __init__(self, configs: AsyncEngineArgs, questionnaire: Questionnaire, logger: Logger):
    """Initialize the server with the given configurations, questionnaire, and logger."""
    self.engine = AsyncLLMEngine.from_engine_args(configs)
    self.questionnaire = questionnaire
    self.logger = logger

  async def evaluate_scenario(self, scenario: Scenario, temperature:float=0) -> List[Tuple[str, str]]:
    """Subscribe a prompt (scenario + preamble + questoin) 
      with constraints on the answer based on the questionnaire.
    """
    prompts = [
        (q.id, f"{self.questionnaire.preamble}\n\n{scenario.desciription}\n\n{q.question}")
        for q in self.questionnaire.questions
    ]

    sampling_params = SamplingParams(
        temperature=0.0, 
        guided_decoding=GuidedDecodingParams(guided_choice=self.questionnaire.default_choices)
    )

    tasks = [
        self.engine.generate(p[1], sampling_params, request_id=f"{scenario.id}=$={p[0]}") 
        for p in (prompts)
    ]
    
    self.logger.info(f"Subscribed {len(tasks)} prompts for scenario {scenario.id}")
    
    results = await asyncio.gather(*tasks)

    return [(r.request_id.split('=$=')[1], r.outputs[0].text) for r in results]

  async def evaluate_scenarios(self, scenario1: Scenario, scenario2: Scenario, 
    greater_than: Callable[[List[Tuple[str, str]], List[Tuple[str, str]]], bool]) -> UUID:
    
    awaitable1 = self.evaluate_scenario(scenario1)
    awaitable2 = self.evaluate_scenario(scenario2)

    results1, results2 = await asyncio.gather(awaitable1, awaitable2)
    self.logger.info(f"Evaluated scenarios {scenario1.id} and {scenario2.id}")
    if greater_than(results1, results2):
      return scenario1.id
    else:
      return scenario2.id