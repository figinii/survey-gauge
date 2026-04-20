# survey-gauge

`survey-gauge` is a lightweight package to assess and evaluate an LLM over a questionnaire with a simple async interface.

The package focuses on three things:

- loading questionnaires and scenarios from YAML
- executing question prompts asynchronously against your model client
- mapping model choices to numeric scores (with support for inverted questions)

## What Problem It Solves

When comparing LLM behavior, you often need a repeatable way to score model outputs over many scenario prompts. `survey-gauge` gives you:

- a strict output schema for choices via Pydantic
- async batched evaluation with optional delay control
- consistent score computation from your questionnaire definition

## Install

```bash
pip install survey-gauge
```

For local development from this repository:

```bash
pip install -e .
```

## Core API

The public API is:

- `Questionnaire`
- `Scenario`
- `Eval`

Typical workflow:

1. Load questionnaire YAML with `Questionnaire.from_yml(...)`
2. Load scenario YAML with `Scenario.from_yml(...)`
3. Build an async client callback
4. Pass callback + questionnaire to `Eval`
5. Run `await evaluator.evaluate_scenario(...)`

## Required Client Callback Contract

`Eval` expects a callback with this signature:

```python
async def prompt_execution(prompt: str, output_model: type[pydantic.BaseModel]) -> pydantic.BaseModel:
	...
```

Important requirements:

- it must return an instance of the provided `output_model`
- the client call should be wrapped with `instructor`
- the underlying LLM SDK client should be async (for example `AsyncOpenAI`)

In practice, the recommended shape is:

```python
import instructor
from openai import AsyncOpenAI

client = instructor.from_openai(
	AsyncOpenAI(api_key="..."),
	mode=instructor.Mode.JSON,
)

async def prompt_execution(prompt, output_model):
	return await client.chat.completions.create(
		model="your-model-name",
		messages=[{"role": "user", "content": prompt}],
		response_model=output_model,
		temperature=0.0,
	)
```

## Minimal Example

```python
import logging
import instructor
from openai import AsyncOpenAI

from survey_gauge import Questionnaire, Scenario, Eval

questionnaire = Questionnaire.from_yml("examples/classification.yml")
scenarios = Scenario.from_yml("examples/intersubjective_scenarios.yml")

client = instructor.from_openai(
	AsyncOpenAI(api_key="YOUR_OPENAI_API_KEY"),
	mode=instructor.Mode.JSON,
)

MODEL = "gpt-4o-mini"

async def prompt_execution(prompt, output_model):
	return await client.chat.completions.create(
		model=MODEL,
		messages=[{"role": "user", "content": prompt}],
		response_model=output_model,
		temperature=0.0,
		top_p=0.01,
		seed=42,
	)

logger = logging.getLogger("Evaluation")
evaluator = Eval(prompt_execution, questionnaire=questionnaire, logger=logger)

scores = await evaluator.evaluate_scenario(scenarios[0], delay=0)
print(scores)
print(sum(s for s in scores if isinstance(s, (int, float))))
```

## Example Notebooks

Colab-friendly examples are available in:

- `examples/example_external_api.ipynb`: OpenAI and Hugging Face Router usage
- `examples/example_vllm.ipynb`: local vLLM server usage through OpenAI-compatible endpoint

## Input Files

The repository already includes examples of YAML files:

- `examples/capitals_scenarios.yml`
- `examples/capitals_questionnaire.yml`
