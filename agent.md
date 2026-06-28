# Agent Reference

Technical reference for the three agents inside `ICRLLearner` and their interaction protocol.

---

## Overview

Every `ICRLLearner` owns exactly three agents. They are instantiated once in `__init__` and reused across all episodes.

```
LearnerAgent   ← generates task outputs
RewardAgent    ← scores each output (0–10)
StrategistAgent ← refines the strategy from the buffer
```

All three inherit from `agno.agent.Agent` and return structured Pydantic objects via `output_schema`.

---

## LearnerAgent

**File:** `src/fasticrl/learner/core.py`  
**Prompt:** `src/fasticrl/learner/prompts.py`  
**Output schema:** `src/fasticrl/learner/models/learn_output.py`

### Role

Generates the actual task response for each episode. Decides whether to explore (try something new) or exploit (refine what worked) based on the reward history in its context.

### Input

```xml
<attempts>
    <attempt>
        Task: ...
        Output: ...
        Reward: ...
    </attempt>
    ...
</attempts>

<task>
    {task}
</task>

<taskdescription>
    {task_description}
</taskdescription>

<strategy>
    {strategy}
</strategy>
```

`attempts` is the full buffer serialized by `ICRLLearner.__attempts_as_xml()`. Empty string on the first episode.

### Output schema

```python
class LearnerOutput(BaseModel):
    decision: Literal["EXPLOIT", "EXPLORE"]
    reflection: str        # brief justification for the decision
    learning_output: str   # the actual answer — no meta-commentary
```

`decision` and `reflection` are chain-of-thought fields that guide structured reasoning; only `learning_output` is used downstream.

### Explore / Exploit heuristic

The system prompt instructs the model to use this table:

| Situation | Decision |
|---|---|
| Recent rewards high and stable | EXPLOIT |
| Recent rewards declining or stagnant | EXPLORE |
| Fewer than 3 attempts exist | EXPLORE |
| High reward variance | EXPLORE |
| Clear winning strategy identified | EXPLOIT |

---

## RewardAgent

**File:** `src/fasticrl/reward/core.py`  
**Prompt:** `src/fasticrl/reward/prompts.py`  
**Output schema:** `src/fasticrl/reward/models/reward_output.py`

### Role

Acts as the reward function. Given a task and the learner's output, scores quality on a 1–10 integer scale. Designed to be replaced with a deterministic function for tasks that have an objective ground truth.

### Input

```xml
<task>
    {task}
</task>

<output>
    {output}
</output>
```

### Output schema

```python
class RewardOutput(BaseModel):
    reward: int  # validated: ge=1, le=10
```

### Scoring rubric (from system prompt)

| Score | Meaning |
|---|---|
| 0–2 | Completely wrong or irrelevant |
| 3–4 | Partial; major omissions or errors |
| 5–6 | Acceptable; meets basics but mediocre |
| 7–8 | Good; accurate and well-crafted |
| 9–10 | Exceptional; flawless |

### Replacing with a deterministic reward

For tasks with a ground truth (e.g. Game of 24, coding challenges), you can bypass the RewardAgent entirely by subclassing `ICRLLearner` and overriding `generate_reward`:

```python
class MyLearner(ICRLLearner):
    def generate_reward(self, task: str, action: LearnerOutput) -> RewardOutput:
        score = my_eval_fn(task, action.learning_output)
        return RewardOutput(reward=score)
```

---

## StrategistAgent

**File:** `src/fasticrl/strategist/core.py`  
**Prompt:** `src/fasticrl/strategist/prompts.py`  
**Output schema:** `src/fasticrl/strategist/models/strategy_output.py`

### Role

Meta-cognitive analyst. Reads the full attempt buffer and the current strategy, identifies reward-driving patterns, and writes an updated strategy for the learner. Never solves tasks directly.

### Input

```xml
<task_description>
    {task_description}
</task_description>

<existing_strategy>
    {strategy}
</existing_strategy>

<attempts>
    {attempts}
</attempts>
```

### Output schema

```python
class StrategyOutput(BaseModel):
    strategy: str   # updated strategy text; replaces ICRLLearner.strategy
```

### When it runs

Controlled by `strategy_update_interval` in `auto_learn`. With `strategy_update_interval=5`, the strategist runs after episodes 5, 10, 15, … The strategy starts as an empty string and is updated in place on `ICRLLearner.strategy`.

```python
learner.auto_learn(
    episodes=20,
    strategy_update_interval=5,   # update strategy every 5 episodes
)
```

---

## Data flow per episode

```
auto_learn (episode i)
  │
  ├─► _generate_attempt_for_task(task)
  │     ├─► generate_action(task)
  │     │     └─► LearnerAgent.generate_learning_output(
  │     │               task, task_description, strategy, buffer_as_xml)
  │     │           → LearnerOutput
  │     │
  │     └─► generate_reward(task, learner_output)
  │           └─► RewardAgent.generate_reward_output(task, output)
  │               → RewardOutput
  │
  ├─► Attempt(task, output, reward) appended to buffer
  │
  └─► [every N episodes] update_strategy()
        └─► StrategistAgent.generate_new_strategy(
                task_description, existing_strategy, buffer_as_xml)
            → StrategyOutput → ICRLLearner.strategy updated
```

---

## Shared data structures

### `Attempt`

```python
class Attempt(BaseModel):
    task: str
    output: str
    reward: int
```

All three agents read the buffer as serialized `<attempt>` XML blocks. The buffer is never trimmed automatically — it grows across all episodes.

### `AgentSaveState`

Persisted by `to_yaml` / loaded by `from_yaml`:

```python
class AgentSaveState(BaseModel):
    task_description: str
    strategy: str
    buffer: list[Attempt]
```

> **Note:** `tasks` is not part of `AgentSaveState`. After `from_yaml`, you must re-supply the task list before calling `auto_learn`:
>
> ```python
> learner = ICRLLearner.from_yaml("state.yaml", ...)
> learner.tasks = ["task A", "task B", ...]
> learner.auto_learn(episodes=10)
> ```

---

## Using different models per agent

Each agent takes any `agno.models.base.Model`. Mix and match freely:

```python
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama

learner = ICRLLearner(
    learner_model=OpenAIChat(id="gpt-4o"),          # best quality for generation
    reward_model=OpenAIChat(id="gpt-4o-mini"),       # cheap for scoring
    strategy_model=OpenAIChat(id="gpt-4o"),          # strong reasoning for meta-analysis
    task_description="...",
    tasks=[...],
)
```

---

## Extending the agents

All three agents expose their prompts as module-level string constants. Override them before instantiation to change agent behavior without subclassing:

```python
from fasticrl.learner import prompts as learner_prompts

learner_prompts.ICRL_LEARNING_SYSTEM_PROMPT = "You are a haiku poet..."
```

Or subclass any agent and pass it directly:

```python
from fasticrl.learner.core import LearnerAgent

class HaikuLearner(LearnerAgent):
    def __init__(self, model):
        super().__init__(model=model)
        self.system_message = "You are a haiku poet..."

# Pass the class, not the instance — ICRLLearner constructs agents internally
# Instead, patch after construction:
learner = ICRLLearner(...)
learner.learner_agent = HaikuLearner(model=my_model)
```
