# FastICRL

<p align="center">
  <a href="https://pypi.org/project/fasticrl/"><img src="https://img.shields.io/pypi/v/fasticrl" alt="PyPI version"></a>
  <a href="https://pypi.org/project/fasticrl/"><img src="https://img.shields.io/pypi/pyversions/fasticrl" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/no%20fine--tuning-required-brightgreen" alt="No fine-tuning required">
  <img src="https://img.shields.io/badge/no%20GPU-required-blue" alt="No GPU required">
  <a href="https://github.com/agno-agi/agno"><img src="https://img.shields.io/badge/powered%20by-agno-8A2BE2" alt="Powered by agno"></a>
</p>

**In-Context Reinforcement Learning for LLMs — no fine-tuning, no gradient updates, no GPU.**

FastICRL implements the ICRL paradigm from [_Reward Is Enough: LLMs Are In-Context Reinforcement Learners_](https://arxiv.org/abs/2506.06303) (Song et al., 2025). A learner LLM improves its outputs purely by reading its own history of attempts and rewards inside the context window — guided by a meta-cognitive strategist. No training, no infrastructure, just inference.

---

## How it works

Three LLM agents collaborate in a feedback loop:

```text
┌──────────────────────────────────────────────────┐
│                   ICRLLearner                    │
│                                                  │
│  Task ──► Learner ──► Output ──► Reward Agent    │
│             ▲                         │          │
│             │        Attempt          │          │
│             │  (task, output, score)  │          │
│             └─────────────────────────┘          │
│                          │                       │
│                  (every N episodes)              │
│                          ▼                       │
│                      Strategist                  │
│                (refines the strategy)            │
└──────────────────────────────────────────────────┘
```

| Agent          | Role                                                                                  |
| -------------- | ------------------------------------------------------------------------------------- |
| **Learner**    | Generates task outputs; balances exploration vs. exploitation based on reward history |
| **Reward**     | Scores each output on a 0–10 scale (acts as the reward function)                      |
| **Strategist** | Analyzes past attempts to synthesize actionable strategies for future episodes        |

Each agent can be backed by a different model — e.g. a cheap model for reward, a powerful one for the learner.

---

## Installation

```bash
pip install fasticrl
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add fasticrl
```

Model provider extras (install whichever you use):

```bash
pip install "fasticrl[openai]"   # OpenAI
pip install "fasticrl[ollama]"   # Ollama (local models)
```

Requires Python ≥ 3.13.

---

## Quick start

```python
from fasticrl import ICRLLearner
from agno.models.openai import OpenAIChat

model = OpenAIChat(id="gpt-4o-mini")

learner = ICRLLearner(
    learner_model=model,
    reward_model=model,
    strategy_model=model,
    task_description="Write a concise, compelling product description for an e-commerce listing.",
    tasks=[
        "Wireless noise-cancelling headphones",
        "Ergonomic standing desk",
        "Portable espresso maker",
    ],
)

# Run 3 episodes, update strategy every 2 steps, show progress bar
learner.auto_learn(episodes=3, batch_size=2, cli_mode=True, strategy_update_interval=2)

# Inspect what the agent learned
print(learner.strategy)
```

---

## API

### `ICRLLearner`

```python
ICRLLearner(
    learner_model,        # agno Model for the learner agent
    reward_model,         # agno Model for the reward agent
    strategy_model,       # agno Model for the strategist agent
    task_description,     # describes the overall task domain (required)
    tasks,                # list of concrete task instances to cycle through
    buffer,               # optional: pre-loaded list of Attempt objects
    strategy,             # optional: pre-loaded strategy string
)
```

#### Key methods

| Method                                                                 | Description                                                                                                                                                                         |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto_learn(episodes, batch_size, cli_mode, strategy_update_interval)` | Run N episodes. `batch_size > 1` parallelizes tasks with a thread pool. `cli_mode=True` shows a progress bar. `strategy_update_interval=K` refreshes the strategy every K episodes. |
| `generate_action(task)`                                                | Run the learner on a single task and return its output                                                                                                                              |
| `generate_reward(task, action)`                                        | Score a learner output with the reward agent                                                                                                                                        |
| `generate_attempt_by_present_task()`                                   | Single step: generate + score the current task                                                                                                                                      |
| `update_strategy()`                                                    | Ask the strategist to refine the strategy from the current buffer                                                                                                                   |
| `to_yaml(path)`                                                        | Persist the full agent state (buffer + strategy) to a YAML file                                                                                                                     |
| `ICRLLearner.from_yaml(path, ...)`                                     | Resume from a saved state                                                                                                                                                           |

---

## Saving and resuming

```python
# Save
learner.to_yaml("my_agent.yaml")

# Resume later
learner = ICRLLearner.from_yaml(
    "my_agent.yaml",
    learner_model=model,
    reward_model=model,
    strategy_model=model,
)
learner.auto_learn(episodes=5)
```

---

## Using Ollama (local models)

```python
from agno.models.ollama import Ollama

learner = ICRLLearner(
    learner_model=Ollama(id="llama3.2"),
    reward_model=Ollama(id="llama3.2"),
    strategy_model=Ollama(id="llama3.2"),
    task_description="...",
    tasks=[...],
)
```

Any [agno](https://github.com/agno-agi/agno)-compatible model works.

---

## Citation

This project is based on and inspired by the following papers:

_Reward Is Enough: LLMs Are In-Context Reinforcement Learners_  
Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, Yanjun Qi  
arXiv:2506.06303 — [https://arxiv.org/abs/2506.06303](https://arxiv.org/abs/2506.06303)

_Large Language Models as Optimizers_  
Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen  
arXiv:2309.03409 — [https://arxiv.org/abs/2309.03409](https://arxiv.org/abs/2309.03409)

_Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs_  
Yifan Zhou, Sachin Grover, Mohamed El Mistiri, Kamalesh Kalirathinam, Pratyush Kerhalkar, Swaroop Mishra, Neelesh Kumar, Sanket Gaurav, Oya Aran, Heni Ben Amor  
NeurIPS 2025 — [https://openreview.net/forum?id=95plu1Mo20](https://openreview.net/forum?id=95plu1Mo20)

---

## License

MIT
