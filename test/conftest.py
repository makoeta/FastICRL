import pytest
from unittest.mock import MagicMock

from fasticrl.learner.models.learn_output import LearnerOutput
from fasticrl.models.attempt import Attempt
from fasticrl.reward.models.reward_output import RewardOutput
from fasticrl.strategist.models.strategy_output import StrategyOutput


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def sample_attempt():
    return Attempt(task="solve 1+1", output="2", reward=9)


@pytest.fixture
def sample_attempts():
    return [
        Attempt(task="solve 1+1", output="2", reward=9),
        Attempt(task="solve 2+2", output="4", reward=8),
    ]


@pytest.fixture
def learner_output():
    return LearnerOutput(
        decision="EXPLOIT",
        reflection="Rewards are stable.",
        learning_output="The answer is 42.",
    )


@pytest.fixture
def reward_output():
    return RewardOutput(reward=8)


@pytest.fixture
def strategy_output():
    return StrategyOutput(strategy="Always be concise and accurate.")
