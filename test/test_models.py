import pytest
from pydantic import ValidationError

from fasticrl.learner.models.learn_output import LearnerOutput
from fasticrl.models.agent_save_state import AgentSaveState
from fasticrl.models.attempt import Attempt
from fasticrl.reward.models.reward_output import RewardOutput
from fasticrl.strategist.models.strategy_output import StrategyOutput


class TestAttempt:
    def test_valid_creation(self):
        attempt = Attempt(task="task", output="output", reward=5)
        assert attempt.task == "task"
        assert attempt.output == "output"
        assert attempt.reward == 5

    def test_serialization(self):
        attempt = Attempt(task="t", output="o", reward=3)
        assert attempt.model_dump() == {"task": "t", "output": "o", "reward": 3}

    def test_roundtrip(self):
        original = Attempt(task="t", output="o", reward=7)
        restored = Attempt.model_validate(original.model_dump())
        assert restored == original

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            Attempt(task="t", output="o")


class TestRewardOutput:
    def test_valid_mid_range(self):
        out = RewardOutput(reward=5)
        assert out.reward == 5

    def test_boundary_min(self):
        out = RewardOutput(reward=1)
        assert out.reward == 1

    def test_boundary_max(self):
        out = RewardOutput(reward=10)
        assert out.reward == 10

    def test_below_min_raises(self):
        with pytest.raises(ValidationError):
            RewardOutput(reward=0)

    def test_above_max_raises(self):
        with pytest.raises(ValidationError):
            RewardOutput(reward=11)

    def test_negative_raises(self):
        with pytest.raises(ValidationError):
            RewardOutput(reward=-1)


class TestLearnerOutput:
    def _make(self, **kwargs):
        defaults = dict(decision="EXPLOIT", reflection="stable rewards", learning_output="answer")
        return LearnerOutput(**{**defaults, **kwargs})

    def test_valid_creation(self):
        out = self._make(learning_output="some response")
        assert out.learning_output == "some response"

    def test_exploit_decision(self):
        out = self._make(decision="EXPLOIT")
        assert out.decision == "EXPLOIT"

    def test_explore_decision(self):
        out = self._make(decision="EXPLORE")
        assert out.decision == "EXPLORE"

    def test_invalid_decision_raises(self):
        with pytest.raises(ValidationError):
            self._make(decision="INVALID")

    def test_reflection_stored(self):
        out = self._make(reflection="unique reflection text")
        assert out.reflection == "unique reflection text"

    def test_empty_learning_output(self):
        out = self._make(learning_output="")
        assert out.learning_output == ""

    def test_multiline_output(self):
        text = "line one\nline two\nline three"
        out = self._make(learning_output=text)
        assert out.learning_output == text


class TestStrategyOutput:
    def test_valid_creation(self):
        out = StrategyOutput(strategy="be concise")
        assert out.strategy == "be concise"

    def test_empty_strategy(self):
        out = StrategyOutput(strategy="")
        assert out.strategy == ""


class TestAgentSaveState:
    def test_defaults(self):
        state = AgentSaveState(task_description="desc")
        assert state.strategy == ""
        assert state.buffer == []

    def test_with_strategy(self):
        state = AgentSaveState(task_description="desc", strategy="do well")
        assert state.strategy == "do well"

    def test_with_buffer(self, sample_attempts):
        state = AgentSaveState(task_description="desc", buffer=sample_attempts)
        assert len(state.buffer) == 2
        assert all(isinstance(a, Attempt) for a in state.buffer)

    def test_coerces_dicts_in_buffer(self):
        state = AgentSaveState.model_validate(
            {
                "task_description": "desc",
                "buffer": [{"task": "t", "output": "o", "reward": 5}],
            }
        )
        assert isinstance(state.buffer[0], Attempt)
        assert state.buffer[0].task == "t"

    def test_serialization_roundtrip(self, sample_attempts):
        original = AgentSaveState(
            task_description="arithmetic",
            strategy="be brief",
            buffer=sample_attempts,
        )
        data = original.model_dump(mode="json")
        restored = AgentSaveState.model_validate(data)
        assert restored.task_description == original.task_description
        assert restored.strategy == original.strategy
        assert len(restored.buffer) == len(original.buffer)
