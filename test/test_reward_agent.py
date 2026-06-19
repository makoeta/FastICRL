import pytest
from unittest.mock import MagicMock, patch

from fasticrl.reward.core import RewardAgent
from fasticrl.reward.models.reward_output import RewardOutput
from fasticrl.reward.prompts import ICRL_REWARD_SYSTEM_PROMPT


@pytest.fixture
def reward_agent(mock_model):
    with patch("agno.agent.Agent.__init__", return_value=None):
        return RewardAgent(model=mock_model)


class TestRewardAgentInit:
    def test_sets_model(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = RewardAgent(model=mock_model)
        assert agent.model is mock_model

    def test_sets_output_schema(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = RewardAgent(model=mock_model)
        assert agent.output_schema is RewardOutput

    def test_sets_system_message(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = RewardAgent(model=mock_model)
        assert agent.system_message == ICRL_REWARD_SYSTEM_PROMPT


class TestGenerateRewardOutput:
    def _mock_run(self, content):
        return MagicMock(content=content)

    def test_returns_reward_output_content(self, reward_agent):
        expected = RewardOutput(reward=7)
        with patch("agno.agent.Agent.run", return_value=self._mock_run(expected)):
            result = reward_agent.generate_reward_output(
                task="solve 1+1", output="2"
            )
        assert result is expected

    def test_prompt_contains_task(self, reward_agent):
        with patch("agno.agent.Agent.run", return_value=self._mock_run(RewardOutput(reward=5))) as mock_run:
            reward_agent.generate_reward_output(task="unique_task_abc", output="some output")
        assert "unique_task_abc" in mock_run.call_args[0][0]

    def test_prompt_contains_output(self, reward_agent):
        with patch("agno.agent.Agent.run", return_value=self._mock_run(RewardOutput(reward=5))) as mock_run:
            reward_agent.generate_reward_output(task="task", output="unique_output_xyz")
        assert "unique_output_xyz" in mock_run.call_args[0][0]

    def test_calls_super_run_once(self, reward_agent):
        with patch("agno.agent.Agent.run", return_value=self._mock_run(RewardOutput(reward=5))) as mock_run:
            reward_agent.generate_reward_output(task="task", output="output")
        mock_run.assert_called_once()

    @pytest.mark.parametrize("reward_value", [1, 5, 10])
    def test_passes_through_reward_value(self, reward_agent, reward_value):
        expected = RewardOutput(reward=reward_value)
        with patch("agno.agent.Agent.run", return_value=self._mock_run(expected)):
            result = reward_agent.generate_reward_output(task="task", output="out")
        assert result.reward == reward_value
