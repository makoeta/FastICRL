import pytest
from unittest.mock import MagicMock, patch

from fasticrl.strategist.core import StrategistAgent
from fasticrl.strategist.models.strategy_output import StrategyOutput
from fasticrl.strategist.prompts import ICRL_STRATEGY_SYSTEM_PROMPT


@pytest.fixture
def strategist_agent(mock_model):
    with patch("agno.agent.Agent.__init__", return_value=None):
        return StrategistAgent(model=mock_model)


class TestStrategistAgentInit:
    def test_sets_model(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = StrategistAgent(model=mock_model)
        assert agent.model is mock_model

    def test_sets_output_schema(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = StrategistAgent(model=mock_model)
        assert agent.output_schema is StrategyOutput

    def test_sets_system_message(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = StrategistAgent(model=mock_model)
        assert agent.system_message == ICRL_STRATEGY_SYSTEM_PROMPT


class TestGenerateNewStrategy:
    def _mock_run(self, content):
        return MagicMock(content=content)

    def test_returns_strategy_output_content(self, strategist_agent):
        expected = StrategyOutput(strategy="focus on precision")
        with patch("agno.agent.Agent.run", return_value=self._mock_run(expected)):
            result = strategist_agent.generate_new_strategy(
                task_description="arithmetic",
                existing_strategy="",
                buffer_as_xml="<attempts/>",
            )
        assert result is expected

    def test_prompt_contains_task_description(self, strategist_agent):
        with patch("agno.agent.Agent.run", return_value=self._mock_run(StrategyOutput(strategy="s"))) as mock_run:
            strategist_agent.generate_new_strategy(
                task_description="unique_description_lmn",
                existing_strategy="",
                buffer_as_xml="",
            )
        assert "unique_description_lmn" in mock_run.call_args[0][0]

    def test_prompt_contains_existing_strategy(self, strategist_agent):
        with patch("agno.agent.Agent.run", return_value=self._mock_run(StrategyOutput(strategy="s"))) as mock_run:
            strategist_agent.generate_new_strategy(
                task_description="desc",
                existing_strategy="unique_strategy_pqr",
                buffer_as_xml="",
            )
        assert "unique_strategy_pqr" in mock_run.call_args[0][0]

    def test_prompt_contains_buffer(self, strategist_agent):
        with patch("agno.agent.Agent.run", return_value=self._mock_run(StrategyOutput(strategy="s"))) as mock_run:
            strategist_agent.generate_new_strategy(
                task_description="desc",
                existing_strategy="",
                buffer_as_xml="<unique_buffer_stu/>",
            )
        assert "<unique_buffer_stu/>" in mock_run.call_args[0][0]

    def test_calls_super_run_once(self, strategist_agent):
        with patch("agno.agent.Agent.run", return_value=self._mock_run(StrategyOutput(strategy="s"))) as mock_run:
            strategist_agent.generate_new_strategy(
                task_description="desc",
                existing_strategy="",
                buffer_as_xml="",
            )
        mock_run.assert_called_once()

    def test_empty_strategy_passthrough(self, strategist_agent):
        expected = StrategyOutput(strategy="")
        with patch("agno.agent.Agent.run", return_value=self._mock_run(expected)):
            result = strategist_agent.generate_new_strategy(
                task_description="desc",
                existing_strategy="",
                buffer_as_xml="",
            )
        assert result.strategy == ""
