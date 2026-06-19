import pytest
from unittest.mock import MagicMock, patch

from fasticrl.learner.core import LearnerAgent
from fasticrl.learner.models.learn_output import LearnerOutput
from fasticrl.learner.prompts import ICRL_LEARNING_SYSTEM_PROMPT


@pytest.fixture
def learner_agent(mock_model):
    with patch("agno.agent.Agent.__init__", return_value=None):
        return LearnerAgent(model=mock_model)


class TestLearnerAgentInit:
    def test_sets_model(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = LearnerAgent(model=mock_model)
        assert agent.model is mock_model

    def test_sets_output_schema(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = LearnerAgent(model=mock_model)
        assert agent.output_schema is LearnerOutput

    def test_sets_system_message(self, mock_model):
        with patch("agno.agent.Agent.__init__", return_value=None):
            agent = LearnerAgent(model=mock_model)
        assert agent.system_message == ICRL_LEARNING_SYSTEM_PROMPT


def _make_learner_output(**kwargs):
    defaults = dict(decision="EXPLOIT", reflection="stable", learning_output="x")
    return LearnerOutput(**{**defaults, **kwargs})


class TestGenerateLearningOutput:
    def _mock_run(self, content):
        return MagicMock(content=content)

    def test_returns_learner_output_content(self, learner_agent):
        expected = _make_learner_output(learning_output="42")
        with patch("agno.agent.Agent.run", return_value=self._mock_run(expected)):
            result = learner_agent.generate_learning_output(
                buffer_as_xml="<attempts/>",
                task="1+1",
                task_description="arithmetic",
                strategy="",
            )
        assert result is expected

    def test_prompt_contains_task(self, learner_agent):
        out = _make_learner_output()
        with patch("agno.agent.Agent.run", return_value=self._mock_run(out)) as mock_run:
            learner_agent.generate_learning_output(
                buffer_as_xml="<attempts/>",
                task="unique_task_xyz",
                task_description="desc",
                strategy="",
            )
        assert "unique_task_xyz" in mock_run.call_args[0][0]

    def test_prompt_contains_task_description(self, learner_agent):
        out = _make_learner_output()
        with patch("agno.agent.Agent.run", return_value=self._mock_run(out)) as mock_run:
            learner_agent.generate_learning_output(
                buffer_as_xml="<attempts/>",
                task="task",
                task_description="unique_description_abc",
                strategy="",
            )
        assert "unique_description_abc" in mock_run.call_args[0][0]

    def test_prompt_contains_strategy(self, learner_agent):
        out = _make_learner_output()
        with patch("agno.agent.Agent.run", return_value=self._mock_run(out)) as mock_run:
            learner_agent.generate_learning_output(
                buffer_as_xml="<attempts/>",
                task="task",
                task_description="desc",
                strategy="unique_strategy_qrs",
            )
        assert "unique_strategy_qrs" in mock_run.call_args[0][0]

    def test_prompt_contains_buffer(self, learner_agent):
        out = _make_learner_output()
        with patch("agno.agent.Agent.run", return_value=self._mock_run(out)) as mock_run:
            learner_agent.generate_learning_output(
                buffer_as_xml="<unique_buffer_tag/>",
                task="task",
                task_description="desc",
                strategy="",
            )
        assert "<unique_buffer_tag/>" in mock_run.call_args[0][0]

    def test_calls_super_run_once(self, learner_agent):
        out = _make_learner_output()
        with patch("agno.agent.Agent.run", return_value=self._mock_run(out)) as mock_run:
            learner_agent.generate_learning_output(
                buffer_as_xml="",
                task="task",
                task_description="desc",
                strategy="",
            )
        mock_run.assert_called_once()
