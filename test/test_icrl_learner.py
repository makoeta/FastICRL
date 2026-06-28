import os
import pytest
from unittest.mock import patch

from fasticrl.icrl_learner import ICRLLearner
from fasticrl.models.attempt import Attempt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_learner(mock_model, tasks=None, buffer=None, strategy="", task_description="Solve arithmetic"):
    """Construct an ICRLLearner with all three sub-agents mocked out."""
    tasks = tasks if tasks is not None else ["1+1", "2+2", "3+3"]
    with patch("fasticrl.icrl_learner.LearnerAgent"), \
         patch("fasticrl.icrl_learner.RewardAgent"), \
         patch("fasticrl.icrl_learner.StrategistAgent"):
        return ICRLLearner(
            learner_model=mock_model,
            reward_model=mock_model,
            strategy_model=mock_model,
            task_description=task_description,
            tasks=tasks,
            buffer=buffer or [],
            strategy=strategy,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def learner(mock_model):
    return make_learner(mock_model)


@pytest.fixture
def learner_with_buffer(mock_model, sample_attempts):
    return make_learner(mock_model, buffer=sample_attempts)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestICRLLearnerInit:
    def test_missing_task_description_raises(self, mock_model):
        with patch("fasticrl.icrl_learner.LearnerAgent"), \
             patch("fasticrl.icrl_learner.RewardAgent"), \
             patch("fasticrl.icrl_learner.StrategistAgent"):
            with pytest.raises(ValueError, match="task_description"):
                ICRLLearner(
                    learner_model=mock_model,
                    reward_model=mock_model,
                    strategy_model=mock_model,
                )

    def test_empty_tasks_list_raises(self, mock_model):
        with patch("fasticrl.icrl_learner.LearnerAgent"), \
             patch("fasticrl.icrl_learner.RewardAgent"), \
             patch("fasticrl.icrl_learner.StrategistAgent"):
            with pytest.raises(ValueError, match="tasks"):
                ICRLLearner(
                    learner_model=mock_model,
                    reward_model=mock_model,
                    strategy_model=mock_model,
                    task_description="desc",
                    tasks=[],
                )

    def test_episode_starts_at_zero_without_buffer(self, learner):
        assert learner.episode == 0

    def test_episode_starts_at_buffer_length(self, mock_model, sample_attempts):
        l = make_learner(mock_model, buffer=sample_attempts)
        assert l.episode == len(sample_attempts)

    def test_default_strategy_is_empty(self, learner):
        assert learner.strategy == ""

    def test_custom_strategy_stored(self, mock_model):
        l = make_learner(mock_model, strategy="be precise")
        assert l.strategy == "be precise"

    def test_tasks_none_defaults_to_empty_list(self, mock_model):
        with patch("fasticrl.icrl_learner.LearnerAgent"), \
             patch("fasticrl.icrl_learner.RewardAgent"), \
             patch("fasticrl.icrl_learner.StrategistAgent"):
            l = ICRLLearner(
                learner_model=mock_model,
                reward_model=mock_model,
                strategy_model=mock_model,
                task_description="desc",
                tasks=None,
            )
        assert l.tasks == []


# ---------------------------------------------------------------------------
# _present_learning_task
# ---------------------------------------------------------------------------

class TestPresentLearningTask:
    def test_returns_first_task_when_episode_zero(self, learner):
        assert learner._present_learning_task == "1+1"

    def test_cycles_through_tasks(self, mock_model):
        tasks = ["a", "b", "c"]
        l = make_learner(mock_model, tasks=tasks)
        assert l._present_learning_task == "a"
        l.episode = 1
        assert l._present_learning_task == "b"
        l.episode = 2
        assert l._present_learning_task == "c"
        l.episode = 3
        assert l._present_learning_task == "a"

    def test_wraps_around_with_buffer_offset(self, mock_model, sample_attempts):
        tasks = ["x", "y"]
        # buffer has 2 attempts → episode starts at 2 → tasks[2 % 2] = tasks[0] = "x"
        l = make_learner(mock_model, tasks=tasks, buffer=sample_attempts)
        assert l._present_learning_task == "x"


# ---------------------------------------------------------------------------
# __attempts_as_xml  (via name-mangled accessor)
# ---------------------------------------------------------------------------

class TestAttemptsAsXml:
    def test_empty_buffer_returns_empty_string(self, learner):
        assert learner._ICRLLearner__attempts_as_xml() == ""

    def test_single_attempt_xml(self, learner, sample_attempt):
        learner.buffer = [sample_attempt]
        xml = learner._ICRLLearner__attempts_as_xml()
        assert "<attempt>" in xml
        assert "Task: solve 1+1" in xml
        assert "Output: 2" in xml
        assert "Reward: 9" in xml

    def test_multiple_attempts_all_present(self, learner, sample_attempts):
        learner.buffer = sample_attempts
        xml = learner._ICRLLearner__attempts_as_xml()
        assert xml.count("<attempt>") == 2
        assert "1+1" in xml
        assert "2+2" in xml


# ---------------------------------------------------------------------------
# generate_action / generate_reward / _generate_attempt_for_task
# ---------------------------------------------------------------------------

class TestGenerateAction:
    def test_calls_learner_agent(self, learner, learner_output):
        learner.learner_agent.generate_learning_output.return_value = learner_output
        result = learner.generate_action("1+1")
        learner.learner_agent.generate_learning_output.assert_called_once()
        assert result is learner_output

    def test_passes_task_to_learner(self, learner, learner_output):
        learner.learner_agent.generate_learning_output.return_value = learner_output
        learner.generate_action("unique_task_123")
        call_kwargs = learner.learner_agent.generate_learning_output.call_args
        assert call_kwargs.kwargs["task"] == "unique_task_123"

    def test_passes_task_description(self, learner, learner_output):
        learner.learner_agent.generate_learning_output.return_value = learner_output
        learner.generate_action("some task")
        call_kwargs = learner.learner_agent.generate_learning_output.call_args
        assert call_kwargs.kwargs["task_description"] == "Solve arithmetic"

    def test_passes_strategy(self, mock_model, learner_output):
        l = make_learner(mock_model, strategy="unique_strategy_456")
        l.learner_agent.generate_learning_output.return_value = learner_output
        l.generate_action("task")
        call_kwargs = l.learner_agent.generate_learning_output.call_args
        assert call_kwargs.kwargs["strategy"] == "unique_strategy_456"


class TestGenerateReward:
    def test_calls_reward_agent(self, learner, learner_output, reward_output):
        learner.reward_agent.generate_reward_output.return_value = reward_output
        result = learner.generate_reward("1+1", learner_output)
        learner.reward_agent.generate_reward_output.assert_called_once()
        assert result is reward_output

    def test_passes_task_and_output(self, learner, learner_output, reward_output):
        learner.reward_agent.generate_reward_output.return_value = reward_output
        learner.generate_reward("solve me", learner_output)
        call_kwargs = learner.reward_agent.generate_reward_output.call_args
        assert call_kwargs.kwargs["task"] == "solve me"
        assert call_kwargs.kwargs["output"] == learner_output.learning_output


class TestGenerateAttemptForTask:
    def test_returns_attempt(self, learner, learner_output, reward_output):
        learner.learner_agent.generate_learning_output.return_value = learner_output
        learner.reward_agent.generate_reward_output.return_value = reward_output
        attempt = learner._generate_attempt_for_task("1+1")
        assert isinstance(attempt, Attempt)
        assert attempt.task == "1+1"
        assert attempt.output == learner_output.learning_output
        assert attempt.reward == reward_output.reward


# ---------------------------------------------------------------------------
# auto_learn
# ---------------------------------------------------------------------------

class TestAutoLearn:
    def test_raises_when_no_tasks(self, mock_model):
        l = make_learner(mock_model, tasks=["placeholder"])
        l.tasks = []
        with pytest.raises(ValueError, match="tasks list is empty"):
            l.auto_learn(episodes=1)

    def test_adds_attempts_to_buffer(self, learner):
        dummy = Attempt(task="1+1", output="2", reward=9)
        with patch.object(learner, "_generate_attempt_for_task", return_value=dummy):
            learner.auto_learn(episodes=3, batch_size=1)
        assert len(learner.buffer) == 3

    def test_increments_episode(self, learner):
        dummy = Attempt(task="t", output="o", reward=5)
        with patch.object(learner, "_generate_attempt_for_task", return_value=dummy):
            learner.auto_learn(episodes=2, batch_size=1)
        assert learner.episode == 2

    def test_defaults_episodes_to_task_count(self, mock_model):
        tasks = ["a", "b", "c"]
        l = make_learner(mock_model, tasks=tasks)
        dummy = Attempt(task="t", output="o", reward=5)
        with patch.object(l, "_generate_attempt_for_task", return_value=dummy):
            l.auto_learn()
        assert len(l.buffer) == len(tasks)

    def test_batch_mode_adds_correct_count(self, learner):
        # auto_learn runs `episodes` iterations each processing `batch_size` tasks,
        # so total attempts = episodes * batch_size
        dummy = Attempt(task="t", output="o", reward=5)
        with patch.object(learner, "_generate_attempt_for_task", return_value=dummy):
            learner.auto_learn(episodes=4, batch_size=2)
        assert len(learner.buffer) == 4 * 2

    def test_strategy_update_interval_triggers_update(self, learner):
        dummy = Attempt(task="t", output="o", reward=5)
        with patch.object(learner, "_generate_attempt_for_task", return_value=dummy), \
             patch.object(learner, "update_strategy") as mock_update:
            learner.auto_learn(episodes=2, batch_size=1, strategy_update_interval=2)
        mock_update.assert_called_once()

    def test_strategy_update_interval_not_triggered_early(self, learner):
        dummy = Attempt(task="t", output="o", reward=5)
        with patch.object(learner, "_generate_attempt_for_task", return_value=dummy), \
             patch.object(learner, "update_strategy") as mock_update:
            learner.auto_learn(episodes=1, batch_size=1, strategy_update_interval=5)
        mock_update.assert_not_called()


# ---------------------------------------------------------------------------
# update_strategy
# ---------------------------------------------------------------------------

class TestUpdateStrategy:
    def test_updates_strategy_from_strategist(self, learner, strategy_output):
        learner.strategist_agent.generate_new_strategy.return_value = strategy_output
        learner.update_strategy()
        assert learner.strategy == strategy_output.strategy

    def test_passes_task_description_to_strategist(self, learner, strategy_output):
        learner.strategist_agent.generate_new_strategy.return_value = strategy_output
        learner.update_strategy()
        call_kwargs = learner.strategist_agent.generate_new_strategy.call_args
        assert call_kwargs.kwargs["task_description"] == "Solve arithmetic"

    def test_passes_existing_strategy(self, mock_model, strategy_output):
        l = make_learner(mock_model, strategy="old_strategy")
        l.strategist_agent.generate_new_strategy.return_value = strategy_output
        l.update_strategy()
        call_kwargs = l.strategist_agent.generate_new_strategy.call_args
        assert call_kwargs.kwargs["existing_strategy"] == "old_strategy"


# ---------------------------------------------------------------------------
# agent_save_state
# ---------------------------------------------------------------------------

class TestAgentSaveState:
    def test_save_state_contains_task_description(self, learner):
        assert learner.agent_save_state.task_description == "Solve arithmetic"

    def test_save_state_contains_strategy(self, mock_model):
        l = make_learner(mock_model, strategy="my strategy")
        assert l.agent_save_state.strategy == "my strategy"

    def test_save_state_buffer_matches(self, learner, sample_attempts):
        learner.buffer = sample_attempts
        state = learner.agent_save_state
        assert len(state.buffer) == 2

    def test_save_state_buffer_preserves_attempt_fields(self, learner, sample_attempt):
        learner.buffer = [sample_attempt]
        state = learner.agent_save_state
        assert state.buffer[0].task == "solve 1+1"
        assert state.buffer[0].reward == 9


# ---------------------------------------------------------------------------
# to_yaml / from_yaml
# ---------------------------------------------------------------------------

class TestYamlPersistence:
    def test_to_yaml_creates_file(self, mock_model, tmp_path):
        l = make_learner(mock_model)
        path = str(tmp_path / "state")
        l.to_yaml(path)
        assert os.path.exists(path + ".yaml")

    def test_to_yaml_appends_extension_if_missing(self, mock_model, tmp_path):
        l = make_learner(mock_model)
        path = str(tmp_path / "state")
        l.to_yaml(path)
        assert os.path.exists(path + ".yaml")

    def test_to_yaml_does_not_double_append_extension(self, mock_model, tmp_path):
        l = make_learner(mock_model)
        path = str(tmp_path / "state.yaml")
        l.to_yaml(path)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".yaml")

    def test_roundtrip_task_description(self, mock_model, tmp_path):
        l = make_learner(mock_model, task_description="unique_description_789")
        path = str(tmp_path / "state")
        l.to_yaml(path)

        with patch("fasticrl.icrl_learner.LearnerAgent"), \
             patch("fasticrl.icrl_learner.RewardAgent"), \
             patch("fasticrl.icrl_learner.StrategistAgent"):
            loaded = ICRLLearner.from_yaml(
                path + ".yaml",
                learner_model=mock_model,
                reward_model=mock_model,
                strategy_model=mock_model,
            )
        assert loaded.task_description == "unique_description_789"

    def test_roundtrip_strategy(self, mock_model, tmp_path):
        l = make_learner(mock_model, strategy="my_saved_strategy")
        path = str(tmp_path / "state")
        l.to_yaml(path)

        with patch("fasticrl.icrl_learner.LearnerAgent"), \
             patch("fasticrl.icrl_learner.RewardAgent"), \
             patch("fasticrl.icrl_learner.StrategistAgent"):
            loaded = ICRLLearner.from_yaml(
                path + ".yaml",
                learner_model=mock_model,
                reward_model=mock_model,
                strategy_model=mock_model,
            )
        assert loaded.strategy == "my_saved_strategy"

    def test_roundtrip_buffer(self, mock_model, tmp_path, sample_attempts):
        l = make_learner(mock_model, buffer=sample_attempts)
        path = str(tmp_path / "state")
        l.to_yaml(path)

        with patch("fasticrl.icrl_learner.LearnerAgent"), \
             patch("fasticrl.icrl_learner.RewardAgent"), \
             patch("fasticrl.icrl_learner.StrategistAgent"):
            loaded = ICRLLearner.from_yaml(
                path + ".yaml",
                learner_model=mock_model,
                reward_model=mock_model,
                strategy_model=mock_model,
            )
        assert len(loaded.buffer) == len(sample_attempts)
        assert loaded.buffer[0].task == sample_attempts[0].task
        assert loaded.buffer[0].reward == sample_attempts[0].reward

    def test_roundtrip_episode_matches_buffer_length(self, mock_model, tmp_path, sample_attempts):
        l = make_learner(mock_model, buffer=sample_attempts)
        path = str(tmp_path / "state")
        l.to_yaml(path)

        with patch("fasticrl.icrl_learner.LearnerAgent"), \
             patch("fasticrl.icrl_learner.RewardAgent"), \
             patch("fasticrl.icrl_learner.StrategistAgent"):
            loaded = ICRLLearner.from_yaml(
                path + ".yaml",
                learner_model=mock_model,
                reward_model=mock_model,
                strategy_model=mock_model,
            )
        assert loaded.episode == len(sample_attempts)
