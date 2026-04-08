"""OpenEnv-compatible environment."""

import random
from typing import Tuple, Dict, Any
from .schema import Observation, Action, Reward, State
from .tasks import get_task
from .grader import compute_incremental_reward


class OpenEnv:
    """
    OpenEnv-compatible agent environment for multi-task evaluation.
    
    Supports three tasks:
    - email_triage (easy)
    - data_cleaning (medium)
    - customer_support (hard)
    """
    
    def __init__(self, task: str = "email_triage", seed: int = 42):
        """Initialize environment."""
        self.task_name = task
        self.task_def = get_task(task)
        
        if not self.task_def:
            raise ValueError(f"Unknown task: {task}")
        
        self.seed = seed
        self.rng = random.Random(seed)
        
        # State tracking
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.history = []
        self.current_input = ""
        self.done = False
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment and return initial observation.
        
        OpenEnv contract: reset() returns observation only (not info tuple).
        """
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.history = []
        self.done = False
        
        # Sample a random example from the task
        example_idx = self.rng.randint(0, len(self.task_def.examples) - 1)
        self.current_input, _ = self.task_def.examples[example_idx]
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation as dict."""
        obs = Observation(
            task=self.task_name,
            input_data=self.current_input,
            history=self.history,
            step_count=self.step_count,
        )
        return obs.model_dump()
    
    def state(self) -> Dict[str, Any]:
        """
        Return full environment state (OpenEnv requirement).
        """
        state = State(
            task=self.task_name,
            input_data=self.current_input,
            history=self.history,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            observation=Observation(
                task=self.task_name,
                input_data=self.current_input,
                history=self.history,
                step_count=self.step_count,
            ),
        )
        return state.model_dump()
    
    def step(self, action_data: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Execute one step.
        
        OpenEnv contract: step(action) returns (observation, reward, done, info).
        
        Args:
            action_data: Dict with 'response' key
        
        Returns:
            (observation, reward, done, info)
        """
        # Validate and parse action
        try:
            action = Action.model_validate(action_data)
        except Exception as e:
            # Invalid action: apply penalty
            reward_val = Reward(
                score=0.0,
                components={"error": "invalid_action"},
                penalty=0.2,
            )
            self.step_count += 1
            self.done = self.step_count >= self.task_def.max_steps
            return self._get_observation(), reward_val.score, self.done, {"error": str(e)}
        
        # Compute reward
        reward_score = compute_incremental_reward(
            action.response,
            self.step_count,
            self.task_name
        )
        
        reward = Reward(
            score=reward_score,
            components={
                "task_grade": self.task_def.grade(action.response),
                "progress": self.step_count / self.task_def.max_steps,
            },
        )
        
        # Update state
        self.step_count += 1
        self.cumulative_reward += reward_score
        self.history.append(action.response)
        self.done = self.step_count >= self.task_def.max_steps
        
        # Return OpenEnv format: (observation, reward, done, info)
        return (
            self._get_observation(),
            reward.score,
            self.done,
            {
                "task_grade": reward.components["task_grade"],
                "cumulative_reward": self.cumulative_reward,
            }
        )
