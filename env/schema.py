"""Pydantic schemas for OpenEnv compliance."""

from pydantic import BaseModel, Field
from typing import Optional, List


class Observation(BaseModel):
    """Observation model: structured input to the agent."""
    task: str = Field(..., description="Task name (email_triage, data_cleaning, customer_support)")
    input_data: str = Field(..., description="The input to process")
    history: List[str] = Field(default_factory=list, description="Previous actions in this episode")
    step_count: int = Field(default=0, description="Current step number")


class Action(BaseModel):
    """Action model: agent's response."""
    response: str = Field(..., description="Agent's response/action")


class Reward(BaseModel):
    """Reward model: incremental feedback."""
    score: float = Field(..., ge=0.0, le=1.0, description="Reward value [0.0, 1.0]")
    components: dict = Field(default_factory=dict, description="Breakdown of reward components")
    penalty: float = Field(default=0.0, ge=0.0, description="Applied penalties")


class State(BaseModel):
    """Full environment state snapshot."""
    task: str
    input_data: str
    history: List[str]
    step_count: int
    cumulative_reward: float
    observation: Observation
