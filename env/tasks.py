"""Task definitions for OpenEnv."""

from dataclasses import dataclass
from typing import List, Tuple
from .grader import (
    grade_email_triage,
    grade_data_cleaning,
    grade_customer_support_reply,
)


@dataclass
class Task:
    """Base task definition."""
    name: str
    description: str
    difficulty: str  # easy, medium, hard
    max_steps: int
    examples: List[Tuple[str, str]]  # (input, expected_output)


class EmailTriageTask(Task):
    """Easy task: Classify email as urgent or not urgent."""
    
    def __init__(self):
        super().__init__(
            name="email_triage",
            description="Classify incoming emails as URGENT or NOT_URGENT",
            difficulty="easy",
            max_steps=5,
            examples=[
                (
                    "Subject: System Down\nBody: Production servers offline, fix needed NOW",
                    "URGENT"
                ),
                (
                    "Subject: Meeting Reminder\nBody: We have a team standup tomorrow at 10am",
                    "NOT_URGENT"
                ),
                (
                    "Subject: Critical Security Alert\nBody: Unauthorized access detected in sales db",
                    "URGENT"
                ),
            ]
        )
    
    def grade(self, action_response: str) -> float:
        """Grade the email triage response."""
        return grade_email_triage(action_response)


class DataCleaningTask(Task):
    """Medium task: Fix inconsistent CSV formatting."""
    
    def __init__(self):
        super().__init__(
            name="data_cleaning",
            description="Clean malformed CSV data into consistent format",
            difficulty="medium",
            max_steps=8,
            examples=[
                (
                    "name|age|email\nJohn;30|john@mail\nSarah 25 sarah@mail",
                    "name,age,email\nJohn,30,john@mail\nSarah,25,sarah@mail"
                ),
                (
                    "id  product  price\n1  Laptop  1000\n2  Phone  500",
                    "id,product,price\n1,Laptop,1000\n2,Phone,500"
                ),
            ]
        )
    
    def grade(self, action_response: str) -> float:
        """Grade the data cleaning response."""
        return grade_data_cleaning(action_response)


class CustomerSupportTask(Task):
    """Hard task: Generate structured customer support response."""
    
    def __init__(self):
        super().__init__(
            name="customer_support",
            description="Generate empathetic and solution-focused customer support reply",
            difficulty="hard",
            max_steps=10,
            examples=[
                (
                    "Customer: I've been trying to reset my password for 2 days but keep getting error 500",
                    "I sincerely apologize for this frustrating experience. "
                    "I understand how critical account access is. "
                    "I've identified the issue in our password reset service and have a solution for you. "
                    "Please try resetting again or contact our support team directly."
                ),
            ]
        )
    
    def grade(self, action_response: str) -> float:
        """Grade the customer support response."""
        return grade_customer_support_reply(action_response)


# Task registry
TASKS = {
    "email_triage": EmailTriageTask(),
    "data_cleaning": DataCleaningTask(),
    "customer_support": CustomerSupportTask(),
}


def get_task(task_name: str) -> Task:
    """Get task by name."""
    return TASKS.get(task_name)


def list_tasks() -> List[str]:
    """List all available tasks."""
    return list(TASKS.keys())
