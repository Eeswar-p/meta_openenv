"""Deterministic graders for each task."""

import re


def grade_email_triage(action_response: str, ground_truth: str = None) -> float:
    """
    Grade email triage task.
    - Correct classification (urgent/not urgent): 1.0
    - Relevant keywords present: 0.5
    - Otherwise: 0.0
    """
    response_lower = action_response.lower()
    
    # If ground truth provided, check exact match
    if ground_truth:
        if ground_truth.lower() in response_lower:
            return 1.0
        else:
            return 0.0
    
    # Otherwise, heuristic scoring
    # Check for explicit classification
    if "urgent" in response_lower and "not" not in response_lower:
        return 0.9  # Classified as URGENT
    elif "not_urgent" in response_lower or (("not" in response_lower or "routine" in response_lower) and "urgent" not in response_lower):
        return 0.7  # Classified as NOT_URGENT
    elif "priority" in response_lower or "critical" in response_lower:
        return 0.85
    elif any(kw in response_lower for kw in ["reply", "respond", "answer"]):
        return 0.5
    else:
        return 0.3


def grade_data_cleaning(action_response: str) -> float:
    """
    Grade data cleaning task.
    - Proper CSV format (commas, consistent structure): 0.6
    - No duplicates/empty lines: 0.2
    - Proper formatting: 0.2
    """
    score = 0.0
    
    # Check for comma-separated format
    if "," in action_response:
        score += 0.6
    
    # Check for proper line structure
    lines = action_response.strip().split("\n")
    if len(lines) > 1 and all(line.count(",") == lines[0].count(",") for line in lines if line):
        score += 0.2
    else:
        score += 0.05
    
    # Check for no excessive whitespace
    if "  " not in action_response and action_response.count("\n\n") == 0:
        score += 0.2
    else:
        score += 0.05
    
    return min(score, 1.0)


def grade_customer_support_reply(action_response: str) -> float:
    """
    Grade customer support reply task.
    - Contains apology/empathy: +0.3
    - Contains clear solution/action: +0.4
    - Professional tone: +0.3
    """
    response_lower = action_response.lower()
    score = 0.0
    
    # Empathy/apology
    empathy_keywords = {"sorry", "apologize", "understand", "appreciate", "frustrat"}
    if any(kw in response_lower for kw in empathy_keywords):
        score += 0.3
    else:
        score += 0.05
    
    # Solution/action
    solution_keywords = {"will", "can", "solution", "fix", "resolv", "help"}
    if any(kw in response_lower for kw in solution_keywords):
        score += 0.4
    else:
        score += 0.1
    
    # Professional tone (length, structure)
    if len(action_response) > 20:
        score += 0.3
    else:
        score += 0.05
    
    return min(score, 1.0)


def compute_incremental_reward(action_response: str, step_count: int, task: str) -> float:
    """
    Compute incremental reward based on step progress and action quality.
    
    - Length bonus (>10 chars): 0.1
    - Non-empty: 0.1
    - Step progress (<5 steps): 0.15
    - Task-specific grader: 0.65 (main component)
    """
    score = 0.0
    penalty = 0.0
    
    # Length and content checks
    if len(action_response) > 10:
        score += 0.1
    if len(action_response.strip()) > 0:
        score += 0.1
    
    # Step progress bonus (encourage quick resolution)
    if step_count < 5:
        score += 0.15
    
    # Task-specific grading
    if task == "email_triage":
        score += 0.65 * grade_email_triage(action_response)
    elif task == "data_cleaning":
        score += 0.65 * grade_data_cleaning(action_response)
    elif task == "customer_support":
        score += 0.65 * grade_customer_support_reply(action_response)
    
    # Apply penalties
    if action_response.lower().count(action_response.lower().split()[0] if action_response.split() else "") > 3:
        penalty += 0.1  # Repetition penalty
    
    if "..." in action_response or "[" in action_response:
        penalty += 0.05  # Lazy generation penalty
    
    return max(min(score - penalty, 1.0), 0.0)
