"""Baseline inference script using OpenAI API."""

import os
import json
from openai import OpenAI
from env import OpenEnv, list_tasks


def run_baseline(task: str, input_text: str, model: str = "gpt-4o-mini") -> str:
    """
    Run baseline agent using OpenAI API.
    
    Args:
        task: Task name (email_triage, data_cleaning, customer_support)
        input_text: Task input
        model: OpenAI model (default: gpt-4o-mini)
    
    Returns:
        Agent response
    """
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Neither HF_TOKEN nor OPENAI_API_KEY environment variable set"
        )
    
    client = OpenAI(api_key=api_key)
    
    # Task-specific prompts
    prompts = {
        "email_triage": (
            "Classify the following email as URGENT or NOT_URGENT.\n"
            "Email:\n{input_text}\n\n"
            "Classification (just output URGENT or NOT_URGENT):"
        ),
        "data_cleaning": (
            "Fix the following CSV data. Return only the cleaned CSV.\n"
            "Data:\n{input_text}\n\n"
            "Cleaned CSV:"
        ),
        "customer_support": (
            "Write a professional customer support response to:\n"
            "{input_text}\n\n"
            "Response:"
        ),
    }
    
    prompt = prompts.get(task, f"Task: {task}\nInput: {input_text}")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt.format(input_text=input_text)
            }
        ],
        temperature=0.7,
        max_tokens=500,
    )
    
    return response.choices[0].message.content


def evaluate_task(task: str, num_episodes: int = 3, model: str = "gpt-4o-mini") -> dict:
    """
    Evaluate baseline agent on a task over multiple episodes.
    
    Args:
        task: Task name
        num_episodes: Number of evaluation runs
        model: OpenAI model
    
    Returns:
        Dict with scores and results
    """
    env = OpenEnv(task=task)
    results = {
        "task": task,
        "episodes": [],
        "avg_reward": 0.0,
        "avg_task_grade": 0.0,
    }
    
    total_reward = 0.0
    total_grade = 0.0
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_grades = []
        
        while not env.done:
            try:
                # Get baseline response
                response = run_baseline(task, env.current_input, model)
                
                # Step environment
                obs, reward, done, info = env.step({"response": response})
                
                episode_reward += reward
                episode_grades.append(info.get("task_grade", 0.0))
                
            except Exception as e:
                print(f"Error in episode {ep}: {e}")
                break
        
        avg_grade = sum(episode_grades) / len(episode_grades) if episode_grades else 0.0
        results["episodes"].append({
            "episode": ep,
            "cumulative_reward": episode_reward,
            "avg_task_grade": avg_grade,
        })
        
        total_reward += episode_reward
        total_grade += avg_grade
    
    results["avg_reward"] = total_reward / num_episodes
    results["avg_task_grade"] = total_grade / num_episodes
    
    return results


if __name__ == "__main__":
    print("OpenEnv Baseline Evaluation")
    print("=" * 50)
    
    for task in list_tasks():
        print(f"\nEvaluating: {task}")
        try:
            result = evaluate_task(task, num_episodes=2)
            print(f"  Avg Reward: {result['avg_reward']:.3f}")
            print(f"  Avg Task Grade: {result['avg_task_grade']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
