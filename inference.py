"""Baseline inference script – strictly follows OpenEnv submission checklist."""

import os
import json
from openai import OpenAI
from env import OpenEnv, list_tasks

# ── Environment variables (checklist-required) ────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")                    # NO default – checklist requirement

# Optional – if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI client configured via env vars (checklist-required) ────────────────
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ── Task prompts ──────────────────────────────────────────────────────────────
PROMPTS = {
    "email_triage": (
        "Classify the following email as URGENT or NOT_URGENT.\n"
        "Email:\n{input_text}\n\n"
        "Classification (output only URGENT or NOT_URGENT):"
    ),
    "data_cleaning": (
        "Fix the following CSV data. Return only the cleaned CSV.\n"
        "Data:\n{input_text}\n\n"
        "Cleaned CSV:"
    ),
    "customer_support": (
        "Write a professional, empathetic customer support response to:\n"
        "{input_text}\n\n"
        "Response:"
    ),
}


def get_llm_response(task: str, input_text: str) -> str:
    """Call the LLM via the OpenAI client (configured from env vars)."""
    prompt = PROMPTS.get(task, f"Task: {task}\nInput: {input_text}")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt.format(input_text=input_text)}],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def run_episode(task: str, episode: int) -> dict:
    """
    Run a single episode with START / STEP / END structured stdout logging.
    This format is required by the OpenEnv submission checklist.
    """
    env = OpenEnv(task=task, seed=episode)
    obs = env.reset()

    # ── START ─────────────────────────────────────────────────────────────────
    print(json.dumps({
        "type": "START",
        "episode": episode,
        "task": task,
        "observation": obs,
    }))

    cumulative_reward = 0.0
    step_count = 0

    while True:
        input_text = obs.get("input_data", "")
        response   = get_llm_response(task, input_text)

        obs, reward, done, info = env.step({"response": response})
        cumulative_reward += reward
        step_count += 1

        # ── STEP ──────────────────────────────────────────────────────────────
        print(json.dumps({
            "type": "STEP",
            "episode": episode,
            "task": task,
            "step": step_count,
            "response": response,
            "reward": round(reward, 4),
            "done": done,
            "info": info,
        }))

        if done:
            break

    # ── END ───────────────────────────────────────────────────────────────────
    print(json.dumps({
        "type": "END",
        "episode": episode,
        "task": task,
        "cumulative_reward": round(cumulative_reward, 4),
        "steps": step_count,
    }))

    return {
        "episode": episode,
        "task": task,
        "cumulative_reward": cumulative_reward,
        "steps": step_count,
    }


if __name__ == "__main__":
    tasks = list_tasks()
    num_episodes = 2

    for task in tasks:
        for ep in range(num_episodes):
            try:
                run_episode(task, episode=ep)
            except Exception as exc:
                print(json.dumps({
                    "type": "ERROR",
                    "episode": ep,
                    "task": task,
                    "error": str(exc),
                }))
