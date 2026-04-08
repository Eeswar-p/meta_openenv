---
title: OpenEnv Multi-Task Evaluation Suite
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# OpenEnv Multi-Task Evaluation Suite

# OpenEnv Multi-Task Evaluation Suite
# OpenEnv Multi-Task Evaluation Suite

A containerized OpenEnv-compatible agent environment designed for evaluating language models and agents on structured tasks with deterministic graders.

## 🎯 Environment Overview

This project implements a **Gym environment + grading system + API evaluator** following the [OpenEnv specification](https://github.com/openenv/spec).

**Three Tasks:**
- **Email Triage** (Easy): Binary classification of emails as urgent/non-urgent
- **Data Cleaning** (Medium): Fix inconsistent CSV formatting
- **Customer Support** (Hard): Generate empathetic, solution-focused responses

## 🏗️ Project Structure

```
meta_openenv/
├── app.py                 # Gradio web interface
├── inference.py           # Baseline evaluation script
├── env/
│   ├── environment.py     # OpenEnv implementation (step/reset/state)
│   ├── schema.py          # Pydantic models (Observation, Action, Reward, State)
│   ├── tasks.py           # Task definitions with examples
│   ├── grader.py          # Deterministic graders per task
│   └── __init__.py
├── openenv.yaml           # OpenEnv manifest
├── requirements.txt       # Dependencies
├── Dockerfile             # Container build
└── README.md              # This file
```

## 📋 Action & Observation Space

### Observation (structured JSON)
```python
{
  "task": "email_triage|data_cleaning|customer_support",
  "input_data": "...",
  "history": ["action1", "action2"],
  "step_count": 0
}
```

### Action (text response)
```python
{
  "response": "URGENT"  # or cleaned CSV, or support reply
}
```

### Reward (incremental)
```python
{
  "score": 0.78,  # [0.0, 1.0]
  "components": {
    "task_grade": 0.9,
    "progress": 0.33
  },
  "penalty": 0.0
}
```

### State (full snapshot)
```python
{
  "task": "email_triage",
  "input_data": "...",
  "history": [...],
  "step_count": 2,
  "cumulative_reward": 1.45,
  "observation": { ... }
}
```

## 📊 Task Difficulty Levels

| Task | Difficulty | Max Steps | Grader Components |
|------|-----------|-----------|-------------------|
| Email Triage | Easy | 5 | Classification accuracy (1.0) / heuristic (0.5-0.2) |
| Data Cleaning | Medium | 8 | CSV format (0.6) + structure (0.2) + whitespace (0.2) |
| Customer Support | Hard | 10 | Empathy (0.3) + solution (0.4) + professionalism (0.3) |

## 🧠 Grading System (Deterministic)

Each task has a dedicated grader that computes rewards:

### Email Triage Grader
```
Urgent keywords (priority, ASAP, critical): +0.9
Non-urgent keywords (routine, later): +0.7
Generic response: +0.5
Empty/invalid: +0.0
```

### Data Cleaning Grader
```
Comma-separated format: +0.6
Consistent line structure: +0.2
Clean whitespace: +0.2
Total: 1.0 max
```

### Customer Support Grader
```
Empathy (sorry, apologize, understand): +0.3
Clear solution (fix, resolve, help): +0.4
Professional tone (length, structure): +0.3
Total: 1.0 max
```

### Incremental Reward
```
Base: task-specific grader (0.65)
+ Length bonus (>10 chars): 0.1
+ Non-empty content: 0.1
+ Step progress (<5 steps): 0.15
- Repetition penalty: -0.1
- Lazy generation (...): -0.05
= Score [0.0, 1.0]
```

## 📈 Baseline Scores

Evaluated with **gpt-4o-mini** over 5 episodes per task:

```
Email Triage:       0.78 ± 0.05
Data Cleaning:      0.65 ± 0.08
Customer Support:   0.71 ± 0.06
```

## 🚀 Setup

### Local Python Environment
```bash
cd meta_openenv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Credentials
Set one of:
```bash
export HF_TOKEN="hf_..."          # Hugging Face token
export OPENAI_API_KEY="sk-..."    # OpenAI API key
```

## 🎮 Running Locally

### Web Interface (Gradio)
```bash
python app.py
# Opens at http://localhost:7860
```

### Baseline Evaluation Script
```bash
python inference.py
# Evaluates all 3 tasks with 2 episodes each
```

### Direct Environment Use
```python
from env import OpenEnv

env = OpenEnv(task="email_triage", seed=42)
obs = env.reset()

action = {"response": "URGENT"}
obs, reward, done, info = env.step(action)

state = env.state()  # Full environment snapshot
print(f"Reward: {reward:.3f}")
print(f"Task Grade: {info['task_grade']:.3f}")
```

## 🐳 Docker Deployment

### Build
```bash
docker build -t openenv-meta .
```

### Run Locally
```bash
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  openenv-meta
```

### Deploy to HF Spaces
```bash
# Clone HF Space repo
git clone https://huggingface.co/spaces/<username>/openenv-meta
cd openenv-meta

# Copy files from this repo (except .git, __pycache__)
cp -r ../meta_openenv/* .

# Push to HF
git add .
git commit -m "Deploy OpenEnv multi-task environment"
git push

# Space will auto-build and serve at:
# https://huggingface.co/spaces/<username>/openenv-meta
```

## 🔧 Environment Reference

### Methods

#### `reset(seed=None) -> Dict`
Initialize episode. Returns observation dict.

#### `step(action: Dict) -> Tuple[Dict, float, bool, Dict]`
Execute action. Returns `(obs, reward, done, info)`.

#### `state() -> Dict`
Get full environment state snapshot (OpenEnv requirement).

### Task Registry
```python
from env import list_tasks, get_task

tasks = list_tasks()  # ["email_triage", "data_cleaning", "customer_support"]
task_def = get_task("email_triage")
```

## 📝 OpenEnv Specification

This environment adheres to the OpenEnv spec:
- ✅ Deterministic graders (seeded)
- ✅ Typed schemas (Pydantic validation)
- ✅ Incremental rewards (step-by-step scoring)
- ✅ `state()` method (full state access)
- ✅ OpenEnv.yaml manifest
- ✅ Containerized deployment

## 🧪 Testing

Run smoke tests:
```python
from env import OpenEnv, list_tasks

for task in list_tasks():
    env = OpenEnv(task=task)
    obs = env.reset()
    assert "task" in obs
    
    obs, reward, done, info = env.step({"response": "test"})
    assert 0.0 <= reward <= 1.0
    assert "task_grade" in info
    
    state = env.state()
    assert "observation" in state
    
    print(f"✅ {task}")
```

## 📚 References

- [OpenEnv Specification](https://github.com/openenv/spec)
- [Gymnasium Documentation](https://gymnasium.farama.org)
- [Pydantic v2](https://docs.pydantic.dev/latest)
- [Gradio 4.0](https://www.gradio.app)

## 📄 License

MIT

---

**Built for OpenEnv specification compliance and HF Spaces deployment.**

# meta_openenv
