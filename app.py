from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from env import OpenEnv, list_tasks


app = FastAPI(title="Meta OpenEnv Space", version="2.0.0")

TASKS: Dict[str, Dict[str, str]] = {
        "email_triage": {
                "description": "Classify an email as URGENT or NOT_URGENT.",
                "example": "Subject: Login issue\nBody: We are blocked, please fix ASAP.",
        },
        "data_cleaning": {
                "description": "Normalize messy CSV-like text.",
                "example": "name, age ,city\nAlice, 22, Seattle\n Bob,31, Austin",
        },
        "customer_support": {
                "description": "Write an empathetic, actionable support response.",
                "example": "My payment failed twice and I am frustrated.",
        },
}

HISTORY: List[dict] = []
LEADERBOARD: List[dict] = []
SESSIONS: Dict[str, OpenEnv] = {}


class EvaluateRequest(BaseModel):
        task: str
        user_input: str
        participant: str = "Guest"


class ResetRequest(BaseModel):
    task: Optional[str] = "email_triage"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    session_id: str
    response: str


def _score_email(text: str) -> tuple[str, float, List[str]]:
        lowered = text.lower()
        urgent_hits = sum(word in lowered for word in ["urgent", "asap", "critical", "immediately", "blocked"])
        label = "URGENT" if urgent_hits > 0 else "NOT_URGENT"
        score = min(1.0, 0.45 + urgent_hits * 0.14)
        feedback = [
                "Detected urgency signals." if urgent_hits else "No urgency signals detected.",
                "Output label is concise and deterministic.",
        ]
        return label, score, feedback


def _score_data(text: str) -> tuple[str, float, List[str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        normalized = "\n".join(",".join(cell.strip() for cell in line.split(",")) for line in lines)
        has_csv = any("," in line for line in lines)
        score = 0.3
        if has_csv:
                score += 0.3
        if len(lines) >= 2:
                score += 0.2
        if all("  " not in line for line in lines):
                score += 0.2
        feedback = [
                "Whitespace normalized.",
                "Row format checked for consistency.",
        ]
        return normalized, min(1.0, score), feedback


def _score_support(text: str) -> tuple[str, float, List[str]]:
        lowered = text.lower()
        empathy = any(w in lowered for w in ["sorry", "understand", "apologize"])
        action = any(w in lowered for w in ["fix", "resolve", "check", "investigate", "help"])
        response = (
                "Thank you for reporting this. I understand the frustration. "
                "I will investigate immediately and share an update with a concrete resolution path."
        )
        score = 0.45 + (0.25 if empathy else 0.0) + (0.3 if action else 0.0)
        feedback = [
                "Empathy detected." if empathy else "Add empathy language to improve score.",
                "Actionability detected." if action else "Add specific next steps.",
        ]
        return response, min(1.0, score), feedback


def _run_task(task: str, text: str) -> tuple[str, float, List[str]]:
        if task == "email_triage":
                return _score_email(text)
        if task == "data_cleaning":
                return _score_data(text)
        return _score_support(text)


def _grade_band(score: float) -> str:
        if score >= 0.85:
                return "Excellent"
        if score >= 0.7:
                return "Strong"
        if score >= 0.5:
                return "Needs Improvement"
        return "Weak"


@app.get("/api/status")
def status_json():
        return {
                "status": "running",
                "app": "meta_openenv",
                "version": "2.0.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "task_count": len(TASKS),
                "history_size": len(HISTORY),
                "endpoints": {
                        "home": "/",
                        "status": "/api/status",
                        "tasks": "/api/tasks",
                        "evaluate": "/api/evaluate",
                        "leaderboard": "/api/leaderboard",
                        "history": "/api/history",
                },
        }


@app.get("/api/tasks")
def tasks_json():
        return TASKS


@app.get("/api/openenv/tasks")
def openenv_tasks_json():
    return {"tasks": list_tasks()}


@app.get("/api/leaderboard")
def leaderboard_json():
        return {"leaderboard": LEADERBOARD[:10]}


@app.get("/api/history")
def history_json():
        return {"history": HISTORY[:20]}


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest):
        task = req.task if req.task in TASKS else "email_triage"
        cleaned = req.user_input.strip()
        if not cleaned:
                return {
                        "ok": False,
                        "error": "Please enter input before submitting.",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                }

        output, score, feedback = _run_task(task, cleaned)
        result = {
                "ok": True,
                "participant": req.participant.strip() or "Guest",
                "task": task,
                "input": cleaned,
                "output": output,
                "score": round(score, 3),
                "grade": _grade_band(score),
                "feedback": feedback,
                "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        HISTORY.insert(0, result)
        del HISTORY[40:]

        LEADERBOARD.insert(
                0,
                {
                        "participant": result["participant"],
                        "task": task,
                        "score": result["score"],
                        "grade": result["grade"],
                        "timestamp": result["timestamp"],
                },
        )
        LEADERBOARD.sort(key=lambda row: row["score"], reverse=True)
        del LEADERBOARD[20:]

        return result


@app.post("/reset")
@app.post("/api/reset")
@app.post("/openenv/reset")
@app.post("/api/openenv/reset")
def openenv_reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()
    task = req.task if req.task in list_tasks() else "email_triage"
    seed = req.seed if req.seed is not None else 42
    env = OpenEnv(task=task, seed=seed)
    observation = env.reset()
    session_id = str(uuid4())
    SESSIONS[session_id] = env
    return {
        "ok": True,
        "session_id": session_id,
        "task": task,
        "observation": observation,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/step")
@app.post("/api/step")
@app.post("/openenv/step")
@app.post("/api/openenv/step")
def openenv_step(req: StepRequest):
    env = SESSIONS.get(req.session_id)
    if env is None:
        return {
            "ok": False,
            "error": "Invalid session_id. Call POST /reset first.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    observation, reward, done, info = env.step({"response": req.response})
    return {
        "ok": True,
        "session_id": req.session_id,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/state/{session_id}")
@app.get("/api/state/{session_id}")
@app.get("/openenv/state/{session_id}")
@app.get("/api/openenv/state/{session_id}")
def openenv_state(session_id: str):
    env = SESSIONS.get(session_id)
    if env is None:
        return {
            "ok": False,
            "error": "Invalid session_id. Call POST /reset first.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    return {
        "ok": True,
        "session_id": session_id,
        "state": env.state(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/", response_class=HTMLResponse)
def home_page():
        return """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Meta OpenEnv</title>
    <style>
        :root {
            --bg: #070d1b;
            --surface: #101a34;
            --surface-2: #142244;
            --text: #eef4ff;
            --muted: #9ab0d9;
            --line: #2a3f6e;
            --good: #3adf95;
            --primary: #4b8bff;
            --primary-2: #2ed3b7;
            --warn: #ffd166;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: "Segoe UI", "Trebuchet MS", Tahoma, sans-serif;
            color: var(--text);
            background:
                radial-gradient(1100px 400px at 0% -10%, #223567 0%, transparent 60%),
                radial-gradient(900px 420px at 100% 0%, #154357 0%, transparent 62%),
                var(--bg);
            padding: 20px;
        }

        .shell {
            max-width: 1120px;
            margin: 0 auto;
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(16,26,52,0.97), rgba(13,22,45,0.96));
            box-shadow: 0 25px 60px rgba(0,0,0,0.35);
        }

        .hero {
            padding: 24px;
            border-bottom: 1px solid var(--line);
            display: grid;
            gap: 10px;
        }

        h1 {
            margin: 0;
            font-size: clamp(1.6rem, 2.4vw, 2.3rem);
        }

        .sub {
            color: var(--muted);
            line-height: 1.45;
            font-size: 1rem;
        }

        .stats {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            padding: 0 24px 16px;
        }

        .stat {
            background: rgba(20,34,68,0.8);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 10px 12px;
        }

        .stat .k { font-size: 0.82rem; color: var(--muted); }
        .stat .v { font-size: 1.1rem; font-weight: 700; }

        .main {
            display: grid;
            gap: 14px;
            grid-template-columns: 1fr 1fr;
            padding: 16px 24px 24px;
        }

        .card {
            background: rgba(15,24,48,0.88);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 14px;
        }

        .label {
            color: #c4d4f7;
            margin-bottom: 8px;
            font-size: 0.88rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input, select, textarea, button {
            font-family: inherit;
            width: 100%;
            border-radius: 10px;
            border: 1px solid #314c85;
            background: #0c1733;
            color: var(--text);
            padding: 10px 12px;
        }

        textarea { min-height: 140px; resize: vertical; }

        .row { display: grid; gap: 10px; grid-template-columns: 1fr 1fr; }

        .btns {
            display: grid;
            gap: 10px;
            grid-template-columns: repeat(4, 1fr);
            margin-top: 10px;
        }

        button {
            cursor: pointer;
            border: none;
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            font-weight: 700;
            transition: transform 120ms ease, filter 120ms ease;
        }

        button:hover { transform: translateY(-1px); filter: brightness(1.05); }

        pre {
            margin: 0;
            background: #07112a;
            border: 1px solid #263d71;
            border-radius: 10px;
            min-height: 180px;
            padding: 12px;
            overflow: auto;
            color: #d8e7ff;
            line-height: 1.42;
            font-size: 0.9rem;
        }

        .meter {
            margin: 10px 0;
            height: 12px;
            border: 1px solid #335088;
            border-radius: 999px;
            overflow: hidden;
            background: #0a1630;
        }

        .meter > div {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #2ed3b7, #4b8bff);
            transition: width 240ms ease;
        }

        .tiny { color: var(--muted); font-size: 0.84rem; margin-top: 8px; }

        .leader {
            margin-top: 12px;
            border-top: 1px dashed #34518b;
            padding-top: 10px;
        }

        ol {
            margin: 6px 0 0 18px;
            padding: 0;
            color: #deebff;
            font-size: 0.92rem;
        }

        @media (max-width: 980px) {
            .stats { grid-template-columns: 1fr 1fr; }
            .main { grid-template-columns: 1fr; }
            .btns { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <h1>Meta OpenEnv Arena</h1>
            <div class="sub">A hackathon-ready evaluation console with task presets, scoring, feedback, history, and leaderboard.</div>
        </section>

        <section class="stats">
            <div class="stat"><div class="k">Service</div><div class="v" id="svc">Running</div></div>
            <div class="stat"><div class="k">Tasks</div><div class="v" id="taskCount">3</div></div>
            <div class="stat"><div class="k">Submissions</div><div class="v" id="subCount">0</div></div>
            <div class="stat"><div class="k">Top Score</div><div class="v" id="topScore">0.000</div></div>
        </section>

        <section class="main">
            <article class="card">
                <div class="label">Input Panel</div>
                <div class="row">
                    <div>
                        <div class="label">Participant</div>
                        <input id="participant" placeholder="Your name" value="Guest" />
                    </div>
                    <div>
                        <div class="label">Task</div>
                        <select id="taskSelect">
                            <option value="email_triage">email_triage</option>
                            <option value="data_cleaning">data_cleaning</option>
                            <option value="customer_support">customer_support</option>
                        </select>
                    </div>
                </div>
                <div class="label" style="margin-top:10px;">Your Input</div>
                <textarea id="userInput" placeholder="Write your response or paste data here..."></textarea>
                <div class="btns">
                    <button id="exampleBtn" type="button">Load Example</button>
                    <button id="submitBtn" type="button">Evaluate</button>
                    <button id="copyBtn" type="button">Copy Output</button>
                    <button id="clearBtn" type="button">Clear</button>
                </div>
                <div class="tiny">Pro tip: aim for score >= 0.85 to appear as Excellent.</div>
            </article>

            <article class="card">
                <div class="label">Output Panel</div>
                <div id="resultSummary" class="tiny">No submission yet.</div>
                <div class="meter"><div id="scoreBar"></div></div>
                <pre id="output">Results will appear here after Evaluate.</pre>

                <div class="leader">
                    <div class="label">Leaderboard</div>
                    <ol id="leaderboard"></ol>
                </div>
            </article>
        </section>
    </main>

    <script>
        let taskMeta = {};

        function setSummary(text) {
            document.getElementById('resultSummary').textContent = text;
        }

        function setScore(score) {
            const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
            document.getElementById('scoreBar').style.width = pct + '%';
        }

        async function loadStatus() {
            const status = await fetch('/api/status').then(r => r.json());
            document.getElementById('svc').textContent = status.status;
            document.getElementById('taskCount').textContent = status.task_count;
            document.getElementById('subCount').textContent = status.history_size;
        }

        async function loadTasks() {
            taskMeta = await fetch('/api/tasks').then(r => r.json());
        }

        async function loadLeaderboard() {
            const data = await fetch('/api/leaderboard').then(r => r.json());
            const list = document.getElementById('leaderboard');
            list.innerHTML = '';
            const rows = data.leaderboard || [];
            if (!rows.length) {
                const li = document.createElement('li');
                li.textContent = 'No submissions yet';
                list.appendChild(li);
                document.getElementById('topScore').textContent = '0.000';
                return;
            }
            document.getElementById('topScore').textContent = Number(rows[0].score).toFixed(3);
            rows.slice(0, 5).forEach(row => {
                const li = document.createElement('li');
                li.textContent = row.participant + ' | ' + row.task + ' | ' + Number(row.score).toFixed(3) + ' | ' + row.grade;
                list.appendChild(li);
            });
        }

        function loadExample() {
            const task = document.getElementById('taskSelect').value;
            const ex = (taskMeta[task] && taskMeta[task].example) ? taskMeta[task].example : 'No example.';
            document.getElementById('userInput').value = ex;
            setSummary('Example loaded for ' + task + '.');
        }

        async function evaluate() {
            const task = document.getElementById('taskSelect').value;
            const participant = document.getElementById('participant').value || 'Guest';
            const userInput = document.getElementById('userInput').value;
            const output = document.getElementById('output');
            output.textContent = 'Evaluating...';

            try {
                const res = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task, participant, user_input: userInput })
                });
                const data = await res.json();
                output.textContent = JSON.stringify(data, null, 2);

                if (data.ok) {
                    setScore(Number(data.score || 0));
                    setSummary('Grade: ' + data.grade + ' | Score: ' + Number(data.score).toFixed(3));
                    await loadStatus();
                    await loadLeaderboard();
                } else {
                    setScore(0);
                    setSummary(data.error || 'Invalid submission.');
                }
            } catch (err) {
                output.textContent = 'Error: ' + err;
                setSummary('Request failed.');
            }
        }

        async function copyOutput() {
            const text = document.getElementById('output').textContent;
            await navigator.clipboard.writeText(text);
            setSummary('Output copied to clipboard.');
        }

        function clearAll() {
            document.getElementById('userInput').value = '';
            document.getElementById('output').textContent = 'Results will appear here after Evaluate.';
            setScore(0);
            setSummary('Cleared. Add new input to continue.');
        }

        document.getElementById('exampleBtn').addEventListener('click', loadExample);
        document.getElementById('submitBtn').addEventListener('click', evaluate);
        document.getElementById('copyBtn').addEventListener('click', copyOutput);
        document.getElementById('clearBtn').addEventListener('click', clearAll);

        (async function bootstrap() {
            await loadTasks();
            await loadStatus();
            await loadLeaderboard();
            setSummary('Ready. Pick a task and click Evaluate.');
        })();
    </script>
</body>
</html>
"""
