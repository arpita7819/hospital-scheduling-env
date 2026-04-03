"""
server/app.py — OpenEnv server entry point.
Called via: python -m server.app  or  the 'server' script in pyproject.toml
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading
import http.server
from urllib.parse import urlparse
from datetime import datetime
from environment import HospitalSchedulingEnv
from tasks import load_task

_lock = threading.Lock()
_env = None
_task = None
_trajectory = []
_results = None


def get_env():
    global _env, _task
    if _env is None:
        _env, _task = load_task("hard")
        _env.reset(seed=42)
    return _env, _task


def run_benchmark():
    global _results
    from baseline import benchmark
    _results = benchmark(n_seeds=5)
    print("Benchmark complete.")


threading.Thread(target=run_benchmark, daemon=True).start()


class Handler(http.server.BaseHTTPRequestHandler):

    def _send(self, code, data, content_type="application/json"):
        body = (json.dumps(data) if content_type == "application/json"
                else data).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/web"):
            html = self._dashboard()
            self._send(200, html, "text/html; charset=utf-8")

        elif path == "/state":
            with _lock:
                env, _ = get_env()
                self._send(200, {"observation": env.state()})

        elif path == "/validate":
            self._send(200, {
                "valid": True,
                "name": "HospitalSchedulingEnv",
                "version": "1.0.0",
                "sdk": "openenv",
                "endpoints": ["/reset", "/step", "/state", "/validate"],
                "tasks": ["basic_triage", "full_scheduling", "crisis_surge"],
                "observation_space": "Dict[str, Any]",
                "action_space": "Dict[str, int]",
            })

        elif path == "/health":
            self._send(200, {"status": "ok"})

        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/reset":
            body = self._body()
            seed = body.get("seed", 42)
            difficulty = body.get("difficulty", "hard")
            with _lock:
                global _env, _task, _trajectory
                _env, _task = load_task(difficulty)
                obs = _env.reset(seed=seed)
                _trajectory = []
            self._send(200, {"observation": obs, "seed": seed, "difficulty": difficulty})

        elif path == "/step":
            body = self._body()
            action = {
                "patient_id":        body.get("patient_id", -1),
                "assign_bed_type":   body.get("assign_bed_type", 2),
                "assign_or":         body.get("assign_or", 0),
                "assign_ventilator": body.get("assign_ventilator", 0),
            }
            with _lock:
                env, task = get_env()
                obs, reward, done, info = env.step(action)
                _trajectory.append({
                    "obs": obs, "action": action,
                    "reward": reward, "info": info,
                })
                score = task.grader(env, _trajectory) if done else None
            self._send(200, {
                "observation": obs,
                "reward": reward,
                "done": done,
                "info": info,
                "score": score,
            })

        else:
            self._send(404, {"error": "not found"})

    def _dashboard(self):
        if _results is None:
            return """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="4">
<title>Hospital Scheduling Env</title>
<style>body{font-family:monospace;background:#0d1117;color:#e6edf3;
max-width:800px;margin:60px auto;text-align:center}h1{color:#58a6ff}
.sp{display:inline-block;width:36px;height:36px;border:4px solid #30363d;
border-top-color:#58a6ff;border-radius:50%;animation:spin 1s linear infinite;margin:20px}
@keyframes spin{to{transform:rotate(360deg)}}p{color:#8b949e}</style>
</head><body><h1>🏥 Hospital Scheduling Env</h1>
<div class="sp"></div><p>Running benchmark...</p></body></html>"""

        rows = ""
        for diff in ["easy", "medium", "hard"]:
            col = {"easy": "#3fb950", "medium": "#d29922", "hard": "#f85149"}[diff]
            best = max(v["mean"] for v in _results[diff].values())
            for agent in ["random", "greedy_fifo", "heuristic"]:
                s = _results[diff][agent]
                bold = "font-weight:bold;color:#58a6ff" if abs(s["mean"] - best) < 0.001 else ""
                rows += (f'<tr><td style="color:{col}">{diff}</td><td>{agent}</td>'
                         f'<td style="{bold}">{s["mean"]:.4f}</td>'
                         f'<td>{s["std"]:.4f}</td></tr>')

        return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Hospital Scheduling Env</title>
<style>body{{font-family:monospace;background:#0d1117;color:#e6edf3;
max-width:860px;margin:40px auto;padding:20px}}
h1{{color:#58a6ff}}h2{{color:#79c0ff;border-bottom:1px solid #21262d;padding-bottom:6px}}
p{{color:#8b949e}}table{{border-collapse:collapse;width:100%}}
th{{background:#161b22;color:#8b949e;padding:8px 16px;text-align:left;
border-bottom:1px solid #30363d;font-size:12px}}
td{{padding:8px 16px;border-bottom:1px solid #161b22;font-size:13px}}
pre{{background:#161b22;padding:16px;border-radius:8px;font-size:12px}}
.badge{{background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb;
border-radius:6px;padding:2px 10px;font-size:12px;margin-left:8px}}</style>
</head><body>
<h1>🏥 Hospital Scheduling Env <span class="badge">Running</span></h1>
<p>OpenEnv hospital resource scheduling · {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}</p>
<h2>Baseline benchmark</h2>
<table><tr><th>Difficulty</th><th>Agent</th><th>Mean score</th><th>Std</th></tr>
{rows}</table>
<h2>API endpoints</h2>
<pre>POST /reset     → reset environment
POST /step      → take action
GET  /state     → current observation
GET  /validate  → OpenEnv compliance</pre>
</body></html>"""

    def log_message(self, fmt, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}]", fmt % args)


def main():
    print(f"===== Hospital Scheduling Env — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    print("Server starting on http://0.0.0.0:7860")
    http.server.HTTPServer(("0.0.0.0", 7860), Handler).serve_forever()


if __name__ == "__main__":
    main()
