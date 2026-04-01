"""
app.py — Hugging Face Spaces entry point.
Starts web server immediately on port 7860, then runs benchmark in background.
"""
import json
import http.server
import threading
from datetime import datetime

results = None
status = "running"

HTML_LOADING = """<!DOCTYPE html>
<html> 
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <title>Hospital Scheduling Env</title>
  <style>
    body { font-family: monospace; background: #0d1117; color: #e6edf3;
           max-width: 800px; margin: 60px auto; padding: 20px; text-align: center; }
    h1   { color: #58a6ff; }
    .spinner { display: inline-block; width: 40px; height: 40px;
               border: 4px solid #30363d; border-top-color: #58a6ff;
               border-radius: 50%; animation: spin 1s linear infinite; margin: 20px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    p { color: #8b949e; }
  </style>
</head>
<body>
  <h1>🏥 Hospital Scheduling Env</h1>
  <div class="spinner"></div>
  <p>Running baseline benchmark...</p>
  <p style="font-size:12px">This page refreshes automatically every 5 seconds.</p>
</body>
</html>"""

def make_results_page(res):
    def rows():
        r = ""
        agent_order = ["random", "greedy_fifo", "heuristic"]
        diff_color = {"easy": "#3fb950", "medium": "#d29922", "hard": "#f85149"}
        for diff in ["easy", "medium", "hard"]:
            agents = res[diff]
            best = max(v["mean"] for v in agents.values())
            for agent in agent_order:
                s = agents[agent]
                bold = "font-weight:bold;color:#58a6ff" if abs(s["mean"] - best) < 0.001 else ""
                r += f'<tr><td style="color:{diff_color[diff]}">{diff}</td><td>{agent}</td>'
                r += f'<td style="{bold}">{s["mean"]:.4f}</td><td>{s["std"]:.4f}</td></tr>\n'
        return r

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hospital Scheduling Env</title>
  <style>
    body {{ font-family: monospace; background: #0d1117; color: #e6edf3;
           max-width: 860px; margin: 40px auto; padding: 20px; }}
    h1   {{ color: #58a6ff; margin-bottom: 4px; }}
    h2   {{ color: #79c0ff; margin-top: 32px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
    p    {{ color: #8b949e; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th   {{ background: #161b22; color: #8b949e; padding: 8px 16px;
            text-align: left; border-bottom: 1px solid #30363d; font-size: 12px; }}
    td   {{ padding: 8px 16px; border-bottom: 1px solid #161b22; font-size: 13px; }}
    pre  {{ background: #161b22; padding: 16px; border-radius: 8px;
            overflow: auto; font-size: 13px; color: #e6edf3; }}
    .tag {{ display:inline-block; padding:2px 10px; border-radius:12px;
            font-size:11px; margin-right:6px; }}
    .easy   {{ background:#1a4728; color:#3fb950; }}
    .medium {{ background:#3d2f00; color:#d29922; }}
    .hard   {{ background:#3d1c1c; color:#f85149; }}
    .badge  {{ display:inline-block; background:#1f6feb22; color:#58a6ff;
               border:1px solid #1f6feb; border-radius:6px;
               padding:2px 10px; font-size:12px; margin-left:8px; }}
  </style>
</head>
<body>
  <h1>🏥 Hospital Scheduling Env <span class="badge">Running</span></h1>
  <p>OpenEnv-compliant hospital resource scheduling · Benchmark ran at {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>

  <h2>Baseline benchmark results</h2>
  <p>5 seeds × 3 agents × 3 difficulty levels</p>
  <table>
    <tr><th>Difficulty</th><th>Agent</th><th>Mean score</th><th>Std</th></tr>
    {rows()}
  </table>

  <h2>Tasks</h2>
  <p>
    <span class="tag easy">Easy</span> <code>basic_triage</code> — 48h episode, bed-type triage<br><br>
    <span class="tag medium">Medium</span> <code>full_scheduling</code> — 96h, all resources + staff<br><br>
    <span class="tag hard">Hard</span> <code>crisis_surge</code> — 168h surge, scarce ICU + ventilators
  </p>

  <h2>Quick start</h2>
  <pre>from tasks import load_task

env, task = load_task("hard")
obs = env.reset(seed=42)
done, trajectory = False, []

while not done:
    action = {{"patient_id": -1, "assign_bed_type": 2,
               "assign_or": 0, "assign_ventilator": 0}}
    obs, reward, done, info = env.step(action)

score = task.grader(env, trajectory)
print(f"Score: {{score:.4f}}")</pre>

  <h2>Action space</h2>
  <table>
    <tr><th>Field</th><th>Values</th><th>Description</th></tr>
    <tr><td>patient_id</td><td>int or -1</td><td>Patient to schedule (-1 = no-op)</td></tr>
    <tr><td>assign_bed_type</td><td>0 / 1 / 2</td><td>ICU / General / None</td></tr>
    <tr><td>assign_or</td><td>0 / 1</td><td>Schedule OR room</td></tr>
    <tr><td>assign_ventilator</td><td>0 / 1</td><td>Assign ventilator</td></tr>
  </table>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if results is None:
            page = HTML_LOADING
        else:
            page = make_results_page(results)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())

    def log_message(self, fmt, *args):
        pass  # suppress noisy access logs


def run_benchmark():
    global results, status
    print("=" * 60)
    print(f"  Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    from baseline import benchmark, print_results
    res = benchmark(n_seeds=5)
    print_results(res)
    with open("baseline_results.json", "w") as f:
        json.dump(res, f, indent=2)
    results = res
    status = "done"
    print("\nBenchmark complete. Results available at http://0.0.0.0:7860")


# Start benchmark in background so server comes up immediately
threading.Thread(target=run_benchmark, daemon=True).start()

# Server starts right away — HF sees port 7860 open within milliseconds
print("Server starting on http://0.0.0.0:7860")
http.server.HTTPServer(("0.0.0.0", 7860), Handler).serve_forever()
