FROM python:3.11-slim

LABEL maintainer="your-username"
LABEL description="HospitalSchedulingEnv - OpenEnv Hospital Resource Scheduling"

WORKDIR /app

# No external deps — pure stdlib
COPY environment.py tasks.py baseline.py openenv.yaml README.md ./

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Default: run benchmark and expose results via simple HTTP server
CMD ["python", "-c", "\
import json, http.server, threading, os; \
from baseline import benchmark; \
print('Running baseline benchmark...'); \
results = benchmark(n_seeds=5); \
with open('baseline_results.json','w') as f: json.dump(results, f, indent=2); \
print('Results written to baseline_results.json'); \
print(json.dumps(results, indent=2)); \
"]
