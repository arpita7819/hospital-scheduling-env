import json
from baseline import benchmark

print("Running baseline benchmark...")

results = benchmark(n_seeds=5)

with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results written to baseline_results.json")
print(json.dumps(results, indent=2))