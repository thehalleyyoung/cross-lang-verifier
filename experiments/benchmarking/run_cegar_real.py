#!/usr/bin/env python3
"""Run CEGAR experiment with real LLM calls."""
import sys, json, time, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get API key
if not os.environ.get("OPENAI_API_KEY"):
    key = os.popen("source ~/.bashrc && echo $OPENAI_API_KEY").read().strip()
    if key:
        os.environ["OPENAI_API_KEY"] = key

from src.cegar_engine import CEGAREngine, CEGARResult, analyze_cegar_results
from src.oracle.oracle import VerificationOracle
from benchmarks.pairs.benchmark_pairs import get_all_pairs
from benchmarks.pairs.scaled_benchmark_pairs import get_scaled_pairs

# Get functions that we know produce divergent results (these need CEGAR repair)
oracle = VerificationOracle(timeout_ms=5000)
core_pairs = get_all_pairs()
scaled_pairs = get_scaled_pairs()
all_pairs = core_pairs + scaled_pairs

# Pre-filter: only use pairs where our oracle can handle the C code
usable_c_funcs = []
for p in all_pairs:
    try:
        r = oracle.verify(p.c_source, "pub fn dummy(x: i32) -> i32 { x }", p.name)
        if r.verdict != "error":
            usable_c_funcs.append((p.name, p.c_source, p.category, p.expected_result))
    except:
        pass

print(f"Usable C functions: {len(usable_c_funcs)}")

# Select a representative sample (up to 40 functions)
from collections import defaultdict
by_cat = defaultdict(list)
for name, code, cat, exp in usable_c_funcs:
    by_cat[cat].append((name, code, cat, exp))

selected = []
for cat, funcs in sorted(by_cat.items()):
    selected.extend(funcs[:5])

selected = selected[:40]
print(f"Selected {len(selected)} functions for CEGAR across {len(set(c for _,_,c,_ in selected))} categories")

# Run with gpt-4.1-nano
print("\n=== CEGAR with gpt-4.1-nano ===")
engine_nano = CEGAREngine(model="gpt-4.1-nano", max_iterations=5, timeout_ms=5000)
nano_results = []
for i, (name, code, cat, exp) in enumerate(selected):
    print(f"  [{i+1}/{len(selected)}] {name} ({cat})...", end=" ", flush=True)
    try:
        r = engine_nano.run(code, name)
        status = "CONVERGED" if r.converged else r.final_verdict
        print(f"{status} ({r.total_iterations} iters, {r.total_time_ms:.0f}ms)")
        nano_results.append(r)
    except Exception as e:
        print(f"ERROR: {e}")
        nano_results.append(CEGARResult(func_name=name, c_code=code, converged=False, final_verdict="error"))

nano_analysis = analyze_cegar_results(nano_results)
print(f"\nNano summary: {nano_analysis['converged']}/{nano_analysis['total_pairs']} converged ({nano_analysis['convergence_rate']}%)")

# Run with gpt-5-chat-latest (smaller sample)
print("\n=== CEGAR with gpt-5-chat-latest ===")
engine_strong = CEGAREngine(model="gpt-5-chat-latest", max_iterations=5, timeout_ms=5000)
strong_selected = selected[:20]
strong_results = []
for i, (name, code, cat, exp) in enumerate(strong_selected):
    print(f"  [{i+1}/{len(strong_selected)}] {name} ({cat})...", end=" ", flush=True)
    try:
        r = engine_strong.run(code, name)
        status = "CONVERGED" if r.converged else r.final_verdict
        print(f"{status} ({r.total_iterations} iters, {r.total_time_ms:.0f}ms)")
        strong_results.append(r)
    except Exception as e:
        print(f"ERROR: {e}")
        strong_results.append(CEGARResult(func_name=name, c_code=code, converged=False, final_verdict="error"))

strong_analysis = analyze_cegar_results(strong_results)
print(f"\nStrong model summary: {strong_analysis['converged']}/{strong_analysis['total_pairs']} converged ({strong_analysis['convergence_rate']}%)")

# Naive retry baseline (no counterexample feedback)
print("\n=== Naive Retry baseline (no counterexample feedback) ===")
from openai import OpenAI
client = OpenAI()

naive_converged = 0
naive_total = 0
naive_results_data = []
for i, (name, code, cat, exp) in enumerate(selected[:20]):
    naive_total += 1
    best_verdict = "unknown"
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "Translate this C function to semantically equivalent Rust. Use wrapping arithmetic where C has UB. Return ONLY the Rust function in a ```rust``` code block."},
                    {"role": "user", "content": f"```c\n{code}\n```"}
                ],
                temperature=0.7 + attempt * 0.1,
                max_tokens=512,
            )
            rust = resp.choices[0].message.content or ""
            m = re.search(r'```rust\s*(.*?)```', rust, re.DOTALL)
            if m:
                rust_code = m.group(1).strip()
            else:
                rust_code = rust.strip()
            
            r = oracle.verify(code, rust_code, name)
            if r.verdict == "equivalent":
                best_verdict = "equivalent"
                break
            elif r.verdict == "divergent":
                best_verdict = "divergent"
        except Exception as e:
            pass
    
    matched = best_verdict == "equivalent"
    if matched:
        naive_converged += 1
    naive_results_data.append({"name": name, "verdict": best_verdict, "converged": matched})
    print(f"  [{i+1}/20] {name}: {best_verdict}")

print(f"\nNaive retry: {naive_converged}/{naive_total} converged ({round(naive_converged/max(naive_total,1)*100,1)}%)")

# Save all results
all_results = {
    "cegar_nano": {
        "model": "gpt-4.1-nano",
        "n_functions": len(selected),
        "analysis": nano_analysis,
        "per_function": [r.to_dict() for r in nano_results],
    },
    "cegar_strong": {
        "model": "gpt-5-chat-latest", 
        "n_functions": len(strong_selected),
        "analysis": strong_analysis,
        "per_function": [r.to_dict() for r in strong_results],
    },
    "naive_retry": {
        "model": "gpt-4.1-nano",
        "n_functions": naive_total,
        "converged": naive_converged,
        "convergence_rate": round(naive_converged/max(naive_total,1)*100, 1),
        "per_function": naive_results_data,
    },
    "comparison": {
        "cegar_nano_rate": nano_analysis['convergence_rate'],
        "cegar_strong_rate": strong_analysis['convergence_rate'],
        "naive_retry_rate": round(naive_converged/max(naive_total,1)*100, 1),
    }
}

with open("experiments/results/cegar_experiment_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n=== COMPARISON ===")
print(f"CEGAR (nano):   {nano_analysis['convergence_rate']}%")
print(f"CEGAR (strong): {strong_analysis['convergence_rate']}%") 
print(f"Naive retry:    {round(naive_converged/max(naive_total,1)*100,1)}%")
print("\nResults saved to experiments/results/cegar_experiment_results.json")
