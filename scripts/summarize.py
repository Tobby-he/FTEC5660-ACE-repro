#!/usr/bin/env python3
"""
Summarize trials written by scripts/run_trials.py into a markdown table.

Usage (from repo root):
    python scripts/summarize.py results/formula_base results/formula_ace_offline

For each experiment folder: per-trial accuracies, mean ± sample std, how many
test samples were actually scored (the upstream evaluator silently drops samples
whose evaluation raised), served model names, token usage and estimated cost.

Prices are USD per 1M tokens and default to DeepSeek's published deepseek-chat
pricing at the time of writing; check https://api-docs.deepseek.com/quick_start/pricing
and override with --price_in_miss / --price_in_hit / --price_out.
"""
import argparse
import glob
import json
import os
import statistics


def find_run_dirs(trial_dir):
    return sorted(glob.glob(os.path.join(trial_dir, "ace_run_*")))


def accuracies(run_dir):
    """Map metric name -> (accuracy, scored, expected) from final_results.json."""
    with open(os.path.join(run_dir, "final_results.json")) as f:
        res = json.load(f)
    with open(os.path.join(run_dir, "run_config.json")) as f:
        cfg = json.load(f)["config"]
    out = {}
    for key, name in [("test_results", "test"),
                      ("initial_test_results", "initial_test"),
                      ("final_test_results", "final_test"),
                      ("online_test_results", "online_test")]:
        if key in res:
            r = res[key]
            out[name] = (r.get("accuracy"), r.get("total"), cfg.get("max_test"))
    if "training_results" in res:
        out["best_val"] = (res["training_results"].get("best_validation_accuracy"), None, None)
    return out


def usage(run_dir):
    calls = prompt = hit = completion = 0
    served = set()
    for path in glob.glob(os.path.join(run_dir, "detailed_llm_logs", "*.json")):
        with open(path) as f:
            c = json.load(f)
        if "prompt_num_tokens" not in c:
            continue
        calls += 1
        prompt += c["prompt_num_tokens"] or 0
        completion += c.get("response_num_tokens") or 0
        hit += c.get("prompt_cache_hit_tokens") or 0
        if c.get("served_model"):
            served.add(c["served_model"])
    return calls, prompt, hit, completion, served


def fmt_mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{100 * xs[0]:.1f}"
    return f"{100 * statistics.mean(xs):.1f} ± {100 * statistics.stdev(xs):.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiments", nargs="+", help="results/<tag> folders")
    p.add_argument("--price_in_miss", type=float, default=0.28)
    p.add_argument("--price_in_hit", type=float, default=0.028)
    p.add_argument("--price_out", type=float, default=0.42)
    args = p.parse_args()

    print("| Experiment | Metric | Trials (%) | Mean ± std (%) | Scored / requested | Calls | Tokens in (cache hit) / out | Est. cost (USD) | Served model |")
    print("|---|---|---|---|---|---|---|---|---|")
    for exp in args.experiments:
        trials = sorted(glob.glob(os.path.join(exp, "trial_*")))
        per_metric, tot = {}, [0, 0, 0, 0]
        served = set()
        for t in trials:
            for run_dir in find_run_dirs(t):
                if not os.path.exists(os.path.join(run_dir, "final_results.json")):
                    continue  # unfinished run
                for name, (acc, scored, expected) in accuracies(run_dir).items():
                    per_metric.setdefault(name, []).append((acc, scored, expected))
                calls, prompt, hit, completion, s = usage(run_dir)
                for i, v in enumerate((calls, prompt, hit, completion)):
                    tot[i] += v
                served |= s
        n_runs = max((len(v) for v in per_metric.values()), default=0)
        calls, prompt, hit, completion = tot
        cost = ((prompt - hit) * args.price_in_miss + hit * args.price_in_hit
                + completion * args.price_out) / 1e6
        first = True
        for name, rows in per_metric.items():
            accs = [a for a, _, _ in rows]
            scored = ", ".join(f"{s}/{e or '?'}" for _, s, e in rows if s is not None) or "-"
            usage_cols = (f"{calls} | {prompt:,} ({hit:,}) / {completion:,} | {cost:.2f} (total, {n_runs} runs) | {', '.join(sorted(served)) or '-'}"
                          if first else " | | | ")
            print(f"| {os.path.basename(exp.rstrip('/'))} | {name} | "
                  f"{', '.join(f'{100 * a:.1f}' for a in accs if a is not None)} | {fmt_mean(accs)} | {scored} | {usage_cols} |")
            first = False


if __name__ == "__main__":
    main()
