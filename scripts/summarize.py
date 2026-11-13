#!/usr/bin/env python3
"""
Summarize trials written by scripts/run_trials.py into a markdown table.

Usage (from repo root):
    python scripts/summarize.py results/formula_base results/formula_ace_offline

For each experiment folder: per-trial accuracies, mean ± sample std, how many
test samples were actually scored (the upstream evaluator silently drops samples
whose evaluation raised), served model names, token usage and estimated cost.

Prices are USD per 1M tokens at DeepSeek's PEAK rate and default to deepseek-flash
(https://api-docs.deepseek.com/quick_start/pricing, read 2026-11-13). Calls made
off-peak are billed at half. Peak is 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri.
Chinese public holidays are also off-peak but are not detected here. Override the
rates with --price_in_miss / --price_in_hit / --price_out.
"""
import argparse
import glob
import json
import os
import statistics
from datetime import datetime, timezone


def is_peak(local_iso):
    """True if a call logged at local_iso (naive local time) fell in DeepSeek peak hours."""
    t = datetime.fromisoformat(local_iso).astimezone(timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


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


def usage(run_dir, prices):
    """Return (calls, prompt, cache_hit, completion, cost_usd, served_models) for one run."""
    price_miss, price_hit, price_out = prices
    calls = prompt = hit = completion = 0
    cost = 0.0
    served = set()
    for path in glob.glob(os.path.join(run_dir, "detailed_llm_logs", "*.json")):
        with open(path) as f:
            c = json.load(f)
        if "prompt_num_tokens" not in c:
            continue
        p, h, o = c["prompt_num_tokens"] or 0, c.get("prompt_cache_hit_tokens") or 0, c.get("response_num_tokens") or 0
        rate = 1.0 if is_peak(c["datetime"]) else 0.5
        cost += rate * ((p - h) * price_miss + h * price_hit + o * price_out) / 1e6
        calls += 1
        prompt += p
        hit += h
        completion += o
        if c.get("served_model"):
            served.add(c["served_model"])
    return calls, prompt, hit, completion, cost, served


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
    p.add_argument("--price_in_miss", type=float, default=0.3)
    p.add_argument("--price_in_hit", type=float, default=0.006)
    p.add_argument("--price_out", type=float, default=1.2)
    p.add_argument("--include_unfinished", action="store_true",
                   help="Also count tokens/cost of runs still in progress (for cost projection)")
    args = p.parse_args()
    prices = (args.price_in_miss, args.price_in_hit, args.price_out)

    print("| Experiment | Metric | Trials (%) | Mean ± std (%) | Scored / requested | Calls | Tokens in (cache hit) / out | Est. cost (USD) | Served model |")
    print("|---|---|---|---|---|---|---|---|---|")
    for exp in args.experiments:
        trials = sorted(glob.glob(os.path.join(exp, "trial_*")))
        per_metric, tot = {}, [0, 0, 0, 0, 0.0]
        served = set()
        n_unfinished = 0
        for t in trials:
            for run_dir in find_run_dirs(t):
                finished = os.path.exists(os.path.join(run_dir, "final_results.json"))
                if not finished and not args.include_unfinished:
                    continue
                if finished:
                    for name, (acc, scored, expected) in accuracies(run_dir).items():
                        per_metric.setdefault(name, []).append((acc, scored, expected))
                else:
                    n_unfinished += 1
                *counts, s = usage(run_dir, prices)
                for i, v in enumerate(counts):
                    tot[i] += v
                served |= s
        n_runs = max((len(v) for v in per_metric.values()), default=0)
        calls, prompt, hit, completion, cost = tot
        if not per_metric and n_unfinished:
            per_metric["(in progress)"] = [(None, None, None)]
        first = True
        for name, rows in per_metric.items():
            accs = [a for a, _, _ in rows]
            scored = ", ".join(f"{s}/{e}" if e else str(s) for _, s, e in rows if s is not None) or "-"
            usage_cols = (f"{calls} | {prompt:,} ({hit:,}) / {completion:,} | {cost:.2f} (total, {n_runs} runs{f' + {n_unfinished} unfinished' if n_unfinished else ''}) | {', '.join(sorted(served)) or '-'}"
                          if first else " | | | ")
            print(f"| {os.path.basename(exp.rstrip('/'))} | {name} | "
                  f"{', '.join(f'{100 * a:.1f}' for a in accs if a is not None)} | {fmt_mean(accs)} | {scored} | {usage_cols} |")
            first = False


if __name__ == "__main__":
    main()
