#!/usr/bin/env python3
"""
Run the finance runner N times with identical arguments, one folder per trial.

Usage (from repo root):
    python scripts/run_trials.py --tag formula_base --trials 3 -- \
        --task_name formula --mode eval_only --api_provider deepseek \
        --generator_model deepseek-chat --max_test 100

Everything after `--` is passed to `python -m eval.finance.run`; --save_path is
set per trial to results/<tag>/trial_<k>. Summarize with scripts/summarize.py.
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("Separate runner args from eval.finance.run args with `--`")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Experiment name, e.g. formula_base")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--start", type=int, default=1, help="First trial index (to resume)")
    parser.add_argument("--results_root", default="results")
    args = parser.parse_args(argv[:split])
    run_args = argv[split + 1:]
    if "--save_path" in run_args:
        sys.exit("Do not pass --save_path; it is set per trial")

    for k in range(args.start, args.start + args.trials):
        save_path = os.path.join(args.results_root, args.tag, f"trial_{k}")
        os.makedirs(save_path, exist_ok=True)
        cmd = [sys.executable, "-m", "eval.finance.run", *run_args, "--save_path", save_path]
        print(f"\n### [{args.tag}] trial {k}: {' '.join(cmd)}\n", flush=True)
        t0 = time.time()
        with open(os.path.join(save_path, "stdout.log"), "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                log.write(line)
                if any(k in line for k in ("Accuracy", "ERROR", "Error", "Traceback", "RUN COMPLETE")):
                    print(line, end="", flush=True)
            proc.wait()
        print(f"### [{args.tag}] trial {k} exited {proc.returncode} after {time.time() - t0:.0f}s", flush=True)
        if proc.returncode != 0:
            sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
