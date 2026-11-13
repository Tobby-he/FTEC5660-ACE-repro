# Reproducing ACE on the finance benchmarks (FTEC5660)

Reproducibility study of **Agentic Context Engineering (ACE)**
([arXiv 2510.04618](https://arxiv.org/abs/2510.04618), upstream repo
[ace-agent/ace](https://github.com/ace-agent/ace) @ `82709de`).

Repository: https://github.com/Tobby-he/FTEC5660-ACE-repro. All changes live on branch
`repro` (the default branch), one commit per change. Use `git log 82709de..repro` to review them.

## Target claim

Paper Table 2 (DeepSeek-V3.1, accuracy %):

| Method | GT labels | FiNER | Formula |
|---|---|---|---|
| Base LLM | – | 70.7 | 67.5 |
| ACE offline | ✓ | 78.3 | 85.5 |
| ACE offline | ✗ | 71.1 | 83.0 |
| ACE online | ✓ | 76.7 | 76.5 |
| ACE online | ✗ | 67.3 | 78.5 |

**Primary target: Formula**, base LLM vs. ACE offline (+18.0 points, the
largest gain in the table). Formula prompts are short (~0.5k characters), so the
full 200-sample test split is cheap. FiNER prompts are ~8k characters, so FiNER is
the secondary target.

Metric: exact-match accuracy computed by the repo's own evaluator
(`eval/finance/data_processor.py`). Formula answers are compared as floats.
FiNER is scored **per entity**: each sample has several tagged entities, and accuracy is
correct entities / total entities.

## Setup

```bash
git clone https://github.com/Tobby-he/FTEC5660-ACE-repro.git && cd FTEC5660-ACE-repro
python3.12 -m venv .venv
.venv/bin/pip install "openai>=1.0.0" "python-dotenv>=1.0.0" "tiktoken>=0.12.0" numpy
cp .env.example .env   # then set DEEPSEEK_API_KEY=...
```

`sentence-transformers` and `faiss-cpu` are only needed for
`--use_bulletpoint_analyzer`, which is off by default and was not used here.
To use upstream's full environment, run `uv sync` instead.

Compute: laptop CPU only, with all inference through the DeepSeek API.

### Model swap (controlled variable)

| | Paper | This reproduction |
|---|---|---|
| Model | DeepSeek-V3.1 (non-thinking) | `deepseek-chat` (non-thinking) |
| Provider / base URL | SambaNova, `https://api.sambanova.ai/v1` | DeepSeek, `https://api.deepseek.com` |
| temperature | not stated in paper | 0.0 (hard-coded in `llm.py`) |
| max_tokens | not stated | 4096 (repo default) |
| Prompts / templates | repo prompts | unchanged |

`deepseek-chat` is an alias. Every call now logs `served_model` and
`system_fingerprint`, and `scripts/summarize.py` prints the served model.

**Verified 2026-11-01:** the official API's `/models` endpoint lists only
`deepseek-flash` and `deepseek-v4-pro`. Requests for `deepseek-chat` come back
with `"model": "deepseek-flash"`, and DeepSeek-V3.1 is no longer served there.
**Results are not directly comparable to the paper because the underlying LLM
differs.** Tasks, splits, prompts and the evaluation script are identical.

## Running

Every experiment runs 3 trials. The model runs at temperature 0, but API outputs
still vary from run to run.

```bash
export PROVIDER="--api_provider deepseek --generator_model deepseek-chat --reflector_model deepseek-chat --curator_model deepseek-chat"

# E1  Base LLM (empty playbook), Formula full test split (200)
.venv/bin/python scripts/run_trials.py --tag formula_base --trials 3 -- \
    --task_name formula --mode eval_only $PROVIDER

# E2  ACE offline, Formula: train subset 200 / val subset 100 / full test 200
.venv/bin/python scripts/run_trials.py --tag formula_ace_offline --trials 3 -- \
    --task_name formula --mode offline $PROVIDER \
    --max_train 200 --max_val 100 --eval_steps 50

# E3  ACE online, Formula test split (adapts on the test stream)
.venv/bin/python scripts/run_trials.py --tag formula_ace_online --trials 3 -- \
    --task_name formula --mode online $PROVIDER

# Summary table (mean ± std, scored/requested, tokens, cost)
.venv/bin/python scripts/summarize.py results/formula_base results/formula_ace_offline results/formula_ace_online
```

E2 also records `initial_test` (empty playbook) in every trial. That gives a second
base-LLM measurement for free.

To test the pipeline without spending credits, point `DEEPSEEK_BASE_URL` at any
OpenAI-compatible mock server. All three modes were checked this way before any paid run.

## Hidden assumptions found so far

1. **Reflection rounds and epochs.** The paper sets the maximum number of Reflector rounds and
   offline epochs to **5**, but the repo defaults to `--max_num_rounds 3 --num_epochs 1`.
   The repo's defaults do not match the paper's setup.
2. **Model identity.** The paper served V3.1 through SambaNova. Whether `deepseek-chat` on
   the official API is the same weights cannot be verified, so we log what was served.
3. **Test split.** The FiNER test file is `finer_test_subset_006_seed42.jsonl`
   (441 samples). The paper only says it follows "the original splits", so it is
   unclear whether Table 2 was computed on this subset.
4. **Silently dropped samples.** In `utils.evaluate_test_set`, a sample whose
   evaluation raises an exception is skipped (`continue`). It is removed from the
   denominator instead of counted as wrong, which can inflate accuracy.
   `summarize.py` reports *scored / requested* so this is visible.
   If every sample fails, the function hits a `NameError` (`final_results` is undefined).
5. **Empty API responses** are scored as wrong using a sentinel answer (`llm.py`).
6. **Decoding:** temperature 0 is hard-coded, and the paper gives no decoding settings.

## Results

### E1: Base LLM on Formula (empty playbook, full test split, 3 trials)

| | Paper (DeepSeek-V3.1) | Ours (deepseek-chat → deepseek-flash) |
|---|---|---|
| Accuracy (%) | 67.5 | **70.3 ± 1.5** (69.0, 72.0, 70.0) |
| Samples scored | 200 | 200 / 200 in every trial |
| Tokens (3 trials) | – | 264k in (129k cache hit) / 136k out |
| Cost (3 trials) | – | ≈ $0.10 |
| Wall time per trial | – | 27–40 s (20 workers) |

**The baseline reproduces approximately.** Our mean is 2.8 points above the paper's
number, about two trial standard deviations. Given the model difference, that is
a plausible model effect, not a pipeline discrepancy.

**Non-determinism at temperature 0.** The three trials disagree. 52 questions are
wrong in all three trials and 68 are wrong in at least one, so 16 questions flip between
runs. A single-trial comparison of two variants on this split is therefore not
meaningful below about ±3 points.

**Most baseline errors are about answer conventions, not finance reasoning.**
Error types in trial 1 (62 wrong answers):

| Error type | Count | Example (pred vs gold) |
|---|---|---|
| ×100 scale: fraction given where a percent is expected (or vice versa) | 18 | `0.25` vs `25.0` |
| Within 1%: rounding/precision mismatch | 11 | `14442.35` vs `14442.36` |
| Other numeric (wrong formula or arithmetic) | 31 | `10652.59` vs `11717.85` |
| Unparseable | 2 | – |

About 47% of errors are convention mismatches with the gold labels. A playbook that
learns "report percentages as 25.0, not 0.25" can fix these without any new financial
knowledge. This is a likely mechanism for ACE's large +18.0 gain on Formula, and
the hypothesis E2 tests.

### E2: ACE offline on Formula (3 trials)

Train on a seeded subset of 200 training questions, validate on 100 every 50 steps,
then test on the full 200-question test split. Uses labels, 1 epoch, up to 3 reflection rounds.

| | Paper (DeepSeek-V3.1) | Ours (deepseek-flash) |
|---|---|---|
| Base LLM | 67.5 | 70.3 ± 1.5 (E1) |
| **ACE offline** | **85.5** | **85.5 ± 0.5** (85.0, 86.0, 85.5) |
| Gain over base | +18.0 | +15.2 |
| Best validation accuracy | – | 93.3 ± 2.1 |
| Wall time per trial | – | ~51 min (one sample at a time) |
| Cost per trial | – | ≈ $0.67 |

**ACE's headline number reproduces exactly**, even with 40% of the paper's training
questions and the repo's default 1 epoch. The *gain* is 2.8 points smaller than the
paper's only because our base model starts 2.8 points higher. Both systems end at
the same accuracy.

Every trial also re-ran the empty-playbook test before training (`initial_test`:
71.0, 70.5, 70.5). Pooled with E1 and the ablation's initial tests, the base LLM
scores **69.8 ± 1.4** over 9 runs.

**Which errors ACE fixed.** Mean wrong answers per trial, by type:

| Error type | Base (E1) | ACE offline (E2) |
|---|---|---|
| ×100 scale (percent vs fraction) | 17.7 | **0.0** |
| Rounding in last digit (within 1%) | 9.3 | 9.0 |
| Other numeric | 31.3 | 17.3 |
| Unparseable | 1.0 | 2.7 |
| **Total wrong** | **59.3** | **29.0** |

The convention hypothesis is **partly confirmed**. ACE removed every scale error in all
3 trials, about 18 of the ~30 fixed answers. The learned playbook contains
formula-specific scale rules, for example:

> `[calc-00013] … Net Profit Margin = (Net Profit / Total Revenue) × 100. Express the result as a percentage (e.g., 15.0), not as a decimal.`
> `[calc-00002] … ROI = Net Profit / Total Investment. Return the result as a decimal ratio (e.g., 0.2), not as a percentage.`

The dataset is not internally consistent about percent vs ratio, so the playbook
learns the convention **per formula**. Rounding mismatches were **not** fixed. The
playbooks repeat "round to the nearest hundredth" (55–66 bullets mention it), but
the labels' rounding is not recoverable from the question. The rest of the gain
(~14 answers) comes from formula and arithmetic corrections. Final best playbooks hold
84–100 bullets and about 9–11k tokens.

## Modification: remove the Reflector (`--no_reflector`)

**Change** (one commit, one flag, off by default): in single-sample training the
Reflector is skipped. The Curator instead receives the raw attempt: the question,
the generator's reasoning and answer, the label, and a match or mismatch signal. Removing
the Reflector necessarily also removes (a) its error diagnosis, (b) the up-to-3
reflect-and-retry rounds and (c) helpful/harmful counter updates on bullets.
Everything else is identical to E2: data subsets, seed, epochs, validation schedule
and test split.

**Why:** the paper ablates the Reflector only on AppWorld (Table 3), where
it adds 1.7 points on average (55.1 → 56.8). It is never measured on the finance
tasks, so this is a new measurement, not a reproduction. Our implementation
of "without Reflector" is our own interpretation.

```bash
.venv/bin/python scripts/run_trials.py --tag formula_ace_offline_noref --trials 3 -- \
    --task_name formula --mode offline $PROVIDER \
    --max_train 200 --max_val 100 --eval_steps 50 --no_reflector
```

| Variant (3 trials each) | Test acc. % | Best val % | LLM calls / trial | Wall time / trial | Cost / trial |
|---|---|---|---|---|---|
| Base LLM (E1) | 70.3 ± 1.5 | – | 200 | ~0.5 min | $0.03 |
| ACE offline, full (E2) | **85.5 ± 0.5** | 93.3 ± 2.1 | 1,645 | ~51 min | $0.67 |
| ACE offline, no Reflector | **85.0 ± 1.0** | 93.7 ± 0.6 | 1,400 | ~26 min | $0.53 |

Call counts include the before and after test passes and validation passes, which are the same
for both variants.

**Result: removing the Reflector makes no measurable difference to accuracy on
Formula.** The difference is 0.5 points, smaller than either variant's trial spread.
It is about **21% cheaper** and **about 2× faster** in wall time. The no-Reflector
variant also eliminates all ×100 scale errors (0 in every trial) and leaves the same ~9–10
rounding errors. The raw label is enough for the Curator to learn
the dataset's per-formula conventions. On this task, the Reflector's diagnosis and retries
buy nothing that a direct "here is the attempt, here is the answer" signal doesn't.

Caveat: this holds for Formula, with labels, 1 epoch and this model. It does not
contradict the paper's AppWorld result, where trajectories are long multi-step tool
use and a diagnosis plausibly matters more.

## Cost summary

| Experiment | Runs | LLM calls | Input tokens (cache hit) | Output tokens | Cost (USD) |
|---|---|---|---|---|---|
| E1 base | 3 | 600 | 0.26M (0.13M) | 0.14M | 0.10 |
| E2 ACE offline | 3 | 4,935 | 29.3M (21.4M) | 1.29M | 2.02 |
| Modification, no Reflector | 3 | 4,200 | 30.4M (24.4M) | 1.05M | 1.60 |
| **Total** | 9 | 9,735 | 60.0M (46.0M) | 2.47M | **3.72** |

Priced per call from the logs at deepseek-flash rates. All runs fell in DeepSeek's
half-price off-peak hours. Context caching matters: 77% of input tokens were
cache hits, billed at 2% of the normal rate, because the playbook prefix repeats
across calls.

## Conclusions

- **Reproduced:** base accuracy (70.3 vs 67.5) and ACE offline accuracy (85.5 vs
  85.5) on Formula, with a different model, 40% of the training data and 1 epoch, for
  under $4.
- **Partly reproduced:** the size of the gain (+15.2 vs +18.0). The difference
  is fully explained by a stronger base model.
- **Not reproducible as specified:** the exact model (V3.1 is no longer on the official
  API) and the paper's setup (up to 5 epochs and rounds; the repo defaults differ).
- **New finding:** on Formula the Reflector is not needed. A Curator fed the raw
  attempt and label matches full ACE at about half the time and 79% of the cost.
  Most of ACE's gain here comes from learning answer conventions (all scale errors fixed)
  and correcting formulas.

## Debug diary

- `uv` was not installed. Used a plain venv with only the core dependencies, because the heavy
  embedding dependencies are only imported by the optional bullet-point analyzer.
- No DeepSeek provider upstream. Added `deepseek` to `initialize_clients`
  (OpenAI-compatible, base URL overridable).
- Full offline runs are expensive because the playbook (up to an 80k-token budget) is
  re-sent on every call. Added seeded subset flags so every trial and variant
  sees identical samples.
- `deepseek-chat` turned out to be served as `deepseek-flash`, and V3.1 is gone from the
  API. Added per-call logging of the served model, and documented the swap as a controlled variable.
- The cost script's first prices were guesses that underpriced output tokens 3×.
  Re-priced from DeepSeek's pricing page, per call, with peak and off-peak rates.
- Offline training is sequential (~15 s per step), so 3 trials take ~2.5 h back to back.
  Ran the trials as parallel processes (separate folders, same seed and subsets), which
  costs the same and finishes in ~50 min.
- My first cost estimate ($6–15 for E2) was ~4× too high. Measuring a cheap baseline
  first and pricing from logs avoided over-cutting the experiment.
