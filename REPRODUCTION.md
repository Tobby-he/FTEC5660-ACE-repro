# Reproducing ACE on the finance benchmarks (FTEC5660)

Reproducibility study of **Agentic Context Engineering (ACE)**
([arXiv 2510.04618](https://arxiv.org/abs/2510.04618), upstream repo
[ace-agent/ace](https://github.com/ace-agent/ace) @ `82709de`).

All changes live on branch `repro`, one commit per change. Use `git log 82709de..repro` to review them.

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
git clone <this repo> && cd <repo> && git checkout repro
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
the hypothesis E2 will test.

## Modification

_To fill in: one isolated change, its motivation, and the before/after numbers._

## Debug diary

- `uv` was not installed. Used a plain venv with only the core dependencies, because the heavy
  embedding dependencies are only imported by the optional bullet-point analyzer.
- No DeepSeek provider upstream. Added `deepseek` to `initialize_clients`
  (OpenAI-compatible, base URL overridable).
- Full offline runs are expensive because the playbook (up to an 80k-token budget) is
  re-sent on every call. Added seeded subset flags so every trial and variant
  sees identical samples.
