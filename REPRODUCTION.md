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

`deepseek-chat` is an alias, and DeepSeek may serve a newer checkpoint than V3.1
under it. Every call now logs `served_model` and `system_fingerprint`, and
`scripts/summarize.py` prints the served model. **Results are not directly
comparable to the paper because the underlying LLM may differ.** Tasks, splits
and the evaluation script are identical.

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

_To fill in after runs: paste the `summarize.py` table and compare it with Table 2._

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
