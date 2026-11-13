# FTEC5660 Reproducibility: Agentic Context Engineering (ACE) on XBRL Formula

Reproduction of **Agentic Context Engineering** (Zhang et al., 2025,
[arXiv 2510.04618](https://arxiv.org/abs/2510.04618)) using the authors' code
([ace-agent/ace](https://github.com/ace-agent/ace) @ `82709de`) with the DeepSeek API.
The target is the Formula column of the paper's Table 2 (finance), plus one modification:
**removing the Reflector**.

## Results at a glance

Accuracy on the full 200-question Formula test split, mean ± std over 3 trials:

| Method | Paper (DeepSeek-V3.1) | This reproduction (deepseek-flash) |
|---|---|---|
| Base LLM (empty playbook) | 67.5 | **70.3 ± 1.5** |
| ACE offline (with labels) | 85.5 | **85.5 ± 0.5** |
| ACE offline, **no Reflector** (my modification) | not reported | **85.0 ± 1.0** |

- **ACE's headline number reproduces exactly**, using 40% of the training questions and 1 epoch.
- **Removing the Reflector makes no measurable difference** on Formula (inside the trial spread). It runs in about half the time at 79% of the cost.
- ACE eliminates every ×100 percent/fraction error (≈18 per run → 0). Rounding mismatches (≈9 per run) stay.
- **Total API cost of all 9 runs: $3.72.**

> Results are not directly comparable to the paper, because the underlying LLM differs.
> DeepSeek-V3.1 is no longer served by the official API, and `deepseek-chat` is answered
> by `deepseek-flash`.

## Quick start

**1. Install** (Python 3.10+; tested with 3.12)

```bash
git clone https://github.com/Tobby-he/FTEC5660-ACE-repro.git
cd FTEC5660-ACE-repro
python3 -m venv .venv
.venv/bin/pip install "openai>=1.0.0" "python-dotenv>=1.0.0" "tiktoken>=0.12.0" numpy
```

**2. Add your API key.** `.env` is gitignored.

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY=...
```

**3. Run the experiments** (each runs 3 trials into `results/<tag>/trial_k/`)

```bash
export PROVIDER="--api_provider deepseek --generator_model deepseek-chat --reflector_model deepseek-chat --curator_model deepseek-chat"

# E1  Base LLM, full Formula test split (~30 s per trial, ~$0.03)
.venv/bin/python scripts/run_trials.py --tag formula_base --trials 3 -- \
    --task_name formula --mode eval_only $PROVIDER

# E2  ACE offline: train 200 / val 100 / test 200 (~50 min per trial, ~$0.67)
.venv/bin/python scripts/run_trials.py --tag formula_ace_offline --trials 3 -- \
    --task_name formula --mode offline $PROVIDER --max_train 200 --max_val 100 --eval_steps 50

# Modification  ACE offline without the Reflector (~26 min per trial, ~$0.53)
.venv/bin/python scripts/run_trials.py --tag formula_ace_offline_noref --trials 3 -- \
    --task_name formula --mode offline $PROVIDER --max_train 200 --max_val 100 --eval_steps 50 --no_reflector
```

**4. Summarize** (mean ± std, questions scored, served model, tokens, cost)

```bash
.venv/bin/python scripts/summarize.py results/formula_base results/formula_ace_offline results/formula_ace_offline_noref
```

Tip: offline training runs one question at a time. To finish faster, run the 3 trials in parallel,
one process each, with `--trials 1 --start 1`, `--start 2` and `--start 3`.

## What I changed

All changes are on branch `repro`, one commit per change (`git log 82709de..repro`).
With no new flags set, the code behaves exactly as upstream.

| Change | Where |
|---|---|
| `deepseek` API provider (OpenAI-compatible; base URL overridable via `DEEPSEEK_BASE_URL`) | `utils.py`, `eval/finance/run.py` |
| Log served model, system fingerprint and cache-hit tokens on every LLM call | `llm.py` |
| Seeded subsets: `--max_train`, `--max_val`, `--max_test`, `--subset_seed` | `eval/finance/run.py` |
| **Modification:** `--no_reflector`. The Curator gets the raw attempt and label; no Reflector, no retry rounds | `ace/ace.py`, `eval/finance/run.py` |
| Multi-trial runner | `scripts/run_trials.py` |
| Results summarizer, with per-call cost at DeepSeek peak/off-peak rates | `scripts/summarize.py` |

## Full write-up

[**REPRODUCTION.md**](REPRODUCTION.md) has the details:

- the target claim and metric definition
- setup and the model-swap table
- hidden assumptions found in the code
- per-trial results and error analysis
- the modification and cost breakdown
- conclusions and the debug diary

## Original project

The original ACE README is kept in [README_ACE.md](README_ACE.md). ACE is by Zhang et al.
and released under the Apache 2.0 license ([LICENSE.txt](LICENSE.txt)).

```bibtex
@misc{zhang2025agenticcontextengineeringevolving,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Qizheng Zhang and Changran Hu and Shubhangi Upasani and Boyuan Ma and Fenglu Hong and Vamsidhar Kamanuru and Jay Rainton and Chen Wu and Mengmeng Ji and Hanchen Li and Urmish Thakker and James Zou and Kunle Olukotun},
  year={2025}, eprint={2510.04618}, archivePrefix={arXiv}, primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.04618}
}
```
