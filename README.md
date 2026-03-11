# dfm-evals

`dfm-evals` is a small eval toolkit for `inspect_ai` and EuroEval workflows:

- local task registries
- packaged multi-task suites
- pairwise tournament runs and the tournament viewer
- Every Eval Ever export from Inspect and EuroEval outputs
- Prime sandbox integration
- LUMI launcher helpers under [`lumi/`](lumi/)

## What Is Here

Local tasks currently exposed through the `dfm_evals` registry include:

- `dfm_evals/multi_wiki_qa`
- `dfm_evals/bfcl-v1`
- `dfm_evals/bfcl-v1-da`
- `dfm_evals/ifeval-da`
- `dfm_evals/piqa`

Packaged suites live in [`dfm_evals/eval-sets.yaml`](dfm_evals/eval-sets.yaml).
Tournament example configs live under [`configs/tournaments/`](configs/tournaments/).
The default Prime sandbox config lives at
[`configs/sandboxes/prime-sandbox.yaml`](configs/sandboxes/prime-sandbox.yaml).

## Install

Base install:

```bash
uv sync
```

Development tools:

```bash
uv sync --group dev
```

Optional extras:

Harbor tasks require `inspect-harbor`, which is currently Python 3.12-only:

```bash
uv sync --extra harbor
```

Compose-aware cloud sandboxes such as Modal and Daytona come from
`inspect_sandboxes`, which is also Python 3.12-only:

```bash
uv sync --extra sandboxes
```

## Quick Start

List available tasks:

```bash
uv run evals tasks
```

Run local tasks:

```bash
uv run evals run dfm_evals/multi_wiki_qa --model openai/gpt-5-mini
uv run evals run dfm_evals/bfcl-v1 --model openai/gpt-5-mini
uv run evals run dfm_evals/ifeval-da --model openai/gpt-5-mini
uv run evals run dfm_evals/piqa --model openai/gpt-5-mini
```

Run a packaged suite:

```bash
uv run evals suite fundamentals \
  --target-model openai/gpt-5-mini \
  --judge-model openai/gpt-5-mini
```

Suites default to the packaged file at
[`dfm_evals/eval-sets.yaml`](dfm_evals/eval-sets.yaml). Use `--file <path>` for
custom suite definitions. Supported suite placeholders are:

- `{{target_model}}`
- `{{target_base_url}}`
- `{{judge_model}}`
- `{{judge_base_url}}`

## Tournaments

Inspect the tournament CLI:

```bash
uv run evals tournament --help
```

Example configs live under [`configs/tournaments/`](configs/tournaments/).

View a completed tournament:

```bash
uv run evals tournament view logs/evals-logs/<tournament_run_label>/state \
  --host 127.0.0.1 \
  --port 7576
```

`evals tournament view` is the main report surface for standings, head-to-heads,
prompt drilldown, and judged responses. `inspect view` is still useful for raw
generation and judge logs, but it does not understand tournament state or final
rankings.

## Every Eval Ever Export

Export Inspect logs:

```bash
uv run evals eee inspect \
  --log-path logs/evals-logs/<run_label> \
  --output-dir out/eee/data
```

Export EuroEval results:

```bash
uv run evals eee euroeval \
  --results-file /path/to/euroeval_benchmark_results.jsonl \
  --output-dir out/eee/data
```

Record inference endpoint metadata when needed:

```bash
uv run evals eee inspect \
  --log-path logs/evals-logs/<run_label> \
  --output-dir out/eee/data \
  --inference-base-url https://inference.example/v1 \
  --inference-provider-name my-provider
```

## Sandbox Providers

### Prime

Run with the built-in Prime sandbox provider:

```bash
PRIME_API_KEY=... \
uv run evals run dfm_evals/multi_wiki_qa \
  --model openai/gpt-5-mini \
  --sandbox prime
```

By default, the Prime sandbox integration will look for
[`configs/sandboxes/prime-sandbox.yaml`](configs/sandboxes/prime-sandbox.yaml)
first, then fall back to legacy root-level filenames if you still have them in
older checkouts. You can also pass an explicit config path with
`--sandbox "prime:/path/to/prime-sandbox.yaml"`.

Clean up stale Inspect-created Prime sandboxes:

```bash
uv run inspect sandbox cleanup prime
```

### Modal and Harbor

Run with Modal through `inspect_sandboxes`:

```bash
uv sync --extra sandboxes
MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \
uv run evals run dfm_evals/multi_wiki_qa \
  --model openai/gpt-5-mini \
  --sandbox modal
```

You can also pass an explicit Dockerfile or Compose file via
`--sandbox "modal:/path/to/Dockerfile"` or
`--sandbox "modal:/path/to/compose.yaml"`.

Harbor-backed tasks emit Inspect `ComposeConfig` sandbox specs from task
Dockerfiles or Compose files. The built-in `prime` sandbox does not translate
those compose specs. For Harbor tasks, use Docker or a compose-aware provider
from `inspect_sandboxes` such as Modal.

OpenThoughts-TBLite example:

```bash
uv sync --extra harbor --extra sandboxes
uv run evals suite openthoughts_tblite \
  --target-model openai/gpt-5-mini \
  -- -T sandbox_env_name=modal
```

The packaged Harbor suite defaults to `--no-fail-on-error --continue-on-fail`
so sample-level failures are logged without aborting the full run.

If you hit Modal image-build failures caused by missing Docker `COPY` sources,
see [`docs/modal-patch.md`](docs/modal-patch.md).

## Model Endpoints

For externally served vLLM or other OpenAI-compatible endpoints, configure the
server first and then point Inspect at it with `OPENAI_BASE_URL`. Example:

```bash
vllm serve ../../post \
  --served-model-name gemma3-4b-hermes \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
uv run evals suite fundamentals \
  --target-model openai/gemma3-4b-hermes \
  --judge-model openai/gemma3-4b-hermes
```

For `openai/*` model ids against a custom endpoint, set both
`OPENAI_API_KEY` and `OPENAI_BASE_URL`.

## LUMI

For Slurm, overlay, vLLM, and EuroEval workflows on LUMI, see
[`lumi/README.md`](lumi/README.md).

Useful entry points:

- `./lumi/submit.sh` for Inspect, suites, and tournaments
- `./lumi/euroeval_submit.sh` for the EuroEval launcher
- `./lumi/view.sh` for Inspect log viewing with the correct data root

Common output locations:

- Slurm logs: `logs/slurm/`
- Inspect logs: `logs/evals-logs/<run_label>/`
- Every Eval Ever exports: `logs/every_eval_ever/data/`

## Repo Layout

- [`dfm_evals/`](dfm_evals/) contains the package, task registry, scorers,
  sandboxes, and tournament modules.
- [`configs/`](configs/) contains checked-in example configs for tournaments and
  sandboxes.
- [`docs/`](docs/) contains small supporting notes such as the Modal patch
  background.
- [`lumi/`](lumi/) contains LUMI-specific launch and operations helpers.
- [`tests/`](tests/) contains the test suite.

## Development

Run lint:

```bash
uv run --group dev ruff check dfm_evals tests
```

Run tests:

```bash
uv run --group dev pytest
```
