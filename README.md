# elt-bench-rlvr

A reinforcement-learning environment with verifiable rewards for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench),
built on [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook). In each rollout a model reads a
task, configures and runs extraction and loading, writes SQL transformations, uses the execution feedback, and submits
a pipeline. The environment grades the warehouse state by execution and returns a reward for Tinker RL training.

The design is described in [docs/DESIGN.md](docs/DESIGN.md).

## How a rollout works

The model acts through seven tools. They are the same for every backend.

| Tool | What it does |
| --- | --- |
| `list_files`, `read_file` | Inspect the task: `config.yaml`, `data_model.yaml`, `schemas/`, `documentation/`. |
| `write_file` | Write the pipeline: `pipeline/extract_load.yaml` and `pipeline/models/<model>.sql`. Nothing else is writable. |
| `run_extract_load` | Run every connection of `extract_load.yaml` and report rows loaded or the error per stream. |
| `run_transform` | Run the model files in dependency order and report rows, columns or the error per model. |
| `query` | Run one read-only `SELECT` over the rollout's own tables. |
| `submit` | End the episode. |

Two layers sit behind the tools and can be replaced independently:

| Layer | Credential-free | Official ELT-Bench tasks |
| --- | --- | --- |
| Warehouse (`elt_rlvr/warehouse`) | DuckDB, in process | Snowflake |
| Extract-load engine (`elt_rlvr/extract_load`) | `LocalEngine` over SQLite, CSV, JSON Lines and document collections | `AirbyteEngine`: Terraform resources of the Airbyte provider plus a sync job |

Every rollout has its own workspace directory and warehouse namespace, so rollouts of one task run in parallel and
start from the same state.

## Reward

`reward = EL * (0.3 + 0.7 * T)`

- `EL`: the share of required source tables whose loaded rows equal the reference load (same columns, same multiset
  of rows), not only the same row count.
- `T`: the mean over the required tables of the share of required columns that are right. The grader runs the
  submitted model files itself in a fresh namespace. A column counts only if it matches on the original data and on
  every held-out sample of the source tables, where the expected output comes from the task's reference models.

`RewardConfig` also offers an additive and a binary shape, the number of held-out samples and a per-turn cost.
Metrics report the benchmark's own measures next to the reward: `official_extract_load` (row counts, stage 1) and
`official_transform` (the benchmark's value tolerance, stage 2).

## Install

```bash
uv sync --extra dev                      # credential-free path
uv sync --extra dev --extra snowflake    # adds the Snowflake driver for official tasks
```

tinker-cookbook's renderer dependency ships wheels for Linux and macOS, so run the commands below on one of those.

## 1. Credential-free local integration test

```bash
uv run pytest -v
uv run python -m elt_rlvr.rollout task=storefront policy=reference
```

`pytest` needs no credentials. The tests in `tests/test_environment.py` run complete rollouts through
tinker-cookbook's rollout loop on DuckDB: a scripted policy emits assistant turns as tokens, the renderer parses the
tool calls, the tools run, and the group is graded. They download one public tokenizer from Hugging Face. The other
test files cover the reward, the reward-hacking cases listed below, and the official-task loader.

The second command plays the reference pipeline of a local task, prints the reward and metrics as JSON and writes
the transcript to `runs/rollout.jsonl`.

The three local tasks (`library`, `storefront`, `clinic`) have the layout of an ELT-Bench task and are generated
from a seed, so they need no download.

## 2. Rollout on an official ELT-Bench task

Prerequisites, all from ELT-Bench's own README: a checkout with Airbyte and the source containers running,
`setup/destination/snowflake_credential.json` and `setup/airbyte/airbyte_credential.json` filled in,
`python setup/write_config.py --destination snowflake` run, and the Snowflake ground truth downloaded to
`ground_truth/gt_snowflake`. This repository also needs `terraform` on the PATH.

```bash
export TINKER_API_KEY=...
uv run python -m elt_rlvr.rollout task=human_resources policy=tinker \
    elt_bench_root=/path/to/ELT-Bench warehouse=snowflake model_name=Qwen/Qwen3.6-35B-A3B
```

`policy=reference` plays the reference pipeline of a task that ships one under
`elt_rlvr/tasks/elt_bench_references/`, which checks the whole stack without sampling.

Settings read from the environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TINKER_API_KEY` | | Tinker credentials. |
| `ELT_RLVR_AIRBYTE_API_URL` | `http://localhost:8000/api/public/v1` | Airbyte's public API as reachable from this process. |
| `ELT_RLVR_AIRBYTE_PROVIDER_URL` | `server_url` of the task's `config.yaml` | The same API as reachable from where Terraform runs. |
| `ELT_RLVR_AIRBYTE_CLIENT_ID`, `ELT_RLVR_AIRBYTE_CLIENT_SECRET` | | Use an application token instead of the username and password of `config.yaml`. |
| `ELT_RLVR_TERRAFORM` | `terraform` | Command that runs Terraform. |
| `ELT_RLVR_DATABASE` | `ELT_RLVR` | Snowflake database that holds the namespaces of this environment. |

No credential is stored in this repository or written to a run's logged configuration.

## 3. Tinker RL optimization step

```bash
export TINKER_API_KEY=...
uv run python -m elt_rlvr.train tasks=library,storefront group_size=4 groups_per_batch=2 max_steps=1
```

This runs tinker-cookbook's `rl.train.main` on the environment: it samples `groups_per_batch * group_size` rollouts,
grades them by execution, and takes optimizer steps on the advantages. Add `elt_bench_root=... warehouse=snowflake`
to train on official tasks.

## Reward-hacking tests

`tests/test_reward_hacking.py` and `tests/test_elt_bench.py` hold one test per way to earn reward without building
the pipeline.

| Attempt | Why it fails |
| --- | --- |
| Return the expected table as constants | Graded again on held-out samples of the source tables. |
| Paste a value looked up with `query` into the SQL | The column fails on held-out samples. |
| Leave a correct table in the warehouse and submit other files | The grader runs the submitted files; it does not read the rollout's model tables. |
| Build a table with the right row count instead of loading the source | Loads are compared by content, and only tables written by the engine count. |
| Append the same stream twice | Duplicated rows fail the content comparison. |
| Point a source at a ground-truth file, a URL or another host | Sources are limited to the locations in the task's `config.yaml`. |
| Read the reference pipeline or grader files | File tools are confined to the workspace; the reference lives outside it. |
| Write outside `pipeline/` | Only the two kinds of pipeline file are writable. |
| Read another rollout's or the grader's tables | SQL is parsed, limited to the rollout's tables and re-emitted with qualified names. |
| `read_csv`, `COPY`, `ATTACH`, DDL, DML, several statements | Rejected by the SQL guard; DuckDB also runs with external access off and a locked configuration. |
| Rely on the benchmark's 1% numeric tolerance | Integers and text must be equal; other numbers may differ by a relative 1e-6. |
| Terraform template in a connector field | Escaped before rendering. |
| Read a secret from an error message | Secrets appear to the model as placeholders and are removed from returned errors. |
| Make the grader fail | Any grader error gives reward 0. |
| Run away with time or output | Statement timeout, row and file limits, a cap on sync runs. |

## Latency

```bash
uv run python -m elt_rlvr.bench --task storefront --scale 50 --group-size 8
```

prints the mean seconds per tool call and per graded rollout, with and without the sync cache. The cache serves a
repeated sync of an identical connection from a snapshot inside the warehouse.

## Adding a warehouse

Subclass `Warehouse` in `elt_rlvr/warehouse/base.py` (connection, catalog lookups, bulk load, table copy and
sampling, row digest) and register it in `WAREHOUSES` in `elt_rlvr/warehouse/__init__.py`. The environment, the
tools, the SQL guard and the grader use only that interface and the dialect name. For Airbyte loads, also add the
destination resource to `DESTINATION_RESOURCES` in `elt_rlvr/extract_load/airbyte.py`.

## Layout

```
elt_rlvr/
  env.py            ELTEnvGroupBuilder and ELTDatasetBuilder for tinker-cookbook
  tools.py          the seven tools
  session.py        one rollout's workspace and namespace; carries out the tool calls
  sql_guard.py      vets model-written SQL
  pipeline.py       the submitted files
  transform.py      runs model files in dependency order
  warehouse/        Warehouse interface, DuckDB, Snowflake
  extract_load/     engine interface, local engine, Airbyte engine, sync cache
  grading/          table comparison, reference state, reward
  tasks/            local tasks, ELT-Bench task loader
  train.py          training entry point
  rollout.py        single group rollout
  bench.py          latency measurement
tests/
docs/DESIGN.md
```

## License

Apache-2.0. ELT-Bench and its data are licensed separately by their authors.
