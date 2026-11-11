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
| `run_extract_load` | Apply `extract_load.yaml`: sync the streams that have not loaded yet, or the named connections again, and report rows loaded or the error per stream. |
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

`reward = 0.3 * EL + 0.7 * T`

- `EL`: the share of required source tables whose loaded rows equal the reference load (same columns, same multiset
  of rows), not only the same row count. The tables are graded as the rollout left them, and a table counts only
  if the submitted `extract_load.yaml` still describes the stream that loaded it. Columns that a sync adds
  (`_airbyte_*`, `_ab_*`) are ignored.
- `T`: the mean over the required tables of the share of required columns that are right. The grader runs the
  submitted model files itself in a fresh namespace, on the tables the pipeline loaded. A column counts only if it
  matches on the original data and on every held-out sample of the source tables, where the expected output comes
  from the task's reference models. A sample keeps a random 70% of the rows of each table, drops the rows whose
  parent row was dropped, and then removes all child rows of one parent row per foreign key: a parent without
  children is the case that tells an outer join from an inner one, and random sampling produces it only by
  chance.

Held-out grading and the content check of loads need a reference pipeline for the task. The local tasks have one;
of the official tasks, three have one so far (`elt_rlvr/tasks/elt_bench_references/`): `human_resources`
(Postgres, S3, flat file), `trains` (REST API, flat file) and `app_store` (MongoDB, REST API). An official task
without one is graded against the benchmark's ground truth and row counts only.

Every required column weighs the same, whether the model computes it or copies it from a source table.

`RewardConfig` also offers a multiplicative shape (`EL * (0.3 + 0.7 * T)`) and a binary one, the number of held-out
samples and a per-turn cost. Metrics report the benchmark's own measures next to the reward:
`official_extract_load` (row counts, stage 1) and `official_transform` (the benchmark's value tolerance, stage 2).

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
test files cover the reward, the reward-hacking cases listed below, the official-task loader, and each module on its
own (SQL guard, pipeline files, transforms, engines, sync cache, warehouse, table comparison, session).

The second command plays the reference pipeline of a local task and prints one JSON line per rollout: the reward,
the metrics, and every source table and required column with whether it counted. It writes the transcript to
`runs/rollout.jsonl` (`log_path=` changes the file).

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
`elt_rlvr/tasks/elt_bench_references/` (`human_resources`, `trains`, `app_store`), which checks the whole stack
without sampling. Any of the 100 tasks can be named in `task=`.

Settings read from the environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TINKER_API_KEY` | | Tinker credentials. |
| `ELT_RLVR_AIRBYTE_API_URL` | `http://localhost:8000/api/public/v1` | Airbyte's public API as reachable from this process. |
| `ELT_RLVR_AIRBYTE_PROVIDER_URL` | `server_url` of the task's `config.yaml` | The same API as reachable from where Terraform runs. |
| `ELT_RLVR_AIRBYTE_CLIENT_ID`, `ELT_RLVR_AIRBYTE_CLIENT_SECRET` | | Use an application token instead of the username and password of `config.yaml`. |
| `ELT_RLVR_AIRBYTE_DEFINITION_IDS` | | JSON object from source type (`postgres`, `mongodb`, `file`, `s3`, `custom_api`) to the connector definition to use instead of the one Airbyte ships. |
| `ELT_RLVR_SYNC_TIMEOUT_SECONDS` | `1200` | A sync job still running after this long is cancelled and reported as failed. |
| `ELT_RLVR_TERRAFORM` | `terraform` | Command that runs Terraform. |
| `ELT_RLVR_DATABASE` | `ELT_RLVR` | Snowflake database that holds the namespaces of this environment. |

No credential is stored in this repository or written to a run's logged configuration.

Signing in and reaching Airbyte:

- *Airbyte application tokens, required on Airbyte 2.x.* Airbyte 2.x accepts only bearer tokens on its API; the
  username and password that ELT-Bench's documentation passes to Terraform and `curl` are answered with 401. Set
  `ELT_RLVR_AIRBYTE_CLIENT_ID` and `ELT_RLVR_AIRBYTE_CLIENT_SECRET` to the values `abctl local credentials` prints,
  and the engine requests a token for each Terraform run and API call. Without the two variables the engine uses
  the username and password of `config.yaml`, which older Airbyte versions accept.
- *Provider URL.* Terraform runs on the machine that runs this environment. If the `server_url` in the task's
  `config.yaml` is an address that only resolves inside Airbyte's own network, set
  `ELT_RLVR_AIRBYTE_PROVIDER_URL` to the address of the public API as this machine reaches it, for example
  `http://localhost:8000/api/public/v1`.
- *Snowflake with a key pair, optional.* If `snowflake_credential.json` has a `private_key_file` entry, the
  environment and the Airbyte destination both sign in as that user with that key, and no Snowflake password is
  needed.

Notes from setting ELT-Bench up on a new machine (Ubuntu 24.04, Airbyte 2.5 installed with `abctl`, November
2026). ELT-Bench documents Terraform provider 0.6.5 from late 2024, and several of these notes come from running
it against connectors that are two years newer.

- `localstack/localstack:latest` exits without a licence token, so the S3 sources do not come up. Pinning the image
  in `elt-docker/docker-compose.yml` to `localstack/localstack:3.8.1` works.
- `mongo:8.0` refuses to start on Linux kernels 6.19 and newer (its log names SERVER-121912). `mongo:8.2` starts.
- Airbyte 2.5 asks for four CPUs per sync job. On a machine with four CPUs the job stays pending; install with
  `abctl local install --low-resource-mode`.
- The S3 source that ships with Airbyte 2.5 (4.17.0) requires a `credentials` field that provider 0.6.5 cannot
  express. Setting the S3 source definition to version 4.12.2 (February 2025) makes the documented fields work:
  `python scripts/airbyte_source_definition.py --elt-bench-root /path/to/ELT-Bench pin S3 4.12.2`.
- The MongoDB source that ships with Airbyte 2.5 (`airbyte/source-mongodb` 3.0) requires `databases` where the
  provider writes `database`. Registering `airbyte/source-mongodb-v2:1.5.19` as a custom source definition makes
  the documented fields work:
  `python scripts/airbyte_source_definition.py --elt-bench-root /path/to/ELT-Bench register "MongoDb 1.5.19" airbyte/source-mongodb-v2 1.5.19`
  prints the definition id to put into `ELT_RLVR_AIRBYTE_DEFINITION_IDS='{"mongodb": "<definition id>"}'`.
- ELT-Bench's README registers its REST API connector in Airbyte's web UI.
  `scripts/register_api_connector.py` does it through the API. If publishing fails with "No declarative manifest
  image version found in database for major version 6", Airbyte could not list the image tags on Docker Hub (it
  got 403 on this machine); adding the row for `6.33.4` to the table `declarative_manifest_image_version` of
  Airbyte's database resolved it.
- ELT-Bench's REST API server ignores `limit` and `offset` and answers every request with the whole table, and
  the connector's paginator stops only after a page of fewer than 9,973 rows. For the 40 API tables with at least
  that many rows (in 30 tasks; `scripts/survey_tasks.py` lists them) the sync reads the table again and again.
  `--single-page` of the script registers the connector without the paginator. The engine cancels any sync job
  that runs past `ELT_RLVR_SYNC_TIMEOUT_SECONDS`.

## 3. Tinker RL optimization step

```bash
export TINKER_API_KEY=...
uv run python -m elt_rlvr.train tasks=storefront,clinic group_size=4 groups_per_batch=2 max_steps=1
```

This runs tinker-cookbook's `rl.train.main` on the environment: it samples `groups_per_batch * group_size` rollouts,
grades them by execution, and takes optimizer steps on the advantages. Add `elt_bench_root=... warehouse=snowflake`
to train on official tasks, for example
`tasks=trains elt_bench_root=/path/to/ELT-Bench warehouse=snowflake group_size=2 groups_per_batch=2 max_steps=1`.
A batch larger than the task list wraps around, so one task can fill several groups of a batch.

## Results

Every command above was run on one machine (4 CPUs, Airbyte 2.5, a Snowflake trial account, Tinker). The outputs
are in [logs/](logs/README.md).

| Run | Result |
| --- | --- |
| `uv run pytest -v` | 472 passed |
| Reference pipeline on a local task (`storefront`) | reward 1.0 |
| Reference pipeline on the official tasks `human_resources`, `trains`, `app_store`, every sync through Airbyte | reward 1.0 each, against the benchmark's ground truth and on held-out data |
| Qwen3.6-35B-A3B sampled through Tinker on `trains`, two rollouts | rewards 1.0 and 1.0 |
| Qwen3.6-35B-A3B sampled through Tinker on `human_resources` | reward 0.61: every load right; the episode reached its 30 turns before `positions` built, so 7 of 16 columns count |
| One optimization step on `storefront` and `clinic`, four rollouts each | rewards (0.94, 0.94, 1.0, 0.94) and (1.0, 0.95, 0.95, 1.0); both groups enter the step |
| One optimization step on the official task `trains`, two groups of two rollouts | rewards (0.8, 1.0) and (0.7, 1.0); both groups enter the step |
| Two rollouts sampled at an earlier commit, graded again with `scripts/regrade_transcript.py` under three held-out seeds | 0.3 (`trains`: an inner join drops a train without cars) and 0.91 (`human_resources`: NULL replaced by 0 in two columns) under every seed. Both match the benchmark's ground truth in every column. |

## Reward-hacking tests

`tests/test_reward_hacking.py` and `tests/test_elt_bench.py` hold one test per way to earn reward without building
the pipeline.

| Attempt | Why it fails |
| --- | --- |
| Return the expected table as constants | Graded again on held-out samples of the source tables. |
| Paste a value looked up with `query` into the SQL | The column fails on held-out samples. |
| Leave a correct table in the warehouse and submit other files | The grader runs the submitted model files; it does not read the rollout's model tables. A loaded table counts only if the submitted extract-load file still describes its stream. |
| Build a table with the right row count instead of loading the source | Loads are compared by content, and only tables written by the engine count. |
| Sync a stream twice in an append mode | Duplicated rows fail the content comparison. |
| Point a source at a ground-truth file, a URL or another host | Sources are limited to the locations in the task's `config.yaml`. |
| Read the reference pipeline or grader files | File tools are confined to the workspace; the reference lives outside it. |
| Write outside `pipeline/` | Only the two kinds of pipeline file are writable. |
| Read another rollout's or the grader's tables | SQL is parsed, limited to the rollout's tables (a CTE name counts only inside its scope) and re-emitted with qualified names. |
| `read_csv`, `COPY`, `ATTACH`, DDL, DML, several statements | Rejected by the SQL guard; DuckDB also runs with external access off and a locked configuration. |
| Rely on the benchmark's 1% numeric tolerance | Integers and text must be equal; other numbers may differ by a relative 1e-6. |
| Terraform template in a connector field, to read a file or a variable of the host | Escaped in values and refused in keys before rendering. |
| Read a secret or the API token from an error message | Secrets appear to the model as placeholders, reach Terraform as sensitive variables, and are removed from returned errors. |
| Make the grader fail | Any grader error gives reward 0. A held-out draw that fails does so for every rollout that needs it. |
| Run away with time or output | Statement timeout, row and file limits on the tools, a budget of connection syncs, a deadline for each sync job. The grader fetches a model table only if it has the expected number of rows. |

## Latency

```bash
uv run python -m elt_rlvr.bench --task storefront --scale 50 --group-size 8
uv run python -m elt_rlvr.bench --task human_resources --elt-bench-root /path/to/ELT-Bench --warehouse snowflake \
    --group-size 3 --direct-rollouts 1
```

prints the mean seconds per tool call and per graded rollout, with and without the sync cache. The cache serves a
repeated sync of an identical connection from a snapshot inside the warehouse. Only a sync in which every stream
loaded becomes a snapshot; a failed one runs again the next time.

| Mean seconds | `human_resources` (Airbyte, Snowflake) | `storefront` (DuckDB, 25,000 source rows) |
| --- | --- | --- |
| `run_extract_load`, synced by the engine | 301 | 0.098 |
| `run_extract_load`, served by the sync cache | 6.7 | 0.031 |
| `run_transform` | 2.8 | 0.020 |
| `query` | 0.1 | 0.002 |
| Grading, per rollout | 8.7 | 0.36 |

The first row is the round without the cache: one rollout with three connections on the official task, eight
rollouts on the local one. The other rows are the round with the cache, three and eight rollouts graded as one
group. A group shares its held-out samples, so one official rollout graded alone takes 25 s.

## Adding a reference pipeline for an official task

Put `extract_load.yaml`, `models/<model>.sql` and `task.yaml` (the foreign keys between source tables, which the
held-out samples respect) into `elt_rlvr/tasks/elt_bench_references/<task>/`. `policy=reference` must then earn
reward 1.0, which shows that the reference reproduces the benchmark's ground truth. From then on the task is
graded with the content check of loads and with held-out data.

## Adding a warehouse

Subclass `Warehouse` in `elt_rlvr/warehouse/base.py` (connection, catalog lookups, bulk load, table copy and
sampling, row digest) and register it in `WAREHOUSES` in `elt_rlvr/warehouse/__init__.py`. The environment, the
tools, the SQL guard and the grader use only that interface and the dialect name. For Airbyte loads, also add an
entry to `DESTINATIONS` in `elt_rlvr/extract_load/airbyte.py`: the Terraform resource of the destination, a
function that builds its configuration from the task's `config.yaml` and ELT-Bench's credential file, and
whether the destination stores unquoted schema names in upper case.

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
scripts/            Airbyte setup helpers for ELT-Bench, survey of the official tasks, regrade of a logged rollout
tests/
docs/DESIGN.md
logs/               outputs of the commands in this file
```

## License

Apache-2.0. ELT-Bench and its data are licensed separately by their authors.
