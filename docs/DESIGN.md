# ELT-Bench as a Tinker RLVR environment: implementation design

Repository: `elt-bench-rlvr`. Numbers come from runs whose commands and outputs are in `logs/`.

## 1. The ELT task

**Components.** Sources of up to five kinds (Postgres, MongoDB, REST API, S3, flat files), a destination warehouse,
`config.yaml` with connection values, `data_model.yaml` with the tables to build and a description of every
column, source schemas, and Airbyte documentation. A solution loads every source table and builds every data
model with SQL.

**Prompt and tools.** The benchmark gives its agents a shell. For RL the model gets seven typed tools:
`list_files`, `read_file`, `write_file`, `run_extract_load`, `run_transform`, `query`, `submit`. It writes
`pipeline/extract_load.yaml` (connections, connector configuration, streams, sync modes) and one SQL file per data
model, runs them, reads the errors and row counts that come back, and submits. On official tasks the connector
configuration is the `configuration` block of the Airbyte Terraform resource, so the benchmark's documentation
applies unchanged. Every ELT decision stays with the model; the shell, where most ways to cheat and most of the
latency come from, is gone.

**Termination.** `submit`, a turn limit, a token budget or a wall-clock limit. The reward is computed from state,
so it does not depend on which one ended the episode.

**State.** Stateful: files, Airbyte objects and warehouse tables change, and a sync in append mode is not
idempotent (a second sync doubles the rows, so `run_extract_load` applies the file like `terraform apply` and
leaves loaded streams alone). Episodes are still independent: each rollout has its own workspace and warehouse
namespace, and sources are read-only. Rollouts of one task start from the same state and run in parallel, which
GRPO-style groups need.

## 2. Reward

**Correct pipeline.** Run from an empty namespace, it loads every source table with exactly the source's rows and
builds every data model as the correct function of the source data: right on today's data and still right when
the data changes.

**Verification**, by execution in the warehouse:

- *Extract-load score* `EL`: the share of source tables whose loaded rows equal a reference load (an
  order-insensitive digest of all rows). The benchmark compares row counts only.
- *Transform score* `T`: the share of required columns that are right. The grader re-runs the submitted SQL in a
  fresh namespace and compares each table with the expected one column by column: on the tables the pipeline
  loaded, against the benchmark's ground truth, and on held-out samples of the source tables, against the output
  of reference SQL. A column counts only if it matches everywhere. Integers and text must be equal; the
  benchmark's 1% tolerance is only a reported metric.

`reward = 0.3 * EL + 0.7 * T`. `T` needs no gate on `EL`: columns are graded on the tables the pipeline itself
loaded, so a wrong load already costs the columns that depend on it. The benchmark scores transformations only
after loading succeeds. The shape that mirrors this, `EL * (0.3 + 0.7 * T)`, is one setting, and it discards
signal: a sampled rollout that loaded every table more than once and had a transform score of 0.25 gets 0.175
from the sum and 0 from the product.

**Outcome or process reward.** A graded outcome reward. tinker-cookbook centres the total reward within a group,
so a group whose rollouts all score the same teaches nothing, and the graded reward separates rollouts that the
benchmark's all-or-nothing criterion scores alike. On the official task `human_resources` a sampled rollout with
every load right and 14 of 16 columns right earns 0.91 where the criterion gives 0; in the logged optimization
step the two groups scored (1.0, 0.94, 0.94, 0.88) and (1.0, 1.0, 0.95, 0.95). The milestones that execution can
verify (a table loaded, a column right) are already in the outcome reward, and paying them earlier would not
change a per-trajectory advantage. A bonus for an event such as "a sync succeeded" can be collected by repeating
the event, so there is none.

## 3. Implementation

A tinker-cookbook recipe: `ELTEnvGroupBuilder` builds one group of rollouts per task with `build_agent_tool_env`
and grades it in `compute_group_rewards`; `ELTDatasetBuilder` feeds `rl.train.main`. Two interfaces carry what
depends on infrastructure: `Warehouse` (DuckDB, Snowflake; a new destination is one subclass) and
`ExtractLoadEngine` (`LocalEngine`, `AirbyteEngine`).

The credential-free path runs DuckDB and `LocalEngine` on three generated tasks laid out like ELT-Bench tasks;
tests drive it through the cookbook's rollout loop with a scripted policy that emits tokens. The official path
was run on Airbyte 2.5 and Snowflake: all 100 tasks load, and on three tasks that together cover the five source
kinds a reference pipeline earns reward 1.0 against the benchmark's ground truth. Rollouts sampled from
Qwen3.6-35B-A3B through Tinker earned 1.0 and 0.9 on `trains` and 0.91 on `human_resources`. One defect of the
benchmark surfaced: its REST API connector never finishes a table of 9,973 rows or more (40 tables in 30 tasks;
details and a workaround are in the README).

## 4. Reward hacking

The policy controls only tool arguments. Each row has a test.

| Loophole | Closed by |
| --- | --- |
| Hard-code the expected table, or paste looked-up values into SQL | Held-out data: constants score 1.0 on the original data and 0 after the held-out check. |
| Leave correct tables behind and submit other SQL | The grader runs the submitted files itself. |
| Fabricate a source table with the right row count; append a stream twice | Content digest; only engine-written tables count. |
| Load the answers through the loader (the ground truth is public) | A source may name only locations in the task's `config.yaml`. |
| Read grader files or other rollouts' tables | File tools are confined to the workspace. SQL is parsed, limited to the rollout's tables and re-emitted from the syntax tree. |
| Use warehouse credentials | Secrets reach the model only as placeholders. |
| Exploit the 1% tolerance, break the grader, exhaust resources | Exact comparison; a grader error gives 0; timeouts and a sync budget. |

Known limits. Held-out grading and the content check need a reference pipeline per task. Three official tasks
have one; the rest fall back to the public ground truth and row counts, where a policy that memorized the
expected tables would be paid. A column description with two readings can fail a reasonable solution.

## 5. Efficiency

| Mean seconds | Official task (Airbyte, Snowflake) | Local task (DuckDB, 25,000 rows) |
| --- | --- | --- |
| `run_extract_load`, synced by the engine | 295 (three connections) | 0.105 |
| `run_extract_load`, served by the sync cache | 7.6 | 0.030 |
| `run_transform` | 3.2 | 0.020 |
| `query` | 0.1 | 0.002 |
| Grading, per rollout | 8.0 | 0.25 |

- *Sync cache.* A sync depends only on the connection configuration while sources are fixed. Each distinct
  configuration is synced once into a snapshot; later syncs copy it inside the warehouse (a zero-copy clone on
  Snowflake) and reproduce the same rows, duplicates and errors. Snapshots persist across runs.
- *Grading.* The reference load is built once per task and kept (5 minutes, then 4 s). Held-out samples are drawn
  once per group, and only if some rollout has a column right. Rollouts are graded in parallel.
- *Transitions.* Tools run in worker threads on one connection, with no dbt or shell start-up per call. What
  remains is Airbyte: about 100 s per connection, paid once per distinct configuration.
