# ELT-Bench as a Tinker RLVR environment: implementation design

Repository: https://github.com/XiaoyuanWang111/elt-bench-rlvr. Numbers come from runs whose commands and outputs
are in its `logs/`. ELT-Bench is used at commit `fcf3129` (branch `elt_bench++`).

## 1. The ELT task

**Components.** Sources of up to five kinds (Postgres, MongoDB, REST API, S3, flat files), a destination
warehouse, and a workspace: connection values (`config.yaml`), the tables to build with a description of every
column (`data_model.yaml`), source schemas and Airbyte documentation.

**Prompt and tools.** The system prompt lists the workspace files, the workflow and what is graded. The benchmark
gives its agents a shell; here the model gets seven typed tools (`list_files`, `read_file`, `write_file`,
`run_extract_load`, `run_transform`, `query`, `submit`). It writes `pipeline/extract_load.yaml` and one SQL file
per data model, runs them, reads the errors and row counts that come back, and submits. On official tasks the
connector configuration is the `configuration` block of the Airbyte Terraform resource, so the benchmark's
documentation applies unchanged. To close loopholes (section 4) the environment does what a benchmark agent does
with a shell and the warehouse credentials: it creates the destination and the connection, applies the
Terraform, runs the sync, and materializes each SQL file as a table. There is no dbt project.

**Termination.** `submit`, or a limit (30 turns, 60 tool calls, 60,000 context tokens, 30 minutes). The reward is
computed from state, not from how the episode ended.

**State.** Stateful: files and warehouse tables change. An append sync adds its rows every time it runs, so a
table appended twice stays wrong until it is overwritten, and `run_extract_load` skips a stream whose table
already holds what that stream loads. Episodes are independent: each rollout has its own workspace and warehouse
namespace, sources are read-only, and rollouts of one task start from the same state and run in parallel, as
GRPO-style groups need.

## 2. Reward

**Correct pipeline.** It loads every source table with exactly the source's rows and builds every data model as
the correct function of the source data: right on today's data and still right when the data changes.

**Verification**, by execution in the warehouse:

- *Extract-load score* `EL`: the share of source tables whose loaded rows equal a reference load (an
  order-insensitive digest of all rows; the benchmark compares row counts). Loads are graded as the rollout left
  them, and count only if the submitted file still describes the streams that loaded them.
- *Transform score* `T`: the mean over data models of the share of required columns that are right. The grader
  re-runs the submitted SQL in a fresh namespace and compares each table with the expected one column by column,
  on the tables the pipeline loaded (against the benchmark's ground truth) and on held-out samples of the source
  tables (against reference SQL). A column counts only if it matches everywhere; a table that fails to build or
  has another number of rows loses all its columns.

A held-out sample keeps a random 70% of each table's rows, consistent with the foreign keys. It then takes all
child rows away from one parent row per key, which tells an outer join from an inner one, and moves the key
values of each parent table to other rows, which defeats values pasted per key. It is stricter than the
benchmark on purpose: it grades cases the benchmark's data lacks, by the reference pipeline's reading of the
column descriptions. Two rollouts sampled from Qwen3.6-35B-A3B match the ground truth in every column, one with
an inner join that drops a train without cars (`trains`), one with NULL replaced by 0 in two columns whose
description does not ask for it (`human_resources`). Under three seeds they earn 0.3 and 0.91 each time; with
random rows only, they earned 1.0 under some seeds.

`reward = 0.3 * EL + 0.7 * T`. The weights put most of the reward on the transformations, where the benchmark's
agents fail most, and keep a correct load above zero. `T` needs no gate on `EL`: columns are graded on the
tables the pipeline itself loaded, so a wrong load already costs the columns that depend on it. A shape that
gates `T` on `EL` as the benchmark's two stages do, `EL * (0.3 + 0.7 * T)`, is an option that discards signal:
it pays 0 to an earlier sampled rollout that loaded its tables more than once (`EL` 0) with `T` 0.25, where the
sum pays 0.175.

**Outcome or process reward.** A graded outcome reward. tinker-cookbook centres the reward within a group, so a
group whose rollouts all score the same teaches nothing; graded rewards differ where pass-or-fail scores tie.
The logged optimization steps had groups (1.0, 0.94, 0.94, 1.0) and (1.0, 1.0, 0.89, 0.95) on two local tasks
and (0.0, 0.9) on the official task `trains`, whose other group, (0.9, 0.9), was dropped for that reason.
Milestones that execution can verify (a table loaded, a column right) are already in the outcome reward, and
paying them earlier would not change a per-trajectory advantage. A bonus for an event such as "a sync
succeeded" can be collected by repeating the event, so there is none.

## 3. Implementation

A tinker-cookbook recipe: `ELTEnvGroupBuilder` builds one group of rollouts per task with `build_agent_tool_env`
and grades it in `compute_group_rewards`; `ELTDatasetBuilder` feeds `rl.train.main`. Infrastructure sits behind
two interfaces, `Warehouse` (DuckDB, Snowflake) and `ExtractLoadEngine` (`LocalEngine`, `AirbyteEngine`); a new
destination is a `Warehouse` subclass and one entry for its Airbyte resource.

The credential-free path runs three generated tasks laid out like ELT-Bench tasks; tests drive it through the
cookbook's rollout loop with a scripted policy that emits tokens. The official path was run on Airbyte 2.5 and
Snowflake: the loader reads all 100 tasks, and on three tasks that cover the five source kinds a reference
pipeline earns 1.0 against the benchmark's ground truth. One benchmark defect surfaced: ELT-Bench's REST API
server ignores the page parameters, and its Airbyte connector asks for pages until one has fewer than 9,973
rows, so a sync of a larger table did not end (40 API tables in 30 tasks are that large). The logged `app_store`
run uses the connector without its paginator (README).

## 4. Reward hacking

The policy controls only tool arguments. Each row has a test.

| Loophole | Closed by |
| --- | --- |
| Hard-code the expected table, or paste looked-up values into SQL | Held-out data, on tasks with a reference pipeline: constants score 0 after the held-out check, and a value pasted per key is wrong once the keys have moved. |
| Leave correct tables behind and submit other files; fabricate a source table; append a stream twice | The grader runs the submitted SQL. A load counts by content digest, only if the engine wrote it and the submitted file describes it. |
| Load the public ground truth as a source; read grader files or other rollouts' tables | A source location must be part of a value in the task's `config.yaml`. File tools stay in the workspace. SQL is parsed, limited to the rollout's tables and re-emitted; `TABLE(...)`, qualified names and catalog functions are refused. |
| Use warehouse credentials, or have Terraform read files of the host | Secrets reach the model as placeholders. Template syntax is escaped in values, refused in keys. |
| Exploit the 1% tolerance, break the grader, exhaust resources | Integers and text must be equal; a grader error gives 0, a failed held-out draw gives the whole group 0; timeouts, a sync budget, no fetch of an oversized table. |

Known limits. Held-out grading and the content check need a reference pipeline per task; three official tasks
have one. On the other 97, loads are checked by row count and models against the public ground truth, so
constants, pasted values and a wrong load with the right row count are paid. Both checks assume the table
layout of the reference load. A sample changes rows and keys, not other values, so a value pasted per row of a
table that no foreign key refers to survives. Isolation rests on the SQL guard, not on a warehouse role per
rollout. A column description with two readings can fail a reasonable solution.

## 5. Efficiency

| Mean seconds | Official task (Airbyte, Snowflake) | Local task (DuckDB, 25,000 rows) |
| --- | --- | --- |
| `run_extract_load`, synced by the engine | 290 (three connections, one run) | 0.099 |
| `run_extract_load`, served by the sync cache | 7.4 | 0.028 |
| `run_transform` | 3.1 | 0.018 |
| `query` | 0.2 | 0.001 |
| Grading, per rollout (groups of 3 and 8) | 18.8 | 0.38 |

- *Sync cache.* A sync depends only on the connection configuration while sources are fixed. Each distinct
  configuration is synced once into a snapshot that persists across runs; later syncs copy it inside the
  warehouse (a zero-copy clone on Snowflake). A failed sync is not kept.
- *Grading.* The reference load is built once per task and kept (5 s to reuse; building it is one load without
  the cache). Held-out samples are drawn once per group, and only if some rollout has a column right; drawing
  them is most of the grading time, and a rollout graded alone takes 37 s.
- *Transitions.* No dbt or shell start-up per call. What remains is Airbyte, about 100 s per connection, paid
  once per distinct configuration. Next: sync the connections of one call at once instead of in turn.
