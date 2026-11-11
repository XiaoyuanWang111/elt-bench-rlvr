# ELT-Bench as a Tinker RLVR environment: implementation design

Repository: `elt-bench-rlvr`. Numbers come from runs whose commands and outputs are in `logs/`.

## 1. The ELT task

**Components.** Sources of up to five kinds (Postgres, MongoDB, REST API, S3, flat files), a destination
warehouse, and a workspace: connection values (`config.yaml`), the tables to build with a description of every
column (`data_model.yaml`), source schemas and Airbyte documentation.

**Prompt and tools.** The system prompt lists the workspace files, the workflow and what is graded. The benchmark
gives its agents a shell; here the model gets seven typed tools (`list_files`, `read_file`, `write_file`,
`run_extract_load`, `run_transform`, `query`, `submit`). It writes `pipeline/extract_load.yaml` and one SQL file
per data model, runs them, reads the errors and row counts that come back, and submits. On official tasks the
connector configuration is the `configuration` block of the Airbyte Terraform resource, so the benchmark's
documentation applies unchanged. The environment does two things a benchmark agent does itself, to close
loopholes (section 4): it creates the destination, with credentials the model never sees, and it runs Terraform,
the sync and the SQL in place of a shell.

**Termination.** `submit`, or a limit (30 turns, 60 tool calls, 60,000 context tokens, 30 minutes). The reward is
computed from state, not from how the episode ended.

**State.** Stateful: files and warehouse tables change, and an append sync is not idempotent, so
`run_extract_load` applies the file as `terraform apply` would and leaves a loaded stream alone. Episodes are
independent: each rollout has its own workspace and warehouse namespace, sources are read-only, and rollouts of
one task start from the same state and run in parallel, as GRPO-style groups need.

## 2. Reward

**Correct pipeline.** It loads every source table with exactly the source's rows and builds every data model as
the correct function of the source data: right on today's data and still right when the data changes.

**Verification**, by execution in the warehouse:

- *Extract-load score* `EL`: the share of source tables whose loaded rows equal a reference load (an
  order-insensitive digest of all rows; the benchmark compares row counts). Loads are graded as the rollout left
  them, and count only if the submitted file still describes the stream that loaded them.
- *Transform score* `T`: the share of required columns that are right. The grader re-runs the submitted SQL in a
  fresh namespace and compares each table with the expected one column by column, on the tables the pipeline
  loaded (against the benchmark's ground truth) and on held-out samples of the source tables (against reference
  SQL). A column counts only if it matches everywhere.

A held-out sample keeps a random 70% of each table's rows, consistent with the foreign keys, and then takes all
child rows away from one parent row per key. Two rollouts sampled from Qwen3.6-35B-A3B show why: both match the
ground truth in every column, one with an inner join that drops a train without cars (`trains`), one with NULL
replaced by 0 in two columns whose description does not ask for it (`human_resources`). The original data has no
train without cars and no position without employees. Every sample has, so the two earn 0.3 and 0.91 under
every seed; with random rows alone they earned 1.0 under some.

`reward = 0.3 * EL + 0.7 * T`. `T` needs no gate on `EL`: columns are graded on the tables the pipeline itself
loaded, so a wrong load already costs the columns that depend on it. The benchmark's two-stage shape,
`EL * (0.3 + 0.7 * T)`, is an option that discards signal: it pays 0 to an earlier sampled rollout that loaded
every table twice (`EL` 0) with `T` 0.25, where the sum pays 0.175.

**Outcome or process reward.** A graded outcome reward. tinker-cookbook centres the reward within a group, so a
group whose rollouts all score the same teaches nothing; graded rewards differ where pass-or-fail scores tie.
The logged optimization steps had groups (0.94, 0.94, 1.0, 0.94) and (1.0, 0.95, 0.95, 1.0) on two local tasks
and (0.8, 1.0) and (0.7, 1.0) on the official task `trains`. Milestones that execution can verify (a table
loaded, a column right) are already in the outcome reward, and paying them earlier would not change a
per-trajectory advantage. A bonus for an event such as "a sync succeeded" can be collected by repeating the
event, so there is none.

## 3. Implementation

A tinker-cookbook recipe: `ELTEnvGroupBuilder` builds one group of rollouts per task with `build_agent_tool_env`
and grades it in `compute_group_rewards`; `ELTDatasetBuilder` feeds `rl.train.main`. Infrastructure sits behind
two interfaces, `Warehouse` (DuckDB, Snowflake) and `ExtractLoadEngine` (`LocalEngine`, `AirbyteEngine`); a new
destination is a `Warehouse` subclass and one entry for its Airbyte resource.

The credential-free path runs three generated tasks laid out like ELT-Bench tasks; tests drive it through the
cookbook's rollout loop with a scripted policy that emits tokens. The official path was run on Airbyte 2.5 and
Snowflake: the loader reads all 100 tasks, and on three tasks that cover the five source kinds a reference
pipeline earns 1.0 against the benchmark's ground truth. One benchmark defect surfaced: its REST API server
ignores the page parameters, so its connector never finishes a table of 9,973 rows or more (40 tables in 30
tasks; workaround in the README).

## 4. Reward hacking

The policy controls only tool arguments. Each row has a test.

| Loophole | Closed by |
| --- | --- |
| Hard-code the expected table, or paste looked-up values into SQL | Held-out data: constants score 1.0 on the original data and 0 after the held-out check. |
| Leave correct tables behind and submit other files; fabricate a source table; append a stream twice | The grader runs the submitted SQL. A load counts by content digest, only if the engine wrote it and the submitted file describes it. |
| Load the public ground truth as a source; read grader files or other rollouts' tables | Sources may name only locations in the task's `config.yaml`. File tools stay in the workspace. SQL is parsed, limited to the rollout's tables and re-emitted. |
| Use warehouse credentials, or have Terraform read files of the host | Secrets reach the model as placeholders. Template syntax is escaped in values, refused in keys. |
| Exploit the benchmark's 1% tolerance, break the grader, exhaust resources | Integers and text must be equal; a grader error gives 0; timeouts, a sync budget, no fetch of an oversized table. |

Known limits. Held-out grading and the content check need a reference pipeline per task. Three official tasks
have one; the rest fall back to the public ground truth and row counts, where a policy that memorized the
expected tables would be paid. Both checks use the table layout of the reference load, so a correct load laid
out differently (MongoDB without a schema, for example) is not credited. A column description with two readings
can fail a reasonable solution.

## 5. Efficiency

| Mean seconds | Official task (Airbyte, Snowflake) | Local task (DuckDB, 25,000 rows) |
| --- | --- | --- |
| `run_extract_load`, synced by the engine | 301 (three connections, one run) | 0.098 |
| `run_extract_load`, served by the sync cache | 6.7 | 0.031 |
| `run_transform` | 2.8 | 0.020 |
| `query` | 0.1 | 0.002 |
| Grading, per rollout (groups of 3 and 8) | 8.7 | 0.36 |

- *Sync cache.* A sync depends only on the connection configuration while sources are fixed. Each distinct
  configuration is synced once into a snapshot that persists across runs; later syncs copy it inside the
  warehouse (a zero-copy clone on Snowflake). A failed sync is not kept.
- *Grading.* The reference load is built once per task and kept: 5 minutes to build, 4 s to reuse. Held-out
  samples are drawn once per group, and only if some rollout has a column right; a rollout graded alone takes
  25 s.
- *Transitions.* No dbt or shell start-up per call. What remains is Airbyte: about 100 s per connection, paid
  once per distinct configuration.
