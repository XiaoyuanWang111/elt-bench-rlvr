# ELT-Bench as a Tinker RLVR environment: implementation design

Repository: [github.com/XiaoyuanWang111/elt-bench-rlvr](https://github.com/XiaoyuanWang111/elt-bench-rlvr); the
commands and outputs behind every number are in its `logs/`. ELT-Bench is used at commit `fcf3129`.

## 1. The ELT task

**Components.** Sources of up to five kinds (Postgres, MongoDB, REST API, S3, flat files), a destination
warehouse, and a workspace: connection values (`config.yaml`), the tables to build with a description of every
column (`data_model.yaml`), source schemas and Airbyte documentation.

**Prompt and tools.** The system prompt lists the workspace files, the workflow and what is graded. In place of
the benchmark's shell the model gets seven typed tools (`list_files`, `read_file`, `write_file`,
`run_extract_load`, `run_transform`, `query`, `submit`). It writes `pipeline/extract_load.yaml`, whose connector
blocks are those of the benchmark's Airbyte documentation, and one SQL file per data model, runs them, reads the
errors and row counts, and submits. To close loopholes (section 4) the environment does what an agent would do
with a shell and the warehouse credentials: it creates the destination and connection, applies the Terraform,
runs the sync and materializes each SQL file. There is no dbt project.

**Termination.** `submit` or a limit (30 turns, 60 tool calls, 60,000 context tokens, 30 minutes). The reward
comes from state, so how the episode ended does not matter.

**State.** Stateful: files and tables change, and an append sync adds its rows each time it runs, so
`run_extract_load` skips a stream whose table already holds what that stream loads. Each rollout has its own
workspace and warehouse namespace over read-only sources, so the rollouts of a group start from the same state
and run in parallel.

## 2. Reward

**Correct pipeline.** It loads every source table with exactly the source's rows and builds every data model as
the correct function of the source data: right on today's data and still right when the data changes.

**Verification**, by execution in the warehouse:

- *Extract-load score* `EL`: the share of source tables whose rows equal a reference load (an order-insensitive
  digest; the benchmark compares row counts), counted only if the submitted file still describes the streams
  that loaded them.
- *Transform score* `T`: the mean over data models of the share of required columns that are right. The grader
  re-runs the submitted SQL on the tables the pipeline loaded (against the benchmark's ground truth) and on
  held-out samples of the sources (against reference SQL). A column counts only if it matches everywhere; a
  table that fails to build or has another number of rows loses all its columns.

A held-out sample keeps a random 70% of each table's rows, consistent with the foreign keys, takes all child rows
away from one parent row per key (an inner join then loses a row) and moves the key values of each parent table to
other rows (a value pasted per key is then wrong). Being stricter than the benchmark, it must not fail sound SQL.
The worked example ELT-Bench ships for `retails`, written by its authors, earns 1.0 under three seeds against our
reference, which was written from the column descriptions; a copy with one LEFT JOIN made an inner join matches
the ground truth and earns 0.65. The check also catches mistakes in sampled rollouts: two that match the ground
truth in every column, one with an inner join that drops trains without cars, one with NULL replaced by 0 where
the description does not ask for it, earn 0.3 and 0.91 under every seed tried.

`reward = 0.3 * EL + 0.7 * T` weights the transformations, where the benchmark's agents fail most, and keeps a
correct load above zero. `T` needs no gate on `EL`, as columns are graded on the tables the pipeline loaded.
Gating as the benchmark's stages do, `EL * (0.3 + 0.7 * T)`, is an option that discards signal: it pays 0 to an
earlier rollout that loaded its tables twice (`EL` 0) with `T` 0.25, where the sum pays 0.175.

**Outcome or process reward.** A graded outcome reward. tinker-cookbook centres rewards within a group, so a
group that scores alike teaches nothing, and graded rewards differ where pass-or-fail scores tie: the logged
optimization steps had groups (1.0, 0.94, 0.94, 1.0) and (1.0, 1.0, 0.89, 0.95) on local tasks and (0.0, 0.9)
on `trains`, whose group (0.9, 0.9) was dropped. Verifiable milestones (a table loaded, a column right) are
already in the outcome, and paying them earlier would not change a trajectory's advantage; a bonus for an event
such as a successful sync could be farmed by repeating it.

## 3. Implementation

A tinker-cookbook recipe: `ELTEnvGroupBuilder` builds a group with `build_agent_tool_env` and grades it in
`compute_group_rewards`; `ELTDatasetBuilder` feeds `rl.train.main`. Infrastructure sits behind `Warehouse`
(DuckDB, Snowflake; a new destination is a subclass and one Airbyte entry) and `ExtractLoadEngine`
(`LocalEngine`, `AirbyteEngine`). Tests run three generated tasks through the cookbook's rollout loop with a
scripted policy, credential-free. On Airbyte 2.5 and Snowflake the loader reads all 100 tasks, and four tasks
that cover the five source kinds have a reference pipeline that earns 1.0 against the ground truth. One
benchmark defect surfaced: the REST API server ignores page parameters while its connector reads pages until one
has fewer than 9,973 rows, so syncs of larger tables (40 in 30 tasks) never end; the README has a workaround.

## 4. Reward hacking

The policy controls only tool arguments. Each row has a test.

| Loophole | Closed by |
| --- | --- |
| Hard-code the output, or paste looked-up values into SQL | Held-out data (tasks with a reference): constants fail, and a value pasted per key is wrong once keys move. |
| Leave correct tables and submit other files; fabricate a source table; append twice | The grader runs the submitted SQL; a load counts by content digest, if the engine wrote it and the file describes it. |
| Load the public ground truth; read grader files or other rollouts' tables | Source locations must come from `config.yaml`. File tools stay in the workspace. SQL is parsed, limited to the rollout's tables and re-emitted; `TABLE(...)`, qualified names and catalog functions are refused. |
| Use warehouse credentials, or have Terraform read host files | Secrets reach the model as placeholders; template syntax is escaped in values, refused in keys. |
| Exploit the 1% tolerance, break the grader, exhaust resources | Exact integers and text; a grader error gives 0, a failed draw gives the group 0; timeouts, a sync budget, no fetch of an oversized table. |

Known limits. Held-out grading and the content check need a reference pipeline, which four official tasks have;
on the other 96, loads are checked by row count and models against the public ground truth, so constants and a
wrong load of the right size are paid. A value pasted per row of a table no foreign key refers to survives a
sample. Isolation rests on the SQL guard, not on a warehouse role per rollout.

## 5. Efficiency

| Mean seconds | Official task (Airbyte, Snowflake) | Local task (DuckDB, 25,000 rows) |
| --- | --- | --- |
| `run_extract_load`, synced by the engine | 266 (three connections, one run) | 0.090 |
| `run_extract_load`, served by the sync cache | 7.8 | 0.026 |
| `run_transform` | 2.6 | 0.017 |
| `query` | 0.1 | 0.001 |
| Grading, per rollout (groups of 3 and 8) | 14.9 | 0.34 |

- *Sync cache.* With fixed sources a sync depends only on its configuration, so each configuration is synced once
  into a snapshot kept across runs and later syncs clone it inside the warehouse. Failed syncs are not kept.
- *Grading.* The reference load is built once per task. Held-out samples are drawn once per group, only if some
  rollout has a column right; they are most of the grading time (38 s for a rollout graded alone).
- *Transitions.* No dbt or shell start-up per call. What remains is Airbyte, about 100 s per connection and once
  per configuration; syncing the connections of a call at once is the next step.
