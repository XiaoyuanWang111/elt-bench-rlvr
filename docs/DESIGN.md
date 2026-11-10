# ELT-Bench as a Tinker RLVR environment: implementation design

Repository: `elt-bench-rlvr`. All measurements below are from the code in it.

## 1. The ELT task

**Components.** An ELT-Bench task has sources of up to five kinds (Postgres, MongoDB, REST API, S3, flat files), a
destination warehouse, `config.yaml` with connection values, `data_model.yaml` with the tables to build and a
description of every column, source schemas, and Airbyte documentation. A solution extracts and loads every source
table and builds every data model with SQL.

**Prompt and tools.** The benchmark gives its agents a shell in a container. For RL the model gets seven typed
tools: `list_files`, `read_file`, `write_file`, `run_extract_load`, `run_transform`, `query`, `submit`. The model
writes `pipeline/extract_load.yaml` (connections, connector configuration, streams, sync modes) and one
`pipeline/models/<table>.sql` per model, runs them, reads the errors and row counts that come back, and submits.
For official tasks the connector configuration is the `configuration` block of the Airbyte Terraform resource, so
the benchmark's documentation applies unchanged; the environment renders the resources, applies them and runs the
sync. A typed interface costs the model no ELT decision (which sources, which fields, which streams, which SQL) and
removes the shell, which is where most ways to cheat and most of the latency come from.

**Termination.** `submit`, a turn limit, a token budget, or a wall-clock limit. The reward does not depend on which
one ended the episode, because it is computed from state.

**State.** The environment is stateful: workspace files, Airbyte objects and warehouse tables all change, and a
sync in append mode is not idempotent. Episodes are made independent instead: each rollout gets its own workspace
and warehouse namespace, sources are read-only, and everything is dropped at the end. Rollouts of one task
therefore start from the same state and can run in parallel, which GRPO-style groups need.

## 2. Reward

**Correct pipeline.** Run from an empty namespace, the pipeline (a) loads every source table with exactly the
source's rows and (b) builds every data model as the correct function of the source data: right on today's data
and still right when the data changes.

**Verification.** Both parts are checked by execution in the warehouse.

- *Extract-load score* `EL`: the share of source tables whose loaded rows equal a reference load, compared by an
  order-insensitive digest of all rows. The benchmark's stage 1 compares row counts only.
- *Transform score* `T`: the grader re-runs the submitted model files in a fresh namespace and compares each table
  with the expected one, column by column after aligning rows on the key. It does this on the original data
  against the benchmark's ground truth, and on held-out data: random samples of the source tables that keep
  foreign keys consistent, with the expected output computed by reference SQL. A column counts only if it matches
  everywhere. Integers and text must be equal; the benchmark's 1% tolerance is kept only as a reported metric.

`reward = EL * (0.3 + 0.7 * T)`, in [0, 1]. Transformation credit is scaled by `EL` because a model built on
unverified raw data is not verified either.

**Outcome or process reward.** Outcome reward is what ELT supports well, and it can be graded rather than binary,
because the task has two natural verifiable stages and each model has several columns. That matters for training:
tinker-cookbook centres the total reward within a group, so a group whose rollouts all score 0 teaches nothing,
and the benchmark's binary criterion gives almost only such groups. Process reward is limited to what execution
can verify. The verified milestones (a table loaded, a column right) are already inside the outcome reward, and
since the advantage is computed per trajectory, paying them earlier would change nothing. Bonuses for events such
as "a sync succeeded" or "a query ran" can be collected by repeating the event, so there are none; process signals
are logged as metrics, and an optional per-turn cost is available.

## 3. Implementation

The environment is a tinker-cookbook recipe: `ELTEnvGroupBuilder` builds one group of rollouts per task with
`build_agent_tool_env`, and grades the group in `compute_group_rewards`; `ELTDatasetBuilder` feeds
`rl.train.main`. Two interfaces carry everything that depends on infrastructure:

- `Warehouse` (DuckDB, Snowflake): namespaces, vetted queries, bulk load, table copy and sampling, row digest. A
  new destination is one subclass.
- `ExtractLoadEngine` (`LocalEngine`, `AirbyteEngine`): run one connection into a namespace.

The credential-free path uses DuckDB and `LocalEngine` over three generated tasks that have the layout of an
ELT-Bench task. Tests drive it through the real rollout loop with a scripted policy that emits tokens.

## 4. Reward hacking

The policy controls only tool arguments; the tools and the grader are trusted. Each item has a test.

| Loophole | Closed by |
| --- | --- |
| Hard-code the expected table, or paste looked-up values into SQL | Held-out data. Measured: constants score 1.0 on the original data and 0 after the held-out check. |
| Leave correct tables behind, submit other SQL | The grader runs the submitted files itself. |
| Fabricate a source table with the right row count; append a stream twice | Content digest; only engine-written tables count. The row-count check accepts the fabricated table. |
| Use the loader to fetch the answers (ELT-Bench's ground truth is public) | A source may only name locations in the task's `config.yaml`. |
| Read grader files or other rollouts' tables | Workspace-confined file tools; SQL is parsed, limited to the rollout's tables, and re-emitted from the syntax tree, so the warehouse never sees unvetted text. DuckDB runs with external access off. |
| Use warehouse credentials | The destination is provisioned by the environment; secrets reach the model only as placeholders. |
| Exploit the 1% tolerance, break the grader, exhaust resources | Exact comparison; any grader error gives 0; timeouts and caps. |

Known limit: held-out grading needs reference SQL for the task, and a column description that admits two
readings can fail a reasonable solution. Sampling keeps foreign keys intact to avoid penalizing solutions that
rely on them.

## 5. Efficiency

Measured on the local backend with 25,000 source rows and a group of 8: a file or query tool call takes 1 to
3 ms, `run_transform` 36 ms, grading 0.19 s per rollout.

- *Sync cache.* A sync depends only on the connection configuration while sources are fixed. Each distinct
  configuration is synced once into a snapshot; later syncs copy the snapshot inside the warehouse (a zero-copy
  clone on Snowflake) and reproduce the same rows, duplicates and errors. Locally this cuts `run_extract_load`
  from 104 ms to 31 ms; with Airbyte it replaces a sync of minutes.
- *Grading.* The reference load and expected tables are built once per task; held-out samples are drawn once per
  group, and only if some rollout has a column right; rollouts are graded in parallel; metadata lookups are
  batched. Together these took grading from 0.49 s to 0.19 s per rollout.
- *Transitions.* Tools run in worker threads on one shared connection, with no process start per call (no dbt or
  shell start-up).
