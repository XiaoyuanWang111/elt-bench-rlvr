# Logs

Outputs of the commands in the top-level README, run on one machine (Ubuntu 24.04, 4 CPUs, Airbyte 2.5 in
low-resource mode, a Snowflake trial account, Tinker) at the commit named in `commit.txt`. Later commits add
`scripts/regrade_transcript.py`, the script that wrote the regrade files, and change only documentation and
these logs. The machine's home directory, the Snowflake account, and the identifiers of
Tinker runs and Snowflake queries are replaced by placeholders; terminal colour codes and the request logging of
the Snowflake and HTTP clients are removed. Every command was given `log_path=runs/acceptance3/...` to choose
where its transcript is written.

A `*.txt` file is what the command printed. A `*.jsonl` file is the transcript that the rollout command writes:
one record per assistant turn with its tool calls and their results, then one summary record per rollout.

## 1. Credential-free local integration test

| File | Command | Result |
| --- | --- | --- |
| `1_local/pytest.txt` | `uv run pytest -v` | 472 passed |
| `1_local/rollout_reference_storefront.*` | `uv run python -m elt_rlvr.rollout task=storefront policy=reference` | reward 1.0 |

## 2. Rollouts on official ELT-Bench tasks

All through Airbyte and Snowflake: `uv run python -m elt_rlvr.rollout task=<task> policy=<policy>
elt_bench_root=/path/to/ELT-Bench warehouse=snowflake`, with `model_name=Qwen/Qwen3.6-35B-A3B` for `policy=tinker`.
The sync cache was emptied before the run, so the reference rollouts wait for real Airbyte syncs (five minutes
for the three connections of `human_resources`).

| File | Task, policy | Result |
| --- | --- | --- |
| `2_official/rollout_reference_human_resources.*` | `human_resources`, `reference` | reward 1.0 |
| `2_official/rollout_reference_trains.*` | `trains`, `reference` | reward 1.0 |
| `2_official/rollout_reference_app_store.*` | `app_store`, `reference` | reward 1.0 |
| `2_official/rollout_tinker_trains.*` | `trains`, `tinker`, `group_size=2` | rewards 1.0 and 1.0 |
| `2_official/rollout_tinker_human_resources.*` | `human_resources`, `tinker` | reward 0.61: every load right, 7 of 16 columns right; the episode reached its 30 turns before the `positions` model built |
| `2_official/survey_tasks.txt` | `uv run python scripts/survey_tasks.py /path/to/ELT-Bench` | the loader reads all 100 tasks; 40 REST API tables in 30 tasks have at least 9,973 rows |

A reference rollout earns 1.0 only if the loaded rows equal the reference load, the models reproduce the
benchmark's ground truth, and they agree with the reference SQL on two held-out samples.

`2_official/earlier/` holds rollouts sampled at earlier commits that the documents cite:

- `rollout_tinker_trains_9381e7c.*` and `rollout_tinker_human_resources_9381e7c.*`. Rollout 1 on `trains` and the
  rollout on `human_resources` match the benchmark's ground truth in every column (`official_transform` 1.0). The
  first joins trains to cars with an inner join; the second replaces NULL with 0 in `per_salary_over_20000` and
  `avg_salary`, whose descriptions do not ask for it. At that commit a held-out sample was a random sample of
  rows only, and whether it held a train without cars or a position without employees depended on the seed.
- `regrade_9381e7c.txt`: the files of these two rollouts graded again with the code of commit 9381e7c under the
  held-out seeds 7, 11 and 13, with `uv run python scripts/regrade_transcript.py /path/to/ELT-Bench <task>
  <transcript> <rollout> 7 11 13`. The `trains` files earn 1.0, 0.3 and 0.3; the `human_resources` files earn
  1.0 under all three (the sample of their own run held a position without employees, and they earned 0.91).
- `regrade.txt`: the same command at the commit in `commit.txt`. Every sample now has a parent row without child
  rows, and the rewards are 0.3 and 0.91 under every seed.
- `rollout_tinker_human_resources_b6dfd96.jsonl`: from commit b6dfd96, when `run_extract_load` still synced every
  connection on every call and the default reward was the product `EL * (0.3 + 0.7 * T)`. The model loaded each
  table more than once (extract-load score 0) and reached a transform score of 0.25, which the product turned
  into reward 0.

## 3. Tinker RL optimization step

| Directory | Command | Rewards |
| --- | --- | --- |
| `3_training/local_tasks/` | `uv run python -m elt_rlvr.train tasks=storefront,clinic group_size=4 groups_per_batch=2 max_steps=1` | `storefront` 0.94, 0.94, 1.0, 0.94; `clinic` 1.0, 0.95, 0.95, 1.0 |
| `3_training/official_task/` | `uv run python -m elt_rlvr.train tasks=trains elt_bench_root=/path/to/ELT-Bench warehouse=snowflake group_size=2 groups_per_batch=2 max_steps=1` | `trains` 0.8, 1.0 and 0.7, 1.0 |

In both steps each group has rollouts with different rewards, so both groups enter the optimizer step. In the
second, one task fills both groups of the batch and its four rollouts run at the same time.

Each directory holds `train_tail.txt` (the last lines the command printed: the metric table of the step),
`metrics.jsonl` (the same metrics as JSON), `train_rollout_summaries.jsonl` (one record per rollout),
`config.json` (the configuration of the run) and `checkpoints.jsonl` (the checkpoint saved after the step).

## Latency

| File | Command |
| --- | --- |
| `latency/latency_local.json` | `uv run python -m elt_rlvr.bench --task storefront --scale 50 --group-size 8` |
| `latency/latency_official.json` | `uv run python -m elt_rlvr.bench --task human_resources --elt-bench-root /path/to/ELT-Bench --warehouse snowflake --group-size 3 --direct-rollouts 1` |
