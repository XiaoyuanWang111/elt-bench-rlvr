# Logs

Outputs of the commands in the top-level README, run on one machine (Ubuntu 24.04, 4 CPUs, Airbyte 2.5 in
low-resource mode, a Snowflake trial account, Tinker). The runs used a checkout with exactly the tree of the
commit named in `commit.txt`; later commits change only documentation and these logs.

What was changed in the files: the machine's home directory, the Snowflake account, and the identifiers of Tinker
runs and Snowflake queries are replaced by placeholders, and terminal colour codes and the request logging of the
Snowflake and HTTP clients are removed. Every command was given `log_path=runs/acceptance4/...` to choose where
its transcript is written. The transcripts hold rows of the benchmark's public data.

A `*.txt` file is what the command printed. A `*.jsonl` file is the transcript that the rollout command writes:
one record per assistant turn with its tool calls and their results, then one summary record per rollout. The
summary lists every source table and required column with whether it counted.

## 1. Credential-free local integration test

| File | Command | Result |
| --- | --- | --- |
| `1_local/pytest.txt` | `uv run pytest -v` | 496 passed |
| `1_local/rollout_reference_storefront.*` | `uv run python -m elt_rlvr.rollout task=storefront policy=reference` | reward 1.0 |

## 2. Rollouts on official ELT-Bench tasks

All through Airbyte and Snowflake: `uv run python -m elt_rlvr.rollout task=<task> policy=<policy>
elt_bench_root=/path/to/ELT-Bench warehouse=snowflake`, with `model_name=Qwen/Qwen3.6-35B-A3B` for `policy=tinker`.
The sync cache was emptied before the run, so the reference rollouts wait for real Airbyte syncs (five minutes
for the three connections of `human_resources`). The REST API connector of this deployment is registered with
`scripts/register_api_connector.py --single-page`; without that the 64,286-row table of `app_store` does not
finish.

| File | Task, policy | Result |
| --- | --- | --- |
| `2_official/rollout_reference_human_resources.*` | `human_resources`, `reference` | reward 1.0 |
| `2_official/rollout_reference_trains.*` | `trains`, `reference` | reward 1.0 |
| `2_official/rollout_reference_app_store.*` | `app_store`, `reference` | reward 1.0 |
| `2_official/rollout_tinker_trains.*` | `trains`, `tinker`, `group_size=2` | rewards 0.9 and 1.0. Rollout 0 matches the ground truth in every column and loses `run_east_and_has_less_than_4_cars` on held-out data: for a train without cars its count is NULL, and `NULL < 4` is not true. |
| `2_official/rollout_tinker_human_resources.*` | `human_resources`, `tinker` | reward 0.96: every load right, 15 of 16 columns right. `per_salary_over_20000` is rounded to two decimals, which the benchmark's 1% tolerance accepts (`official_transform` 1.0) and the exact comparison does not. The episode ended at its 30 turns without `submit`. |
| `2_official/survey_tasks.txt` | `uv run python scripts/survey_tasks.py /path/to/ELT-Bench` | the loader reads all 100 tasks; 40 REST API tables in 30 tasks have at least 9,973 rows |

A reference rollout earns 1.0 only if the loaded rows equal the reference load and the models reproduce the
benchmark's ground truth. Its held-out check compares the reference models with themselves and shows only that
the samples can be drawn.

`2_official/earlier/` holds rollouts sampled at earlier commits that the documents cite:

- `rollout_tinker_trains_9381e7c.*` and `rollout_tinker_human_resources_9381e7c.*`. Rollout 1 on `trains` and the
  rollout on `human_resources` match the benchmark's ground truth in every column (`official_transform` 1.0). The
  first joins trains to cars with an inner join; the second replaces NULL with 0 in `per_salary_over_20000` and
  `avg_salary`, whose descriptions do not ask for it. Rollout 0 on `trains` is correct.
- `regrade_9381e7c.txt`: the files of the first two graded again with the code of commit 9381e7c, where a
  held-out sample was a random sample of rows, under the seeds 7, 11 and 13. The `trains` files earn 1.0, 0.3
  and 0.3; the `human_resources` files earn 1.0 three times (the sample of their own run held a position without
  employees, and they earned 0.91). Its first line was printed by a one-line script that names the checkout in
  use.
- `regrade.txt`: all three graded again at the commit in `commit.txt` with `uv run python
  scripts/regrade_transcript.py /path/to/ELT-Bench <task> <transcript> <rollout> 7 11 13`. A sample now takes
  the child rows of one parent row per foreign key away and moves the parent keys to other rows. The rewards are
  0.3, 0.91 and 1.0 under each of the three seeds.
- `rollout_tinker_human_resources_b6dfd96.jsonl`: from commit b6dfd96, when `run_extract_load` still synced every
  connection on every call and the default reward was the product `EL * (0.3 + 0.7 * T)`. The model loaded each
  table more than once (extract-load score 0) and reached a transform score of 0.25, which the product turned
  into reward 0.

## 3. Tinker RL optimization step

| Directory | Command | Rewards |
| --- | --- | --- |
| `3_training/local_tasks/` | `uv run python -m elt_rlvr.train tasks=storefront,clinic group_size=4 groups_per_batch=2 max_steps=1` | `storefront` 1.0, 0.94, 0.94, 1.0; `clinic` 1.0, 1.0, 0.89, 0.95 |
| `3_training/official_task/` | `uv run python -m elt_rlvr.train tasks=trains elt_bench_root=/path/to/ELT-Bench warehouse=snowflake group_size=2 groups_per_batch=2 max_steps=1` | `trains` 0.9, 0.9 and 0.0, 0.9 |

In the first step both groups have rollouts with different rewards and enter the optimizer step. In the second,
one task fills both groups of the batch and its four rollouts run at the same time; tinker-cookbook drops the
group whose two rollouts earned the same (`cost/discarded_usd` in `metrics.jsonl`) and takes the step on the
other. The rollout with reward 0 stopped calling tools after it wrote its extract-load file.

Each directory holds `train_tail.txt` (the last two tables the command printed: the metrics of the step and its
cost), `metrics.jsonl` (the same metrics as JSON), `train_rollout_summaries.jsonl` (one record per rollout),
`config.json` (the configuration of the run) and `checkpoints.jsonl` (the checkpoint saved after the step).

## Latency

| File | Command |
| --- | --- |
| `latency/latency_local.json` | `uv run python -m elt_rlvr.bench --task storefront --scale 50 --group-size 8` |
| `latency/latency_official.json` | `uv run python -m elt_rlvr.bench --task human_resources --elt-bench-root /path/to/ELT-Bench --warehouse snowflake --group-size 3 --direct-rollouts 1` |
