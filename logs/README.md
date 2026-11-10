# Logs

Outputs of the commands in the top-level README, run on one machine (Ubuntu 24.04, 4 CPUs, Airbyte 2.5 in
low-resource mode, a Snowflake trial account, Tinker) at the commit named in `commit.txt`. The machine's home
directory, the Snowflake account and the identifiers of the Tinker runs are replaced by placeholders, and the
request logging of the Snowflake and HTTP clients is removed.

A `*.txt` file is what the command printed. A `*.jsonl` file is the transcript that the rollout command writes:
one record per assistant turn with its tool calls and their results, then one summary record per rollout.

## 1. Credential-free local integration test

| File | Command | Result |
| --- | --- | --- |
| `1_local/pytest.txt` | `uv run pytest -v` | 442 passed |
| `1_local/rollout_reference_storefront.*` | `uv run python -m elt_rlvr.rollout task=storefront policy=reference` | reward 1.0 |

## 2. Rollouts on official ELT-Bench tasks

All through Airbyte and Snowflake: `uv run python -m elt_rlvr.rollout task=<task> policy=<policy>
elt_bench_root=/path/to/ELT-Bench warehouse=snowflake`, with `model_name=Qwen/Qwen3.6-35B-A3B` for `policy=tinker`.

| File | Task, policy | Result |
| --- | --- | --- |
| `2_official/rollout_reference_human_resources.*` | `human_resources`, `reference` | reward 1.0 |
| `2_official/rollout_reference_trains.*` | `trains`, `reference` | reward 1.0 |
| `2_official/rollout_reference_app_store.*` | `app_store`, `reference` | reward 1.0 |
| `2_official/rollout_tinker_trains.*` | `trains`, `tinker`, `group_size=2` | rewards 1.0 and 0.9 (one of seven columns wrong) |
| `2_official/rollout_tinker_human_resources.*` | `human_resources`, `tinker` | reward 0.91 (every load right, 14 of 16 columns right) |

A reference rollout earns 1.0 only if the loaded rows equal the reference load, the models reproduce the
benchmark's ground truth, and they agree with the reference SQL on two held-out samples. In these runs the syncs
went through Airbyte (five minutes for the three connections of `human_resources`).

`2_official/earlier/rollout_tinker_human_resources_b6dfd96.jsonl` is a rollout from commit b6dfd96, when
`run_extract_load` still synced every connection on every call and the default reward was the product
`EL * (0.3 + 0.7 * T)`. The model loaded each table more than once (extract-load score 0) and reached a
transform score of 0.25, which the product turned into reward 0. docs/DESIGN.md cites it.

## 3. Tinker RL optimization step

`uv run python -m elt_rlvr.train tasks=storefront,clinic group_size=4 groups_per_batch=2 max_steps=1`

| File | Content |
| --- | --- |
| `3_training/train_tail.txt` | The last lines the command printed: the metric table of the step. |
| `3_training/metrics.jsonl` | The same metrics as JSON. |
| `3_training/train_rollout_summaries.jsonl` | One record per rollout. Rewards: `storefront` 1.0, 0.94, 0.94, 0.88; `clinic` 1.0, 1.0, 0.95, 0.95. |
| `3_training/config.json`, `checkpoints.jsonl` | The configuration of the run and the checkpoint saved after the step. |

## Latency

| File | Command |
| --- | --- |
| `latency/latency_local.json` | `uv run python -m elt_rlvr.bench --task storefront --scale 50 --group-size 8` |
| `latency/latency_official.json` | `uv run python -m elt_rlvr.bench --task human_resources --elt-bench-root /path/to/ELT-Bench --warehouse snowflake --group-size 3 --direct-rollouts 1` |
